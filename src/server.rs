//! Daemon socket loop: accepts connections, dispatches each request, and streams query
//! output back over the same connection.
use std::path::PathBuf;
use std::sync::Arc;

use tokio::net::{UnixListener, UnixStream};

use crate::daemon::{Daemon, map_engine_error};
use crate::engine::Engine;
use crate::error::{self, CliError};
use crate::ipc::{self, Cancel, ErrorPayload, Response, ServerMessage};
use crate::lock::DaemonLock;
use crate::storage::Storage;

/// Runs the daemon in the foreground: acquires the single-instance lock, binds the
/// socket, reconciles state left over from a previous run, then serves forever. If
/// another daemon already holds the lock, this returns immediately — a working daemon
/// is already up, so there is nothing for this process to do.
pub async fn run_daemon<E: Engine + 'static>(
    ralph_home: PathBuf,
    make_engine: fn(PathBuf) -> E,
) -> std::io::Result<()> {
    std::fs::create_dir_all(&ralph_home)?;
    let lock_path = ralph_home.join("daemon.lock");
    let Some(_lock) = DaemonLock::acquire(&lock_path)? else {
        return Ok(());
    };

    let socket_path = ralph_home.join("daemon.sock");
    // Holding the lock means any leftover socket file is stale (from a prior crash),
    // never a live daemon — safe to remove before binding.
    let _ = std::fs::remove_file(&socket_path);

    let db_path = ralph_home.join("ralph.db");
    let storage = Storage::open(&db_path).map_err(std::io::Error::other)?;
    let daemon = Daemon::new(storage, ralph_home, make_engine);
    let _ = daemon.reconcile_on_startup();

    let listener = UnixListener::bind(&socket_path)?;
    loop {
        let (stream, _) = listener.accept().await?;
        let daemon = daemon.clone();
        tokio::spawn(async move {
            let _ = handle_connection(daemon, stream).await;
        });
    }
}

async fn handle_connection<E: Engine + 'static>(
    daemon: Arc<Daemon<E>>,
    mut stream: UnixStream,
) -> std::io::Result<()> {
    let Some(request) = ipc::read_frame(&mut stream).await? else {
        return Ok(());
    };
    match request {
        ipc::Request::Run { model, name } => {
            let result = daemon.run(model, name).await.map(Response::Run);
            send_result(&mut stream, result).await
        }
        ipc::Request::Ps => send_result(&mut stream, daemon.ps().map(Response::Ps)).await,
        ipc::Request::Inspect { session } => {
            send_result(&mut stream, daemon.inspect(&session).map(Response::Inspect)).await
        }
        ipc::Request::Query { session, prompt } => {
            handle_query(&daemon, &mut stream, &session, &prompt).await
        }
    }
}

async fn send_result(
    stream: &mut UnixStream,
    result: Result<Response, CliError>,
) -> std::io::Result<()> {
    let response = result.unwrap_or_else(|e| Response::Error(to_error_payload(&e)));
    ipc::write_frame(stream, &ServerMessage::Result(Box::new(response))).await
}

fn to_error_payload(e: &CliError) -> ErrorPayload {
    let envelope = error::envelope(e);
    ErrorPayload {
        exit_code: error::exit_code(e),
        summary: envelope.summary,
        detail: envelope.detail,
        next: envelope.next,
    }
}

async fn handle_query<E: Engine + 'static>(
    daemon: &Arc<Daemon<E>>,
    stream: &mut UnixStream,
    session: &str,
    prompt: &str,
) -> std::io::Result<()> {
    let running = match daemon.begin_query(session).await {
        Ok(r) => r,
        Err(e) => return send_result(stream, Err(e)).await,
    };

    let generation = {
        let engine = running.engine.lock().await;
        engine.generate(prompt).await
    };
    let handle = match generation {
        Ok(h) => h,
        Err(e) => return send_result(stream, Err(map_engine_error(e))).await,
    };

    let mut tokens = handle.tokens;
    let mut cancel = Some(handle.cancel);
    let mut token_count: i64 = 0;
    loop {
        tokio::select! {
            chunk = tokens.recv() => {
                match chunk {
                    Some(Ok(text)) => {
                        token_count += 1;
                        ipc::write_frame(stream, &ServerMessage::Chunk(text)).await?;
                    }
                    _ => break,
                }
            }
            cancel_msg = ipc::read_frame::<_, Cancel>(stream) => {
                if matches!(cancel_msg, Ok(Some(_))) {
                    if let Some(tx) = cancel.take() {
                        let _ = tx.send(());
                    }
                } else {
                    break;
                }
            }
        }
    }
    let _ = daemon.record_tokens(&running.session_id, token_count);
    ipc::write_frame(stream, &ServerMessage::Done { token_count }).await
}
