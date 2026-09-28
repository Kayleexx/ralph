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

mod query;
#[cfg(test)]
mod tests;
use query::handle_query;

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
    daemon
        .reconcile_on_startup()
        .map_err(std::io::Error::other)?;

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
            lifecycle(&mut stream, |cancel| {
                daemon.run_cancellable(model, name, Some(cancel))
            })
            .await
        }
        ipc::Request::Ps => send_result(&mut stream, daemon.ps().map(Response::Ps)).await,
        ipc::Request::Inspect { session } => {
            send_result(&mut stream, daemon.inspect(&session).map(Response::Inspect)).await
        }
        ipc::Request::Query { session, prompt } => {
            handle_query(&daemon, &mut stream, &session, &prompt).await
        }
        ipc::Request::Recover { session } => {
            lifecycle(&mut stream, |cancel| {
                daemon.recover_cancellable(&session, Some(cancel))
            })
            .await
        }
    }
}

async fn send_result<W: tokio::io::AsyncWrite + Unpin>(
    stream: &mut W,
    result: Result<Response, CliError>,
) -> std::io::Result<()> {
    let response = result.unwrap_or_else(|e| Response::Error(to_error_payload(&e)));
    ipc::write_frame(stream, &ServerMessage::Result(Box::new(response))).await
}

fn to_error_payload(e: &CliError) -> ErrorPayload {
    let envelope = error::envelope(e);
    ErrorPayload {
        diagnostic: e.diagnostic().map(str::to_string),
        exit_code: error::exit_code(e),
        summary: envelope.summary,
        detail: envelope.detail,
        next: envelope.next,
    }
}

async fn lifecycle<F: std::future::Future<Output = Result<crate::ipc::SessionInfo, CliError>>>(
    stream: &mut UnixStream,
    operation: impl FnOnce(tokio::sync::oneshot::Receiver<()>) -> F,
) -> std::io::Result<()> {
    let (mut reader, mut writer) = stream.split();
    let (cancel_tx, cancel_rx) = tokio::sync::oneshot::channel();
    let operation = operation(cancel_rx);
    tokio::pin!(operation);
    let result = tokio::select! {
        result=&mut operation => result,
        _=ipc::read_frame::<_,Cancel>(&mut reader) => {
            // The receiver may already have completed; cancellation then has no effect.
            let _=cancel_tx.send(());
            operation.await
        }
    };
    send_result(&mut writer, result.map(Response::Run)).await
}
