//! CLI-side transport: resolving where ralph keeps its state, auto-starting the daemon
//! when it isn't already running, and driving the query stream (including Ctrl-C).
use std::io::Write;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use tokio::net::UnixStream;

use crate::ipc::{self, Cancel, ErrorPayload, Request, Response, ServerMessage};

const AUTOSTART_TIMEOUT: Duration = Duration::from_secs(10);
const AUTOSTART_POLL_INTERVAL: Duration = Duration::from_millis(100);

pub fn ralph_home() -> PathBuf {
    if let Ok(dir) = std::env::var("XDG_DATA_HOME") {
        return PathBuf::from(dir).join("ralph");
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home)
        .join(".local")
        .join("share")
        .join("ralph")
}

/// Connects to the daemon's socket, spawning it first if it isn't already listening.
/// Spawning here is a plain detached child process, not a full double-forked Unix
/// daemon — sufficient for a local single-user tool, and avoids any unsafe fork/setsid.
pub async fn connect_or_start(ralph_home: &Path) -> std::io::Result<UnixStream> {
    let socket_path = ralph_home.join("daemon.sock");
    if let Ok(stream) = UnixStream::connect(&socket_path).await {
        warn_if_daemon_is_stale(ralph_home);
        return Ok(stream);
    }

    let _ = std::fs::remove_file(&socket_path);
    let exe = std::env::current_exe()?;
    std::process::Command::new(exe)
        .arg("__daemon")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        // Gives the daemon its own process group so it outlives this CLI invocation even
        // when something (a shell job, a supervising harness) tears down this process's
        // whole group — a real background daemon must not die with its launcher.
        .process_group(0)
        .spawn()?;

    let deadline = std::time::Instant::now() + AUTOSTART_TIMEOUT;
    loop {
        if let Ok(stream) = UnixStream::connect(&socket_path).await {
            return Ok(stream);
        }
        if std::time::Instant::now() >= deadline {
            return Err(std::io::Error::other(
                "daemon did not become reachable in time",
            ));
        }
        tokio::time::sleep(AUTOSTART_POLL_INTERVAL).await;
    }
}

/// Best-effort, never fatal: an already-running daemon started from an older binary than
/// this CLI silently keeps running that old code indefinitely (see README's "a running
/// daemon keeps its current build" note) — that's exactly what turned an already-fixed
/// bug back into a confusing raw crash once. Reads the daemon's own recorded pid and
/// compares `/proc/<pid>/exe`'s mtime to this process's own binary; any failure (no pid
/// file, process gone, permission denied, pid reused by something unrelated) is silently
/// ignored rather than guessed at.
fn warn_if_daemon_is_stale(ralph_home: &Path) {
    let mtime = |path: &Path| std::fs::metadata(path).and_then(|m| m.modified()).ok();
    let Some(current) = std::env::current_exe().ok().as_deref().and_then(mtime) else {
        return;
    };
    let Some(pid) = std::fs::read_to_string(ralph_home.join("daemon.pid"))
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
    else {
        return;
    };
    let Some(daemon) = std::fs::read_link(format!("/proc/{pid}/exe"))
        .ok()
        .as_deref()
        .and_then(mtime)
    else {
        return;
    };
    if daemon < current {
        eprintln!(
            "warning: the running ralph daemon was started from an older build; restart it (stop the process, then run any ralph command) to pick up recent changes"
        );
    }
}

pub async fn send_request(
    stream: &mut UnixStream,
    request: &Request,
) -> std::io::Result<ipc::Response> {
    ipc::write_frame(stream, request).await?;
    match ipc::read_frame(stream).await? {
        Some(ServerMessage::Result(response)) => Ok(*response),
        _ => Err(std::io::Error::other(
            "daemon closed the connection unexpectedly",
        )),
    }
}

pub struct QueryOutcome {
    pub text: String,
    pub token_count: i64,
    pub cancelled: bool,
}

pub enum QueryResult {
    Outcome(QueryOutcome),
    /// The daemon rejected the query outright (e.g. session not active) before any
    /// generation started.
    Failed(ErrorPayload),
}

/// Streams a query's tokens, printing each one as it arrives unless `quiet` is set (JSON
/// mode buffers separately and passes quiet=true here). The first Ctrl-C cancels the
/// generation but leaves the session active; already-printed output is never rewound.
pub async fn run_query(
    stream: UnixStream,
    session: &str,
    prompt: &str,
    quiet: bool,
) -> std::io::Result<QueryResult> {
    let (mut reader, mut writer) = stream.into_split();
    ipc::write_frame(
        &mut writer,
        &Request::Query {
            session: session.to_string(),
            prompt: prompt.to_string(),
        },
    )
    .await?;

    let mut text = String::new();
    let mut cancelled = false;
    let mut ctrl_c = Box::pin(tokio::signal::ctrl_c());

    let token_count = loop {
        tokio::select! {
            frame = ipc::read_frame::<_, ServerMessage>(&mut reader) => {
                match frame? {
                    Some(ServerMessage::Chunk(chunk)) => {
                        if !quiet {
                            print!("{chunk}");
                            let _ = std::io::stdout().flush();
                        }
                        text.push_str(&chunk);
                    }
                    Some(ServerMessage::Done { token_count }) => break token_count,
                    Some(ServerMessage::Result(boxed)) => match *boxed {
                        Response::Error(payload) => return Ok(QueryResult::Failed(payload)),
                        _ => return Err(std::io::Error::other("daemon sent an unexpected message")),
                    },
                    _ => return Err(std::io::Error::other("daemon sent an unexpected message")),
                }
            }
            _ = &mut ctrl_c, if !cancelled => {
                cancelled = true;
                ipc::write_frame(&mut writer, &Cancel).await?;
                eprintln!("^C cancelled");
            }
        }
    };
    Ok(QueryResult::Outcome(QueryOutcome {
        text,
        token_count,
        cancelled,
    }))
}
