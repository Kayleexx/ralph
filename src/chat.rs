//! `ralph chat`: an interactive human-facing session interface. One-shot/scriptable use
//! stays on `ralph query` — this just loops it, reusing the exact same connect/stream/
//! cancel plumbing so there is no separate chat-specific protocol or architecture.
use std::io::{IsTerminal, Write};
use std::path::Path;

use crate::cli::Flags;
use crate::client;
use crate::commands::{print_error, print_io_error, print_protocol_error, style};
use crate::ipc::{Request, Response};

pub async fn run_chat(home: &Path, cli: Flags, session: String) -> i32 {
    let mut stream = match client::connect_or_start(home).await {
        Ok(s) => s,
        Err(e) => return print_io_error(cli, e),
    };
    match client::send_request(
        &mut stream,
        &Request::Inspect {
            session: session.clone(),
        },
    )
    .await
    {
        // Active, or a state `ensure_ready` below can transparently fix — anything
        // else (moving/moved/failed/not found) can't be chatted with at all, so it's
        // worth failing before the "chatting with..." banner rather than after.
        Ok(Response::Inspect(info))
            if matches!(
                info.session.state.as_str(),
                "active" | "recovering" | "paused" | "hibernated"
            ) => {}
        Ok(Response::Inspect(info)) => {
            return print_error(
                cli,
                &crate::ipc::ErrorPayload {
                    diagnostic: None,
                    exit_code: 4,
                    summary: format!("session is {}, cannot chat", info.session.state),
                    detail: vec![],
                    next: Some(format!(
                        "run: ralph run <model> --name <name> (or inspect {session})"
                    )),
                },
            );
        }
        Ok(Response::Error(payload)) => return print_error(cli, &payload),
        Ok(_) => return print_protocol_error(cli),
        Err(e) => return print_io_error(cli, e),
    }

    // Transparently recover/restore before the first prompt, rather than paying for
    // that plus the first turn's generation latency together with no feedback. A fresh
    // connection: the daemon handles exactly one request per connection, and the one
    // above already used its turn.
    let mut ready_stream = match client::connect_or_start(home).await {
        Ok(s) => s,
        Err(e) => return print_io_error(cli, e),
    };
    match client::ensure_ready(&mut ready_stream, &session, false).await {
        Ok(Response::Run(_)) => {}
        Ok(Response::Error(payload)) => return print_error(cli, &payload),
        Ok(_) => return print_protocol_error(cli),
        Err(e) => return print_io_error(cli, e),
    }

    let is_tty = std::io::stdout().is_terminal();
    if is_tty {
        println!(
            "{}\n",
            style(
                cli,
                "2",
                &format!("chatting with {session} — /exit or Ctrl-D to leave")
            )
        );
    }

    loop {
        if is_tty {
            print!("{} ", style(cli, "1;36", "you \u{203a}"));
            let _ = std::io::stdout().flush();
        }
        let Some(line) = read_line().await else {
            break; // Ctrl-D
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "/exit" {
            break;
        }

        let stream = match client::connect_or_start(home).await {
            Ok(s) => s,
            Err(e) => {
                print_io_error(cli, e);
                continue;
            }
        };
        let reply_prefix = Some(style(cli, "1;35", "ralph \u{203a} "));
        match client::run_query(stream, &session, line, false, reply_prefix).await {
            // A blank line after the response, not just a newline, so turns read as
            // distinct blocks instead of running into the next prompt.
            Ok(client::QueryResult::Outcome(_)) => println!("\n"),
            Ok(client::QueryResult::Failed(payload)) => {
                print_error(cli, &payload);
            }
            Err(e) => {
                print_io_error(cli, e);
            }
        }
    }
    0
}

/// `None` on EOF (Ctrl-D) or a read error — either way, the chat loop exits, never the
/// Ralph session itself.
async fn read_line() -> Option<String> {
    tokio::task::spawn_blocking(|| {
        let mut buf = String::new();
        match std::io::stdin().read_line(&mut buf) {
            Ok(0) => None,
            Ok(_) => Some(buf),
            Err(_) => None,
        }
    })
    .await
    .unwrap_or(None)
}
