#![forbid(unsafe_code)]

mod cli;
mod client;
mod daemon;
mod doctor;
mod engine;
mod error;
mod ipc;
mod lock;
mod server;
mod session;
mod state;
mod storage;

use std::io::Read;
use std::path::Path;

use clap::Parser;

use cli::{Cli, Command};
use engine::vllm::VllmEngine;
use error::Envelope;
use ipc::{ErrorPayload, Request, Response};

/// The subset of `Cli` the render/dispatch helpers need, split out so matching on
/// `cli.command` by value doesn't fight borrowing the global flags at the same time.
#[derive(Clone, Copy)]
struct Flags {
    json: bool,
    quiet: bool,
    no_color: bool,
    verbose: bool,
}

impl From<&Cli> for Flags {
    fn from(cli: &Cli) -> Self {
        Flags {
            json: cli.json,
            quiet: cli.quiet,
            no_color: cli.no_color,
            verbose: cli.verbose,
        }
    }
}

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    let runtime = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("x failed to start: {e}");
            return std::process::ExitCode::from(1);
        }
    };
    let code = runtime.block_on(dispatch(cli));
    std::process::ExitCode::from(code as u8)
}

async fn dispatch(cli: Cli) -> i32 {
    let home = client::ralph_home();
    let flags = Flags::from(&cli);
    match cli.command {
        Command::InternalDaemon => match server::run_daemon(home, VllmEngine::new).await {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("daemon error: {e}");
                1
            }
        },
        Command::Run { model, name } => run_run(&home, flags, model, name).await,
        Command::Query { session, prompt } => run_query(&home, flags, session, prompt).await,
        Command::Ps => run_ps(&home, flags).await,
        Command::Inspect { session } => run_inspect(&home, flags, session).await,
        Command::Doctor => run_doctor(&home, flags),
    }
}

fn ok_mark(cli: Flags) -> &'static str {
    if cli.no_color { "OK" } else { "\u{2713}" }
}

fn arrow(cli: Flags) -> &'static str {
    if cli.no_color { "->" } else { "\u{2192}" }
}

fn print_json<T: serde::Serialize>(value: &T) {
    println!(
        "{}",
        serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string())
    );
}

fn print_error(cli: Flags, payload: &ErrorPayload) -> i32 {
    if cli.json {
        print_json(&serde_json::json!({ "error": payload }));
    } else {
        let envelope = Envelope {
            summary: payload.summary.clone(),
            detail: payload.detail.clone(),
            next: payload.next.clone(),
        };
        eprintln!("{}", envelope.render(cli.no_color));
    }
    payload.exit_code
}

fn print_io_error(cli: Flags, e: std::io::Error) -> i32 {
    print_error(
        cli,
        &ErrorPayload {
            exit_code: 1,
            summary: "could not reach the ralph daemon".to_string(),
            detail: vec![e.to_string()],
            next: Some("try again, or check: ralph doctor".to_string()),
        },
    )
}

fn print_protocol_error(cli: Flags) -> i32 {
    print_io_error(
        cli,
        std::io::Error::other("daemon sent an unexpected response"),
    )
}

async fn connect(home: &Path, cli: Flags) -> Result<tokio::net::UnixStream, i32> {
    client::connect_or_start(home)
        .await
        .map_err(|e| print_io_error(cli, e))
}

async fn run_run(home: &Path, cli: Flags, model: String, name: Option<String>) -> i32 {
    let mut stream = match connect(home, cli).await {
        Ok(s) => s,
        Err(code) => return code,
    };
    match client::send_request(&mut stream, &Request::Run { model, name }).await {
        Ok(Response::Run(info)) => {
            if cli.json {
                print_json(&info);
            } else if cli.quiet {
                println!("{}", info.name);
            } else {
                println!("{} {} ready", ok_mark(cli), info.name);
                println!("  model: {}@{}", info.model, info.model_revision);
            }
            0
        }
        Ok(Response::Error(payload)) => print_error(cli, &payload),
        Ok(_) => print_protocol_error(cli),
        Err(e) => print_io_error(cli, e),
    }
}

async fn run_query(home: &Path, cli: Flags, session: String, prompt_arg: Option<String>) -> i32 {
    let prompt = match prompt_arg.as_deref() {
        None | Some("-") => {
            let mut buf = String::new();
            if let Err(e) = std::io::stdin().read_to_string(&mut buf) {
                return print_io_error(cli, e);
            }
            buf
        }
        Some(p) => p.to_string(),
    };
    let stream = match connect(home, cli).await {
        Ok(s) => s,
        Err(code) => return code,
    };
    match client::run_query(stream, &session, &prompt, cli.json).await {
        Ok(client::QueryResult::Outcome(outcome)) => {
            if cli.json {
                print_json(
                    &serde_json::json!({ "session": session, "text": outcome.text, "tokens": outcome.token_count }),
                );
            } else {
                println!();
            }
            i32::from(outcome.cancelled)
        }
        Ok(client::QueryResult::Failed(payload)) => print_error(cli, &payload),
        Err(e) => print_io_error(cli, e),
    }
}

async fn run_ps(home: &Path, cli: Flags) -> i32 {
    let mut stream = match connect(home, cli).await {
        Ok(s) => s,
        Err(code) => return code,
    };
    match client::send_request(&mut stream, &Request::Ps).await {
        Ok(Response::Ps(sessions)) => {
            if cli.json {
                print_json(&sessions);
            } else if cli.quiet {
                for s in &sessions {
                    println!("{}", s.name);
                }
            } else if sessions.is_empty() {
                println!(
                    "no sessions yet\n\n{} ralph run <model> --name <name>",
                    arrow(cli)
                );
            } else {
                println!(
                    "{:<16} {:<28} {:>8} {:<10} LOCATION",
                    "NAME", "MODEL", "TOKENS", "STATE"
                );
                for s in &sessions {
                    println!(
                        "{:<16} {:<28} {:>8} {:<10} {}",
                        s.name, s.model, s.token_count, s.state, s.location
                    );
                    if cli.verbose {
                        println!(
                            "  id: {}  pid: {}",
                            s.id,
                            s.pid
                                .map(|p| p.to_string())
                                .unwrap_or_else(|| "-".to_string())
                        );
                    }
                }
            }
            0
        }
        Ok(Response::Error(payload)) => print_error(cli, &payload),
        Ok(_) => print_protocol_error(cli),
        Err(e) => print_io_error(cli, e),
    }
}

async fn run_inspect(home: &Path, cli: Flags, session: String) -> i32 {
    let mut stream = match connect(home, cli).await {
        Ok(s) => s,
        Err(code) => return code,
    };
    match client::send_request(&mut stream, &Request::Inspect { session }).await {
        Ok(Response::Inspect(info)) => {
            if cli.json {
                print_json(&info);
            } else if cli.quiet {
                println!("{}", info.session.state);
            } else {
                println!("{} ({})", info.session.name, info.session.state);
                println!(
                    "  model: {}@{}",
                    info.session.model, info.session.model_revision
                );
                println!("  recoverability: {}", info.recoverability);
                println!("  fast restore: {}", info.fast_restore);
                println!("  portable state: {}", info.portable_state);
            }
            0
        }
        Ok(Response::Error(payload)) => print_error(cli, &payload),
        Ok(_) => print_protocol_error(cli),
        Err(e) => print_io_error(cli, e),
    }
}

// Runs entirely in the CLI process rather than round-tripping through the daemon: doctor
// exists to diagnose why the daemon/environment *isn't* working, so it must not depend on
// the daemon being startable, and must not create the data directory as a side effect of
// connecting to one.
fn run_doctor(home: &Path, cli: Flags) -> i32 {
    let checks = doctor::run_checks(home);
    let any_fail = checks.iter().any(|c| c.status == "fail");
    if cli.json {
        print_json(&checks);
    } else if cli.quiet {
        println!("{}", if any_fail { "fail" } else { "pass" });
    } else {
        for check in &checks {
            println!(
                "{:<5} {:<16} {}",
                check.status.to_uppercase(),
                check.name,
                check.detail
            );
        }
    }
    if any_fail { 6 } else { 0 }
}
