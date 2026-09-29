#![forbid(unsafe_code)]

mod chat;
mod checkpoint;
mod cli;
mod client;
mod commands;
mod daemon;
mod doctor;
mod engine;
mod error;
mod handoff_recv;
mod ipc;
mod lock;
mod picker;
mod portable;
mod server;
mod session;
mod state;
mod storage;
mod typo;

use clap::Parser;

use cli::{Cli, Command, Flags};
use engine::vllm::VllmEngine;

/// Rust's `println!`/`print!`/`eprintln!` macros panic on any write failure, including a
/// closed pipe (piping into `head`, a command that isn't installed, any reader that exits
/// early) — a normal, routine condition every well-behaved Unix CLI exits quietly on
/// rather than dumping a panic backtrace for. The usual fix (reset SIGPIPE to its default
/// disposition via libc) needs an `unsafe` FFI call, forbidden crate-wide here, so this
/// intercepts the resulting panic by its stable, documented message instead.
fn install_broken_pipe_handler() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let is_broken_pipe = info
            .payload()
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| info.payload().downcast_ref::<&str>().copied())
            .is_some_and(|msg| msg.contains("Broken pipe"));
        if is_broken_pipe {
            std::process::exit(0);
        }
        default_hook(info);
    }));
}

fn main() -> std::process::ExitCode {
    install_broken_pipe_handler();
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
        Command::Run { model, name } => commands::run_run(&home, flags, model, name).await,
        Command::Query { session, prompt } => {
            commands::run_query(&home, flags, session, prompt).await
        }
        Command::Chat { session } => chat::run_chat(&home, flags, session).await,
        Command::Ps => commands::run_ps(&home, flags).await,
        Command::Inspect { session } => commands::run_inspect(&home, flags, session).await,
        Command::Recover { session } => commands::run_recover(&home, flags, session).await,
        Command::Checkpoint { session } => commands::run_checkpoint(&home, flags, session).await,
        Command::Pause { session } => commands::run_pause(&home, flags, session).await,
        Command::Hibernate { session } => commands::run_hibernate(&home, flags, session).await,
        Command::Resume {
            session,
            fast_only,
            portable,
        } => commands::run_resume(&home, flags, session, fast_only, portable).await,
        Command::Export {
            session,
            output,
            force,
            with_accel,
        } => commands::run_export(&home, flags, session, output, force, with_accel).await,
        Command::Import { path, name } => commands::run_import(&home, flags, path, name).await,
        Command::Handoff {
            session,
            destination,
            name,
        } => commands::run_handoff(&home, flags, session, destination, name).await,
        Command::Drain { location, to, yes } => {
            commands::run_drain(&home, flags, location, to, yes).await
        }
        Command::Doctor => commands::run_doctor(&home, flags),
        Command::Completions { shell } => {
            cli::print_completions(shell);
            0
        }
        Command::InternalHandoffProbe => handoff_recv::run_probe(&home).await,
        Command::InternalHandoffRecv { name } => handoff_recv::run_recv(&home, name).await,
    }
}
