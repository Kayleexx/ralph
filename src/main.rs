#![forbid(unsafe_code)]

mod chat;
mod cli;
mod client;
mod commands;
mod daemon;
mod doctor;
mod engine;
mod error;
mod ipc;
mod lock;
mod picker;
mod server;
mod session;
mod state;
mod storage;
mod typo;

use clap::Parser;

use cli::{Cli, Command, Flags};
use engine::vllm::VllmEngine;

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
        Command::Run { model, name } => commands::run_run(&home, flags, model, name).await,
        Command::Query { session, prompt } => {
            commands::run_query(&home, flags, session, prompt).await
        }
        Command::Chat { session } => chat::run_chat(&home, flags, session).await,
        Command::Ps => commands::run_ps(&home, flags).await,
        Command::Inspect { session } => commands::run_inspect(&home, flags, session).await,
        Command::Recover { session } => commands::run_recover(&home, flags, session).await,
        Command::Doctor => commands::run_doctor(&home, flags),
    }
}
