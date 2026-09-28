//! Command-line surface. Only run/query/ps/inspect/doctor are registered — an
//! unregistered subcommand produces clap's own honest "unrecognized subcommand" error
//! rather than a hand-written "not implemented yet" message.
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "ralph", version)]
pub struct Cli {
    #[arg(long, global = true)]
    pub json: bool,
    #[arg(long, global = true)]
    pub quiet: bool,
    #[arg(long, global = true)]
    pub no_color: bool,
    #[arg(short, long, global = true)]
    pub verbose: bool,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Start a model and create a session for it.
    Run {
        model: String,
        #[arg(long)]
        name: Option<String>,
    },
    /// Send a prompt to a session and stream the response.
    Query {
        session: String,
        /// Read from stdin when omitted or "-".
        prompt: Option<String>,
    },
    /// List sessions.
    Ps,
    /// Show a session's lifecycle and recoverability.
    Inspect { session: String },
    /// Check the local environment.
    Doctor,
    /// Internal: run the daemon loop in the foreground. Not part of the public surface.
    #[command(hide = true, name = "__daemon")]
    InternalDaemon,
}
