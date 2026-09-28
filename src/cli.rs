//! Command-line surface. Only run/query/ps/inspect/doctor/chat are registered — an
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
    /// Start a model and create a session for it. With no model and an interactive
    /// terminal, shows a small picker instead of failing.
    Run {
        model: Option<String>,
        #[arg(long)]
        name: Option<String>,
    },
    /// Send a prompt to a session and stream the response.
    Query {
        session: String,
        /// Read from stdin when omitted or "-".
        prompt: Option<String>,
    },
    /// Interactive human-facing session interface.
    Chat { session: String },
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

/// The subset of `Cli` the render/dispatch helpers need, split out so matching on
/// `cli.command` by value doesn't fight borrowing the global flags at the same time.
#[derive(Clone, Copy)]
pub struct Flags {
    pub json: bool,
    pub quiet: bool,
    pub no_color: bool,
    pub verbose: bool,
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
