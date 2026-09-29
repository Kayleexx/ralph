//! Command-line surface. Only run/query/ps/inspect/doctor/chat are registered — an
//! unregistered subcommand produces clap's own honest "unrecognized subcommand" error
//! rather than a hand-written "not implemented yet" message.
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "ralph",
    version,
    about = "A continuity runtime for stateful LLM inference."
)]
pub struct Cli {
    /// Machine-readable output on stdout only; no human decoration, stable field names.
    #[arg(long, global = true)]
    pub json: bool,
    /// Print only the essential result (e.g. just the session name or state).
    #[arg(long, global = true)]
    pub quiet: bool,
    /// Never emit color/unicode glyphs, even on a terminal that supports them.
    #[arg(long, global = true)]
    pub no_color: bool,
    /// Show internal details (worker id, paths) — never prompt content.
    #[arg(short, long, global = true)]
    pub verbose: bool,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Start a model and create a session for it (no model = interactive picker)
    Run {
        /// Model id vLLM can serve (e.g. Qwen/Qwen2.5-0.5B-Instruct). Omit for the
        /// interactive picker on a real terminal.
        model: Option<String>,
        /// Session name. Omit to get a generated (or picker-suggested) one.
        #[arg(long)]
        name: Option<String>,
    },
    /// Send a prompt to a session and stream the response (one-shot, scriptable)
    Query {
        /// Session name, id, or a unique id prefix.
        session: String,
        /// The prompt text. Read from stdin when omitted or "-".
        prompt: Option<String>,
    },
    /// Interactive back-and-forth with a session (/exit or Ctrl-D to leave)
    Chat {
        /// Session name, id, or a unique id prefix.
        session: String,
    },
    /// List sessions
    Ps,
    /// Show a session's lifecycle and recoverability
    Inspect {
        /// Session name, id, or a unique id prefix. Typos get a "did you mean" suggestion.
        session: String,
    },
    /// Reconstruct a session onto a fresh worker after its worker was lost
    Recover {
        /// Session name, id, or a unique id prefix.
        session: String,
    },
    /// Record a durable pointer to the session's current KV cache, for a faster resume
    Checkpoint {
        /// Session name, id, or a unique id prefix.
        session: String,
    },
    /// Stop generation, flush durable state, and free the session's GPU worker
    Pause {
        /// Session name, id, or a unique id prefix.
        session: String,
    },
    /// A stronger pause: same GPU release, but best-effort if acceleration state can't
    /// be saved
    Hibernate {
        /// Session name, id, or a unique id prefix.
        session: String,
    },
    /// Reattach a paused session, fast (native KV) if a compatible checkpoint exists
    Resume {
        /// Session name, id, or a unique id prefix.
        session: String,
        /// Fail rather than fall back to portable reconstruction if no compatible
        /// checkpoint is available.
        #[arg(long, conflicts_with = "portable")]
        fast_only: bool,
        /// Always reconstruct from the durable token log, ignoring any checkpoint.
        #[arg(long)]
        portable: bool,
    },
    /// Write a session's durable history (and optionally its KV checkpoint) to a
    /// portable `.ralph` artifact
    Export {
        /// Session name, id, or a unique id prefix.
        session: String,
        /// Output path. Defaults to ./<session-name>.ralph in the current directory.
        #[arg(long)]
        output: Option<String>,
        /// Overwrite the output path if it already exists.
        #[arg(long)]
        force: bool,
        /// Also include the session's KV checkpoint, if one exists.
        #[arg(long)]
        with_accel: bool,
    },
    /// Reconstruct a session from a `.ralph` artifact, landing it paused
    Import {
        /// Path to a `.ralph` artifact.
        path: String,
        /// Session name for the imported session. Defaults to the exported name.
        #[arg(long)]
        name: Option<String>,
    },
    /// Check the local environment (GPU, vLLM, disk, sqlite) — never mutates anything
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
