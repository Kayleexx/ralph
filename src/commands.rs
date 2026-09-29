//! CLI command handlers: connects to the daemon (or runs checks directly, for `doctor`),
//! sends the request, and renders the human/JSON/quiet output for each command.
mod generation;
mod lifecycle;
use std::io::{IsTerminal, Read, Write};
use std::path::Path;
use std::time::{Duration, Instant};

use tokio::net::UnixStream;

use crate::cli::Flags;
use crate::client;
use crate::doctor;
use crate::error::Envelope;
use crate::ipc::{ErrorPayload, Request, Response};
use crate::picker;

pub(crate) fn ok_mark(cli: Flags) -> &'static str {
    if cli.no_color { "OK" } else { "\u{2713}" }
}

pub(crate) fn arrow(cli: Flags) -> &'static str {
    if cli.no_color { "->" } else { "\u{2192}" }
}

/// Wraps `text` in an ANSI SGR code (e.g. `"36"` for cyan, `"2"` for dim) unless
/// `--no-color` is set or stdout isn't a terminal — color must never leak into piped or
/// redirected output, `--no-color` aside.
pub(crate) fn style(cli: Flags, code: &str, text: &str) -> String {
    if cli.no_color || !std::io::stdout().is_terminal() {
        text.to_string()
    } else {
        format!("\x1b[{code}m{text}\x1b[0m")
    }
}

/// `model@revision` when a real, distinct revision is known; otherwise just the model id
/// — never a fake revision equal to the model id itself.
pub(crate) fn format_model_ref(model: &str, revision: Option<&str>) -> String {
    match revision {
        Some(rev) => format!("{model}@{rev}"),
        None => model.to_string(),
    }
}

/// Display-only shortening (last path segment) — the canonical id is always what's
/// stored and what `--json` reports; this never becomes a persistent alias.
pub(crate) fn short_model_name(model: &str) -> &str {
    model.rsplit('/').next().unwrap_or(model)
}

pub(crate) fn print_json<T: serde::Serialize>(value: &T) {
    println!(
        "{}",
        serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string())
    );
}

pub(crate) fn print_error(cli: Flags, payload: &ErrorPayload) -> i32 {
    if cli.json {
        let mut payload = serde_json::json!(payload);
        if !cli.verbose
            && let Some(object) = payload.as_object_mut()
        {
            object.remove("diagnostic");
        }
        print_json(&serde_json::json!({ "error": payload }));
    } else {
        let envelope = Envelope {
            summary: payload.summary.clone(),
            detail: payload.detail.clone(),
            next: payload.next.clone(),
        };
        eprintln!("{}", envelope.render(cli.no_color));
    }
    if cli.verbose
        && let Some(diagnostic) = &payload.diagnostic
    {
        eprintln!("{diagnostic}");
    }
    payload.exit_code
}

pub(crate) fn print_io_error(cli: Flags, e: std::io::Error) -> i32 {
    print_error(
        cli,
        &ErrorPayload {
            diagnostic: None,
            exit_code: 1,
            summary: "could not reach the ralph daemon".to_string(),
            detail: vec![e.to_string()],
            next: Some("try again, or check: ralph doctor".to_string()),
        },
    )
}

pub(crate) fn print_protocol_error(cli: Flags) -> i32 {
    print_io_error(
        cli,
        std::io::Error::other("daemon sent an unexpected response"),
    )
}

pub(crate) async fn connect(home: &Path, cli: Flags) -> Result<tokio::net::UnixStream, i32> {
    client::connect_or_start(home)
        .await
        .map_err(|e| print_io_error(cli, e))
}

async fn existing_session_names(home: &Path) -> Vec<String> {
    let Ok(mut stream) = client::connect_or_start(home).await else {
        return Vec::new();
    };
    match client::send_request(&mut stream, &Request::Ps).await {
        Ok(Response::Ps(sessions)) => sessions.into_iter().map(|s| s.name).collect(),
        _ => Vec::new(),
    }
}

/// No model on an interactive terminal: shows the picker. No model on a script/CI
/// invocation: a concise, actionable error instead of a confusing hang on stdin.
async fn resolve_model_and_name(
    home: &Path,
    cli: Flags,
    model: Option<String>,
    name: Option<String>,
) -> Result<(String, Option<String>), i32> {
    let Some(model) = model else {
        if !picker::should_prompt(cli.json) {
            return Err(print_error(
                cli,
                &ErrorPayload {
                    diagnostic: None,
                    exit_code: 2,
                    summary: "no model specified".to_string(),
                    detail: vec![],
                    next: Some("run: ralph run <model> --name <name>".to_string()),
                },
            ));
        }
        let Some(picked) = picker::pick_model() else {
            return Err(print_error(
                cli,
                &ErrorPayload {
                    diagnostic: None,
                    exit_code: 1,
                    summary: "cancelled".to_string(),
                    detail: vec![],
                    next: None,
                },
            ));
        };
        let name = match name {
            Some(n) => Some(n),
            None => {
                let existing = existing_session_names(home).await;
                let default = picker::unique_name(&picker::slugify(&picked), &existing);
                Some(picker::pick_session_name(&default).unwrap_or(default))
            }
        };
        return Ok((picked, name));
    };
    Ok((model, name))
}

/// Starting a model has no other feedback for the 10-60+ seconds it typically takes
/// (model download, vLLM/CUDA warmup) — an elapsed-time counter on stderr, TTY-only,
/// beats total silence without pretending to know a real percentage.
async fn send_with_progress(
    stream: &mut UnixStream,
    request: &Request,
    operation: &str,
) -> std::io::Result<Response> {
    let request_fut = client::send_request(stream, request);
    tokio::pin!(request_fut);
    let mut ticks = tokio::time::interval(Duration::from_secs(1));
    ticks.tick().await; // the first tick fires immediately; skip it
    let start = Instant::now();
    let mut printed = false;
    let result = loop {
        tokio::select! {
            biased;
            result = &mut request_fut => break result,
            _ = ticks.tick() => {
                let elapsed = start.elapsed().as_secs();
                eprint!("\r{operation}... {elapsed}s");
                let _ = std::io::stderr().flush();
                printed = true;
            }
        }
    };
    if printed {
        eprintln!();
    }
    result
}

pub async fn run_run(home: &Path, cli: Flags, model: Option<String>, name: Option<String>) -> i32 {
    let (model, name) = match resolve_model_and_name(home, cli, model, name).await {
        Ok(pair) => pair,
        Err(code) => return code,
    };
    let mut stream = match connect(home, cli).await {
        Ok(s) => s,
        Err(code) => return code,
    };
    let request = Request::Run { model, name };
    let show_progress = !cli.json && !cli.quiet && std::io::stdout().is_terminal();
    let result = if show_progress {
        send_with_progress(&mut stream, &request, "starting").await
    } else {
        client::send_request(&mut stream, &request).await
    };
    match result {
        Ok(Response::Run(info)) => {
            if cli.json {
                print_json(&info);
            } else if cli.quiet {
                println!("{}", info.name);
            } else {
                println!(
                    "{} {} ready",
                    style(cli, "32", ok_mark(cli)),
                    style(cli, "1", &info.name)
                );
                println!(
                    "  model: {}",
                    format_model_ref(&info.model, info.model_revision.as_deref())
                );
            }
            0
        }
        Ok(Response::Error(payload)) => print_error(cli, &payload),
        Ok(_) => print_protocol_error(cli),
        Err(e) => print_io_error(cli, e),
    }
}

pub use generation::{run_query, run_recover};
pub use lifecycle::{run_checkpoint, run_pause, run_resume};

pub async fn run_ps(home: &Path, cli: Flags) -> i32 {
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
                let header = format!(
                    "{:<16} {:<24} {:>8} {:<10} LOCATION",
                    "NAME", "MODEL", "TOKENS", "STATE"
                );
                println!("{}", style(cli, "2", &header));
                for s in &sessions {
                    println!(
                        "{:<16} {:<24} {:>8} {:<10} {}",
                        s.name,
                        short_model_name(&s.model),
                        s.token_count,
                        s.state,
                        s.location
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

pub async fn run_inspect(home: &Path, cli: Flags, session: String) -> i32 {
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
                    "  model: {}",
                    format_model_ref(&info.session.model, info.session.model_revision.as_deref())
                );
                println!("  recoverability: {}", info.recoverability);
                println!("  fast restore: {}", info.fast_restore);
                println!("  portable state: {}", info.portable_state);
                if let Some(failure) = info.last_failure {
                    println!("  last worker failure: {failure}");
                }
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
pub fn run_doctor(home: &Path, cli: Flags) -> i32 {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_model_ref_omits_suffix_when_unresolved() {
        assert_eq!(
            format_model_ref("Qwen/Qwen2.5-0.5B-Instruct", None),
            "Qwen/Qwen2.5-0.5B-Instruct"
        );
    }

    #[test]
    fn format_model_ref_appends_real_revision() {
        assert_eq!(
            format_model_ref("Qwen/Qwen2.5-0.5B-Instruct", Some("abc123")),
            "Qwen/Qwen2.5-0.5B-Instruct@abc123"
        );
    }

    #[test]
    fn short_model_name_strips_org_prefix() {
        assert_eq!(
            short_model_name("Qwen/Qwen2.5-0.5B-Instruct"),
            "Qwen2.5-0.5B-Instruct"
        );
    }

    #[test]
    fn short_model_name_is_unchanged_without_a_slash() {
        assert_eq!(short_model_name("gpt2"), "gpt2");
    }
}
