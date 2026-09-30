//! CLI command handlers: connects to the daemon (or runs checks directly, for `doctor`),
//! sends the request, and renders the human/JSON/quiet output for each command.
mod generation;
mod handoff;
mod lifecycle;
mod migrate;
mod portable;
mod status;
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
/// — never a fake revision equal to the model id itself. Shown abbreviated (first 8
/// chars, matching a short git SHA) — the full revision is always in `--json`.
pub(crate) fn format_model_ref(model: &str, revision: Option<&str>) -> String {
    match revision {
        Some(rev) => format!("{model}@{}", rev.chars().take(8).collect::<String>()),
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

pub async fn run_run(
    home: &Path,
    cli: Flags,
    model: Option<String>,
    name: Option<String>,
    gpu: Option<u32>,
    policy: Option<crate::continuity::ContinuityPolicy>,
    continuity_target_ms: Option<u64>,
) -> i32 {
    let (model, name) = match resolve_model_and_name(home, cli, model, name).await {
        Ok(pair) => pair,
        Err(code) => return code,
    };
    let mut stream = match connect(home, cli).await {
        Ok(s) => s,
        Err(code) => return code,
    };
    let request = Request::Run {
        model,
        name,
        gpu,
        policy,
        continuity_target_ms,
    };
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
                if let Some(demoted) = &info.made_room_for {
                    println!("making room \u{2014} hibernated {demoted}");
                }
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
pub use handoff::{run_drain, run_handoff};
pub use lifecycle::{run_checkpoint, run_hibernate, run_pause, run_resume};
pub use migrate::run_migrate;
pub use portable::{run_export, run_import};

pub use status::{run_doctor, run_inspect, run_ps};

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
    fn format_model_ref_abbreviates_a_full_length_sha() {
        assert_eq!(
            format_model_ref(
                "Qwen/Qwen2.5-0.5B-Instruct",
                Some("7ae557604adf67be50417f59c2c2f167def9a775")
            ),
            "Qwen/Qwen2.5-0.5B-Instruct@7ae55760"
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
