//! vLLM adapter: spawns the vLLM server as a subprocess and talks to it only over its
//! localhost HTTP API. One subprocess per session, reused for every query against it —
//! never shelled out to per request.
use std::fs::File;
use std::net::TcpListener;
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Stdio};
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, oneshot};

use super::{Engine, EngineError, GenerationHandle, HealthStatus, ModelSpec, ResolvedModel, hub};

#[cfg(test)]
mod tests;

const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(500);
// vLLM's cold start (import torch, init CUDA, spawn the EngineCore subprocess, load
// weights) routinely takes 60-150s+ even for a sub-1B model on a modest single GPU —
// 120s was cutting that close enough to fail on a healthy, still-loading worker. This is
// a ceiling against a genuinely stuck process, not a tuned "typical" duration.
const HEALTH_TIMEOUT: Duration = Duration::from_secs(300);
const STOP_GRACE_PERIOD: Duration = Duration::from_secs(3);
// vLLM's own default (0.9) pre-allocates most of the GPU's memory up front, which fails
// outright on a workstation where something else (a desktop compositor, browser, etc.) is
// already holding a slice of a small GPU's VRAM. A lower default leaves headroom for that
// on exactly the kind of single-GPU laptop this project targets.
const GPU_MEMORY_UTILIZATION: &str = "0.6";

pub struct VllmEngine {
    log_path: PathBuf,
    client: reqwest::Client,
    base_url: String,
    // The exact tag vLLM was told to serve — required on every completion request, since
    // vLLM validates it against the served model and 404s on a mismatch.
    model: String,
    child: Option<Child>,
}

impl VllmEngine {
    pub fn new(log_path: PathBuf) -> Self {
        Self {
            log_path,
            client: reqwest::Client::new(),
            base_url: String::new(),
            model: String::new(),
            child: None,
        }
    }
}

/// Checks, in order: an activated venv (`$VIRTUAL_ENV`, works from any directory once
/// `source .venv/bin/activate` has run), a `.venv` in the current directory (works when
/// `ralph` is run from inside the project without activating anything), then whatever
/// `vllm` resolves to on `PATH`. `ralph` is often installed globally and run from
/// wherever the user happens to be, so a CWD-relative check alone isn't enough.
pub(crate) fn resolve_vllm_binary() -> PathBuf {
    resolve_vllm_binary_from(
        std::env::var("VIRTUAL_ENV").ok(),
        Path::new(".venv/bin/vllm"),
    )
}

/// Split out so tests can control both inputs directly, rather than mutating the real
/// `VIRTUAL_ENV` process environment (`std::env::set_var` is `unsafe` on this toolchain).
fn resolve_vllm_binary_from(virtual_env: Option<String>, cwd_venv: &Path) -> PathBuf {
    if let Some(venv) = virtual_env {
        let candidate = PathBuf::from(venv).join("bin/vllm");
        if candidate.exists() {
            return candidate;
        }
    }
    if cwd_venv.exists() {
        return cwd_venv.to_path_buf();
    }
    PathBuf::from("vllm")
}

fn pick_free_port() -> Result<u16, EngineError> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    Ok(listener.local_addr()?.port())
}

impl Engine for VllmEngine {
    async fn start_model(&mut self, spec: &ModelSpec) -> Result<ResolvedModel, EngineError> {
        let port = pick_free_port()?;
        self.base_url = format!("http://127.0.0.1:{port}");
        self.model = spec.model.clone();

        let stdout_log = File::create(&self.log_path)?;
        let stderr_log = stdout_log.try_clone()?;

        let mut cmd = Command::new(resolve_vllm_binary());
        cmd.args([
            "serve",
            &spec.model,
            "--port",
            &port.to_string(),
            "--gpu-memory-utilization",
            GPU_MEMORY_UTILIZATION,
            // Skips CUDA graph capture, which otherwise dominates startup time (tens of
            // seconds even for a small model) — favors fast, repeatable session startup
            // over the last bit of decode throughput.
            "--enforce-eager",
        ]);
        if let Some(revision) = &spec.revision {
            cmd.args(["--revision", revision]);
        }
        // FlashInfer's sampler JIT-compiles a CUDA kernel on first use, which needs the
        // full CUDA toolkit (nvcc) rather than just the driver — not something Ralph
        // should require on a workstation that only has the driver installed.
        cmd.env("VLLM_USE_FLASHINFER_SAMPLER", "0");
        cmd.stdout(Stdio::from(stdout_log));
        cmd.stderr(Stdio::from(stderr_log));
        cmd.kill_on_drop(true);
        // vLLM spawns its own EngineCore worker as a separate OS process (Python
        // multiprocessing), not just a thread — signaling only the direct child would
        // orphan that worker, permanently leaking GPU memory. Giving vLLM its own process
        // group lets `stop_model` signal the whole group at once.
        cmd.process_group(0);
        self.child = Some(cmd.spawn()?);

        self.wait_until_healthy().await?;

        // vLLM's own /v1/models doesn't expose a resolvable git revision distinct from
        // the model id, so the real commit sha is resolved from the Hub API instead;
        // `None` (never the model id itself) when that can't be done.
        let revision =
            hub::resolve_revision(&self.client, &spec.model, spec.revision.as_deref()).await;
        let engine_version = self.fetch_version().await;

        Ok(ResolvedModel {
            revision,
            engine_version,
        })
    }

    async fn health(&self) -> HealthStatus {
        match self
            .client
            .get(format!("{}/health", self.base_url))
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => HealthStatus::Healthy,
            Ok(resp) => HealthStatus::Unhealthy(format!("status {}", resp.status())),
            Err(e) => HealthStatus::Unhealthy(e.to_string()),
        }
    }

    async fn generate(&self, prompt: &str) -> Result<GenerationHandle, EngineError> {
        let (tx, rx) = mpsc::channel(32);
        let (cancel_tx, cancel_rx) = oneshot::channel();
        // Chat completions, not raw completions: a bare prompt string gives an
        // instruction-tuned model no chat template and no natural stopping point, so it
        // just fills the token budget with unrelated filler. Sending it as a one-turn
        // conversation lets vLLM apply the model's own chat template and stop tokens.
        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = serde_json::json!({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": true,
            "max_tokens": 128,
        });
        let client = self.client.clone();
        tokio::spawn(stream_completion(client, url, body, tx, cancel_rx));
        Ok(GenerationHandle {
            tokens: rx,
            cancel: cancel_tx,
        })
    }

    async fn stop_model(&mut self) -> Result<(), EngineError> {
        let Some(child) = &mut self.child else {
            return Ok(());
        };
        if let Some(pid) = child.id() {
            // Signaling the process group (negative pid), not just the direct child,
            // reaches vLLM's separately-spawned EngineCore worker too — otherwise it's
            // orphaned and keeps holding GPU memory. SIGTERM first via a `kill`
            // subprocess (avoids any unsafe libc call), then SIGKILL the group if it
            // doesn't exit within the grace period.
            let _ = Command::new("kill")
                .args(["-TERM", &format!("-{pid}")])
                .status()
                .await;
            let deadline = Instant::now() + STOP_GRACE_PERIOD;
            while Instant::now() < deadline {
                if matches!(child.try_wait(), Ok(Some(_))) {
                    return Ok(());
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            let _ = Command::new("kill")
                .args(["-KILL", &format!("-{pid}")])
                .status()
                .await;
        }
        child.wait().await?;
        Ok(())
    }

    fn try_wait_for_exit(&mut self) -> Option<ExitStatus> {
        match &mut self.child {
            Some(child) => child.try_wait().ok().flatten(),
            // No child was ever spawned; treat as already-exited rather than reporting
            // "still running" forever.
            None => Some(ExitStatus::from_raw(-1)),
        }
    }

    fn pid(&self) -> Option<u32> {
        self.child.as_ref().and_then(|c| c.id())
    }
}

impl VllmEngine {
    async fn wait_until_healthy(&mut self) -> Result<(), EngineError> {
        let deadline = Instant::now() + HEALTH_TIMEOUT;
        let last_reason = loop {
            if let Some(child) = &mut self.child
                && let Ok(Some(status)) = child.try_wait()
            {
                return Err(self.detect_oom().unwrap_or_else(|| {
                    EngineError::WorkerExited(format!("exited during startup: {status}"))
                }));
            }
            let reason = match self.health().await {
                HealthStatus::Healthy => return Ok(()),
                HealthStatus::Unhealthy(reason) => reason,
            };
            if Instant::now() >= deadline {
                break reason;
            }
            tokio::time::sleep(HEALTH_POLL_INTERVAL).await;
        };
        let _ = self.stop_model().await;
        Err(self
            .detect_oom()
            .unwrap_or(EngineError::HealthTimeout(last_reason)))
    }

    async fn fetch_version(&self) -> Option<String> {
        let resp = self
            .client
            .get(format!("{}/version", self.base_url))
            .send()
            .await
            .ok()?;
        let json: serde_json::Value = resp.json().await.ok()?;
        json.get("version")
            .and_then(|v| v.as_str())
            .map(String::from)
    }

    /// Scans the worker's own log for an out-of-memory message. Best-effort: vLLM's exact
    /// OOM wording can vary between versions, so this only sharpens the error message —
    /// a miss still falls back to the generic startup-failure error.
    fn detect_oom(&self) -> Option<EngineError> {
        let contents = std::fs::read_to_string(&self.log_path).ok()?;
        let line = contents
            .lines()
            .find(|l| l.to_lowercase().contains("out of memory"))?;
        Some(EngineError::OutOfMemory(line.trim().to_string()))
    }
}

async fn stream_completion(
    client: reqwest::Client,
    url: String,
    body: serde_json::Value,
    tx: mpsc::Sender<Result<String, EngineError>>,
    mut cancel_rx: oneshot::Receiver<()>,
) {
    let resp = match client.post(&url).json(&body).send().await {
        Ok(r) => r,
        Err(e) => {
            let _ = tx.send(Err(EngineError::Request(e))).await;
            return;
        }
    };
    if !resp.status().is_success() {
        let status = resp.status();
        let body_text = resp.text().await.unwrap_or_default();
        let _ = tx
            .send(Err(classify_error_response(status, &body_text)))
            .await;
        return;
    }
    let mut stream = resp.bytes_stream();
    let mut buf = String::new();
    loop {
        tokio::select! {
            _ = &mut cancel_rx => return,
            chunk = stream.next() => {
                match chunk {
                    Some(Ok(bytes)) => {
                        buf.push_str(&String::from_utf8_lossy(&bytes));
                        if !forward_complete_events(&mut buf, &tx).await {
                            return;
                        }
                    }
                    Some(Err(e)) => {
                        let _ = tx.send(Err(EngineError::Request(e))).await;
                        return;
                    }
                    None => return,
                }
            }
        }
    }
}

/// Best-effort classification of a non-success completion response. Context-length wording
/// isn't standardized across vLLM versions, so this is a heuristic, not a guarantee — an
/// unrecognized error still reaches the user as `BadResponse` rather than being dropped.
fn classify_error_response(status: reqwest::StatusCode, body: &str) -> EngineError {
    let lower = body.to_lowercase();
    if lower.contains("maximum context length") || lower.contains("context_length_exceeded") {
        let limit = first_number(body).unwrap_or(0);
        return EngineError::ContextWindowExceeded { limit };
    }
    EngineError::BadResponse(format!(
        "{status}: {}",
        body.chars().take(300).collect::<String>()
    ))
}

fn first_number(text: &str) -> Option<u32> {
    let digits: String = text
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

/// Drains complete SSE events (`\n\n`-terminated) out of `buf`, forwarding each chat
/// completion chunk's text delta. Returns false once the receiver is gone or a `[DONE]`
/// marker is seen. The first chunk carries only a role with empty content, and the last
/// carries only a finish reason with no `content` key at all — both are silently skipped.
async fn forward_complete_events(
    buf: &mut String,
    tx: &mpsc::Sender<Result<String, EngineError>>,
) -> bool {
    while let Some(pos) = buf.find("\n\n") {
        let event = buf[..pos].to_string();
        buf.drain(..pos + 2);
        for line in event.lines() {
            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };
            if data == "[DONE]" {
                return false;
            }
            let Ok(json) = serde_json::from_str::<serde_json::Value>(data) else {
                continue;
            };
            let Some(text) = json["choices"][0]["delta"]["content"].as_str() else {
                continue;
            };
            if text.is_empty() {
                continue;
            }
            if tx.send(Ok(text.to_string())).await.is_err() {
                return false;
            }
        }
    }
    true
}
