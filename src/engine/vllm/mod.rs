//! vLLM adapter: spawns the vLLM server as a subprocess and talks to it only over its
//! localhost HTTP API. One subprocess shared by compatible sessions and queries —
//! never shelled out to per request.
use std::fs::OpenOptions;
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::process::ExitStatusExt;
use std::path::PathBuf;
use std::process::{ExitStatus, Stdio};
use std::time::{Duration, Instant};

use tokio::process::{Child, Command};
use tokio::sync::{mpsc, oneshot};

use super::{
    ChatMessage, Engine, EngineError, GenerationHandle, HealthStatus, ModelSpec, ResolvedModel, hub,
};
pub(crate) use spawn::resolve_vllm_binary;
use spawn::{pick_free_port, vllm_serve_args};
use stream::stream_completion;

mod diagnostics;
pub mod kv_offload;
pub(crate) mod ownership;
mod spawn;
mod stream;
#[cfg(test)]
mod tests;
mod tokenize;

const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(500);
// Cold imports, CUDA initialization and uncached weights can take several minutes.
const HEALTH_TIMEOUT: Duration = Duration::from_secs(300);
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
    owned: Option<ownership::Ownership>,
    profile: Option<super::profiles::WorkerProfile>,
}

impl VllmEngine {
    pub fn new(log_path: PathBuf) -> Self {
        Self {
            log_path,
            client: reqwest::Client::new(),
            base_url: String::new(),
            model: String::new(),
            child: None,
            owned: None,
            profile: None,
        }
    }
}

impl Engine for VllmEngine {
    async fn start_model(&mut self, spec: &ModelSpec) -> Result<ResolvedModel, EngineError> {
        let port = pick_free_port()?;
        self.base_url = format!("http://127.0.0.1:{port}");
        self.model = spec.model.clone();
        self.profile = spec.profile;

        let revision = match &spec.revision {
            Some(revision) => Some(revision.clone()),
            None => hub::resolve_revision(&self.client, &spec.model, None).await,
        };
        if revision.is_none() {
            return Err(EngineError::BadResponse("cannot resolve immutable model revision; retry when the model cache or Hub is available".into()));
        }
        let nonce = ulid::Ulid::new().to_string();
        let mut stdout_log = OpenOptions::new()
            .create(true)
            .append(true)
            .mode(0o600)
            .open(&self.log_path)?;
        writeln!(stdout_log, "\n--- Ralph startup {nonce} ---")?;
        let stderr_log = stdout_log.try_clone()?;

        let mut cmd = Command::new(resolve_vllm_binary());
        cmd.args(vllm_serve_args(
            &spec.model,
            port,
            revision.as_deref(),
            spec.profile,
            spec.kv_offload.as_ref(),
        ));
        // FlashInfer's sampler JIT-compiles a CUDA kernel on first use, which needs the
        // full CUDA toolkit (nvcc) rather than just the driver — not something Ralph
        // should require on a workstation that only has the driver installed.
        cmd.env("RALPH_WORKER_OWNER", &nonce);
        // setproctitle otherwise overwrites /proc/environ and erases ownership markers.
        cmd.env("SPT_NOENV", "1");
        cmd.env("VLLM_USE_FLASHINFER_SAMPLER", "0");
        // The /sleep, /wake_up, /is_sleeping routes are only mounted in vLLM's "dev"
        // router when this is set — without it they 404 even with --enable-sleep-mode.
        cmd.env("VLLM_SERVER_DEV_MODE", "1");
        cmd.stdout(Stdio::from(stdout_log));
        cmd.stderr(Stdio::from(stderr_log));
        cmd.kill_on_drop(true);
        // vLLM spawns its own EngineCore worker as a separate OS process (Python
        // multiprocessing), not just a thread — signaling only the direct child would
        // orphan that worker, permanently leaking GPU memory. Giving vLLM its own process
        // group lets `stop_model` signal the whole group at once.
        cmd.process_group(0);
        self.child = Some(cmd.spawn()?);

        let pid = self
            .child
            .as_ref()
            .and_then(|c| c.id())
            .ok_or_else(|| EngineError::BadResponse("spawned worker has no PID".into()))?;
        self.owned = Some(ownership::Ownership::new(
            pid,
            nonce,
            self.base_url.clone(),
            self.model.clone(),
            revision.clone(),
            spec.profile,
        )?);
        if let Some(owned) = &self.owned {
            owned.save(&self.log_path.with_file_name("worker.json"))?;
        }
        if let Err(error) = self.wait_until_healthy().await {
            let tail = diagnostics::tail(&self.log_path);
            let cause = diagnostics::cause(&tail).unwrap_or_else(|| error.to_string());
            self.stop_model().await?;
            return Err(EngineError::Startup {
                model: self.model.clone(),
                cause,
                diagnostic: format!("log: {}\n{tail}", self.log_path.display()),
            });
        }

        // Model and tokenizer were pinned to the immutable revision before spawn.
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

    async fn generate(&self, messages: &[ChatMessage]) -> Result<GenerationHandle, EngineError> {
        let (tx, rx) = mpsc::channel(32);
        let (cancel_tx, cancel_rx) = oneshot::channel();
        // Chat completions, not raw completions: a bare prompt string gives an
        // instruction-tuned model no chat template and no natural stopping point, so it
        // just fills the token budget with unrelated filler. Sending the full turn
        // history lets vLLM apply the model's own chat template and stop tokens, and lets
        // a recovered session continue as one conversation rather than starting fresh.
        let url = format!("{}/v1/chat/completions", self.base_url);
        let messages: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| serde_json::json!({"role": m.role.as_str(), "content": m.content}))
            .collect();
        // No `max_tokens`: omitting it entirely (rather than hardcoding a value that's
        // wrong for every model with a different context window) lets vLLM compute the
        // real per-request ceiling itself — max_model_len minus the prompt's token count
        // — so generation runs until the model's own stop token or its actual context
        // limit, whichever comes first, instead of an arbitrary fixed cutoff.
        let body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "stream": true,
            "return_token_ids": true,
            // Deterministic (greedy) decoding: without this, vLLM falls back to whatever
            // sampling defaults the model's own generation_config.json ships with, which
            // can differ per model and makes the same prompt give a different — and
            // sometimes wrong — answer on every query. A durable continuity runtime
            // should not compound that with its own unconfigured randomness.
            "temperature": 0,
        });
        let client = self.client.clone();
        tokio::spawn(stream_completion(client, url, body, tx, cancel_rx));
        Ok(GenerationHandle {
            tokens: rx,
            cancel: cancel_tx,
        })
    }

    async fn sleep(&mut self) -> Result<(), EngineError> {
        let resp = self
            .client
            .post(format!("{}/sleep?level=1", self.base_url))
            .send()
            .await?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(EngineError::BadResponse(format!(
                "sleep request failed: {}",
                resp.status()
            )))
        }
    }

    async fn wake_up(&mut self) -> Result<(), EngineError> {
        let resp = self
            .client
            .post(format!("{}/wake_up", self.base_url))
            .send()
            .await?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(EngineError::BadResponse(format!(
                "wake_up request failed: {}",
                resp.status()
            )))
        }
    }

    /// Best-effort: a request failure (worker briefly unreachable, dev-mode route
    /// missing) reads as "not sleeping" rather than propagating an error, since callers
    /// use this only to decide whether a wake-up call is needed before generating.
    async fn is_sleeping(&self) -> bool {
        let Ok(resp) = self
            .client
            .get(format!("{}/is_sleeping", self.base_url))
            .send()
            .await
        else {
            return false;
        };
        let Ok(json) = resp.json::<serde_json::Value>().await else {
            return false;
        };
        json.get("is_sleeping")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    }

    async fn prefill(&self, messages: &[ChatMessage]) -> Result<(), EngineError> {
        self.prefill_chat(messages).await
    }

    async fn tokenize(&self, messages: &[ChatMessage]) -> Result<super::Tokenized, EngineError> {
        self.tokenize_chat(messages).await
    }
    async fn encode(&self, text: &str) -> Result<Vec<u32>, EngineError> {
        self.encode_text(text).await
    }
    async fn stop_model(&mut self) -> Result<(), EngineError> {
        if let Some(owned) = &self.owned {
            owned.cleanup()?;
        }
        if let Some(child) = &mut self.child {
            child.wait().await?;
        }
        self.child = None;
        self.owned = None;
        let path = self.log_path.with_file_name("worker.json");
        match std::fs::remove_file(path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
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
                return Err(EngineError::WorkerExited(format!(
                    "exited during startup: {status}"
                )));
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
        Err(EngineError::HealthTimeout(last_reason))
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
}
