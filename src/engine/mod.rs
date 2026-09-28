//! Engine adapter: the boundary between ralph and whatever process actually runs
//! inference. Only the methods this phase calls are declared here — pause/resume/
//! checkpoint and a session-object concept get added once something actually implements
//! and calls them.
pub mod hub;
pub mod vllm;

use std::future::Future;
use std::process::ExitStatus;

use thiserror::Error;
use tokio::sync::{mpsc, oneshot};

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("no usable GPU is available")]
    NoGpu,
    #[error("out of GPU memory: {0}")]
    OutOfMemory(String),
    #[error("worker did not become healthy within the startup timeout: {0}")]
    HealthTimeout(String),
    #[error("worker exited unexpectedly: {0}")]
    WorkerExited(String),
    #[error("request exceeds the model's context window ({limit} tokens)")]
    ContextWindowExceeded { limit: u32 },
    #[error("engine request failed: {0}")]
    Request(#[from] reqwest::Error),
    #[error("engine returned an unreadable response: {0}")]
    BadResponse(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub model: String,
    pub revision: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ResolvedModel {
    /// `None` when the exact revision genuinely couldn't be resolved (offline, private
    /// repo, etc.) — never a fake value pretending to be one.
    pub revision: Option<String>,
    pub engine_version: Option<String>,
}

pub enum HealthStatus {
    Healthy,
    Unhealthy(String),
}

/// A single in-flight generation. `cancel` lets the caller stop it early (Ctrl-C);
/// dropping `tokens` without cancelling has the same effect once the sender notices.
pub struct GenerationHandle {
    pub tokens: mpsc::Receiver<Result<String, EngineError>>,
    pub cancel: oneshot::Sender<()>,
}

// Spelled out as `-> impl Future<...> + Send` (rather than `async fn`) so the futures
// this produces are Send and can be driven from a spawned task; an `async fn` impl body
// still satisfies this signature.
pub trait Engine: Send {
    fn start_model(
        &mut self,
        spec: &ModelSpec,
    ) -> impl Future<Output = Result<ResolvedModel, EngineError>> + Send;
    fn health(&self) -> impl Future<Output = HealthStatus> + Send;
    fn generate(
        &self,
        prompt: &str,
    ) -> impl Future<Output = Result<GenerationHandle, EngineError>> + Send;
    fn stop_model(&mut self) -> impl Future<Output = Result<(), EngineError>> + Send;
    /// Non-blocking: `None` while the worker is still alive. Supervision polls this
    /// instead of awaiting exit while holding the engine's lock — generate/health need
    /// that same lock, and a blocking wait would starve them for the whole session.
    fn try_wait_for_exit(&mut self) -> Option<ExitStatus>;
    /// Internal worker identity only — never surfaced to the user without -v.
    fn pid(&self) -> Option<u32>;
}
