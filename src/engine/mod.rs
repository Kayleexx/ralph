//! Engine adapter: the boundary between ralph and whatever process actually runs
//! inference. Only methods used by the current implementation are declared here — pause/resume/
//! checkpoint and a session-object concept get added once something actually implements
//! and calls them.
pub mod hub;
pub mod profiles;
pub mod vllm;

use std::future::Future;
use std::path::PathBuf;
use std::process::ExitStatus;

use thiserror::Error;
use tokio::sync::{mpsc, oneshot};

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("{model}: {cause}")]
    Startup {
        model: String,
        cause: String,
        diagnostic: String,
    },
    #[error("no usable GPU is available")]
    NoGpu,
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

/// Points a worker at a per-session, fingerprint-gated KV directory. Passing this is
/// opportunistic acceleration only — never required for correctness, since a missing or
/// stale directory just means an ordinary cold prefill (see `daemon/lifecycle.rs`).
#[derive(Debug, Clone)]
pub struct KvOffloadSpec {
    pub root_dir: PathBuf,
    pub cpu_bytes: u64,
    /// Pinned rather than left to vLLM's own random default so a crashed worker's
    /// `/dev/shm/vllm_offload_<engine_id>.mmap` can be found and unlinked afterward.
    pub engine_id: String,
}

#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub model: String,
    pub revision: Option<String>,
    pub profile: Option<profiles::WorkerProfile>,
    pub kv_offload: Option<KvOffloadSpec>,
}

#[derive(Debug, Clone)]
pub struct ResolvedModel {
    /// `None` when the exact revision genuinely couldn't be resolved (offline, private
    /// repo, etc.) — never a fake value pretending to be one.
    pub revision: Option<String>,
    pub engine_version: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

impl Role {
    fn as_str(self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

pub enum HealthStatus {
    Healthy,
    Unhealthy(String),
}

/// A single in-flight generation. `cancel` lets the caller stop it early (Ctrl-C);
/// dropping `tokens` without cancelling has the same effect once the sender notices.
#[derive(Debug)]
pub struct TokenChunk {
    pub text: String,
    pub ids: Vec<u32>,
}

pub struct Tokenized {
    pub ids: Vec<u32>,
    pub limit: u32,
}

pub struct GenerationHandle {
    pub tokens: mpsc::Receiver<Result<TokenChunk, EngineError>>,
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
        messages: &[ChatMessage],
    ) -> impl Future<Output = Result<GenerationHandle, EngineError>> + Send;
    fn prefill(
        &self,
        messages: &[ChatMessage],
    ) -> impl Future<Output = Result<(), EngineError>> + Send;
    fn tokenize(
        &self,
        messages: &[ChatMessage],
    ) -> impl Future<Output = Result<Tokenized, EngineError>> + Send;
    fn encode(&self, text: &str) -> impl Future<Output = Result<Vec<u32>, EngineError>> + Send;
    fn stop_model(&mut self) -> impl Future<Output = Result<(), EngineError>> + Send;
    /// Offloads weights to host RAM and frees the GPU allocation without exiting the
    /// process — much cheaper to undo than a cold start. A no-op is never assumed; the
    /// caller checks `is_sleeping` rather than tracking sleep state itself, since the
    /// engine is the only source of truth for it.
    fn sleep(&mut self) -> impl Future<Output = Result<(), EngineError>> + Send;
    fn wake_up(&mut self) -> impl Future<Output = Result<(), EngineError>> + Send;
    fn is_sleeping(&self) -> impl Future<Output = bool> + Send;
    /// Non-blocking: `None` while the worker is still alive. Supervision polls this
    /// instead of awaiting exit while holding the engine's lock — generate/health need
    /// that same lock, and a blocking wait would starve them for the whole session.
    fn try_wait_for_exit(&mut self) -> Option<ExitStatus>;
    /// Internal worker identity only — never surfaced to the user without -v.
    fn pid(&self) -> Option<u32>;
}
