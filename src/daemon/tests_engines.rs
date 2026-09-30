//! Purpose-built fake `Engine`s for specific failure-mode tests — `tests.rs`'s
//! `FakeEngine` covers ordinary daemon plumbing; these model one narrow failure shape
//! each so the test using them doesn't need a real process to get there.
use std::os::unix::process::ExitStatusExt;
use std::path::PathBuf;
use std::process::ExitStatus;

use super::tests::FakeEngine;
use crate::engine::{
    ChatMessage, Engine, EngineError, GenerationHandle, HealthStatus, ModelSpec, ResolvedModel,
};

/// An `Engine` whose `start_model` always fails — models a worker that can never come
/// back up, for testing that the bounded restart policy still converges instead of
/// retrying forever.
pub(super) struct AlwaysFailingEngine;

impl AlwaysFailingEngine {
    pub(crate) fn new(_log_path: PathBuf) -> Self {
        AlwaysFailingEngine
    }
}

impl Engine for AlwaysFailingEngine {
    async fn start_model(&mut self, _spec: &ModelSpec) -> Result<ResolvedModel, EngineError> {
        Err(EngineError::HealthTimeout(
            "stub: never comes up".to_string(),
        ))
    }

    async fn health(&self) -> HealthStatus {
        HealthStatus::Unhealthy("never started".to_string())
    }

    async fn generate(&self, _messages: &[ChatMessage]) -> Result<GenerationHandle, EngineError> {
        Err(EngineError::HealthTimeout(
            "stub: never comes up".to_string(),
        ))
    }

    async fn sleep(&mut self) -> Result<(), EngineError> {
        Ok(())
    }

    async fn wake_up(&mut self) -> Result<(), EngineError> {
        Ok(())
    }

    async fn is_sleeping(&self) -> bool {
        false
    }

    async fn prefill(&self, _messages: &[ChatMessage]) -> Result<(), EngineError> {
        Ok(())
    }
    async fn tokenize(
        &self,
        messages: &[ChatMessage],
    ) -> Result<crate::engine::Tokenized, EngineError> {
        Ok(crate::engine::Tokenized {
            ids: messages
                .iter()
                .flat_map(|m| m.content.bytes().map(u32::from))
                .collect(),
            limit: 4096,
        })
    }
    async fn encode(&self, text: &str) -> Result<Vec<u32>, EngineError> {
        Ok(text.bytes().map(u32::from).collect())
    }
    async fn stop_model(&mut self) -> Result<(), EngineError> {
        Ok(())
    }

    fn try_wait_for_exit(&mut self) -> Option<ExitStatus> {
        Some(ExitStatus::from_raw(-1))
    }

    fn pid(&self) -> Option<u32> {
        None
    }
}

/// Fails startup only when asked to attach native KV offload — models a checkpoint
/// directory whose fingerprint matches but whose content vLLM can't actually restore
/// from, so tests can exercise the resume-time portable fallback without a real engine.
pub(super) struct FlakyNativeEngine {
    inner: FakeEngine,
}

impl FlakyNativeEngine {
    pub(crate) fn new(log_path: PathBuf) -> Self {
        FlakyNativeEngine {
            inner: FakeEngine::new(log_path),
        }
    }
}

impl Engine for FlakyNativeEngine {
    async fn start_model(&mut self, spec: &ModelSpec) -> Result<ResolvedModel, EngineError> {
        if spec.kv_offload.is_some() {
            return Err(EngineError::BadResponse("corrupt kv checkpoint".into()));
        }
        self.inner.start_model(spec).await
    }
    async fn health(&self) -> HealthStatus {
        self.inner.health().await
    }
    async fn generate(&self, messages: &[ChatMessage]) -> Result<GenerationHandle, EngineError> {
        self.inner.generate(messages).await
    }
    async fn sleep(&mut self) -> Result<(), EngineError> {
        self.inner.sleep().await
    }
    async fn wake_up(&mut self) -> Result<(), EngineError> {
        self.inner.wake_up().await
    }
    async fn is_sleeping(&self) -> bool {
        self.inner.is_sleeping().await
    }
    async fn prefill(&self, messages: &[ChatMessage]) -> Result<(), EngineError> {
        self.inner.prefill(messages).await
    }
    async fn tokenize(
        &self,
        messages: &[ChatMessage],
    ) -> Result<crate::engine::Tokenized, EngineError> {
        self.inner.tokenize(messages).await
    }
    async fn encode(&self, text: &str) -> Result<Vec<u32>, EngineError> {
        self.inner.encode(text).await
    }
    async fn stop_model(&mut self) -> Result<(), EngineError> {
        self.inner.stop_model().await
    }
    fn try_wait_for_exit(&mut self) -> Option<ExitStatus> {
        self.inner.try_wait_for_exit()
    }
    fn pid(&self) -> Option<u32> {
        self.inner.pid()
    }
}

/// Fails startup on any GPU index other than 0 — models a destination GPU that's
/// unreachable/unusable, so migration rollback can be tested deterministically without
/// real second-GPU hardware.
pub(super) struct GpuBoundEngine {
    inner: FakeEngine,
}

impl GpuBoundEngine {
    pub(crate) fn new(log_path: PathBuf) -> Self {
        GpuBoundEngine {
            inner: FakeEngine::new(log_path),
        }
    }
}

impl Engine for GpuBoundEngine {
    async fn start_model(&mut self, spec: &ModelSpec) -> Result<ResolvedModel, EngineError> {
        if spec.gpu != 0 {
            return Err(EngineError::BadResponse(format!(
                "gpu{} unreachable",
                spec.gpu
            )));
        }
        self.inner.start_model(spec).await
    }
    async fn health(&self) -> HealthStatus {
        self.inner.health().await
    }
    async fn generate(&self, messages: &[ChatMessage]) -> Result<GenerationHandle, EngineError> {
        self.inner.generate(messages).await
    }
    async fn sleep(&mut self) -> Result<(), EngineError> {
        self.inner.sleep().await
    }
    async fn wake_up(&mut self) -> Result<(), EngineError> {
        self.inner.wake_up().await
    }
    async fn is_sleeping(&self) -> bool {
        self.inner.is_sleeping().await
    }
    async fn prefill(&self, messages: &[ChatMessage]) -> Result<(), EngineError> {
        self.inner.prefill(messages).await
    }
    async fn tokenize(
        &self,
        messages: &[ChatMessage],
    ) -> Result<crate::engine::Tokenized, EngineError> {
        self.inner.tokenize(messages).await
    }
    async fn encode(&self, text: &str) -> Result<Vec<u32>, EngineError> {
        self.inner.encode(text).await
    }
    async fn stop_model(&mut self) -> Result<(), EngineError> {
        self.inner.stop_model().await
    }
    fn try_wait_for_exit(&mut self) -> Option<ExitStatus> {
        self.inner.try_wait_for_exit()
    }
    fn pid(&self) -> Option<u32> {
        self.inner.pid()
    }
}
