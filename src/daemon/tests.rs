use std::os::unix::process::ExitStatusExt;
use std::process::ExitStatus;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use tokio::sync::{mpsc, oneshot};

use super::*;
use crate::engine::{GenerationHandle, HealthStatus, ResolvedModel};

/// An `Engine` that never touches a real process, so daemon plumbing (locking, state
/// transitions, reconciliation) can be tested without vLLM or a GPU.
struct FakeEngine {
    exited: Arc<AtomicBool>,
}

impl FakeEngine {
    fn new(_log_path: PathBuf) -> Self {
        FakeEngine {
            exited: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl Engine for FakeEngine {
    async fn start_model(&mut self, spec: &ModelSpec) -> Result<ResolvedModel, EngineError> {
        Ok(ResolvedModel {
            revision: spec.model.clone(),
            engine_version: Some("fake".to_string()),
        })
    }

    async fn health(&self) -> HealthStatus {
        HealthStatus::Healthy
    }

    async fn generate(&self, _prompt: &str) -> Result<GenerationHandle, EngineError> {
        let (tx, rx) = mpsc::channel(4);
        let (cancel_tx, _cancel_rx) = oneshot::channel();
        let _ = tx.send(Ok("hi".to_string())).await;
        Ok(GenerationHandle {
            tokens: rx,
            cancel: cancel_tx,
        })
    }

    async fn stop_model(&mut self) -> Result<(), EngineError> {
        Ok(())
    }

    fn try_wait_for_exit(&mut self) -> Option<ExitStatus> {
        if self.exited.load(Ordering::SeqCst) {
            Some(ExitStatus::from_raw(0))
        } else {
            None
        }
    }

    fn pid(&self) -> Option<u32> {
        Some(1)
    }
}

fn test_daemon(ralph_home: &std::path::Path) -> Arc<Daemon<FakeEngine>> {
    let storage = Storage::open_in_memory().unwrap();
    Daemon::with_gpu_check(storage, ralph_home.to_path_buf(), FakeEngine::new, || true)
}

#[tokio::test]
async fn run_rejects_duplicate_name() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    daemon
        .run("model-a".to_string(), Some("demo".to_string()))
        .await
        .unwrap();
    let err = daemon
        .run("model-b".to_string(), Some("demo".to_string()))
        .await
        .unwrap_err();
    assert!(matches!(err, CliError::DuplicateName(_)));
}

#[tokio::test]
async fn run_fails_fast_without_gpu() {
    let dir = tempfile::tempdir().unwrap();
    let storage = Storage::open_in_memory().unwrap();
    let daemon =
        Daemon::with_gpu_check(storage, dir.path().to_path_buf(), FakeEngine::new, || false);
    let err = daemon.run("model".to_string(), None).await.unwrap_err();
    assert!(matches!(err, CliError::Resource(_)));
}

#[tokio::test]
async fn run_creates_an_active_session() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    let info = daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();
    assert_eq!(info.name, "demo");
    assert_eq!(info.state, "active");
    assert_eq!(info.model_revision, "model");
}

#[tokio::test]
async fn inspect_unknown_session_is_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    let err = daemon.inspect("nope").unwrap_err();
    assert!(matches!(err, CliError::NotFound(_)));
}

#[tokio::test]
async fn concurrent_query_on_same_session_fails_fast() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();

    let first = daemon.begin_query("demo").await.unwrap();
    let second = daemon.begin_query("demo").await;
    assert!(matches!(second, Err(CliError::InvalidState(_))));

    drop(first);
    assert!(daemon.begin_query("demo").await.is_ok());
}

#[tokio::test]
async fn query_on_non_active_session_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    // No session named "demo" exists at all yet, so this exercises the not-active path
    // via NotFound rather than a state check — a session can only reach Active through
    // `run`, so this also implicitly covers "not yet started".
    let result = daemon.begin_query("demo").await;
    assert!(matches!(result, Err(CliError::NotFound(_))));
}

#[tokio::test]
async fn reconciliation_demotes_active_session_after_daemon_restart() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("ralph.db");
    let ralph_home = dir.path().to_path_buf();

    let storage1 = Storage::open(&db_path).unwrap();
    let daemon1 = Daemon::with_gpu_check(storage1, ralph_home.clone(), FakeEngine::new, || true);
    let info = daemon1
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();
    assert_eq!(info.state, "active");

    // A fresh daemon process has no handle to daemon1's FakeEngine, mirroring a real
    // crash-and-restart: the old worker (if any) is orphaned, not reattached.
    let storage2 = Storage::open(&db_path).unwrap();
    let daemon2 = Daemon::with_gpu_check(storage2, ralph_home, FakeEngine::new, || true);
    let demoted = daemon2.reconcile_on_startup().unwrap();
    assert_eq!(demoted, vec![info.id]);

    let inspected = daemon2.inspect("demo").unwrap();
    assert_eq!(inspected.session.state, "stopped");
    assert_eq!(inspected.recoverability, "degraded");
}

/// Regression test: the supervisor task must not hold the engine's mutex for the whole
/// session lifetime — `generate` needs that same lock, and previously the supervisor held
/// it for as long as the worker stayed alive, deadlocking every query forever.
#[tokio::test]
async fn generate_does_not_deadlock_with_supervisor_running() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();

    // Give the supervisor's poll loop a chance to run (and, before the fix, to acquire
    // and never release the lock) before we try to use it ourselves.
    tokio::time::sleep(Duration::from_millis(50)).await;

    let running = daemon.begin_query("demo").await.unwrap();
    let result = tokio::time::timeout(Duration::from_secs(2), async {
        let engine = running.engine.lock().await;
        engine.generate("hello").await
    })
    .await;
    assert!(
        result.is_ok(),
        "generate() timed out acquiring the engine lock"
    );
}
