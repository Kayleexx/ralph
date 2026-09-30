//! Deterministic lifecycle/rollback tests for `migrate_cancellable`, using `FakeEngine`/
//! `GpuBoundEngine` and a fake `gpu_memory` override — real 2-GPU hardware behavior is
//! only proven by `tests/e2e_migration.rs` against actual CUDA devices; this file only
//! covers the state-machine/rollback logic Stage C's own design doesn't need hardware
//! to verify.
use super::admission::GpuMemory;
use super::tests::FakeEngine;
use super::tests_engines::GpuBoundEngine;
use super::*;
use std::collections::HashMap;

fn two_gpus(index: u32) -> Result<GpuMemory, CliError> {
    Ok(GpuMemory {
        index,
        name: "fake-gpu".into(),
        free_mib: 8000,
        allocations: HashMap::new(),
    })
}

fn daemon_with<E: crate::engine::Engine + 'static>(
    dir: &std::path::Path,
    make_engine: fn(std::path::PathBuf) -> E,
) -> Arc<Daemon<E>> {
    let mut d = Daemon::with_gpu_check(
        Storage::open_in_memory().unwrap(),
        dir.to_path_buf(),
        make_engine,
        || true,
    );
    Arc::get_mut(&mut d).unwrap().gpu_memory = two_gpus;
    d
}

#[tokio::test]
async fn migrate_moves_location_and_commits_active() {
    let dir = tempfile::tempdir().unwrap();
    let d = daemon_with(dir.path(), FakeEngine::new);
    let info = d
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();
    assert_eq!(info.location, "local/gpu0");

    let report = d.migrate_cancellable("demo", 1, None).await.unwrap();
    assert_eq!(report.source_gpu, 0);
    assert_eq!(report.destination_gpu, 1);
    assert!(!report.native);
    assert_eq!(d.inspect("demo").unwrap().session.state, "active");
    assert_eq!(d.inspect("demo").unwrap().session.location, "local/gpu1");
    assert!(d.workers.lock().await.contains_key(&WorkerKey {
        model: "model".to_string(),
        gpu: 1
    }));
}

#[tokio::test]
async fn migrate_to_the_same_gpu_is_a_zero_cost_noop() {
    let dir = tempfile::tempdir().unwrap();
    let d = daemon_with(dir.path(), FakeEngine::new);
    d.run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();

    let report = d.migrate_cancellable("demo", 0, None).await.unwrap();
    assert_eq!(report.total_ms, 0.0);
    assert_eq!(d.inspect("demo").unwrap().session.location, "local/gpu0");
}

#[tokio::test]
async fn migrate_requires_an_active_session() {
    let dir = tempfile::tempdir().unwrap();
    let d = daemon_with(dir.path(), FakeEngine::new);
    let info = d
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();
    d.pause_cancellable("demo", None).await.unwrap();

    let err = d.migrate_cancellable("demo", 1, None).await.unwrap_err();
    assert!(matches!(err, CliError::InvalidState(_)));
    assert_eq!(d.inspect("demo").unwrap().session.id, info.id);
}

/// Destination attach fails (GPU unreachable) — the session must roll back to the
/// source GPU and land `Active` again, never stuck `Paused`/`Resuming`, never two
/// authoritative workers.
#[tokio::test]
async fn migrate_rolls_back_to_source_when_destination_is_unreachable() {
    let dir = tempfile::tempdir().unwrap();
    let d = daemon_with(dir.path(), GpuBoundEngine::new);
    d.run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();

    d.migrate_cancellable("demo", 1, None).await.unwrap_err();
    assert_eq!(d.inspect("demo").unwrap().session.state, "active");
    assert_eq!(d.inspect("demo").unwrap().session.location, "local/gpu0");
    assert_eq!(d.workers.lock().await.len(), 1);
}
