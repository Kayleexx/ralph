use super::admission::GpuMemory;
use super::tests::FakeEngine;
use super::*;
use crate::engine::profiles::{QWEN3, QWEN25, measured};
use std::cell::Cell;

thread_local! { static FREE: Cell<u64> = const { Cell::new(7000) }; }

fn memory() -> Result<GpuMemory, CliError> {
    Ok(GpuMemory {
        name: "NVIDIA GeForce RTX 5050 Laptop GPU".into(),
        total_mib: 8151,
        free_mib: FREE.get(),
        allocations: HashMap::new(),
    })
}
fn partial_memory() -> Result<GpuMemory, CliError> {
    let mut snapshot = memory()?;
    snapshot.allocations.insert(1, 1950);
    Ok(snapshot)
}
fn daemon(dir: &std::path::Path) -> Arc<Daemon<FakeEngine>> {
    let mut daemon = tests::test_daemon(dir);
    Arc::get_mut(&mut daemon).unwrap().gpu_memory = memory;
    daemon
}

#[tokio::test]
async fn largest_profiles_coexist_and_same_model_inherits_budget() {
    let dir = tempfile::tempdir().unwrap();
    let d = daemon(dir.path());
    let a = d.run(QWEN25.into(), Some("a".into())).await.unwrap();
    let b = d.run(QWEN3.into(), Some("b".into())).await.unwrap();
    let c = d.run(QWEN25.into(), Some("c".into())).await.unwrap();
    assert_eq!(d.workers.lock().await.len(), 2);
    assert_eq!(
        d.storage().worker_profile(&a.id).unwrap(),
        Some(measured(QWEN25).unwrap().1[0])
    );
    assert_eq!(
        d.storage().worker_profile(&b.id).unwrap(),
        Some(measured(QWEN3).unwrap().1[0])
    );
    let a_profile = d.storage().worker_profile(&a.id).unwrap();
    assert_eq!(d.storage().worker_profile(&c.id).unwrap(), a_profile);
    assert!(
        d.run(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0".into(),
            Some("tiny".into())
        )
        .await
        .is_err()
    );
}

#[tokio::test]
async fn smaller_profile_at_exact_boundary_and_rejection_before_spawn() {
    let dir = tempfile::tempdir().unwrap();
    let d = daemon(dir.path());
    FREE.set(3123);
    assert_eq!(
        crate::error::exit_code(&d.run(QWEN25.into(), Some("low".into())).await.unwrap_err()),
        6
    );
    assert_eq!(d.inspect("low").unwrap().session.state, "stopped");
    assert!(d.workers.lock().await.is_empty());
    FREE.set(3124);
    let a = d.run(QWEN25.into(), Some("fits".into())).await.unwrap();
    assert_eq!(
        d.storage().worker_profile(&a.id).unwrap(),
        Some(measured(QWEN25).unwrap().1[1])
    );
    FREE.set(7000);
    let b = d.run(QWEN25.into(), Some("reuse".into())).await.unwrap();
    let a_profile = d.storage().worker_profile(&a.id).unwrap();
    assert_eq!(d.storage().worker_profile(&b.id).unwrap(), a_profile);
}

#[tokio::test]
async fn recovery_keeps_budget_and_refuses_a_smaller_replacement() {
    let dir = tempfile::tempdir().unwrap();
    let d = daemon(dir.path());
    let a = d.run(QWEN25.into(), Some("a".into())).await.unwrap();
    let saved = d.storage().worker_profile(&a.id).unwrap();
    d.workers.lock().await.clear();
    d.transition(&a.id, SessionState::Active, SessionState::Stopped, None)
        .unwrap();
    FREE.set(3124);
    assert_eq!(
        crate::error::exit_code(&d.recover("a").await.unwrap_err()),
        6
    );
    assert_eq!(d.inspect("a").unwrap().session.state, "stopped");
    assert_eq!(d.storage().worker_profile(&a.id).unwrap(), saved);
    FREE.set(7000);
    assert_eq!(d.recover("a").await.unwrap().id, a.id);
    assert_eq!(d.workers.lock().await[QWEN25].profile, saved);
}

#[tokio::test]
async fn sleeping_worker_keeps_reservation_and_wake_checks_live_headroom() {
    let dir = tempfile::tempdir().unwrap();
    let d = daemon(dir.path());
    d.run(QWEN25.into(), Some("a".into())).await.unwrap();
    let engine = d.workers.lock().await[QWEN25].engine.clone();
    engine.lock().await.sleep().await.unwrap();
    // 2292 reserved for sleeping Qwen2.5 + 3014 for Qwen3 + 1024 headroom.
    FREE.set(5881);
    assert!(d.run(QWEN3.into(), Some("no-room".into())).await.is_err());
    FREE.set(6330);
    d.run(QWEN3.into(), Some("fits".into())).await.unwrap();
    FREE.set(6329);
    assert!(matches!(
        d.begin_query("a", "hi").await,
        Err(CliError::Resource(_))
    ));
    assert!(engine.lock().await.is_sleeping().await);
    FREE.set(6330);
    d.begin_query("a", "hi").await.unwrap();
    assert!(!engine.lock().await.is_sleeping().await);
}

#[tokio::test]
async fn concurrent_profiled_start_has_one_owner_and_retry_reuses() {
    let dir = tempfile::tempdir().unwrap();
    let d = daemon(dir.path());
    let (a, b) = tokio::join!(
        d.run(QWEN3.into(), Some("a".into())),
        d.run(QWEN3.into(), Some("b".into()))
    );
    assert!(a.is_ok() ^ b.is_ok());
    assert_eq!(d.workers.lock().await.len(), 1);
    let name = if a.is_err() { "a" } else { "b" };
    d.recover(name).await.unwrap();
    assert_eq!(d.workers.lock().await.len(), 1);
}

#[tokio::test]
async fn profile_survives_daemon_restart_and_corruption_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ralph.db");
    let mut d = Daemon::with_gpu_check(
        Storage::open(&path).unwrap(),
        dir.path().into(),
        FakeEngine::new,
        || true,
    );
    Arc::get_mut(&mut d).unwrap().gpu_memory = memory;
    let a = d.run(QWEN25.into(), Some("a".into())).await.unwrap();
    let b = d.run(QWEN3.into(), Some("b".into())).await.unwrap();
    let stored = Storage::open(&path).unwrap();
    assert_eq!(
        stored.worker_profile(&a.id).unwrap(),
        Some(measured(QWEN25).unwrap().1[0])
    );
    assert_eq!(
        stored.worker_profile(&b.id).unwrap(),
        Some(measured(QWEN3).unwrap().1[0])
    );
    stored.test_execute("UPDATE session_profiles SET profile = '{\"max_context\":16384,\"kv_mib\":448,\"peak_mib\":1}'");
    let row = stored.resolve(&a.id).unwrap();
    d.workers.lock().await.clear();
    assert!(d.admitted_spec(&row, true).await.is_err());
}

#[tokio::test]
async fn accounted_allocations_choose_qwen3_small_bucket_without_resizing_qwen25() {
    let dir = tempfile::tempdir().unwrap();
    let mut d = daemon(dir.path());
    Arc::get_mut(&mut d).unwrap().gpu_memory = partial_memory;
    FREE.set(4120);
    let a = d.run(QWEN25.into(), Some("a".into())).await.unwrap();
    let b = d.run(QWEN3.into(), Some("b".into())).await.unwrap();
    assert_eq!(
        d.storage().worker_profile(&a.id).unwrap(),
        Some(measured(QWEN25).unwrap().1[0])
    );
    assert_eq!(
        d.storage().worker_profile(&b.id).unwrap(),
        Some(measured(QWEN3).unwrap().1[1])
    );
}

#[tokio::test]
async fn stored_profile_cannot_be_silently_replaced_by_resident_budget() {
    let dir = tempfile::tempdir().unwrap();
    let d = daemon(dir.path());
    let a = d.run(QWEN25.into(), Some("a".into())).await.unwrap();
    let historical = measured(QWEN25).unwrap().1[1];
    d.storage().set_worker_profile(&a.id, historical).unwrap();
    d.transition(&a.id, SessionState::Active, SessionState::Stopped, None)
        .unwrap();
    let error = d.recover("a").await.unwrap_err();
    assert_eq!(crate::error::exit_code(&error), 4);
    assert_eq!(d.storage().worker_profile(&a.id).unwrap(), Some(historical));
    assert_eq!(
        d.workers.lock().await[QWEN25].profile,
        Some(measured(QWEN25).unwrap().1[0])
    );
}
