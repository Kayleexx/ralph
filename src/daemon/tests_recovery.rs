use super::tests::*;
use super::tests_engines::AlwaysFailingEngine;
use super::*;
use std::sync::atomic::Ordering;

/// `FakeEngine::start_model` always succeeds, so the automatic bounded restart succeeds
/// too — every session sharing that worker lands back `Active` on its own, not just
/// `Recovering` with a runnable worker nobody told it about.
#[tokio::test]
async fn worker_crash_demotes_every_attached_session_and_auto_restart_reactivates_them() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    daemon
        .run("shared-model".to_string(), Some("a".to_string()))
        .await
        .unwrap();
    daemon
        .run("shared-model".to_string(), Some("b".to_string()))
        .await
        .unwrap();

    let exited = {
        let workers = daemon.workers.lock().await;
        let entry = workers.get(&wk("shared-model")).unwrap();
        entry.engine.lock().await.exited.clone()
    };
    exited.store(true, Ordering::SeqCst);

    wait_for_replacement(&daemon, "a", "shared-model").await;
    wait_until_active(&daemon, "a").await;
    wait_until_active(&daemon, "b").await;
}

/// A session left `Active`-but-sleeping (idle-supervisor path, not `Pausing`/
/// `Hibernating`) must demote to `Recovering` on daemon restart exactly like any other
/// `Active` session — `reconcile_after_restart`'s unconditional `Active|Starting
/// -> Recovering` demotion doesn't special-case sleep state, so nothing about being
/// asleep should trip it up.
#[tokio::test]
async fn daemon_restart_while_worker_is_sleeping_demotes_to_recovering_and_recovers() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();

    {
        let workers = daemon.workers.lock().await;
        let entry = workers.get(&wk("model")).unwrap();
        entry.engine.lock().await.sleep().await.unwrap();
        assert!(entry.engine.lock().await.is_sleeping().await);
    }
    assert_eq!(daemon.inspect("demo").unwrap().session.state, "active");

    daemon.reconcile_on_startup().unwrap();
    assert_eq!(daemon.inspect("demo").unwrap().session.state, "recovering");

    let recovered = daemon.recover("demo").await.unwrap();
    assert_eq!(recovered.state, "active");
}

/// A worker that can never come back up (every cold-start attempt fails) must still
/// converge to a terminal, non-looping outcome — `Stopped`, not an infinite retry loop,
/// and not `Failed` (a worker crash is never a session failure).
#[tokio::test]
async fn repeated_crash_converges_to_stopped_without_looping_forever() {
    let dir = tempfile::tempdir().unwrap();
    let storage = Storage::open_in_memory().unwrap();
    let daemon = Daemon::with_gpu_check(
        storage,
        dir.path().to_path_buf(),
        AlwaysFailingEngine::new,
        || true,
    );

    let now = "2026-01-01T00:00:00Z".to_string();
    daemon
        .storage()
        .insert(&SessionRow {
            id: "01AAA".to_string(),
            name: "demo".to_string(),
            model: "broken-model".to_string(),
            model_revision: None,
            tokenizer_revision: None,
            engine: "vllm".to_string(),
            engine_version: None,
            state: SessionState::Active,
            pid: None,
            location: "local/gpu0".to_string(),
            token_count: 0,
            created_at: now.clone(),
            updated_at: now,
            continuity_policy: crate::continuity::ContinuityPolicy::default(),
            continuity_target_ms: None,
        })
        .unwrap();

    recovery::handle_worker_loss(&daemon, wk("broken-model"), vec!["01AAA".to_string()], None)
        .await;

    assert_eq!(daemon.inspect("demo").unwrap().session.state, "stopped");

    // A one-shot manual `recover` afterward still tries, and still fails plainly, rather
    // than being silently swallowed or retried forever.
    assert!(daemon.recover("demo").await.is_err());
    assert_eq!(daemon.inspect("demo").unwrap().session.state, "stopped");
}

/// Worker dies before any query was ever made against a session: the session has no
/// token history at all, and recovery from an empty history must still succeed cleanly.
#[tokio::test]
async fn crash_before_any_query_recovers_with_empty_history() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();

    let exited = {
        let workers = daemon.workers.lock().await;
        workers
            .get(&wk("model"))
            .unwrap()
            .engine
            .lock()
            .await
            .exited
            .clone()
    };
    exited.store(true, Ordering::SeqCst);
    wait_for_replacement(&daemon, "demo", "model").await;

    let recovered = daemon.recover("demo").await.unwrap();
    assert_eq!(recovered.state, "active");
    assert!(daemon.replay_messages(&recovered.id).unwrap().is_empty());
}

/// Worker dies after a turn was already durably flushed: recovery must replay exactly
/// that turn — no duplication, no loss.
#[tokio::test]
async fn crash_after_a_flushed_turn_replays_it_after_recovery() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    let info = daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();

    let seq = daemon.record_user_turn(&info.id, "hello", &[1]).unwrap();
    daemon
        .flush_assistant_turn(&info.id, seq, "hi ", &[2])
        .unwrap();
    daemon
        .flush_assistant_turn(&info.id, seq, "hi there", &[2, 3])
        .unwrap();

    let exited = {
        let workers = daemon.workers.lock().await;
        workers
            .get(&wk("model"))
            .unwrap()
            .engine
            .lock()
            .await
            .exited
            .clone()
    };
    exited.store(true, Ordering::SeqCst);
    wait_for_replacement(&daemon, "demo", "model").await;

    daemon.recover("demo").await.unwrap();
    let messages = daemon.replay_messages(&info.id).unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].content, "hello");
    assert_eq!(messages[1].content, "hi there");
}

#[tokio::test]
async fn stopped_recovery_and_active_noop_preserve_identity() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    let info = daemon
        .run("model".into(), Some("demo".into()))
        .await
        .unwrap();
    daemon
        .transition(&info.id, SessionState::Active, SessionState::Stopped, None)
        .unwrap();
    let recovered = daemon.recover("demo").await.unwrap();
    assert_eq!(recovered.id, info.id);
    assert_eq!(recovered.model_revision, info.model_revision);
    assert_eq!(daemon.recover("demo").await.unwrap().state, "active");
}

#[tokio::test]
async fn concurrent_start_reserves_one_worker_and_conflicting_model_is_stopped() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    let (a, b) = tokio::join!(
        daemon.run("model".into(), Some("a".into())),
        daemon.run("model".into(), Some("b".into()))
    );
    assert!(a.is_ok() ^ b.is_ok());
    assert_eq!(daemon.workers.lock().await.len(), 1);
    assert!(daemon.run("other".into(), Some("c".into())).await.is_err());
    assert_eq!(daemon.inspect("c").unwrap().session.state, "stopped");
}

#[tokio::test]
async fn context_rejection_and_revision_mismatch_do_not_mutate_history() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    let info = daemon
        .run("model".into(), Some("demo".into()))
        .await
        .unwrap();
    assert!(matches!(
        daemon.begin_query("demo", &"x".repeat(4096)).await,
        Err(CliError::Usage(_))
    ));
    assert!(daemon.replay_messages(&info.id).unwrap().is_empty());
    daemon
        .transition(&info.id, SessionState::Active, SessionState::Stopped, None)
        .unwrap();
    daemon
        .storage()
        .set_resolved(&info.id, Some("different"), Some("fake"), "t")
        .unwrap();
    assert!(daemon.recover("demo").await.is_err());
    assert_eq!(daemon.inspect("demo").unwrap().session.state, "stopped");
    assert_eq!(
        daemon
            .inspect("demo")
            .unwrap()
            .session
            .model_revision
            .as_deref(),
        Some("different")
    );
}

#[tokio::test]
async fn immediate_replacement_crashes_share_one_retry_budget() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    let info = daemon
        .run("model".into(), Some("demo".into()))
        .await
        .unwrap();
    for _ in 0..4 {
        let engine = daemon
            .workers
            .lock()
            .await
            .remove(&wk("model"))
            .unwrap()
            .engine;
        engine.lock().await.stop_model().await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(3),
            recovery::handle_worker_loss(&daemon, wk("model"), vec![info.id.clone()], None),
        )
        .await
        .unwrap();
    }
    assert_eq!(daemon.inspect("demo").unwrap().session.state, "stopped");
    assert_eq!(daemon.storage().restart_attempts("model").unwrap(), 3);
    assert!(daemon.workers.lock().await.is_empty());
}

/// Regression test for startup GC: a leftover `handoff-recv-*.ralph`
/// staging file and a session directory whose row never made it to `insert` (the
/// crash-mid-`import` window) are both orphans by the time a fresh daemon boots — the
/// live session's own directory must survive the same sweep untouched.
#[tokio::test]
async fn reconcile_on_startup_sweeps_orphaned_handoff_recv_files_and_rowless_session_dirs() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    let info = daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();

    let sessions_root = dir.path().join("sessions");
    let stale_recv = sessions_root.join("handoff-recv-01ORPHAN.ralph");
    std::fs::write(&stale_recv, b"partial").unwrap();
    let rowless_dir = sessions_root.join("01ROWLESSORPHAN");
    std::fs::create_dir_all(&rowless_dir).unwrap();

    daemon.reconcile_on_startup().unwrap();

    assert!(!stale_recv.exists());
    assert!(!rowless_dir.exists());
    assert!(sessions_root.join(&info.id).exists());
}

#[tokio::test]
async fn concurrent_recovery_gets_busy_and_preserves_history() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    let info = daemon
        .run("model".into(), Some("demo".into()))
        .await
        .unwrap();
    daemon
        .transition(&info.id, SessionState::Active, SessionState::Stopped, None)
        .unwrap();
    daemon.workers.lock().await.clear();
    let (a, b) = tokio::join!(daemon.recover("demo"), daemon.recover("demo"));
    assert!(a.is_ok() ^ b.is_ok());
    assert_eq!(daemon.inspect("demo").unwrap().session.state, "active");
}

async fn wait_for_replacement(daemon: &Arc<Daemon<FakeEngine>>, session: &str, model: &str) {
    let _ = session;
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            if daemon.workers.lock().await.contains_key(&wk(model)) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("worker loss and replacement must not deadlock");
}

async fn wait_until_active(daemon: &Arc<Daemon<FakeEngine>>, session: &str) {
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            if daemon.inspect(session).unwrap().session.state == "active" {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("automatic restart must reactivate the session");
}
