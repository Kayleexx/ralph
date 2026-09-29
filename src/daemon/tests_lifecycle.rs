use super::tests::*;
use super::*;
use lifecycle::ResumeMode;

#[tokio::test]
async fn checkpoint_requires_an_active_session() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    let info = daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();
    daemon.checkpoint("demo").await.unwrap();

    daemon
        .transition(&info.id, SessionState::Active, SessionState::Stopped, None)
        .unwrap();
    assert!(matches!(
        daemon.checkpoint("demo").await,
        Err(CliError::InvalidState(_))
    ));
}

#[tokio::test]
async fn pause_then_resume_round_trips_back_to_active() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();

    let paused = daemon.pause_cancellable("demo", None).await.unwrap();
    assert_eq!(paused.state, "paused");
    assert!(daemon.workers.lock().await.is_empty(), "GPU worker freed");

    let resumed = daemon
        .resume_cancellable("demo", ResumeMode::Auto, None)
        .await
        .unwrap();
    assert_eq!(resumed.session.state, "active");
}

#[tokio::test]
async fn resuming_an_already_active_session_is_a_noop() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();
    let resumed = daemon
        .resume_cancellable("demo", ResumeMode::Auto, None)
        .await
        .unwrap();
    assert_eq!(resumed.session.state, "active");
}

/// `pause` always records a checkpoint pointer, so the only way `--fast-only` sees "no
/// compatible checkpoint" is a fingerprint that no longer matches (moved model revision,
/// different engine, ...) — never trust a stale pointer (Invariant 3).
#[tokio::test]
async fn resume_fast_only_with_an_incompatible_fingerprint_fails_clearly_and_stays_paused() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();
    let info = daemon.pause_cancellable("demo", None).await.unwrap();
    daemon
        .storage()
        .set_resolved(&info.id, Some("moved-revision"), Some("fake"), "t")
        .unwrap();

    let err = daemon
        .resume_cancellable("demo", ResumeMode::FastOnly, None)
        .await
        .unwrap_err();
    assert!(matches!(err, CliError::Resource(_)));
    assert_eq!(daemon.inspect("demo").unwrap().session.state, "paused");
}

/// Pausing one session sharing a worker must not disturb the other still-active session.
#[tokio::test]
async fn pausing_one_of_two_sessions_sharing_a_worker_keeps_the_other_active() {
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

    daemon.pause_cancellable("a", None).await.unwrap();
    assert_eq!(daemon.inspect("a").unwrap().session.state, "paused");
    assert_eq!(daemon.inspect("b").unwrap().session.state, "active");
    assert!(
        daemon.workers.lock().await.contains_key("shared-model"),
        "worker must stay up for the still-active session"
    );

    daemon.pause_cancellable("b", None).await.unwrap();
    assert!(
        daemon.workers.lock().await.is_empty(),
        "worker is freed once the last attached session pauses"
    );
}

/// Two concurrent `resume` calls on the same session must have clearly defined behavior
/// (per-session lock, RALPH_SPEC.md §16.7): exactly one proceeds, the other fails fast
/// with a clear "already in progress" rather than racing the state machine.
#[tokio::test]
async fn concurrent_resume_on_the_same_session_serializes() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();
    daemon.pause_cancellable("demo", None).await.unwrap();

    let (a, b) = tokio::join!(
        daemon.resume_cancellable("demo", ResumeMode::Auto, None),
        daemon.resume_cancellable("demo", ResumeMode::Auto, None)
    );
    assert!(a.is_ok() ^ b.is_ok(), "exactly one resume wins the lock");
    assert_eq!(daemon.inspect("demo").unwrap().session.state, "active");
}

#[tokio::test]
async fn hibernate_then_resume_round_trips_back_to_active() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();

    let hibernated = daemon.hibernate_cancellable("demo", None).await.unwrap();
    assert_eq!(hibernated.state, "hibernated");
    assert!(daemon.workers.lock().await.is_empty(), "GPU worker freed");

    let resumed = daemon
        .resume_cancellable("demo", ResumeMode::Auto, None)
        .await
        .unwrap();
    assert_eq!(resumed.session.state, "active");
}

/// Unlike pause, a checkpoint write failure must not block hibernation — §16.7
/// "Hibernation cannot save acceleration state" explicitly allows logical-only.
#[tokio::test]
async fn hibernate_succeeds_even_when_the_checkpoint_write_fails() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();
    daemon.inject_storage_fault("DROP TABLE checkpoints");

    let hibernated = daemon.hibernate_cancellable("demo", None).await.unwrap();
    assert_eq!(hibernated.state, "hibernated");
}

#[tokio::test]
async fn enforce_kv_quota_evicts_oldest_other_sessions_first() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    let write_checkpoint = |session_id: &str, created_at: &str, bytes: usize| {
        let kv_dir = dir.path().join("sessions").join(session_id).join("kvcache");
        std::fs::create_dir_all(&kv_dir).unwrap();
        std::fs::write(kv_dir.join("block.bin"), vec![0u8; bytes]).unwrap();
        daemon
            .storage()
            .upsert_checkpoint(&crate::storage::checkpoints::CheckpointRow {
                session_id: session_id.to_string(),
                fingerprint_json: "{}".to_string(),
                kv_dir: kv_dir.to_string_lossy().into_owned(),
                engine_id: format!("ralph-{session_id}"),
                created_at: created_at.to_string(),
            })
            .unwrap();
        kv_dir
    };
    let oldest = write_checkpoint("old", "2020-01-01T00:00:00Z", 3 * 1024);
    write_checkpoint("new", "2020-01-02T00:00:00Z", 3 * 1024);

    daemon.enforce_kv_quota_within("new", 4 * 1024);

    assert!(!oldest.exists(), "oldest checkpoint is evicted over quota");
    assert!(daemon.storage().get_checkpoint("old").unwrap().is_none());
    assert!(daemon.storage().get_checkpoint("new").unwrap().is_some());
}

/// A stale `Recovering` session for the same model (crashed earlier, never explicitly
/// `ralph recover`ed) must never count as "still needs this worker" — regression test
/// for a real bug found via manual CLI testing: `attach_or_start_worker`'s cold start
/// used to opportunistically glue every same-model `Recovering` session onto the fresh
/// worker's `session_ids`, so `hibernate`/`pause` could never see the count reach zero
/// and silently left the GPU worker running forever.
#[tokio::test]
async fn hibernate_releases_the_worker_despite_an_unrelated_recovering_session() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    daemon
        .run("shared-model".to_string(), Some("stale".to_string()))
        .await
        .unwrap();
    let stale = daemon.inspect("stale").unwrap().session;
    daemon
        .transition(
            &stale.id,
            SessionState::Active,
            SessionState::Recovering,
            None,
        )
        .unwrap();
    {
        let mut workers = daemon.workers.lock().await;
        workers.remove("shared-model");
    }

    daemon
        .run("shared-model".to_string(), Some("newt".to_string()))
        .await
        .unwrap();
    let hibernated = daemon.hibernate_cancellable("newt", None).await.unwrap();
    assert_eq!(hibernated.state, "hibernated");
    assert!(
        daemon.workers.lock().await.is_empty(),
        "worker must be freed: newt was the only session actually attached"
    );
}

#[tokio::test]
async fn pause_on_a_non_active_session_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();
    daemon.pause_cancellable("demo", None).await.unwrap();
    assert!(matches!(
        daemon.pause_cancellable("demo", None).await,
        Err(CliError::InvalidState(_))
    ));
}
