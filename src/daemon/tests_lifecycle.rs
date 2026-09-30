use super::tests::*;
use super::*;
use lifecycle::ResumeMode;

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
/// different engine, ...) — never trust a stale pointer.
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
        daemon
            .workers
            .lock()
            .await
            .contains_key(&wk("shared-model")),
        "worker must stay up for the still-active session"
    );

    daemon.pause_cancellable("b", None).await.unwrap();
    assert!(
        daemon.workers.lock().await.is_empty(),
        "worker is freed once the last attached session pauses"
    );
}

/// Two concurrent `resume` calls on the same session must have clearly defined behavior
/// (per-session lock): exactly one proceeds, the other fails fast
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

/// Same lock, different transient state per op — exactly one of pause/hibernate must
/// win when both race the same active session.
#[tokio::test]
async fn concurrent_pause_and_hibernate_on_the_same_session_serializes() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();

    let (a, b) = tokio::join!(
        daemon.pause_cancellable("demo", None),
        daemon.hibernate_cancellable("demo", None)
    );
    assert!(a.is_ok() ^ b.is_ok(), "exactly one op wins the lock");
    let state = daemon.inspect("demo").unwrap().session.state;
    assert!(state == "paused" || state == "hibernated");
}

/// A session mid-`Moving` must stay opaque to concurrent local lifecycle ops — no
/// split-brain where one op thinks the session moved and another thinks it's live.
#[tokio::test]
async fn concurrent_recover_and_pause_on_a_moving_session_both_fail_cleanly() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    let info = daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();
    daemon
        .transition(&info.id, SessionState::Active, SessionState::Moving, None)
        .unwrap();

    let (a, b) = tokio::join!(
        daemon.recover_cancellable("demo", None),
        daemon.pause_cancellable("demo", None)
    );
    assert!(a.is_err() && b.is_err(), "neither op applies mid-move");
    assert_eq!(daemon.inspect("demo").unwrap().session.state, "moving");
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
        workers.remove(&wk("shared-model"));
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

/// A daemon killed mid-`pause`/`hibernate`/`resume` must not leave a session stuck in
/// that transient state forever — `reconcile_on_startup` rolls each back to a stable
/// landing state, same as it already does for a killed `handoff`/`drain`.
#[tokio::test]
async fn daemon_restart_mid_pause_hibernate_or_resume_rolls_back_instead_of_sticking_forever() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    daemon
        .run("model".to_string(), Some("pausing".to_string()))
        .await
        .unwrap();
    let id = daemon.inspect("pausing").unwrap().session.id;
    daemon
        .transition(&id, SessionState::Active, SessionState::Pausing, None)
        .unwrap();

    daemon
        .run("model".to_string(), Some("hibernating".to_string()))
        .await
        .unwrap();
    let id = daemon.inspect("hibernating").unwrap().session.id;
    daemon
        .transition(&id, SessionState::Active, SessionState::Hibernating, None)
        .unwrap();

    daemon
        .run("model".to_string(), Some("resuming".to_string()))
        .await
        .unwrap();
    daemon.pause_cancellable("resuming", None).await.unwrap();
    let id = daemon.inspect("resuming").unwrap().session.id;
    daemon
        .transition(&id, SessionState::Paused, SessionState::Resuming, None)
        .unwrap();

    daemon.reconcile_on_startup().unwrap();
    assert_eq!(daemon.inspect("pausing").unwrap().session.state, "paused");
    assert_eq!(
        daemon.inspect("hibernating").unwrap().session.state,
        "hibernated"
    );
    assert_eq!(daemon.inspect("resuming").unwrap().session.state, "paused");
}
