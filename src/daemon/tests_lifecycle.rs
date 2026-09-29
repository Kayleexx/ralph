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
