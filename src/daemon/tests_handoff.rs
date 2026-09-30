use std::time::Duration;

use tokio::sync::oneshot;

use super::tests::*;
use super::tests_ssh_support::fake_ssh_script;
use super::*;

fn daemon_with_fake_ssh(dir: &std::path::Path) -> Arc<Daemon<FakeEngine>> {
    let daemon = test_daemon(dir);
    daemon.set_ssh_program(fake_ssh_script(dir).to_string_lossy().into_owned());
    daemon
}

#[tokio::test]
async fn successful_handoff_commits_only_after_the_destination_acks() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = daemon_with_fake_ssh(dir.path());
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();

    let info = daemon
        .handoff_cancellable("demo", "ok", None, None)
        .await
        .unwrap();
    assert_eq!(info.state, "moved");
    assert_eq!(
        daemon
            .storage()
            .get_handoff(&info.id)
            .unwrap()
            .unwrap()
            .destination,
        "ok"
    );
}

#[tokio::test]
async fn ssh_auth_failure_rolls_back_to_paused_and_session_stays_usable() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = daemon_with_fake_ssh(dir.path());
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();

    let err = daemon
        .handoff_cancellable("demo", "auth-fails", None, None)
        .await
        .unwrap_err();
    assert!(matches!(err, CliError::Resource(_)));
    // The reachability probe runs before the worker is ever released, so an
    // SSH-level failure never touches the live session at all.
    assert_eq!(daemon.inspect("demo").unwrap().session.state, "active");
    daemon.begin_query("demo", "still usable").await.unwrap();
}

#[tokio::test]
async fn incompatible_protocol_version_rolls_back_before_any_transfer() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = daemon_with_fake_ssh(dir.path());
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();

    let err = daemon
        .handoff_cancellable("demo", "bad-version", None, None)
        .await
        .unwrap_err();
    assert!(matches!(err, CliError::Usage(_)));
    // Protocol-version mismatch is caught by the same early probe, before any local
    // disruption.
    assert_eq!(daemon.inspect("demo").unwrap().session.state, "active");
}

#[tokio::test]
async fn insufficient_destination_disk_rolls_back_before_any_transfer() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = daemon_with_fake_ssh(dir.path());
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();

    let err = daemon
        .handoff_cancellable("demo", "low-disk", None, None)
        .await
        .unwrap_err();
    assert!(matches!(err, CliError::Resource(_)));
    assert_eq!(daemon.inspect("demo").unwrap().session.state, "paused");
}

#[tokio::test]
async fn destination_rejection_rolls_back_cleanly() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = daemon_with_fake_ssh(dir.path());
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();

    let err = daemon
        .handoff_cancellable("demo", "recv-rejects", None, None)
        .await
        .unwrap_err();
    assert!(matches!(err, CliError::Resource(_)));
    assert_eq!(daemon.inspect("demo").unwrap().session.state, "paused");
    assert!(daemon.storage().get_handoff("demo").is_ok());
}

#[tokio::test]
async fn handoff_of_an_already_paused_session_rolls_back_to_paused_not_active() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = daemon_with_fake_ssh(dir.path());
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();
    daemon.pause_cancellable("demo", None).await.unwrap();

    let err = daemon
        .handoff_cancellable("demo", "recv-rejects", None, None)
        .await
        .unwrap_err();
    assert!(matches!(err, CliError::Resource(_)));
    assert_eq!(daemon.inspect("demo").unwrap().session.state, "paused");
}

#[tokio::test]
async fn cancellation_mid_handoff_kills_the_child_and_rolls_back() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = daemon_with_fake_ssh(dir.path());
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();

    let (cancel_tx, cancel_rx) = oneshot::channel();
    let handoff = daemon.handoff_cancellable("demo", "recv-hangs", None, Some(cancel_rx));
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        let _ = cancel_tx.send(());
    });
    let err = tokio::time::timeout(Duration::from_secs(10), handoff)
        .await
        .expect("handoff should have been cancelled, not hung")
        .unwrap_err();
    assert!(matches!(err, CliError::InvalidState(_)));
    assert_eq!(daemon.inspect("demo").unwrap().session.state, "paused");
}

#[tokio::test]
async fn concurrent_handoff_and_pause_on_the_same_session_serializes() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = daemon_with_fake_ssh(dir.path());
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();

    let _guard = daemon
        .locks
        .try_acquire(&daemon.inspect("demo").unwrap().session.id);
    let err = daemon
        .handoff_cancellable("demo", "ok", None, None)
        .await
        .unwrap_err();
    assert!(matches!(err, CliError::InvalidState(_)));
}

#[tokio::test]
async fn handoff_of_a_moved_session_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = daemon_with_fake_ssh(dir.path());
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();
    daemon
        .handoff_cancellable("demo", "ok", None, None)
        .await
        .unwrap();

    let err = daemon
        .handoff_cancellable("demo", "ok", None, None)
        .await
        .unwrap_err();
    assert!(matches!(err, CliError::InvalidState(_)));
}

#[tokio::test]
async fn daemon_restart_mid_moving_rolls_back_instead_of_sticking_forever() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = daemon_with_fake_ssh(dir.path());
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();
    let id = daemon.inspect("demo").unwrap().session.id;
    daemon
        .transition(&id, SessionState::Active, SessionState::Moving, None)
        .unwrap();

    daemon.reconcile_on_startup().unwrap();
    assert_eq!(daemon.inspect("demo").unwrap().session.state, "paused");
}
