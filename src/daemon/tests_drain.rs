use std::time::Duration;

use super::tests::*;
use super::tests_ssh_support::fake_ssh_script;
use super::*;

fn daemon_with_fake_ssh(dir: &std::path::Path) -> Arc<Daemon<FakeEngine>> {
    let daemon = test_daemon(dir);
    daemon.set_ssh_program(fake_ssh_script(dir).to_string_lossy().into_owned());
    daemon
}

#[tokio::test]
async fn drain_with_two_sessions_requires_yes_and_is_a_no_op_without_it() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    daemon
        .run("model".to_string(), Some("a".to_string()))
        .await
        .unwrap();
    daemon
        .run("model".to_string(), Some("b".to_string()))
        .await
        .unwrap();

    let report = daemon
        .drain_cancellable("local/gpu0", None, false, None)
        .await
        .unwrap();
    assert!(!report.executed);
    assert_eq!(report.sessions.len(), 2);
    assert_eq!(daemon.inspect("a").unwrap().session.state, "active");
    assert_eq!(daemon.inspect("b").unwrap().session.state, "active");
}

#[tokio::test]
async fn drain_with_yes_hibernates_every_active_session_on_the_location() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    daemon
        .run("model".to_string(), Some("a".to_string()))
        .await
        .unwrap();
    daemon
        .run("model".to_string(), Some("b".to_string()))
        .await
        .unwrap();

    let report = daemon
        .drain_cancellable("local/gpu0", None, true, None)
        .await
        .unwrap();
    assert!(report.executed);
    assert!(report.all_safe);
    assert_eq!(daemon.inspect("a").unwrap().session.state, "hibernated");
    assert_eq!(daemon.inspect("b").unwrap().session.state, "hibernated");
}

#[tokio::test]
async fn drain_skips_a_locked_session_and_still_reports_the_others() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    daemon
        .run("model".to_string(), Some("a".to_string()))
        .await
        .unwrap();
    daemon
        .run("model".to_string(), Some("b".to_string()))
        .await
        .unwrap();

    let blocked_id = daemon.inspect("a").unwrap().session.id;
    let _guard = daemon.locks.try_acquire(&blocked_id);
    let report = daemon
        .drain_cancellable("local/gpu0", None, true, None)
        .await
        .unwrap();

    assert!(!report.all_safe);
    let a = report.sessions.iter().find(|o| o.name == "a").unwrap();
    assert!(!a.ok);
    let b = report.sessions.iter().find(|o| o.name == "b").unwrap();
    assert!(b.ok);
    assert_eq!(daemon.inspect("a").unwrap().session.state, "active");
    assert_eq!(daemon.inspect("b").unwrap().session.state, "hibernated");
}

#[tokio::test]
async fn drain_in_progress_blocks_new_admissions_and_clears_the_flag_afterward() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = daemon_with_fake_ssh(dir.path());
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();

    let draining_daemon = daemon.clone();
    let handle = tokio::spawn(async move {
        draining_daemon
            .drain_cancellable("local/gpu0", Some("slow".to_string()), true, None)
            .await
    });
    tokio::time::sleep(Duration::from_millis(50)).await;

    let err = daemon
        .run_cancellable("model".to_string(), Some("other".to_string()), None)
        .await
        .unwrap_err();
    assert!(matches!(
        err,
        CliError::SessionOperation { source, .. } if matches!(*source, CliError::Resource(_))
    ));

    let report = handle.await.unwrap().unwrap();
    assert!(report.all_safe);

    daemon
        .run_cancellable("model".to_string(), Some("after-drain".to_string()), None)
        .await
        .unwrap();
}

#[tokio::test]
async fn drain_with_a_destination_hands_sessions_off_instead_of_hibernating() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = daemon_with_fake_ssh(dir.path());
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();

    let report = daemon
        .drain_cancellable("local/gpu0", Some("ok".to_string()), true, None)
        .await
        .unwrap();
    assert!(report.all_safe);
    assert_eq!(report.sessions[0].action, "handoff");
    assert_eq!(daemon.inspect("demo").unwrap().session.state, "moved");
}

/// Regression test: `SessionRow.location` is stored as `"local/gpu0"`, but
/// RALPH_SPEC.md's own examples call it bare `gpu0` (`ralph drain gpu0`) — the bare
/// form must match, not just the exact stored string.
#[tokio::test]
async fn drain_accepts_the_bare_gpu_name_from_the_spec_examples() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();

    let report = daemon
        .drain_cancellable("gpu0", None, true, None)
        .await
        .unwrap();
    assert_eq!(report.sessions.len(), 1);
    assert!(report.all_safe);
    assert_eq!(daemon.inspect("demo").unwrap().session.state, "hibernated");
}

#[tokio::test]
async fn drain_of_an_unrelated_location_touches_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();

    let report = daemon
        .drain_cancellable("gpu99", None, true, None)
        .await
        .unwrap();
    assert!(report.sessions.is_empty());
    assert_eq!(daemon.inspect("demo").unwrap().session.state, "active");
}
