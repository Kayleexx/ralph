use super::tests::*;
use super::*;

#[tokio::test]
async fn export_without_a_checkpoint_still_succeeds() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();
    let output = dir.path().join("demo.ralph");

    daemon.export("demo", &output, false, true).await.unwrap();

    let bytes = std::fs::read(&output).unwrap();
    let extract = tempfile::tempdir().unwrap();
    let parsed = crate::portable::parse_archive(&bytes, extract.path()).unwrap();
    assert!(!parsed.manifest.has_accel);
}

/// Regression test: `ralph run` + `ralph checkpoint` with no prior pause/resume records
/// a checkpoint row whose kvcache directory was never actually created (the first
/// worker never has the KV-offload connector attached — see `kv_offload_for`).
/// `--with-accel` must fall back to logical-only here, not fail with a raw I/O error.
#[tokio::test]
async fn export_with_accel_tolerates_a_checkpoint_with_no_kvcache_directory_yet() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();
    daemon.checkpoint("demo").await.unwrap();
    let output = dir.path().join("demo.ralph");

    let exported = daemon.export("demo", &output, false, true).await.unwrap();
    assert_eq!(exported.name, "demo");

    let bytes = std::fs::read(&output).unwrap();
    let extract = tempfile::tempdir().unwrap();
    let parsed = crate::portable::parse_archive(&bytes, extract.path()).unwrap();
    assert!(!parsed.manifest.has_accel);
}

#[tokio::test]
async fn export_refuses_an_existing_output_without_force() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();
    let output = dir.path().join("demo.ralph");
    std::fs::write(&output, b"pre-existing").unwrap();

    let err = daemon
        .export("demo", &output, false, false)
        .await
        .unwrap_err();
    assert!(matches!(err, CliError::Usage(_)));
    assert_eq!(std::fs::read(&output).unwrap(), b"pre-existing");

    daemon.export("demo", &output, true, false).await.unwrap();
    assert_ne!(std::fs::read(&output).unwrap(), b"pre-existing");
}

#[tokio::test]
async fn import_lands_in_paused_and_is_immediately_visible() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();
    let output = dir.path().join("demo.ralph");
    daemon.export("demo", &output, false, false).await.unwrap();

    let imported = daemon
        .import(&output, Some("demo-copy".to_string()))
        .await
        .unwrap();
    assert_eq!(imported.state, "paused");
    assert_ne!(imported.id, daemon.inspect("demo").unwrap().session.id);

    let inspected = daemon.inspect("demo-copy").unwrap();
    assert_eq!(inspected.session.state, "paused");
    assert!(daemon.ps().unwrap().iter().any(|s| s.name == "demo-copy"));
}

#[tokio::test]
async fn import_preserves_turn_history() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();
    let id = daemon.inspect("demo").unwrap().session.id;
    let seq = daemon
        .record_user_turn(&id, "remember guava", &[1, 2])
        .unwrap();
    daemon
        .flush_assistant_turn(&id, seq, "ok, guava", &[3, 4])
        .unwrap();
    daemon.finish_generation(&id, seq, true).unwrap();
    let output = dir.path().join("demo.ralph");
    daemon.export("demo", &output, false, false).await.unwrap();

    let imported = daemon
        .import(&output, Some("demo-copy".to_string()))
        .await
        .unwrap();

    let original_id = daemon.inspect("demo").unwrap().session.id;
    let imported_turns = daemon.storage().replay_turns(&imported.id).unwrap();
    let original_turns = daemon.storage().replay_turns(&original_id).unwrap();
    assert_eq!(imported_turns, original_turns);
}

#[tokio::test]
async fn import_with_a_name_collision_fails_and_creates_no_visible_session() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();
    let output = dir.path().join("demo.ralph");
    daemon.export("demo", &output, false, false).await.unwrap();

    let before = daemon.storage().list().unwrap().len();
    let err = daemon.import(&output, None).await.unwrap_err();
    assert!(matches!(err, CliError::DuplicateName(_)));
    assert_eq!(daemon.storage().list().unwrap().len(), before);
}

#[tokio::test]
async fn import_of_a_corrupted_artifact_is_rejected_and_creates_no_session() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();
    let output = dir.path().join("demo.ralph");
    daemon.export("demo", &output, false, false).await.unwrap();
    let mut bytes = std::fs::read(&output).unwrap();
    let pos = bytes.len() / 2;
    bytes[pos] ^= 0xff;
    std::fs::write(&output, &bytes).unwrap();

    let before = daemon.storage().list().unwrap().len();
    let err = daemon
        .import(&output, Some("corrupt-copy".to_string()))
        .await
        .unwrap_err();
    assert!(matches!(err, CliError::Corrupt(_)));
    assert_eq!(daemon.storage().list().unwrap().len(), before);
}

/// Export vs. pause on the same session: same per-session lock contract every other
/// mutating pair (e.g. `concurrent_resume_on_the_same_session_serializes`) already relies
/// on — one wins, the other fails fast rather than racing.
#[tokio::test]
async fn concurrent_export_and_pause_on_the_same_session_serializes() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();
    let output = dir.path().join("demo.ralph");

    let _guard = daemon
        .locks
        .try_acquire(&daemon.inspect("demo").unwrap().session.id);
    let err = daemon
        .export("demo", &output, false, false)
        .await
        .unwrap_err();
    assert!(matches!(err, CliError::InvalidState(_)));
}
