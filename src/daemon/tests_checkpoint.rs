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

/// Unlike pause, a checkpoint write failure must not block hibernation — hibernation is
/// allowed to go logical-only when acceleration state can't be saved.
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

#[tokio::test]
async fn checkpoint_refuses_and_leaves_the_session_unaffected_when_disk_is_full() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();
    daemon.set_disk_full(true);

    assert!(matches!(
        daemon.checkpoint("demo").await,
        Err(CliError::Resource(_))
    ));
    assert_eq!(daemon.inspect("demo").unwrap().session.state, "active");
}

#[tokio::test]
async fn pause_rolls_back_to_active_when_disk_is_full() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();
    daemon.set_disk_full(true);

    assert!(matches!(
        daemon.pause_cancellable("demo", None).await,
        Err(CliError::Resource(_))
    ));
    assert_eq!(daemon.inspect("demo").unwrap().session.state, "active");
    assert!(
        daemon.workers.lock().await.contains_key(&wk("model")),
        "worker must stay attached: nothing was released"
    );
}

/// Hibernation cannot always save acceleration state — a full disk degrades to
/// logical-only, it never blocks the GPU release the way `pause` fails closed.
#[tokio::test]
async fn hibernate_continues_logical_only_when_disk_is_full() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();
    daemon.set_disk_full(true);

    let hibernated = daemon.hibernate_cancellable("demo", None).await.unwrap();
    assert_eq!(hibernated.state, "hibernated");
    assert!(daemon.workers.lock().await.is_empty());
}

/// A checkpoint must not interleave with an in-flight generation on the same session —
/// both go through the same per-session lock `begin_query` already holds.
#[tokio::test]
async fn checkpoint_fails_while_a_query_is_in_flight() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();
    let guard = daemon.begin_query("demo", "hello").await.unwrap();

    assert!(matches!(
        daemon.checkpoint("demo").await,
        Err(CliError::InvalidState(_))
    ));

    drop(guard);
    daemon.checkpoint("demo").await.unwrap();
}

/// A present, fingerprint-compatible checkpoint whose bytes are actually garbage fails
/// at engine startup, not at the fingerprint gate — resume must retry portable instead
/// of surfacing the raw engine error, and drop the now-known-bad pointer row.
#[tokio::test]
async fn resume_falls_back_to_portable_when_the_native_checkpoint_is_corrupt() {
    let dir = tempfile::tempdir().unwrap();
    let storage = Storage::open_in_memory().unwrap();
    let daemon = Daemon::with_gpu_check(
        storage,
        dir.path().to_path_buf(),
        super::tests_engines::FlakyNativeEngine::new,
        || true,
    );
    daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();
    let paused = daemon.pause_cancellable("demo", None).await.unwrap();
    let checkpoint = daemon
        .storage()
        .get_checkpoint(&paused.id)
        .unwrap()
        .unwrap();
    std::fs::create_dir_all(&checkpoint.kv_dir).unwrap();
    std::fs::write(
        std::path::Path::new(&checkpoint.kv_dir).join("garbage.bin"),
        b"junk",
    )
    .unwrap();

    let resumed = daemon
        .resume_cancellable("demo", ResumeMode::Auto, None)
        .await
        .unwrap();
    assert!(!resumed.native);
    assert_eq!(resumed.session.state, "active");
    assert!(
        daemon
            .storage()
            .get_checkpoint(&paused.id)
            .unwrap()
            .is_none(),
        "corrupt pointer row should be dropped, not retried forever"
    );
}

/// The content-signature check (recorded once the worker that wrote it has actually
/// stopped) must reject an offer of native resume before vLLM ever sees tampered bytes
/// — real vLLM 0.30.0 does not reliably error on corrupted `OffloadingConnector` tier
/// files (confirmed live), so this has to happen on ralph's side, before attach.
#[tokio::test]
async fn kv_offload_is_not_offered_when_the_content_signature_no_longer_matches() {
    let dir = tempfile::tempdir().unwrap();
    let daemon = test_daemon(dir.path());
    let info = daemon
        .run("model".to_string(), Some("demo".to_string()))
        .await
        .unwrap();
    let paused = daemon.pause_cancellable("demo", None).await.unwrap();
    let row = daemon.storage().resolve(&paused.id).unwrap();
    let checkpoint = daemon.storage().get_checkpoint(&row.id).unwrap().unwrap();
    // Mirrors vLLM's own real `TieringOffloadingSpec` layout, which nests block files
    // several directories deep (`<hash>_r0/32d/78_g0/*.bin`) — a signature scheme that
    // only looked at the top level would never see real content at all.
    let kv_dir = std::path::Path::new(&checkpoint.kv_dir);
    let nested = kv_dir.join("shard").join("block");
    std::fs::create_dir_all(&nested).unwrap();
    std::fs::write(nested.join("data.bin"), b"real content").unwrap();
    crate::engine::vllm::kv_offload::record_content_signature(kv_dir);
    assert!(
        daemon.kv_offload_for(&row).is_some(),
        "an intact signature must not block an otherwise-valid checkpoint"
    );

    std::fs::write(nested.join("data.bin"), b"tampered").unwrap();
    assert!(
        daemon.kv_offload_for(&row).is_none(),
        "tampered content must never be offered for native resume"
    );
    assert_eq!(info.id, row.id);
}
