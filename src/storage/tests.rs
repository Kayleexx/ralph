use super::*;

fn sample_row(id: &str, name: &str) -> SessionRow {
    SessionRow {
        id: id.to_string(),
        name: name.to_string(),
        model: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
        model_revision: Some("abc123".to_string()),
        tokenizer_revision: None,
        engine: "vllm".to_string(),
        engine_version: None,
        state: SessionState::Created,
        pid: None,
        location: "local/gpu0".to_string(),
        token_count: 0,
        created_at: "2026-01-01T00:00:00Z".to_string(),
        updated_at: "2026-01-01T00:00:00Z".to_string(),
    }
}

#[test]
fn insert_and_resolve_by_name() {
    let storage = Storage::open_in_memory().unwrap();
    storage.insert(&sample_row("01AAA", "demo")).unwrap();
    let row = storage.resolve("demo").unwrap();
    assert_eq!(row.id, "01AAA");
}

#[test]
fn duplicate_name_is_rejected() {
    let storage = Storage::open_in_memory().unwrap();
    storage.insert(&sample_row("01AAA", "demo")).unwrap();
    let err = storage.insert(&sample_row("01BBB", "demo")).unwrap_err();
    assert!(matches!(err, StorageError::DuplicateName(_)));
}

#[test]
fn resolve_by_unique_id_prefix() {
    let storage = Storage::open_in_memory().unwrap();
    storage.insert(&sample_row("01AAABBB", "demo")).unwrap();
    let row = storage.resolve("01AAA").unwrap();
    assert_eq!(row.name, "demo");
}

#[test]
fn ambiguous_prefix_is_rejected() {
    let storage = Storage::open_in_memory().unwrap();
    storage.insert(&sample_row("01AAABBB", "demo1")).unwrap();
    storage.insert(&sample_row("01AAACCC", "demo2")).unwrap();
    let err = storage.resolve("01AAA").unwrap_err();
    assert!(matches!(err, StorageError::AmbiguousPrefix(_, _)));
}

#[test]
fn unknown_session_is_not_found() {
    let storage = Storage::open_in_memory().unwrap();
    let err = storage.resolve("nope").unwrap_err();
    assert!(matches!(err, StorageError::NotFound { .. }));
}

#[test]
fn typo_in_session_name_is_suggested() {
    let storage = Storage::open_in_memory().unwrap();
    storage.insert(&sample_row("01AAA", "demo")).unwrap();
    let err = storage.resolve("dem").unwrap_err();
    match err {
        StorageError::NotFound { suggestion, .. } => {
            assert_eq!(suggestion, Some("demo".to_string()))
        }
        other => panic!("expected NotFound, got {other:?}"),
    }
}

#[test]
fn set_state_updates_atomically() {
    let storage = Storage::open_in_memory().unwrap();
    storage.insert(&sample_row("01AAA", "demo")).unwrap();
    storage
        .set_state(
            "01AAA",
            SessionState::Active,
            Some(4242),
            "2026-01-01T00:01:00Z",
        )
        .unwrap();
    let row = storage.resolve("demo").unwrap();
    assert_eq!(row.state, SessionState::Active);
    assert_eq!(row.pid, Some(4242));
}

#[test]
fn reconciliation_demotes_active_session_with_dead_pid() {
    let storage = Storage::open_in_memory().unwrap();
    storage.insert(&sample_row("01AAA", "demo")).unwrap();
    storage
        .set_state(
            "01AAA",
            SessionState::Active,
            Some(999999),
            "2026-01-01T00:01:00Z",
        )
        .unwrap();
    let demoted = storage
        .reconcile_after_restart(
            |_pid| false,
            SessionState::Recovering,
            "2026-01-01T00:02:00Z",
        )
        .unwrap();
    assert_eq!(demoted, vec!["01AAA".to_string()]);
    assert_eq!(
        storage.resolve("demo").unwrap().state,
        SessionState::Recovering
    );
}

#[test]
fn reconciliation_leaves_active_session_with_live_pid_alone() {
    let storage = Storage::open_in_memory().unwrap();
    storage.insert(&sample_row("01AAA", "demo")).unwrap();
    storage
        .set_state(
            "01AAA",
            SessionState::Active,
            Some(1),
            "2026-01-01T00:01:00Z",
        )
        .unwrap();
    let demoted = storage
        .reconcile_after_restart(
            |_pid| true,
            SessionState::Recovering,
            "2026-01-01T00:02:00Z",
        )
        .unwrap();
    assert!(demoted.is_empty());
    assert_eq!(storage.resolve("demo").unwrap().state, SessionState::Active);
}

#[test]
fn reconciliation_demotes_stuck_starting_session() {
    let storage = Storage::open_in_memory().unwrap();
    storage.insert(&sample_row("01AAA", "demo")).unwrap();
    storage
        .set_state(
            "01AAA",
            SessionState::Starting,
            None,
            "2026-01-01T00:01:00Z",
        )
        .unwrap();
    let demoted = storage
        .reconcile_after_restart(
            |_pid| true,
            SessionState::Recovering,
            "2026-01-01T00:02:00Z",
        )
        .unwrap();
    assert_eq!(demoted, vec!["01AAA".to_string()]);
}

#[test]
fn malformed_database_is_a_corruption_error_and_is_preserved() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("broken.db");
    let bytes = b"this is not a sqlite database";
    std::fs::write(&path, bytes).unwrap();
    assert!(matches!(
        Storage::open(&path),
        Err(StorageError::Corrupt(_))
    ));
    assert_eq!(std::fs::read(path).unwrap(), bytes);
}
