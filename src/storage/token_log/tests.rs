use super::*;

#[test]
fn cumulative_flush_is_idempotent_and_preserves_prefix() {
    let storage = Storage::open_in_memory().unwrap();
    let seq = storage.record_user_turn("s", "hello", &[1], "t").unwrap();
    storage
        .flush_assistant_turn("s", seq, "hi", &[2], "t")
        .unwrap();
    storage
        .flush_assistant_turn("s", seq, "hi", &[2], "t")
        .unwrap();
    storage
        .flush_assistant_turn("s", seq, "hi there", &[2, 3], "t")
        .unwrap();
    assert_eq!(storage.replay_turns("s").unwrap()[1].content, "hi there");
    assert!(
        storage
            .flush_assistant_turn("s", seq, "changed", &[2], "t")
            .is_err()
    );
}
#[test]
fn corruption_is_rejected() {
    let storage = Storage::open_in_memory().unwrap();
    storage.record_user_turn("s", "hello", &[1], "t").unwrap();
    storage
        .conn
        .execute("UPDATE token_turns SET role='invalid' WHERE seq=0", [])
        .unwrap();
    assert!(matches!(
        storage.replay_turns("s"),
        Err(StorageError::Corrupt(_))
    ));
}
#[test]
fn empty_input_is_retained_and_missing_reservation_is_corrupt() {
    let storage = Storage::open_in_memory().unwrap();
    storage.record_user_turn("s", "", &[], "t").unwrap();
    assert_eq!(storage.replay_turns("s").unwrap().len(), 1);
    storage
        .conn
        .execute("DELETE FROM token_turns WHERE seq=1", [])
        .unwrap();
    assert!(matches!(
        storage.replay_turns("s"),
        Err(StorageError::Corrupt(_))
    ));
}
#[test]
fn valid_looking_payload_corruption_is_rejected_without_repair() {
    for sql in [
        "UPDATE token_turns SET content='hullo' WHERE seq=0",
        "UPDATE token_turns SET token_ids='[2]' WHERE seq=0",
        "UPDATE token_turns SET complete=1 WHERE seq=1",
    ] {
        let storage = Storage::open_in_memory().unwrap();
        storage.record_user_turn("s", "hello", &[1], "t").unwrap();
        storage.conn.execute(sql, []).unwrap();
        assert!(matches!(
            storage.replay_turns("s"),
            Err(StorageError::Corrupt(_))
        ));
    }
}
#[test]
fn interrupted_transaction_keeps_committed_history() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.db");
    let storage = Storage::open(&path).unwrap();
    let seq = storage.record_user_turn("s", "hello", &[1], "t").unwrap();
    storage
        .flush_assistant_turn("s", seq, "hi", &[2], "t")
        .unwrap();
    {
        let tx = storage.conn.unchecked_transaction().unwrap();
        tx.execute("UPDATE token_turns SET content='uncommitted'", [])
            .unwrap();
    }
    drop(storage);
    let storage = Storage::open(&path).unwrap();
    assert_eq!(storage.replay_turns("s").unwrap()[1].content, "hi");
}
