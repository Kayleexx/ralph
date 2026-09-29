//! `.ralph` artifact export/import for `token_turns` (`daemon::portable`). Split out of
//! `token_log.rs` to keep that file focused on the live-table read/write path; this file
//! reuses its checksum scheme and row validator rather than duplicating either.
use super::token_log::{RawTurn, update_counts, validate_and_collect};
use super::{Storage, StorageError};
use crate::engine::ChatMessage;
use rusqlite::params;
use serde::{Deserialize, Serialize};

/// Raw row shape for a `.ralph` artifact's turn history. `validate_turns` runs the same
/// integrity checks `replay_turns` runs on a live table, but directly over an in-memory
/// slice — so an import can be fully validated before anything is written to the
/// canonical database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnRow {
    pub seq: i64,
    pub role: String,
    pub content: String,
    pub token_count: i64,
    pub committed_at: String,
    pub token_ids: Option<String>,
    pub complete: Option<i64>,
    pub checksum: Option<String>,
}

impl From<&TurnRow> for RawTurn {
    fn from(row: &TurnRow) -> Self {
        RawTurn {
            seq: row.seq,
            role: row.role.clone(),
            content: row.content.clone(),
            count: row.token_count,
            ids: row.token_ids.clone(),
            complete: row.complete,
            checksum: row.checksum.clone(),
        }
    }
}

/// Validates a `.ralph` artifact's exported turn rows before any of them are written to
/// the database — the same integrity gate `replay_turns` applies to a live table, run
/// here against an in-memory slice instead (`daemon::portable::import`).
pub fn validate_turns(rows: &[TurnRow]) -> Result<Vec<ChatMessage>, StorageError> {
    let raw = rows.iter().map(RawTurn::from).collect();
    validate_and_collect(raw).map(|(messages, _total)| messages)
}

impl Storage {
    /// Raw rows, unvalidated — for `.ralph` export (`daemon::portable`).
    pub fn export_turns(&self, session: &str) -> Result<Vec<TurnRow>, StorageError> {
        let mut stmt = self.conn.prepare("SELECT seq,role,content,token_count,committed_at,token_ids,complete,checksum FROM token_turns WHERE session_id=?1 ORDER BY seq")?;
        let rows = stmt.query_map([session], |r| {
            Ok(TurnRow {
                seq: r.get(0)?,
                role: r.get(1)?,
                content: r.get(2)?,
                token_count: r.get(3)?,
                committed_at: r.get(4)?,
                token_ids: r.get(5)?,
                complete: r.get(6)?,
                checksum: r.get(7)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(StorageError::from)
    }

    /// Reinserts previously-exported rows under a new `session`. Each row's `checksum`
    /// is carried over verbatim — it's a function of `(seq, role, content, count, ids,
    /// complete)` only, never `session_id` (see `token_log::checksum`), so it stays
    /// valid.
    pub fn import_turns(
        &self,
        session: &str,
        rows: &[TurnRow],
        now: &str,
    ) -> Result<(), StorageError> {
        let tx = self.conn.unchecked_transaction()?;
        for row in rows {
            tx.execute(
                "INSERT INTO token_turns
                 (session_id, seq, role, content, token_count, committed_at, token_ids, complete, checksum)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",
                params![
                    session,
                    row.seq,
                    row.role,
                    row.content,
                    row.token_count,
                    row.committed_at,
                    row.token_ids,
                    row.complete,
                    row.checksum,
                ],
            )?;
        }
        update_counts(&tx, session, now)?;
        tx.commit()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn export_then_import_under_a_new_session_id_replays_cleanly() {
        let storage = Storage::open_in_memory().unwrap();
        let seq = storage.record_user_turn("a", "hello", &[1], "t").unwrap();
        storage
            .flush_assistant_turn("a", seq, "hi", &[2], "t")
            .unwrap();
        storage.finish_turn("a", seq, true).unwrap();

        let rows = storage.export_turns("a").unwrap();
        storage.import_turns("b", &rows, "t2").unwrap();

        assert_eq!(
            storage.replay_turns("b").unwrap(),
            storage.replay_turns("a").unwrap()
        );
    }

    #[test]
    fn validate_turns_rejects_a_tampered_row() {
        let storage = Storage::open_in_memory().unwrap();
        let seq = storage.record_user_turn("a", "hello", &[1], "t").unwrap();
        storage
            .flush_assistant_turn("a", seq, "hi", &[2], "t")
            .unwrap();
        storage.finish_turn("a", seq, true).unwrap();

        let mut rows = storage.export_turns("a").unwrap();
        rows[0].content = "tampered".to_string();
        assert!(matches!(
            validate_turns(&rows),
            Err(StorageError::Corrupt(_))
        ));
    }

    // Confirms the checksum scheme is genuinely session-id-independent, which is what
    // makes `import_turns` safe to reuse an archived row's checksum verbatim.
    #[test]
    fn export_import_reuses_the_original_checksum_under_a_different_session_id() {
        let storage = Storage::open_in_memory().unwrap();
        storage.record_user_turn("a", "hi", &[1], "t").unwrap();
        let rows = storage.export_turns("a").unwrap();
        storage.import_turns("b", &rows, "t2").unwrap();
        assert_eq!(
            storage.export_turns("a").unwrap()[0].checksum,
            storage.export_turns("b").unwrap()[0].checksum,
        );
    }
}
