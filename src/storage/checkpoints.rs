//! Checkpoint metadata: a pointer to a per-session KV directory plus the fingerprint that
//! gates reusing it, not the KV bytes themselves (those are vLLM's, written continuously
//! and already atomically during ordinary generation — see `engine::vllm::kv_offload`).
//! One row per session, upserted in a single WAL transaction exactly like every other
//! table here, so a checkpoint write is Ctrl-C/SIGKILL-safe for free: readers never see a
//! half-written row, and the fingerprint gate (not a separate "valid" flag) is what
//! decides whether the pointed-at directory is still trustworthy to hand back to vLLM.
use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointRow {
    pub session_id: String,
    pub fingerprint_json: String,
    pub kv_dir: String,
    pub engine_id: String,
    pub created_at: String,
}

pub(super) const SCHEMA: &str = "CREATE TABLE IF NOT EXISTS checkpoints (
    session_id      TEXT PRIMARY KEY,
    fingerprint     TEXT NOT NULL,
    kv_dir          TEXT NOT NULL,
    engine_id       TEXT NOT NULL,
    created_at      TEXT NOT NULL
)";

impl Storage {
    pub fn get_checkpoint(&self, session_id: &str) -> Result<Option<CheckpointRow>, StorageError> {
        self.conn
            .query_row(
                "SELECT session_id, fingerprint, kv_dir, engine_id, created_at
                 FROM checkpoints WHERE session_id = ?1",
                [session_id],
                |r| {
                    Ok(CheckpointRow {
                        session_id: r.get(0)?,
                        fingerprint_json: r.get(1)?,
                        kv_dir: r.get(2)?,
                        engine_id: r.get(3)?,
                        created_at: r.get(4)?,
                    })
                },
            )
            .optional()
            .map_err(StorageError::from)
    }

    pub fn upsert_checkpoint(&self, row: &CheckpointRow) -> Result<(), StorageError> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "INSERT INTO checkpoints (session_id, fingerprint, kv_dir, engine_id, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(session_id) DO UPDATE SET
                fingerprint = excluded.fingerprint,
                kv_dir = excluded.kv_dir,
                engine_id = excluded.engine_id,
                created_at = excluded.created_at",
            params![
                row.session_id,
                row.fingerprint_json,
                row.kv_dir,
                row.engine_id,
                row.created_at,
            ],
        )?;
        tx.commit()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(session_id: &str) -> CheckpointRow {
        CheckpointRow {
            session_id: session_id.to_string(),
            fingerprint_json: "{}".to_string(),
            kv_dir: "/tmp/kv".to_string(),
            engine_id: "ralph-01AAA".to_string(),
            created_at: "t".to_string(),
        }
    }

    #[test]
    fn missing_checkpoint_is_none() {
        let storage = Storage::open_in_memory().unwrap();
        assert_eq!(storage.get_checkpoint("nope").unwrap(), None);
    }

    #[test]
    fn upsert_then_get_round_trips() {
        let storage = Storage::open_in_memory().unwrap();
        let row = sample("s");
        storage.upsert_checkpoint(&row).unwrap();
        assert_eq!(storage.get_checkpoint("s").unwrap(), Some(row));
    }

    #[test]
    fn upsert_replaces_the_previous_row_for_the_same_session() {
        let storage = Storage::open_in_memory().unwrap();
        storage.upsert_checkpoint(&sample("s")).unwrap();
        let mut updated = sample("s");
        updated.kv_dir = "/tmp/kv2".to_string();
        storage.upsert_checkpoint(&updated).unwrap();
        assert_eq!(storage.get_checkpoint("s").unwrap(), Some(updated));
    }
}
