//! Where a handed-off session went. One row per session, written only once
//! `ralph handoff` has a destination ACK in hand (`daemon::handoff::handoff_cancellable`)
//! — this table never records an attempted-but-not-committed move.
use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HandoffRow {
    pub session_id: String,
    pub destination: String,
    pub remote_name: String,
    pub committed_at: String,
}

pub(super) const SCHEMA: &str = "CREATE TABLE IF NOT EXISTS handoffs (
    session_id      TEXT PRIMARY KEY,
    destination     TEXT NOT NULL,
    remote_name     TEXT NOT NULL,
    committed_at    TEXT NOT NULL
)";

impl Storage {
    pub fn get_handoff(&self, session_id: &str) -> Result<Option<HandoffRow>, StorageError> {
        self.conn
            .query_row(
                "SELECT session_id, destination, remote_name, committed_at
                 FROM handoffs WHERE session_id = ?1",
                [session_id],
                |r| {
                    Ok(HandoffRow {
                        session_id: r.get(0)?,
                        destination: r.get(1)?,
                        remote_name: r.get(2)?,
                        committed_at: r.get(3)?,
                    })
                },
            )
            .optional()
            .map_err(StorageError::from)
    }

    pub fn upsert_handoff(&self, row: &HandoffRow) -> Result<(), StorageError> {
        self.conn.execute(
            "INSERT INTO handoffs (session_id, destination, remote_name, committed_at)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(session_id) DO UPDATE SET
                destination = excluded.destination,
                remote_name = excluded.remote_name,
                committed_at = excluded.committed_at",
            params![
                row.session_id,
                row.destination,
                row.remote_name,
                row.committed_at
            ],
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(session_id: &str) -> HandoffRow {
        HandoffRow {
            session_id: session_id.to_string(),
            destination: "user@gpu-box".to_string(),
            remote_name: "research".to_string(),
            committed_at: "t".to_string(),
        }
    }

    #[test]
    fn missing_handoff_is_none() {
        let storage = Storage::open_in_memory().unwrap();
        assert_eq!(storage.get_handoff("nope").unwrap(), None);
    }

    #[test]
    fn upsert_then_get_round_trips() {
        let storage = Storage::open_in_memory().unwrap();
        let row = sample("s");
        storage.upsert_handoff(&row).unwrap();
        assert_eq!(storage.get_handoff("s").unwrap(), Some(row));
    }

    #[test]
    fn upsert_replaces_the_previous_row_for_the_same_session() {
        let storage = Storage::open_in_memory().unwrap();
        storage.upsert_handoff(&sample("s")).unwrap();
        let mut updated = sample("s");
        updated.destination = "user@other-box".to_string();
        storage.upsert_handoff(&updated).unwrap();
        assert_eq!(storage.get_handoff("s").unwrap(), Some(updated));
    }
}
