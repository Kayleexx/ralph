use super::*;
use crate::engine::profiles::WorkerProfile;

impl Storage {
    pub fn worker_profile(&self, id: &str) -> Result<Option<WorkerProfile>, StorageError> {
        let json: Option<String> = self
            .conn
            .query_row(
                "SELECT profile FROM session_profiles WHERE session_id = ?1",
                [id],
                |r| r.get(0),
            )
            .optional()?;
        json.map(|value| {
            serde_json::from_str(&value)
                .map_err(|e| StorageError::Corrupt(format!("worker profile: {e}")))
        })
        .transpose()
    }

    pub fn set_worker_profile(&self, id: &str, profile: WorkerProfile) -> Result<(), StorageError> {
        let json = serde_json::to_string(&profile)
            .map_err(|e| StorageError::Corrupt(format!("worker profile: {e}")))?;
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "INSERT INTO session_profiles (session_id, profile) VALUES (?1, ?2)
                    ON CONFLICT(session_id) DO UPDATE SET profile = excluded.profile",
            params![id, json],
        )?;
        tx.commit()?;
        Ok(())
    }
}
