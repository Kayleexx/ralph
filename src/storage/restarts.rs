use super::*;
impl Storage {
    pub fn restart_attempts(&self, model: &str) -> Result<u32, StorageError> {
        Ok(self
            .conn
            .query_row(
                "SELECT attempts FROM worker_restarts WHERE model=?1",
                [model],
                |r| r.get(0),
            )
            .optional()?
            .unwrap_or(0))
    }
    pub fn record_restart(
        &self,
        model: &str,
        attempts: u32,
        failure: &str,
    ) -> Result<(), StorageError> {
        self.conn.execute("INSERT INTO worker_restarts VALUES (?1,?2,?3) ON CONFLICT(model) DO UPDATE SET attempts=excluded.attempts, failure=excluded.failure", params![model, attempts, failure])?;
        Ok(())
    }
    pub fn reset_restarts(&self, session: &str) -> Result<(), StorageError> {
        self.conn.execute("UPDATE worker_restarts SET attempts=0 WHERE model=(SELECT model FROM sessions WHERE id=?1)", [session])?;
        Ok(())
    }
    pub fn last_worker_failure(&self, model: &str) -> Result<Option<String>, StorageError> {
        Ok(self
            .conn
            .query_row(
                "SELECT failure FROM worker_restarts WHERE model=?1",
                [model],
                |r| r.get(0),
            )
            .optional()?)
    }
}
