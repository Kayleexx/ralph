use super::*;

impl<E: Engine + 'static> Daemon<E> {
    /// Resolves a session for querying and reserves it for exclusive use for the
    /// duration of the request; a second concurrent query on the *same session* fails
    /// fast rather than queuing (sessions sharing a worker can still run concurrently —
    /// this lock is per-session, not per-worker). Fails before any state changes if
    /// `prompt` would exceed the model's context window on top of what's already
    /// committed (no auto-truncation).
    pub async fn begin_query(
        &self,
        identifier: &str,
        prompt: &str,
    ) -> Result<RunningQuery<E>, CliError> {
        let row = self.storage().resolve(identifier)?;
        if row.state != SessionState::Active {
            return Err(CliError::InvalidState(format!(
                "session is {}, cannot query it",
                row.state
            )));
        }
        let guard = self.locks.try_acquire(&row.id).ok_or_else(|| {
            CliError::InvalidState("operation already in progress for this session".to_string())
        })?;
        let row = self.storage().resolve(&row.id)?;
        if row.state != SessionState::Active {
            return Err(CliError::InvalidState(format!(
                "session is {}; retry recovery",
                row.state
            )));
        }
        let mut workers = self.workers.lock().await;
        let entry = workers.get_mut(&row.model).ok_or_else(|| {
            CliError::InvalidState("worker is not running in this daemon".to_string())
        })?;
        entry.last_active = Instant::now();
        let engine = entry.engine.clone();
        let profile = entry.profile;

        drop(workers);
        self.wake_worker(&engine, profile).await?;

        let mut messages = self.storage().replay_turns(&row.id)?;
        messages.push(ChatMessage {
            role: crate::engine::Role::User,
            content: prompt.into(),
        });
        let tokenized = engine
            .lock()
            .await
            .tokenize(&messages)
            .await
            .map_err(map_engine_error)?;
        if tokenized.ids.len() >= tokenized.limit as usize {
            return Err(CliError::Usage(format!(
                "complete chat context needs {} tokens; model limit is {} tokens (request was not accepted)",
                tokenized.ids.len(),
                tokenized.limit
            )));
        }
        let input_ids = engine
            .lock()
            .await
            .encode(prompt)
            .await
            .map_err(map_engine_error)?;

        Ok(RunningQuery {
            session_id: row.id,
            engine,
            messages,
            input_ids,
            _guard: guard,
        })
    }

    #[cfg(test)]
    pub fn replay_messages(&self, session_id: &str) -> Result<Vec<ChatMessage>, CliError> {
        Ok(self.storage().replay_turns(session_id)?)
    }

    pub fn record_user_turn(
        &self,
        session_id: &str,
        content: &str,
        ids: &[u32],
    ) -> Result<i64, CliError> {
        Ok(self
            .storage()
            .record_user_turn(session_id, content, ids, &now_rfc3339())?)
    }

    pub fn flush_assistant_turn(
        &self,
        session_id: &str,
        seq: i64,
        content: &str,
        ids: &[u32],
    ) -> Result<(), CliError> {
        Ok(self
            .storage()
            .flush_assistant_turn(session_id, seq, content, ids, &now_rfc3339())?)
    }

    pub fn finish_generation(
        &self,
        session: &str,
        seq: i64,
        complete: bool,
    ) -> Result<(), CliError> {
        self.storage().finish_turn(session, seq, complete)?;
        if complete {
            self.storage().reset_restarts(session)?;
        }
        Ok(())
    }
}
