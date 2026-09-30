//! `ralph pause`/`resume`/`hibernate`. `checkpoint.rs` owns the checkpoint pointer
//! itself and the disk-quota/fingerprint bookkeeping around it; this module is the
//! session-lifecycle transitions layered on top of the worker-attach/session-lock
//! machinery elsewhere in this module tree.
use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::oneshot;

use super::{Daemon, cancelled, map_engine_error, now_rfc3339, to_session_info};
use crate::engine::Engine;
use crate::error::CliError;
use crate::ipc::{ResumeInfo, SessionInfo};
use crate::session;
use crate::state::SessionState;
use crate::storage::SessionRow;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResumeMode {
    Auto,
    FastOnly,
    Portable,
}

impl<E: Engine + 'static> Daemon<E> {
    /// Best-effort, honest-only signal for the UX ("native" vs "portable" resume
    /// message) — never claims native restore unless the directory actually holds
    /// something to restore from.
    fn kv_dir_has_content(dir: &PathBuf) -> bool {
        std::fs::read_dir(dir).is_ok_and(|mut entries| entries.next().is_some())
    }

    /// Detaches `row` from its shared worker, stopping it only once no other session
    /// still needs it (sessions can share one worker per model — see `WorkerEntry`).
    /// Shared by `pause_cancellable` and `hibernate_cancellable`: both release the GPU
    /// worker identically, differing only in what happens to the checkpoint pointer and
    /// which terminal state they land in.
    pub(super) async fn release_worker(&self, row: &SessionRow) -> Result<(), CliError> {
        let key = super::WorkerKey {
            model: row.model.clone(),
            gpu: crate::session::gpu_index_of(&row.location),
        };
        let stop_engine = {
            let mut workers = self.workers.lock().await;
            match workers.get_mut(&key) {
                Some(entry) => {
                    entry.session_ids.retain(|id| id != &row.id);
                    if entry.session_ids.is_empty() {
                        workers.remove(&key).map(|removed| removed.engine)
                    } else {
                        None
                    }
                }
                None => None,
            }
        };
        if let Some(engine) = stop_engine {
            engine
                .lock()
                .await
                .stop_model()
                .await
                .map_err(map_engine_error)?;
        }
        Ok(())
    }

    pub async fn pause_cancellable(
        self: &Arc<Self>,
        identifier: &str,
        _cancel: Option<oneshot::Receiver<()>>,
    ) -> Result<SessionInfo, CliError> {
        let row = self.storage().resolve(identifier)?;
        // Holding this for the whole operation is what gives "stop accepting new query
        // work": a query in flight already holds it (see `begin_query`), so pause simply
        // fails fast with the same "operation already in progress" a concurrent query
        // would — the caller retries once the in-flight query reaches its own boundary.
        let _guard = self.locks.try_acquire(&row.id).ok_or_else(|| {
            CliError::InvalidState("operation already in progress for this session".into())
        })?;
        let row = self.storage().resolve(&row.id)?;
        if row.state != SessionState::Active {
            return Err(CliError::InvalidState(format!(
                "session is {}, cannot pause it",
                row.state
            )));
        }
        self.transition(&row.id, SessionState::Active, SessionState::Pausing, None)?;

        // Durable token history is already flushed continuously; record the checkpoint
        // pointer before releasing the worker.
        self.enforce_kv_quota(&row.id);
        if !self.has_checkpoint_room(&row) {
            self.transition(&row.id, SessionState::Pausing, SessionState::Active, None)?;
            return Err(CliError::Resource(
                "not enough disk space to checkpoint; session is unaffected".into(),
            ));
        }
        if let Err(error) = self
            .storage()
            .upsert_checkpoint(&self.checkpoint_row_for(&row))
        {
            self.transition(&row.id, SessionState::Pausing, SessionState::Active, None)?;
            return Err(error.into());
        }

        if let Err(error) = self.release_worker(&row).await {
            self.transition(&row.id, SessionState::Pausing, SessionState::Active, None)?;
            return Err(error.for_session_state(&row.name, &row.model, "active"));
        }
        // Only now, with the worker actually stopped, is the directory's content stable
        // enough to fingerprint — before this it's still being written.
        crate::engine::vllm::kv_offload::record_content_signature(&session::kvcache_dir(
            &session::session_dir(&self.sessions_root, &row.id),
        ));
        self.transition(&row.id, SessionState::Pausing, SessionState::Paused, None)?;
        Ok(to_session_info(&self.storage().resolve(&row.id)?))
    }

    /// A "stronger pause": same GPU release as `pause_cancellable`, but the checkpoint
    /// write is best-effort rather than fail-closed — a logical-only hibernation is fine
    /// when the durable token log is already complete, rather than blocking the whole
    /// operation on a KV write that isn't the source of truth anyway.
    pub async fn hibernate_cancellable(
        self: &Arc<Self>,
        identifier: &str,
        _cancel: Option<oneshot::Receiver<()>>,
    ) -> Result<SessionInfo, CliError> {
        let row = self.storage().resolve(identifier)?;
        let _guard = self.locks.try_acquire(&row.id).ok_or_else(|| {
            CliError::InvalidState("operation already in progress for this session".into())
        })?;
        let row = self.storage().resolve(&row.id)?;
        if row.state != SessionState::Active {
            return Err(CliError::InvalidState(format!(
                "session is {}, cannot hibernate it",
                row.state
            )));
        }
        self.transition(
            &row.id,
            SessionState::Active,
            SessionState::Hibernating,
            None,
        )?;

        self.enforce_kv_quota(&row.id);
        if !self.has_checkpoint_room(&row) {
            eprintln!("hibernate: disk full, continuing logical-only");
        } else if let Err(error) = self
            .storage()
            .upsert_checkpoint(&self.checkpoint_row_for(&row))
        {
            eprintln!("hibernate: checkpoint unavailable, continuing logical-only: {error}");
        }

        if let Err(error) = self.release_worker(&row).await {
            self.transition(
                &row.id,
                SessionState::Hibernating,
                SessionState::Active,
                None,
            )?;
            return Err(error.for_session_state(&row.name, &row.model, "active"));
        }
        crate::engine::vllm::kv_offload::record_content_signature(&session::kvcache_dir(
            &session::session_dir(&self.sessions_root, &row.id),
        ));
        self.transition(
            &row.id,
            SessionState::Hibernating,
            SessionState::Hibernated,
            None,
        )?;
        Ok(to_session_info(&self.storage().resolve(&row.id)?))
    }

    pub async fn resume_cancellable(
        self: &Arc<Self>,
        identifier: &str,
        mode: ResumeMode,
        mut cancel: Option<oneshot::Receiver<()>>,
    ) -> Result<ResumeInfo, CliError> {
        let row = self.storage().resolve(identifier)?;
        let _guard = self.locks.try_acquire(&row.id).ok_or_else(|| {
            CliError::InvalidState("operation already in progress for this session".into())
        })?;
        let row = self.storage().resolve(&row.id)?;
        if row.state == SessionState::Active {
            return Ok(ResumeInfo {
                session: to_session_info(&row),
                native: false,
                already_active: true,
            });
        }
        if row.state != SessionState::Paused && row.state != SessionState::Hibernated {
            return Err(CliError::InvalidState(format!(
                "session is {}, nothing to resume",
                row.state
            )));
        }
        let origin = row.state;
        let origin_label = if origin == SessionState::Hibernated {
            "hibernated"
        } else {
            "paused"
        };

        let allow_native = mode != ResumeMode::Portable;
        let candidate = allow_native.then(|| self.kv_offload_for(&row)).flatten();
        if mode == ResumeMode::FastOnly && candidate.is_none() {
            return Err(CliError::Resource(
                "no valid native checkpoint for this session; session preserved, run: ralph checkpoint <session> first, or resume without --fast-only".into(),
            ));
        }
        let mut native = candidate.is_some_and(|spec| Self::kv_dir_has_content(&spec.root_dir));

        self.transition(&row.id, origin, SessionState::Resuming, None)?;
        let messages = match self.storage().replay_turns(&row.id) {
            Ok(messages) => messages,
            Err(error) => {
                self.transition(&row.id, SessionState::Resuming, origin, None)?;
                return Err(CliError::from(error).for_session_state(
                    &row.name,
                    &row.model,
                    origin_label,
                ));
            }
        };
        let gpu_index = crate::session::gpu_index_of(&row.location);
        let attach = self
            .attach_or_start_worker(&row.model, &row.id, &mut cancel, allow_native, gpu_index)
            .await;
        // A present, fingerprint-compatible checkpoint whose content is actually corrupt
        // fails here at engine startup, not at the `kv_offload_for` compatibility gate
        // (which only checks the fingerprint, not the bytes) — retry once portable rather
        // than surfacing a raw engine error for damage `ralph checkpoint` can just redo.
        let attach = if native && let Err(error) = &attach {
            eprintln!("resume: native checkpoint attach failed, retrying portable: {error}");
            native = false;
            // Drop the pointer row so a genuinely corrupt directory isn't retried on
            // every future resume; a later `ralph checkpoint` just writes a fresh one.
            let _ = self.storage().delete_checkpoint(&row.id);
            self.attach_or_start_worker(&row.model, &row.id, &mut cancel, false, gpu_index)
                .await
        } else {
            attach
        };
        let (engine, resolved, pid) = match attach {
            Ok(v) => v,
            Err(e) => {
                self.transition(&row.id, SessionState::Resuming, origin, None)?;
                return Err(e.for_session_state(&row.name, &row.model, origin_label));
            }
        };
        let prefill = {
            let engine = engine.lock().await;
            tokio::select! {
                biased;
                _ = cancelled(&mut cancel) => Err(crate::engine::EngineError::BadResponse("resume cancelled; history preserved".into())),
                result = engine.prefill(&messages) => result,
            }
        };
        if let Err(error) = prefill {
            self.transition(&row.id, SessionState::Resuming, origin, None)?;
            return Err(map_engine_error(error).for_session_state(
                &row.name,
                &row.model,
                origin_label,
            ));
        }
        if row.model_revision.is_none() {
            self.storage().set_resolved(
                &row.id,
                resolved.revision.as_deref(),
                resolved.engine_version.as_deref(),
                &now_rfc3339(),
            )?;
        }
        self.transition(&row.id, SessionState::Resuming, SessionState::Active, pid)?;
        Ok(ResumeInfo {
            session: to_session_info(&self.storage().resolve(&row.id)?),
            native,
            already_active: false,
        })
    }
}
