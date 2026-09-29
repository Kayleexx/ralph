//! `ralph checkpoint`/`pause`/`resume`. Mostly metadata operations layered on Phase 2's
//! worker-attach/session-lock machinery: the KV bytes themselves are already written
//! continuously and atomically by vLLM's own offload connector during ordinary
//! generation (see `engine::vllm::kv_offload`), so there is no separate bulk transfer to
//! stage or commit here — only a small, already-atomic (SQLite WAL) pointer+fingerprint
//! row, and the session-lifecycle transition around it.
use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::oneshot;

use super::{Daemon, cancelled, map_engine_error, now_rfc3339, to_session_info};
use crate::checkpoint::Fingerprint;
use crate::engine::vllm::kv_offload;
use crate::engine::{Engine, KvOffloadSpec};
use crate::error::CliError;
use crate::ipc::{ResumeInfo, SessionInfo};
use crate::session;
use crate::state::SessionState;
use crate::storage::SessionRow;
use crate::storage::checkpoints::CheckpointRow;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResumeMode {
    Auto,
    FastOnly,
    Portable,
}

/// Coarse total across every session's KV directory combined — good enough to bound
/// disk growth from accumulating checkpoints (§16.12) without per-model/per-user
/// accounting no part of this codebase has today.
const KV_QUOTA_BYTES: u64 = 4 * 1024 * 1024 * 1024;

fn dir_size(dir: &std::path::Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .map(|entry| {
            let path = entry.path();
            if path.is_dir() {
                dir_size(&path)
            } else {
                entry.metadata().map(|m| m.len()).unwrap_or(0)
            }
        })
        .sum()
}

impl<E: Engine + 'static> Daemon<E> {
    fn current_fingerprint(&self, row: &SessionRow) -> Fingerprint {
        Fingerprint {
            model_revision: row.model_revision.clone(),
            tokenizer_revision: row.tokenizer_revision.clone(),
            engine: row.engine.clone(),
            engine_version: row.engine_version.clone(),
            gpu_name: (self.gpu_memory)().ok().map(|g| g.name),
            tensor_parallel: 1,
            adapter: None,
        }
    }

    pub(super) fn checkpoint_row_for(&self, row: &SessionRow) -> CheckpointRow {
        let dir = session::kvcache_dir(&session::session_dir(&self.sessions_root, &row.id));
        CheckpointRow {
            session_id: row.id.clone(),
            fingerprint_json: self.current_fingerprint(row).to_json(),
            kv_dir: dir.to_string_lossy().into_owned(),
            engine_id: kv_offload::engine_id_for_session(&row.id),
            created_at: now_rfc3339(),
        }
    }

    /// `None` unless a durably-recorded checkpoint exists whose fingerprint still matches
    /// this session's current one, and there's enough disk headroom to justify attempting
    /// it — never trusts vLLM's own directory-naming hash as the compatibility gate
    /// (it omits model/tokenizer revision and engine version; see `crate::checkpoint`).
    pub(super) fn kv_offload_for(&self, row: &SessionRow) -> Option<KvOffloadSpec> {
        let checkpoint = self.storage().get_checkpoint(&row.id).ok().flatten()?;
        let stored = Fingerprint::from_json(&checkpoint.fingerprint_json)?;
        if stored != self.current_fingerprint(row) {
            return None;
        }
        let dir = PathBuf::from(&checkpoint.kv_dir);
        if !kv_offload::has_room(&dir, kv_offload::DEFAULT_CPU_BYTES) {
            return None;
        }
        Some(KvOffloadSpec {
            root_dir: dir,
            cpu_bytes: kv_offload::DEFAULT_CPU_BYTES,
            engine_id: checkpoint.engine_id,
        })
    }

    /// Best-effort, honest-only signal for the UX ("native" vs "portable" resume
    /// message) — never claims native restore unless the directory actually holds
    /// something to restore from.
    fn kv_dir_has_content(dir: &PathBuf) -> bool {
        std::fs::read_dir(dir).is_ok_and(|mut entries| entries.next().is_some())
    }

    /// Deletes the oldest *other* sessions' KV directories and checkpoint rows until the
    /// combined total is back under budget (§16.12 "old acceleration snapshots
    /// accumulate", §16.7 "NVMe tier full"). Only ever touches `checkpoints`/on-disk KV
    /// directories — never the durable `sessions`/`token_turns` tables, so eviction can
    /// never make a session unrecoverable (Invariant 2).
    pub(super) fn enforce_kv_quota(&self, keep_session_id: &str) {
        self.enforce_kv_quota_within(keep_session_id, KV_QUOTA_BYTES);
    }

    /// Split out so tests can exercise real eviction against a small quota instead of
    /// writing multiple gigabytes of dummy data to disk.
    pub(super) fn enforce_kv_quota_within(&self, keep_session_id: &str, quota_bytes: u64) {
        let Ok(mut checkpoints) = self.storage().list_checkpoints() else {
            return;
        };
        let mut total: u64 = checkpoints
            .iter()
            .map(|c| dir_size(std::path::Path::new(&c.kv_dir)))
            .sum();
        checkpoints.retain(|c| c.session_id != keep_session_id);
        for checkpoint in checkpoints {
            if total <= quota_bytes {
                break;
            }
            let dir = std::path::Path::new(&checkpoint.kv_dir);
            let freed = dir_size(dir);
            let _ = std::fs::remove_dir_all(dir);
            let _ = self.storage().delete_checkpoint(&checkpoint.session_id);
            total = total.saturating_sub(freed);
        }
    }

    /// Graceful SIGTERM preemption (RALPH_SPEC.md §6.11): one best-effort checkpoint
    /// pass over every currently `Active` session before this process exits. Durable
    /// token history is already flushed continuously (Phase 2), so this only improves
    /// the odds of a *fast* resume elsewhere/after restart — the actual recovery path
    /// is the existing `reconcile_on_startup` → `ralph recover`/`ralph resume` story,
    /// unchanged. Never blocks on a fresh worker admission or state transition.
    pub(crate) fn checkpoint_active_sessions_best_effort(&self) {
        let Ok(rows) = self.storage().list() else {
            return;
        };
        for row in rows {
            if row.state != SessionState::Active {
                continue;
            }
            if let Err(error) = self
                .storage()
                .upsert_checkpoint(&self.checkpoint_row_for(&row))
            {
                eprintln!("SIGTERM checkpoint failed for {}: {error}", row.name);
            }
        }
    }

    pub async fn checkpoint(self: &Arc<Self>, identifier: &str) -> Result<SessionInfo, CliError> {
        let row = self.storage().resolve(identifier)?;
        let _guard = self.locks.try_acquire(&row.id).ok_or_else(|| {
            CliError::InvalidState("operation already in progress for this session".into())
        })?;
        let row = self.storage().resolve(&row.id)?;
        if row.state != SessionState::Active {
            return Err(CliError::InvalidState(format!(
                "session is {}, checkpoint requires an active session",
                row.state
            )));
        }
        self.enforce_kv_quota(&row.id);
        self.storage()
            .upsert_checkpoint(&self.checkpoint_row_for(&row))?;
        Ok(to_session_info(&row))
    }

    /// Detaches `row` from its shared worker, stopping it only once no other session
    /// still needs it (sessions can share one worker per model — see `WorkerEntry`).
    /// Shared by `pause_cancellable` and `hibernate_cancellable`: both release the GPU
    /// worker identically, differing only in what happens to the checkpoint pointer and
    /// which terminal state they land in.
    pub(super) async fn release_worker(&self, row: &SessionRow) -> Result<(), CliError> {
        let stop_engine = {
            let mut workers = self.workers.lock().await;
            match workers.get_mut(&row.model) {
                Some(entry) => {
                    entry.session_ids.retain(|id| id != &row.id);
                    if entry.session_ids.is_empty() {
                        workers.remove(&row.model).map(|removed| removed.engine)
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

        // Durable token history is already flushed continuously (Phase 2); record the
        // checkpoint pointer before releasing the worker.
        self.enforce_kv_quota(&row.id);
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
        self.transition(&row.id, SessionState::Pausing, SessionState::Paused, None)?;
        Ok(to_session_info(&self.storage().resolve(&row.id)?))
    }

    /// A "stronger pause": same GPU release as `pause_cancellable`, but the checkpoint
    /// write is best-effort rather than fail-closed — §16.7 "Hibernation cannot save
    /// acceleration state" explicitly allows a logical-only hibernation when the durable
    /// token log is already complete, rather than blocking the whole operation on a KV
    /// write that isn't the source of truth anyway.
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
        if let Err(error) = self
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
        let native = candidate.is_some_and(|spec| Self::kv_dir_has_content(&spec.root_dir));

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
        let attach = self
            .attach_or_start_worker(&row.model, &row.id, &mut cancel, allow_native)
            .await;
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
        })
    }
}
