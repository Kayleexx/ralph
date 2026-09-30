//! Crash detection, bounded automatic restart, and `ralph recover`. A worker crash
//! demotes its sessions to `Recovering`, not `Stopped` — a worker crash is not a session
//! failure. The daemon then makes a bounded, contextless attempt to get a
//! fresh worker running for that model; reattaching a specific session's conversation is
//! always the explicit, user-driven `ralph recover`, never automatic.
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::Mutex as AsyncMutex;

use super::restart::RestartPolicy;
use super::{Daemon, WorkerEntry, WorkerKey, cancelled, map_engine_error, to_session_info};
use crate::engine::{Engine, ResolvedModel};
use crate::error::CliError;
use crate::ipc::SessionInfo;
use crate::session;
use crate::state::SessionState;
use crate::storage::StorageError;

impl<E: Engine + 'static> Daemon<E> {
    /// This daemon process starts with no in-memory worker handles, so any session left
    /// `Starting`/`Active` from a previous run is marked `Recovering` — never silently
    /// `Active` again — and left for an explicit `ralph recover` rather than an eager
    /// auto-restart of every previously-active session on daemon startup. A session
    /// stuck `Moving`/`Pausing`/`Hibernating`/`Resuming` from a killed operation rolls
    /// back to a stable landing state instead (see `Storage::rollback_stuck_transitions`).
    pub fn reconcile_on_startup(&self) -> Result<Vec<String>, StorageError> {
        crate::engine::vllm::ownership::reconcile(
            &self.sessions_root,
            |id, model, revision, profile| {
                let row = { self.storage().resolve(id) };
                row.is_ok_and(|row| {
                    row.model == model
                        && (row.model_revision.is_none()
                            || row.model_revision.as_deref() == revision)
                        && self
                            .storage()
                            .worker_profile(id)
                            .is_ok_and(|saved| saved == profile)
                })
            },
        )
        .map_err(|e| StorageError::Corrupt(format!("worker ownership: {e}")))?;
        let demoted = self.storage().reconcile_after_restart(
            |_pid| false,
            SessionState::Recovering,
            &super::now_rfc3339(),
        )?;
        self.storage()
            .rollback_stuck_transitions(&super::now_rfc3339())?;
        self.gc_orphaned_artifacts();
        Ok(demoted)
    }

    /// Removes on-disk leftovers no current daemon state ever cleans up on its own:
    /// `/dev/shm` offload segments (`ownership::reconcile`, just above, has already
    /// killed every leftover worker process this run, so any such segment is
    /// unconditionally orphaned — killed workers no longer hold vLLM's own shm mapping),
    /// staged `handoff-recv-*.ralph` files (a crash between write and the one-shot
    /// `__handoff-recv` process's own cleanup), and session directories with no matching
    /// row (only reachable via a crash mid-`import`, between `create_session_dir` and
    /// the row `insert` that publishes it — see `portable::import`'s ordering comment).
    fn gc_orphaned_artifacts(&self) {
        if let Ok(entries) = std::fs::read_dir("/dev/shm") {
            for entry in entries.flatten() {
                if entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with("vllm_offload_ralph-")
                {
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }

        let known: std::collections::HashSet<String> = self
            .storage()
            .list()
            .map(|rows| rows.into_iter().map(|r| r.id).collect())
            .unwrap_or_default();
        let Ok(entries) = std::fs::read_dir(&self.sessions_root) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if path.is_file() && name.starts_with("handoff-recv-") {
                let _ = std::fs::remove_file(&path);
            } else if path.is_dir() && !known.contains(name) {
                let _ = std::fs::remove_dir_all(&path);
            }
        }
    }

    /// Attaches to a live worker for `model` (waking it if asleep), or cold-starts one,
    /// logging to `session_id`'s directory if a cold start is needed. Shared by `run()`
    /// and `recover()` — recovery reuses this same attach path rather than a new one.
    pub(super) async fn attach_or_start_worker(
        self: &Arc<Self>,
        model: &str,
        session_id: &str,
        cancel: &mut Option<tokio::sync::oneshot::Receiver<()>>,
        allow_native: bool,
        gpu_index: u32,
    ) -> Result<(Arc<AsyncMutex<E>>, ResolvedModel, Option<i64>), CliError> {
        if self.is_draining() {
            return Err(CliError::Resource(
                "this location is draining; new sessions are not being admitted, retry after the drain completes".into(),
            ));
        }
        // This guard reserves startup; registry locks are released before I/O.
        let _startup = self.startup.try_lock().map_err(|_| {
            CliError::Resource(
                "a worker is starting or attaching; session preserved, retry shortly".into(),
            )
        })?;
        // Cleared per admission attempt (not just on read) so a make-room event from an
        // earlier, unrelated call — one that never happened to read this slot, e.g.
        // `resume`/`recover` — can never be misattributed to this one.
        *self
            .last_make_room
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
        let row = self.storage().resolve(session_id)?;
        let key = WorkerKey {
            model: model.to_string(),
            gpu: gpu_index,
        };
        let workers = self.workers.lock().await;
        let reuse = workers
            .get(&key)
            .map(|entry| (entry.engine.clone(), entry.resolved.clone(), entry.profile));
        drop(workers);

        if let Some((engine_handle, resolved, profile)) = reuse {
            if row.model_revision.is_some() && row.model_revision != resolved.revision {
                return Err(CliError::InvalidState(
                    "resident worker has a different model revision; history preserved".into(),
                ));
            }
            if row.tokenizer_revision.is_some() && row.tokenizer_revision != resolved.revision {
                return Err(CliError::InvalidState(
                    "resident worker has a different tokenizer revision; history preserved".into(),
                ));
            }
            if engine_handle.lock().await.try_wait_for_exit().is_some() {
                return Err(CliError::Resource(
                    "worker exited; session preserved, retry after crash detection".into(),
                ));
            }
            self.share_profile(session_id, profile)?;
            self.wake_reserved(&engine_handle, profile, gpu_index)
                .await?;
            let pid = engine_handle.lock().await.pid().map(i64::from);
            let mut workers = self.workers.lock().await;
            let Some(entry) = workers.get_mut(&key) else {
                return Err(CliError::Resource(
                    "worker exited while attaching this session".to_string(),
                ));
            };
            if !entry.session_ids.iter().any(|id| id == session_id) {
                entry.session_ids.push(session_id.to_string());
            }
            entry.last_active = Instant::now();

            return Ok((engine_handle, resolved, pid));
        }

        let dir = session::session_dir(&self.sessions_root, session_id);
        let mut engine = (self.make_engine)(session::vllm_log_path(&dir));
        let spec = self.admitted_spec(&row, allow_native, gpu_index).await?;
        let started = {
            let startup = engine.start_model(&spec);
            tokio::pin!(startup);
            tokio::select! {
                biased;
                _ = cancelled(cancel) => Err(crate::engine::EngineError::BadResponse("startup cancelled; history preserved".into())),
                result = &mut startup => result,
            }
        };
        let resolved = match started {
            Ok(resolved) => resolved,
            Err(error) => {
                engine.stop_model().await.map_err(map_engine_error)?;
                return Err(map_engine_error(error));
            }
        };
        if (spec.revision.is_some() && spec.revision != resolved.revision)
            || (row.tokenizer_revision.is_some() && row.tokenizer_revision != resolved.revision)
        {
            engine.stop_model().await.map_err(map_engine_error)?;
            return Err(CliError::InvalidState(
                "replacement model identity differs; history preserved".into(),
            ));
        }
        let pid = engine.pid().map(i64::from);
        let handle = Arc::new(AsyncMutex::new(engine));
        self.workers.lock().await.insert(
            key.clone(),
            WorkerEntry {
                engine: handle.clone(),
                resolved: resolved.clone(),
                profile: spec.profile,
                kv_engine_id: spec.kv_offload.map(|kv| kv.engine_id),
                session_ids: vec![session_id.to_string()],
                last_active: Instant::now(),
            },
        );
        self.spawn_worker_supervisor(key, handle.clone());
        Ok((handle, resolved, pid))
    }

    /// Reconstructs `session`'s context from the durable token log and attaches it to a
    /// live worker. Accepts sessions left `Recovering` (crash/restart) or `Stopped`;
    /// `Failed` is never accepted — that's reserved for a
    /// genuinely unrecoverable session, not infrastructure flakiness.
    #[cfg(test)]
    pub async fn recover(self: &Arc<Self>, identifier: &str) -> Result<SessionInfo, CliError> {
        self.recover_cancellable(identifier, None).await
    }

    pub async fn recover_cancellable(
        self: &Arc<Self>,
        identifier: &str,
        mut cancel: Option<tokio::sync::oneshot::Receiver<()>>,
    ) -> Result<SessionInfo, CliError> {
        let row = self.storage().resolve(identifier)?;
        let _guard = self.locks.try_acquire(&row.id).ok_or_else(|| {
            CliError::InvalidState("operation already in progress for this session".into())
        })?;
        let row = self.storage().resolve(&row.id)?;
        if row.state == SessionState::Active {
            return Ok(to_session_info(&row));
        }
        if !matches!(row.state, SessionState::Recovering | SessionState::Stopped) {
            return Err(CliError::InvalidState(format!(
                "session is {}, nothing to recover",
                row.state
            )));
        }

        let history = { self.storage().replay_turns(&row.id) };
        let messages = match history {
            Ok(messages) => messages,
            Err(error) => {
                if row.state == SessionState::Recovering {
                    self.transition(&row.id, row.state, SessionState::Stopped, None)?;
                }
                return Err(CliError::from(error).for_session(&row.name, &row.model));
            }
        };
        if row.state == SessionState::Stopped {
            self.transition(&row.id, row.state, SessionState::Recovering, None)?;
        }
        let attach = self
            .attach_or_start_worker(
                &row.model,
                &row.id,
                &mut cancel,
                true,
                session::gpu_index_of(&row.location),
            )
            .await;
        let (engine, resolved, pid) = match attach {
            Ok(v) => v,
            Err(e) => {
                self.transition(
                    &row.id,
                    SessionState::Recovering,
                    SessionState::Stopped,
                    None,
                )?;
                return Err(e.for_session(&row.name, &row.model));
            }
        };

        let prefill = {
            let engine = engine.lock().await;
            tokio::select! {
                biased;
                _ = cancelled(&mut cancel) => Err(crate::engine::EngineError::BadResponse("recovery cancelled; history preserved".into())),
                result = engine.prefill(&messages) => result,
            }
        };
        if let Err(error) = prefill {
            self.transition(
                &row.id,
                SessionState::Recovering,
                SessionState::Stopped,
                None,
            )?;
            return Err(map_engine_error(error).for_session(&row.name, &row.model));
        }
        if row.model_revision.is_none() {
            self.storage().set_resolved(
                &row.id,
                resolved.revision.as_deref(),
                resolved.engine_version.as_deref(),
                &super::now_rfc3339(),
            )?;
        }
        self.transition(&row.id, SessionState::Recovering, SessionState::Active, pid)?;
        Ok(to_session_info(&self.storage().resolve(&row.id)?))
    }
}

/// Called by the worker supervisor when a live worker's process exits unexpectedly.
pub(super) async fn handle_worker_loss<E: Engine + 'static>(
    daemon: &Arc<Daemon<E>>,
    key: WorkerKey,
    session_ids: Vec<String>,
    kv_engine_id: Option<String>,
) {
    let model = key.model.clone();
    // vLLM does not unlink its own `/dev/shm/vllm_offload_<engine_id>.mmap` on SIGKILL
    // (confirmed by reproducing it) — the crashed worker can no longer do it itself, so
    // the crash handler owns this cleanup.
    if let Some(engine_id) = &kv_engine_id {
        crate::engine::vllm::kv_offload::cleanup_shm(engine_id);
    }
    for id in &session_ids {
        let _guard = daemon.locks.acquire(id).await;
        let row = { daemon.storage().resolve(id) };
        if let Ok(row) = row
            && matches!(row.state, SessionState::Active | SessionState::Starting)
            && let Err(error) = daemon.transition(id, row.state, SessionState::Recovering, None)
        {
            eprintln!("cannot record worker loss: {error}");
            return;
        }
    }

    let policy = RestartPolicy::default();
    let mut cancel = None;
    let mut attempt = daemon.storage().restart_attempts(&model).unwrap_or(3);
    let restarted = loop {
        if policy.exhausted(attempt) {
            break false;
        }
        tokio::time::sleep(policy.backoff_for(attempt)).await;
        let Some(session_id) = session_ids.first() else {
            break false;
        };
        attempt += 1;
        if let Err(error) = daemon
            .storage()
            .record_restart(&model, attempt, "worker exited")
        {
            eprintln!("cannot persist restart budget: {error}");
            break false;
        }
        match daemon
            .attach_or_start_worker(&model, session_id, &mut cancel, true, key.gpu)
            .await
        {
            Ok(_) => break true,
            Err(error) => {
                if let Err(write_error) =
                    daemon
                        .storage()
                        .record_restart(&model, attempt, &error.to_string())
                {
                    eprintln!("cannot persist startup failure: {write_error}");
                    break false;
                }
            }
        }
    };

    if restarted {
        let mut workers = daemon.workers.lock().await;
        if let Some(entry) = workers.get_mut(&key) {
            for id in &session_ids {
                if !entry.session_ids.contains(id) {
                    entry.session_ids.push(id.clone());
                }
            }
        }
    } else {
        for id in &session_ids {
            let _guard = daemon.locks.acquire(id).await;
            let row = { daemon.storage().resolve(id) };
            if matches!(row, Ok(row) if row.state == SessionState::Recovering)
                && let Err(error) =
                    daemon.transition(id, SessionState::Recovering, SessionState::Stopped, None)
            {
                eprintln!("cannot record exhausted recovery: {error}");
            }
        }
    }
}
