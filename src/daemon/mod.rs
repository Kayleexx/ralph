//! Daemon business logic: session lifecycle, the in-memory worker registry, and startup
//! reconciliation. Generic over `Engine` so tests can substitute a fake worker without
//! touching this logic.
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex as SyncMutex};
use std::time::{Duration, Instant};

use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};

use crate::doctor;
use crate::engine::{Engine, EngineError, ModelSpec, ResolvedModel};
use crate::error::CliError;
use crate::ipc::{InspectInfo, SessionInfo};
use crate::lock::SessionLocks;
use crate::session;
use crate::state::{self, SessionState};
use crate::storage::{SessionRow, Storage, StorageError};

#[cfg(test)]
mod tests;

const SUPERVISOR_POLL_INTERVAL: Duration = Duration::from_millis(500);
// How long a worker can go with no query against any of its sessions before it's put to
// sleep. Sessions themselves never expire in Phase 1 (there's no `ralph stop`), so this
// is keyed on activity, not on the attached-session count reaching zero.
const IDLE_SLEEP_AFTER: Duration = Duration::from_secs(300);

/// One live vLLM process, potentially serving several Ralph sessions that all requested
/// the same model. Keyed by model id in `Daemon::workers` — Phase 1 has no way to
/// request a specific revision, so the model id alone is currently a unique enough key;
/// pinning a revision later would need to become part of the key too.
struct WorkerEntry<E: Engine> {
    engine: Arc<AsyncMutex<E>>,
    resolved: ResolvedModel,
    session_ids: Vec<String>,
    last_active: Instant,
}

fn now_rfc3339() -> String {
    time::OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string())
}

pub struct RunningQuery<E: Engine> {
    pub session_id: String,
    pub engine: Arc<AsyncMutex<E>>,
    _guard: OwnedMutexGuard<()>,
}

pub struct Daemon<E: Engine + 'static> {
    // rusqlite::Connection isn't Sync, so Storage needs its own lock to live behind the
    // Arc this daemon is shared through.
    storage: SyncMutex<Storage>,
    sessions_root: PathBuf,
    locks: SessionLocks,
    // Keyed by model id, not session id — see `WorkerEntry`.
    workers: AsyncMutex<HashMap<String, WorkerEntry<E>>>,
    make_engine: fn(PathBuf) -> E,
    has_gpu: fn() -> bool,
}

fn real_gpu_check() -> bool {
    !matches!(doctor::gpu_count(), None | Some(0))
}

impl<E: Engine + 'static> Daemon<E> {
    pub fn new(storage: Storage, ralph_home: PathBuf, make_engine: fn(PathBuf) -> E) -> Arc<Self> {
        Self::with_gpu_check(storage, ralph_home, make_engine, real_gpu_check)
    }

    /// Lets tests substitute the GPU check — `run`'s pre-flight `nvidia-smi` probe would
    /// otherwise make every test depend on real GPU hardware being present.
    pub fn with_gpu_check(
        storage: Storage,
        ralph_home: PathBuf,
        make_engine: fn(PathBuf) -> E,
        has_gpu: fn() -> bool,
    ) -> Arc<Self> {
        let sessions_root = ralph_home.join("sessions");
        Arc::new(Self {
            storage: SyncMutex::new(storage),
            sessions_root,
            locks: SessionLocks::new(),
            workers: AsyncMutex::new(HashMap::new()),
            make_engine,
            has_gpu,
        })
    }

    // Recovers the guard on poison rather than panicking — a panic in one caller while
    // holding this lock should not take down every other session's storage access.
    fn storage(&self) -> std::sync::MutexGuard<'_, Storage> {
        self.storage
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn transition(
        &self,
        id: &str,
        from: SessionState,
        to: SessionState,
        pid: Option<i64>,
    ) -> Result<(), CliError> {
        state::validate_transition(from, to)?;
        self.storage().set_state(id, to, pid, &now_rfc3339())?;
        Ok(())
    }

    /// This daemon process starts with no in-memory worker handles, so any session left
    /// `Starting`/`Active` from a previous run cannot be reattached — Phase 1 has no
    /// recovery path back to a worker it didn't spawn itself.
    pub fn reconcile_on_startup(&self) -> Result<Vec<String>, StorageError> {
        self.storage()
            .reconcile_after_restart(|_pid| false, &now_rfc3339())
    }

    pub async fn run(
        self: &Arc<Self>,
        model: String,
        name: Option<String>,
    ) -> Result<SessionInfo, CliError> {
        if !(self.has_gpu)() {
            return Err(map_engine_error(EngineError::NoGpu));
        }

        let name = name.unwrap_or_else(session::generate_name);
        if self.storage().name_taken(&name)? {
            return Err(CliError::DuplicateName(name));
        }

        let id = session::new_session_id();
        let now = now_rfc3339();
        let row = SessionRow {
            id: id.clone(),
            name,
            model: model.clone(),
            // Unresolved until `start_model` succeeds and reports a real revision.
            model_revision: None,
            tokenizer_revision: None,
            engine: "vllm".to_string(),
            engine_version: None,
            state: SessionState::Created,
            pid: None,
            location: "local/gpu0".to_string(),
            token_count: 0,
            created_at: now.clone(),
            updated_at: now,
        };
        self.storage().insert(&row)?;
        let dir = session::create_session_dir(&self.sessions_root, &row)
            .map_err(|e| CliError::Other(e.into()))?;
        self.transition(&id, SessionState::Created, SessionState::Starting, None)?;

        // Reuse a worker already serving this model (waking it first if it's asleep)
        // instead of cold-starting a second vLLM process for the same weights.
        let workers = self.workers.lock().await;
        let reuse = workers
            .get(&model)
            .map(|entry| (entry.engine.clone(), entry.resolved.clone()));
        drop(workers); // never held across the wake-up network round-trip below

        if let Some((engine_handle, resolved)) = reuse {
            if let Err(e) = wake_if_sleeping(&engine_handle).await {
                self.transition(&id, SessionState::Starting, SessionState::Failed, None)?;
                return Err(map_engine_error(e));
            }
            // Only attach this session to the worker once it's confirmed usable; the
            // worker could have crashed and been evicted between the lookup above and
            // now, in which case there's nothing to attach to.
            let mut workers = self.workers.lock().await;
            let Some(entry) = workers.get_mut(&model) else {
                drop(workers);
                self.transition(&id, SessionState::Starting, SessionState::Failed, None)?;
                return Err(CliError::Resource(
                    "worker exited while starting this session".to_string(),
                ));
            };
            entry.session_ids.push(id.clone());
            entry.last_active = Instant::now();
            let pid = entry.engine.lock().await.pid().map(i64::from);
            drop(workers);
            self.transition(&id, SessionState::Starting, SessionState::Active, pid)?;
            self.storage().set_resolved(
                &id,
                resolved.revision.as_deref(),
                resolved.engine_version.as_deref(),
                &now_rfc3339(),
            )?;
            return Ok(to_session_info(&self.storage().resolve(&id)?));
        }

        let mut engine = (self.make_engine)(session::vllm_log_path(&dir));
        let spec = ModelSpec {
            model: model.clone(),
            revision: None,
        };
        match engine.start_model(&spec).await {
            Ok(resolved) => {
                let pid = engine.pid().map(i64::from);
                self.transition(&id, SessionState::Starting, SessionState::Active, pid)?;
                self.storage().set_resolved(
                    &id,
                    resolved.revision.as_deref(),
                    resolved.engine_version.as_deref(),
                    &now_rfc3339(),
                )?;
                let handle = Arc::new(AsyncMutex::new(engine));
                self.workers.lock().await.insert(
                    model.clone(),
                    WorkerEntry {
                        engine: handle.clone(),
                        resolved,
                        session_ids: vec![id.clone()],
                        last_active: Instant::now(),
                    },
                );
                self.spawn_worker_supervisor(model, handle);
                Ok(to_session_info(&self.storage().resolve(&id)?))
            }
            Err(e) => {
                self.transition(&id, SessionState::Starting, SessionState::Failed, None)?;
                Err(map_engine_error(e))
            }
        }
    }

    /// Watches a worker for unexpected exit — demoting every session attached to it to
    /// `Stopped` (never `Failed`: a worker crash is not a session failure) — and puts it
    /// to sleep after `IDLE_SLEEP_AFTER` with no query against any of its sessions.
    fn spawn_worker_supervisor(self: &Arc<Self>, model: String, handle: Arc<AsyncMutex<E>>) {
        let daemon = self.clone();
        tokio::spawn(async move {
            loop {
                // Briefly locking to poll, rather than holding the lock across a
                // blocking wait, is what lets `generate`/`health` still get the lock
                // while the worker is alive.
                if handle.lock().await.try_wait_for_exit().is_some() {
                    break;
                }
                let idle = daemon
                    .workers
                    .lock()
                    .await
                    .get(&model)
                    .map(|e| e.last_active.elapsed() >= IDLE_SLEEP_AFTER)
                    .unwrap_or(false);
                if idle {
                    let mut engine = handle.lock().await;
                    if !engine.is_sleeping().await {
                        let _ = engine.sleep().await;
                    }
                }
                tokio::time::sleep(SUPERVISOR_POLL_INTERVAL).await;
            }
            let session_ids = daemon
                .workers
                .lock()
                .await
                .remove(&model)
                .map(|e| e.session_ids)
                .unwrap_or_default();
            for id in session_ids {
                let _ = daemon.transition(&id, SessionState::Active, SessionState::Stopped, None);
            }
        });
    }

    pub fn ps(&self) -> Result<Vec<SessionInfo>, CliError> {
        Ok(self.storage().list()?.iter().map(to_session_info).collect())
    }

    pub fn inspect(&self, identifier: &str) -> Result<InspectInfo, CliError> {
        let row = self.storage().resolve(identifier)?;
        let recoverability = match row.state {
            SessionState::Active | SessionState::Created | SessionState::Starting => "safe",
            SessionState::Failed => "broken",
            // metadata intact, worker gone, no recovery mechanism yet.
            _ => "degraded",
        };
        Ok(InspectInfo {
            session: to_session_info(&row),
            recoverability: recoverability.to_string(),
            fast_restore: "unavailable".to_string(),
            portable_state: "complete".to_string(),
        })
    }

    /// Resolves a session for querying and reserves it for exclusive use for the
    /// duration of the request; a second concurrent query on the *same session* fails
    /// fast rather than queuing (sessions sharing a worker can still run concurrently —
    /// this lock is per-session, not per-worker).
    pub async fn begin_query(&self, identifier: &str) -> Result<RunningQuery<E>, CliError> {
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
        let mut workers = self.workers.lock().await;
        let entry = workers.get_mut(&row.model).ok_or_else(|| {
            CliError::InvalidState("worker is not running in this daemon".to_string())
        })?;
        entry.last_active = Instant::now();
        let engine = entry.engine.clone();
        drop(workers);
        wake_if_sleeping(&engine).await.map_err(map_engine_error)?;
        Ok(RunningQuery {
            session_id: row.id,
            engine,
            _guard: guard,
        })
    }

    pub fn record_tokens(&self, session_id: &str, token_count: i64) -> Result<(), CliError> {
        Ok(self
            .storage()
            .set_token_count(session_id, token_count, &now_rfc3339())?)
    }
}

/// Wakes `engine` only if it's actually asleep — checking first (rather than always
/// calling `wake_up`) keeps the common case (an already-awake worker) to one cheap
/// `is_sleeping` round-trip instead of a wake request the engine has to no-op internally.
async fn wake_if_sleeping<E: Engine>(engine: &Arc<AsyncMutex<E>>) -> Result<(), EngineError> {
    let mut engine = engine.lock().await;
    if engine.is_sleeping().await {
        engine.wake_up().await?;
    }
    Ok(())
}

fn to_session_info(row: &SessionRow) -> SessionInfo {
    SessionInfo {
        id: row.id.clone(),
        name: row.name.clone(),
        model: row.model.clone(),
        model_revision: row.model_revision.clone(),
        engine_version: row.engine_version.clone(),
        state: row.state.to_string(),
        pid: row.pid,
        location: row.location.clone(),
        token_count: row.token_count,
    }
}

pub(crate) fn map_engine_error(e: EngineError) -> CliError {
    match e {
        EngineError::NoGpu => CliError::Resource("no usable gpu is available".to_string()),
        EngineError::OutOfMemory(detail) => {
            CliError::Resource(format!("out of gpu memory: {detail}"))
        }
        EngineError::HealthTimeout(reason) => {
            CliError::Resource(format!("worker did not become healthy in time: {reason}"))
        }
        EngineError::WorkerExited(detail) => {
            CliError::Resource(format!("worker exited before becoming ready: {detail}"))
        }
        EngineError::ContextWindowExceeded { limit } => CliError::Usage(format!(
            "request exceeds the model's context window ({limit} tokens)"
        )),
        EngineError::Request(e) => CliError::Other(e.into()),
        EngineError::BadResponse(detail) => CliError::Other(anyhow::anyhow!(detail)),
        EngineError::Io(e) => CliError::Other(e.into()),
    }
}
