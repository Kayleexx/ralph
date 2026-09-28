//! Daemon business logic: session lifecycle, the in-memory worker registry, and startup
//! reconciliation. Generic over `Engine` so tests can substitute a fake worker without
//! touching this logic.
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex as SyncMutex};
use std::time::Duration;

use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};

use crate::doctor;
use crate::engine::{Engine, EngineError, ModelSpec};
use crate::error::CliError;
use crate::ipc::{InspectInfo, SessionInfo};
use crate::lock::SessionLocks;
use crate::session;
use crate::state::{self, SessionState};
use crate::storage::{SessionRow, Storage, StorageError};

#[cfg(test)]
mod tests;

const SUPERVISOR_POLL_INTERVAL: Duration = Duration::from_millis(500);

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
    workers: AsyncMutex<HashMap<String, Arc<AsyncMutex<E>>>>,
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
            model_revision: model.clone(),
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

        let mut engine = (self.make_engine)(session::vllm_log_path(&dir));
        let spec = ModelSpec {
            model,
            revision: None,
        };
        match engine.start_model(&spec).await {
            Ok(resolved) => {
                let pid = engine.pid().map(i64::from);
                self.transition(&id, SessionState::Starting, SessionState::Active, pid)?;
                self.storage().set_resolved(
                    &id,
                    &resolved.revision,
                    resolved.engine_version.as_deref(),
                    &now_rfc3339(),
                )?;
                let handle = Arc::new(AsyncMutex::new(engine));
                self.workers.lock().await.insert(id.clone(), handle.clone());
                self.spawn_supervisor(id.clone(), handle);
                Ok(to_session_info(&self.storage().resolve(&id)?))
            }
            Err(e) => {
                self.transition(&id, SessionState::Starting, SessionState::Failed, None)?;
                Err(map_engine_error(e))
            }
        }
    }

    /// Watches a worker for unexpected exit and demotes the session to `Stopped` (never
    /// `Failed` — a worker crash is not a session failure) once it does.
    fn spawn_supervisor(self: &Arc<Self>, id: String, handle: Arc<AsyncMutex<E>>) {
        let daemon = self.clone();
        tokio::spawn(async move {
            loop {
                // Briefly locking to poll, rather than holding the lock across a
                // blocking wait, is what lets `generate`/`health` still get the lock
                // while a session is alive.
                if handle.lock().await.try_wait_for_exit().is_some() {
                    break;
                }
                tokio::time::sleep(SUPERVISOR_POLL_INTERVAL).await;
            }
            daemon.workers.lock().await.remove(&id);
            let _ = daemon.transition(&id, SessionState::Active, SessionState::Stopped, None);
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
    /// duration of the request; a second concurrent query on the same session fails
    /// fast rather than queuing.
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
        let workers = self.workers.lock().await;
        let engine = workers.get(&row.id).cloned().ok_or_else(|| {
            CliError::InvalidState("worker is not running in this daemon".to_string())
        })?;
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
