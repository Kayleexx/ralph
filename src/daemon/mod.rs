//! Daemon business logic: session lifecycle, the in-memory worker registry, and startup
//! reconciliation. Generic over `Engine` so tests can substitute a fake worker without
//! touching this logic.
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex as SyncMutex};
use std::time::{Duration, Instant};

use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};

use crate::doctor;
use crate::engine::{ChatMessage, Engine, EngineError, ResolvedModel};
use crate::error::CliError;
use crate::ipc::{InspectInfo, SessionInfo};
use crate::lock::SessionLocks;
use crate::session;
use crate::state::{self, SessionState};
use crate::storage::{SessionRow, Storage, StorageError};

mod admission;
pub(crate) mod lifecycle;
mod query;
mod recovery;
mod restart;
#[cfg(test)]
pub(crate) mod tests;
#[cfg(test)]
mod tests_admission;
#[cfg(test)]
mod tests_lifecycle;
#[cfg(test)]
mod tests_recovery;

const SUPERVISOR_POLL_INTERVAL: Duration = Duration::from_millis(500);
// How long a worker can go with no query against any of its sessions before it's put to
// sleep. Sessions themselves never expire (there's no `ralph stop`), so this
// is keyed on activity, not on the attached-session count reaching zero.
const IDLE_SLEEP_AFTER: Duration = Duration::from_secs(300);

/// One worker per model id. Attachment checks the resolved model/tokenizer revision
/// before sharing it with a historical session.
struct WorkerEntry<E: Engine> {
    engine: Arc<AsyncMutex<E>>,
    resolved: ResolvedModel,
    profile: Option<crate::engine::profiles::WorkerProfile>,
    /// Set only when this worker was started with `--kv-transfer-config`; lets a crash
    /// handler unlink the matching `/dev/shm/vllm_offload_<engine_id>.mmap` (vLLM itself
    /// does not on SIGKILL — confirmed by reproducing it).
    kv_engine_id: Option<String>,
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
    pub messages: Vec<ChatMessage>,
    pub input_ids: Vec<u32>,
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
    startup: AsyncMutex<()>,
    make_engine: fn(PathBuf) -> E,
    has_gpu: fn() -> bool,
    gpu_memory: fn() -> Result<admission::GpuMemory, CliError>,
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
            startup: AsyncMutex::new(()),
            make_engine,
            has_gpu,
            gpu_memory: admission::gpu_memory,
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
    /// `Starting`/`Active` from a previous run is marked `Recovering` — never silently
    /// `Active` again — and left for an explicit `ralph recover` rather than an eager
    /// auto-restart of every previously-active session on daemon startup.
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
        self.storage().reconcile_after_restart(
            |_pid| false,
            SessionState::Recovering,
            &now_rfc3339(),
        )
    }

    #[cfg(test)]
    pub async fn run(
        self: &Arc<Self>,
        model: String,
        name: Option<String>,
    ) -> Result<SessionInfo, CliError> {
        self.run_cancellable(model, name, None).await
    }

    pub async fn run_cancellable(
        self: &Arc<Self>,
        model: String,
        name: Option<String>,
        mut cancel: Option<tokio::sync::oneshot::Receiver<()>>,
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
        session::create_session_dir(&self.sessions_root, &row)
            .map_err(|e| CliError::Other(e.into()))?;
        self.transition(&id, SessionState::Created, SessionState::Starting, None)?;

        match self
            .attach_or_start_worker(&model, &id, &mut cancel, true)
            .await
        {
            Ok((_, resolved, pid)) => {
                self.storage().set_resolved(
                    &id,
                    resolved.revision.as_deref(),
                    resolved.engine_version.as_deref(),
                    &now_rfc3339(),
                )?;
                self.transition(&id, SessionState::Starting, SessionState::Active, pid)?;
                Ok(to_session_info(&self.storage().resolve(&id)?))
            }
            Err(e) => {
                self.transition(&id, SessionState::Starting, SessionState::Stopped, None)?;
                let attempts = self.storage().restart_attempts(&model)?;
                self.storage()
                    .record_restart(&model, attempts, &e.to_string())?;
                Err(e.for_session(&row.name, &row.model))
            }
        }
    }

    /// Watches a worker for unexpected exit — handing off to `recovery::handle_worker_loss`
    /// for every attached session (never `Failed` directly: a worker crash is not a
    /// session failure) — and puts it to sleep after `IDLE_SLEEP_AFTER` with no query
    /// against any of its sessions.
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
                    let _startup = daemon.startup.lock().await;
                    // Activity may have changed while another worker was starting.
                    let still_idle = daemon
                        .workers
                        .lock()
                        .await
                        .get(&model)
                        .is_some_and(|e| e.last_active.elapsed() >= IDLE_SLEEP_AFTER);
                    if !still_idle {
                        continue;
                    }
                    let mut engine = handle.lock().await;
                    if !engine.is_sleeping().await
                        && let Err(error) = engine.sleep().await
                    {
                        eprintln!("idle worker sleep failed: {error}");
                    }
                }
                tokio::time::sleep(SUPERVISOR_POLL_INTERVAL).await;
            }
            let startup = daemon.startup.lock().await;
            if let Err(error) = handle.lock().await.stop_model().await {
                eprintln!("worker cleanup failed: {error}");
                return;
            }
            let (session_ids, kv_engine_id) = {
                let mut workers = daemon.workers.lock().await;
                if workers
                    .get(&model)
                    .is_some_and(|entry| Arc::ptr_eq(&entry.engine, &handle))
                {
                    let entry = workers.remove(&model).unwrap();
                    (entry.session_ids, entry.kv_engine_id)
                } else {
                    (Vec::new(), None)
                }
            };
            drop(startup);
            recovery::handle_worker_loss(&daemon, model, session_ids, kv_engine_id).await;
        });
    }

    #[cfg(test)]
    pub(crate) fn inject_storage_fault(&self, sql: &str) {
        self.storage().test_execute(sql);
    }

    pub fn ps(&self) -> Result<Vec<SessionInfo>, CliError> {
        Ok(self.storage().list()?.iter().map(to_session_info).collect())
    }

    pub fn inspect(&self, identifier: &str) -> Result<InspectInfo, CliError> {
        let row = self.storage().resolve(identifier)?;
        let history = self.storage().replay_turns(&row.id);
        let complete = history
            .as_ref()
            .is_ok_and(|messages| row.token_count == 0 || !messages.is_empty());
        let recoverability = match row.state {
            SessionState::Active | SessionState::Created | SessionState::Starting => "safe",
            SessionState::Failed => "broken",
            _ => "degraded",
        };
        let fast_restore = if self.kv_offload_for(&row).is_some() {
            "available"
        } else {
            "unavailable"
        };
        Ok(InspectInfo {
            session: to_session_info(&row),
            recoverability: if complete { recoverability } else { "degraded" }.to_string(),
            fast_restore: fast_restore.to_string(),
            last_failure: self.storage().last_worker_failure(&row.model)?,
            portable_state: if complete { "complete" } else { "incomplete" }.to_string(),
        })
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

/// Shared by every cancellable long-running operation (`run`, `recover`, `resume`):
/// resolves when the caller sends `Cancel`, or never if there's nothing to cancel.
pub(crate) async fn cancelled(cancel: &mut Option<tokio::sync::oneshot::Receiver<()>>) {
    match cancel {
        Some(cancel) => {
            let _ = cancel.await;
        }
        None => std::future::pending::<()>().await,
    }
}

pub(crate) fn map_engine_error(e: EngineError) -> CliError {
    match e {
        EngineError::Startup {
            model,
            cause,
            diagnostic,
        } => CliError::Startup {
            summary: format!("{model}: {cause}"),
            diagnostic,
        },
        EngineError::NoGpu => CliError::Resource("no usable gpu is available".to_string()),
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
