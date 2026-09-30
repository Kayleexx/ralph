//! Daemon business logic: session lifecycle, the in-memory worker registry, and startup
//! reconciliation. Generic over `Engine` so tests can substitute a fake worker without
//! touching this logic.
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex as SyncMutex};
use std::time::{Duration, Instant};

use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};

use crate::continuity::ContinuityPolicy;
use crate::doctor;
use crate::engine::{ChatMessage, Engine, EngineError, ResolvedModel};
use crate::error::CliError;
use crate::ipc::SessionInfo;
use crate::lock::SessionLocks;
use crate::session;
use crate::state::{self, SessionState};
use crate::storage::{SessionRow, Storage};

mod admission;
mod checkpoint;
mod continuity;
pub(crate) mod drain;
pub(crate) mod handoff;
mod idle;
mod inspect;
pub(crate) mod lifecycle;
pub(crate) mod migrate;
mod portable;
mod query;
mod recovery;
mod restart;
#[cfg(test)]
pub(crate) mod tests;
#[cfg(test)]
mod tests_admission;
#[cfg(test)]
mod tests_checkpoint;
#[cfg(test)]
mod tests_drain;
#[cfg(test)]
mod tests_engines;
#[cfg(test)]
mod tests_handoff;
#[cfg(test)]
mod tests_lifecycle;
#[cfg(test)]
mod tests_migrate;
#[cfg(test)]
mod tests_portable;
#[cfg(test)]
mod tests_recovery;
#[cfg(test)]
mod tests_ssh_support;

const SUPERVISOR_POLL_INTERVAL: Duration = Duration::from_millis(500);
// How long a worker can go with no query against any of its sessions before it's put to
// sleep. Sessions themselves never expire (there's no `ralph stop`), so this
// is keyed on activity, not on the attached-session count reaching zero.
const IDLE_SLEEP_AFTER: Duration = Duration::from_secs(300);

/// One worker per (model id, GPU index). Two sessions on the same model but different
/// GPUs must never share a worker — this pair is the whole of worker identity.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct WorkerKey {
    model: String,
    gpu: u32,
}

/// Attachment checks the resolved model/tokenizer revision before sharing it with a
/// historical session.
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
    // Keyed by (model id, gpu index), not session id — see `WorkerKey`/`WorkerEntry`.
    workers: AsyncMutex<HashMap<WorkerKey, WorkerEntry<E>>>,
    startup: AsyncMutex<()>,
    make_engine: fn(PathBuf) -> E,
    has_gpu: fn() -> bool,
    gpu_memory: fn(u32) -> Result<admission::GpuMemory, CliError>,
    /// Set for the duration of one `ralph drain` call, naming the location being
    /// drained — this codebase is single-GPU only, so any drain blocks every new
    /// admission system-wide until it returns (`drain::drain_cancellable`,
    /// `recovery::attach_or_start_worker`'s admission check).
    draining: SyncMutex<Option<String>>,
    /// The `ssh` executable `handoff::run_ssh` invokes — always `"ssh"` in production;
    /// tests point this at a fake script so handoff/drain tests never touch the network.
    ssh_program: SyncMutex<String>,
    /// Forces `lifecycle::has_checkpoint_room` to report no space, for deterministic
    /// disk-full tests without needing to actually exhaust a filesystem. `false` (real
    /// check) in production.
    force_disk_full: SyncMutex<bool>,
    /// (demoted session name, MiB freed) from the most recent `continuity::make_room`,
    /// taken by `run_cancellable` right after a successful admission. Admission is
    /// already fully serialized by `startup`'s lock (`make_room` only ever runs while
    /// it's held), so one shared slot is safe — the same pattern `draining` already uses
    /// to signal across the call stack without threading a return value through every
    /// intermediate layer.
    last_make_room: SyncMutex<Option<(String, u64)>>,
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
            draining: SyncMutex::new(None),
            ssh_program: SyncMutex::new("ssh".to_string()),
            force_disk_full: SyncMutex::new(false),
            last_make_room: SyncMutex::new(None),
        })
    }

    #[cfg(test)]
    pub fn set_disk_full(&self, full: bool) {
        *self
            .force_disk_full
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = full;
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

    #[cfg(test)]
    pub async fn run(
        self: &Arc<Self>,
        model: String,
        name: Option<String>,
    ) -> Result<SessionInfo, CliError> {
        self.run_cancellable(model, name, 0, ContinuityPolicy::default(), None, None)
            .await
    }

    pub async fn run_cancellable(
        self: &Arc<Self>,
        model: String,
        name: Option<String>,
        gpu: u32,
        continuity_policy: ContinuityPolicy,
        continuity_target_ms: Option<i64>,
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
            location: session::location_for(gpu),
            token_count: 0,
            created_at: now.clone(),
            updated_at: now,
            continuity_policy,
            continuity_target_ms,
        };
        self.storage().insert(&row)?;
        session::create_session_dir(&self.sessions_root, &row)
            .map_err(|e| CliError::Other(e.into()))?;
        self.transition(&id, SessionState::Created, SessionState::Starting, None)?;

        match self
            .attach_or_start_worker(&model, &id, &mut cancel, true, gpu)
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
                let made_room_for = self
                    .last_make_room
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .take();
                let mut info = to_session_info(&self.storage().resolve(&id)?);
                info.made_room_for = made_room_for.map(|(name, _)| name);
                Ok(info)
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
    fn spawn_worker_supervisor(self: &Arc<Self>, key: WorkerKey, handle: Arc<AsyncMutex<E>>) {
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
                    .get(&key)
                    .map(|e| e.last_active.elapsed() >= IDLE_SLEEP_AFTER)
                    .unwrap_or(false);
                if idle {
                    let _startup = daemon.startup.lock().await;
                    // Activity may have changed while another worker was starting.
                    let still_idle = daemon
                        .workers
                        .lock()
                        .await
                        .get(&key)
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
                    .get(&key)
                    .is_some_and(|entry| Arc::ptr_eq(&entry.engine, &handle))
                {
                    let entry = workers.remove(&key).unwrap();
                    (entry.session_ids, entry.kv_engine_id)
                } else {
                    (Vec::new(), None)
                }
            };
            drop(startup);
            recovery::handle_worker_loss(&daemon, key, session_ids, kv_engine_id).await;
        });
    }

    #[cfg(test)]
    pub(crate) fn inject_storage_fault(&self, sql: &str) {
        self.storage().test_execute(sql);
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
        made_room_for: None,
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
