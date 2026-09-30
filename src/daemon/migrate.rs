//! `ralph migrate`: moves a session's live worker from one GPU to another on the same
//! machine, same daemon, same session id — never a change of ownership (unlike
//! `handoff`, which moves to a different Ralph installation). Mechanically this is
//! `pause_cancellable` composed with `resume_cancellable` targeting a different GPU
//! index, plus a `location` update at the commit point; no new `SessionState` is
//! needed because every intermediate state (`Pausing`/`Paused`/`Resuming`) already
//! exists and is already covered by `storage::rollback_stuck_transitions` on crash.
//!
//! First correct implementation only: always portable reconstruction (replay durable
//! history into a fresh destination worker), never native KV transfer — vLLM has no
//! proven-safe mechanism to move KV bytes between GPU-bound processes, and native KV
//! must never cross physical GPUs any more than it crosses engines.
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::oneshot;

use super::{Daemon, cancelled, map_engine_error, now_rfc3339, to_session_info};
use crate::engine::Engine;
use crate::error::CliError;
use crate::ipc::MigrationReport;
use crate::session;
use crate::state::SessionState;
use crate::storage::SessionRow;

impl<E: Engine + 'static> Daemon<E> {
    pub async fn migrate_cancellable(
        self: &Arc<Self>,
        identifier: &str,
        destination_gpu: u32,
        mut cancel: Option<oneshot::Receiver<()>>,
    ) -> Result<MigrationReport, CliError> {
        let row = self.storage().resolve(identifier)?;
        let _guard = self.locks.try_acquire(&row.id).ok_or_else(|| {
            CliError::InvalidState("operation already in progress for this session".into())
        })?;
        let row = self.storage().resolve(&row.id)?;
        if row.state != SessionState::Active {
            return Err(CliError::InvalidState(format!(
                "session is {}, cannot migrate it",
                row.state
            )));
        }
        let source_gpu = session::gpu_index_of(&row.location);
        if source_gpu == destination_gpu {
            return Ok(MigrationReport {
                session: to_session_info(&row),
                source_gpu,
                destination_gpu,
                native: false,
                total_ms: 0.0,
                interruption_ms: 0.0,
                replay_prefill_ms: 0.0,
                token_count: row.token_count,
                source_vram_freed_mib: None,
                destination_vram_used_mib: None,
                bytes_transferred: None,
                ttft_ms: None,
            });
        }
        // Real pre-flight: the destination must actually be measurable before anything
        // is mutated (per "unsupported target capabilities must fail before mutating
        // ownership/state") — `gpu_memory` is the same injectable check `available_vram`
        // uses, so this refuses cleanly on both real unavailable hardware and in tests.
        if let Err(error) = (self.gpu_memory)(destination_gpu) {
            return Err(CliError::Resource(format!(
                "gpu{destination_gpu} is not usable ({error}); session unaffected"
            )));
        }

        let total_start = Instant::now();
        let vram_before = (self.gpu_memory)(source_gpu).ok().map(|g| g.free_mib);

        self.transition(&row.id, SessionState::Active, SessionState::Pausing, None)?;
        if let Err(error) = self.release_worker(&row).await {
            self.transition(&row.id, SessionState::Pausing, SessionState::Active, None)?;
            return Err(error.for_session_state(&row.name, &row.model, "active"));
        }
        let interruption_start = Instant::now();
        self.transition(&row.id, SessionState::Pausing, SessionState::Paused, None)?;

        let source_vram_freed_mib = match (
            (self.gpu_memory)(source_gpu).ok().map(|g| g.free_mib),
            vram_before,
        ) {
            (Some(after), Some(before)) => Some(after.saturating_sub(before)),
            _ => None,
        };

        self.transition(&row.id, SessionState::Paused, SessionState::Resuming, None)?;
        let messages = match self.storage().replay_turns(&row.id) {
            Ok(messages) => messages,
            Err(error) => {
                self.transition(&row.id, SessionState::Resuming, SessionState::Paused, None)?;
                return Err(
                    CliError::from(error).for_session_state(&row.name, &row.model, "paused")
                );
            }
        };

        let replay_start = Instant::now();
        // Never native: a differing GPU is exactly the kind of acceleration-state
        // mismatch the fingerprint/content-signature gate already exists to refuse —
        // `allow_native = false` makes that refusal unconditional rather than relying
        // on the gate to happen to agree.
        let attach = self
            .attach_or_start_worker(&row.model, &row.id, &mut cancel, false, destination_gpu)
            .await;
        let (engine, resolved, pid) = match attach {
            Ok(v) => v,
            Err(e) => {
                return Err(self
                    .rollback_to_source(&row, source_gpu, &mut cancel, e)
                    .await);
            }
        };
        let prefill = {
            let engine = engine.lock().await;
            tokio::select! {
                biased;
                _ = cancelled(&mut cancel) => Err(crate::engine::EngineError::BadResponse("migration cancelled; rolling back".into())),
                result = engine.prefill(&messages) => result,
            }
        };
        if let Err(error) = prefill {
            return Err(self
                .rollback_to_source(&row, source_gpu, &mut cancel, map_engine_error(error))
                .await);
        }
        let replay_prefill_ms = replay_start.elapsed().as_secs_f64() * 1000.0;
        let destination_vram_used_mib = (self.gpu_memory)(destination_gpu).ok().map(|g| g.free_mib);

        if row.model_revision.is_none() {
            self.storage().set_resolved(
                &row.id,
                resolved.revision.as_deref(),
                resolved.engine_version.as_deref(),
                &now_rfc3339(),
            )?;
        }
        // Commit point: only after the destination worker is healthy and primed does the
        // session's location change — a crash before this leaves `location` naming the
        // source GPU, so ordinary crash recovery rebuilds the source worker, never a
        // phantom destination one.
        self.storage().set_location(
            &row.id,
            &session::location_for(destination_gpu),
            &now_rfc3339(),
        )?;
        self.transition(&row.id, SessionState::Resuming, SessionState::Active, pid)?;

        let interruption_ms = interruption_start.elapsed().as_secs_f64() * 1000.0;
        let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        Ok(MigrationReport {
            session: to_session_info(&self.storage().resolve(&row.id)?),
            source_gpu,
            destination_gpu,
            native: false,
            total_ms,
            interruption_ms,
            replay_prefill_ms,
            token_count: row.token_count,
            source_vram_freed_mib,
            destination_vram_used_mib,
            bytes_transferred: None,
            ttft_ms: None,
        })
    }

    /// Reattaches the source GPU and returns the session to `Active`, for any failure
    /// past the point the source worker was already stopped. Mirrors
    /// `resume_cancellable`'s own attach+prefill+transition shape exactly, targeting
    /// `source_gpu` instead of whatever the caller was trying to reach — if this also
    /// fails, the session is left `Paused` (still recoverable via ordinary `ralph
    /// resume`), never worse than before the migration attempt.
    async fn rollback_to_source(
        self: &Arc<Self>,
        row: &SessionRow,
        source_gpu: u32,
        cancel: &mut Option<oneshot::Receiver<()>>,
        cause: CliError,
    ) -> CliError {
        let attach = self
            .attach_or_start_worker(&row.model, &row.id, cancel, false, source_gpu)
            .await;
        let Ok((engine, _, pid)) = attach else {
            eprintln!(
                "migrate: rollback to source gpu{source_gpu} also failed; session left paused, recoverable via ralph resume"
            );
            return cause;
        };
        let messages = match self.storage().replay_turns(&row.id) {
            Ok(messages) => messages,
            Err(_) => return cause,
        };
        let prefill = engine.lock().await.prefill(&messages).await;
        if prefill.is_err() {
            eprintln!(
                "migrate: rollback prefill on source gpu{source_gpu} failed; session left paused, recoverable via ralph resume"
            );
            return cause;
        }
        if let Err(error) =
            self.transition(&row.id, SessionState::Resuming, SessionState::Active, pid)
        {
            eprintln!("migrate: rollback transition failed: {error}");
        }
        cause
    }
}
