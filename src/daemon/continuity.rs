//! The residency-demotion executor: turns a `continuity` cost-model decision into a real
//! action via the existing, already lock-guarded lifecycle primitives (`hibernate_cancellable`).
//! No new mechanism — this module only chooses *which* session to demote.
use std::sync::Arc;

use super::Daemon;
use crate::continuity::{self, ContinuityAction};
use crate::engine::Engine;
use crate::session;
use crate::state::SessionState;
use crate::storage::SessionRow;

impl<E: Engine + 'static> Daemon<E> {
    fn resource_state(
        &self,
        row: &SessionRow,
        worker_reservation_mib: u64,
    ) -> continuity::SessionResourceState {
        let dir = session::kvcache_dir(&session::session_dir(&self.sessions_root, &row.id));
        continuity::SessionResourceState {
            token_count: row.token_count,
            worker_reservation_mib,
            kv_bytes: super::checkpoint::dir_size(&dir),
            restore_readiness: self.restore_readiness(row),
        }
    }

    /// Finds the single cheapest-to-rebuild, single-tenant-worker, `Active` session on
    /// `gpu_index` (never `exclude_session_id`, the one trying to be admitted) and
    /// hibernates it. Only `Hibernate` actually releases a worker's admission
    /// reservation (`admission::reserved_deficit`); `Sleep` keeps it — so make-room
    /// always hibernates, picking the candidate whose `Hibernate` cost the model rates
    /// cheapest, i.e. least valuable to keep resident right now.
    ///
    /// ponytail: only single-tenant workers are eligible — evicting one session off a
    /// worker shared by several would need a fairness policy this phase doesn't have yet.
    pub(super) async fn make_room(
        self: &Arc<Self>,
        gpu_index: u32,
        exclude_session_id: &str,
    ) -> Option<u64> {
        let candidates: Vec<(SessionRow, u64)> = {
            let workers = self.workers.lock().await;
            let mut out = Vec::new();
            for (key, entry) in workers.iter() {
                if key.gpu != gpu_index || entry.session_ids.len() != 1 {
                    continue;
                }
                let id = &entry.session_ids[0];
                if id == exclude_session_id {
                    continue;
                }
                let Ok(row) = self.storage().resolve(id) else {
                    continue;
                };
                if row.state != SessionState::Active {
                    continue;
                }
                // Never raid a session just because it's momentarily quiet — only one
                // that's idle long enough the idle sweep would hibernate it anyway
                // (`idle::IDLE_HIBERNATE_AFTER`). "Never kill useful state blindly."
                let idle = super::idle::idle_for(&row.updated_at).unwrap_or_default();
                if idle < super::idle::IDLE_HIBERNATE_AFTER {
                    continue;
                }
                let reservation = entry
                    .profile
                    .map(crate::engine::profiles::WorkerProfile::reservation_mib)
                    .unwrap_or(0);
                out.push((row, reservation));
            }
            out
        };
        if candidates.is_empty() {
            return None;
        }
        let costs = continuity::MeasuredCosts::default();
        // Rank by (policy eviction preference, hibernate latency): an `Ephemeral`
        // session is sacrificed before a `Warm` one, a `Warm` one before a `Durable`
        // one; cheapest-to-rebuild breaks ties within the same policy. A candidate
        // whose own `--continuity-target` the demotion would violate is never chosen —
        // making room for someone else never breaks another session's own promise.
        let mut best: Option<(String, String, u8, f64, u64)> = None;
        for (row, reservation) in &candidates {
            let state = self.resource_state(row, *reservation);
            let actions = continuity::candidate_actions(&state, &costs);
            let Some(hibernate) = actions
                .iter()
                .find(|a| a.action == ContinuityAction::Hibernate)
            else {
                continue;
            };
            if let Some(target) = row.continuity_target_ms
                && hibernate.estimated_latency_ms > target as f64
            {
                continue;
            }
            let rank = row.continuity_policy.eviction_rank();
            let latency = hibernate.estimated_latency_ms;
            let better = match &best {
                None => true,
                Some((_, _, best_rank, best_latency, _)) => {
                    rank < *best_rank || (rank == *best_rank && latency < *best_latency)
                }
            };
            if better {
                best = Some((
                    row.id.clone(),
                    row.name.clone(),
                    rank,
                    latency,
                    hibernate.vram_freed_mib,
                ));
            }
        }
        let (id, name, _, _, freed) = best?;
        self.hibernate_cancellable(&id, None).await.ok()?;
        *self
            .last_make_room
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some((name, freed));
        Some(freed)
    }
}
