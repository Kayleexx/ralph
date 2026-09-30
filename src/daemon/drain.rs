//! `ralph drain`: empties one location of its actively-running sessions, either by
//! handing them off to a configured destination (`--to`) or, absent one, hibernating
//! them in place. Single-GPU codebase (see `admission.rs`), so `location` names the one
//! GPU this daemon manages and a drain in progress blocks every new admission
//! system-wide for its duration (`recovery::attach_or_start_worker`'s draining check).
use std::sync::Arc;

use tokio::sync::oneshot;

use super::Daemon;
use crate::engine::Engine;
use crate::error::CliError;
use crate::ipc::{DrainOutcome, DrainReport};
use crate::state::SessionState;

/// `SessionRow.location` is stored as `"local/gpu<N>"`, but `ralph drain gpu0` names
/// just the bare GPU — accept either the exact stored value or its bare GPU suffix.
fn location_matches(stored: &str, requested: &str) -> bool {
    stored == requested || stored.rsplit('/').next() == Some(requested)
}

impl<E: Engine + 'static> Daemon<E> {
    pub(super) fn is_draining(&self) -> bool {
        self.draining
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .is_some()
    }

    pub async fn drain_cancellable(
        self: &Arc<Self>,
        location: &str,
        to: Option<String>,
        yes: bool,
        mut cancel: Option<oneshot::Receiver<()>>,
    ) -> Result<DrainReport, CliError> {
        let candidates: Vec<_> = self
            .storage()
            .list()?
            .into_iter()
            .filter(|row| {
                location_matches(&row.location, location) && row.state == SessionState::Active
            })
            .collect();

        let action = if to.is_some() { "handoff" } else { "hibernate" };
        let plan: Vec<DrainOutcome> = candidates
            .iter()
            .map(|row| DrainOutcome {
                name: row.name.clone(),
                action: action.to_string(),
                ok: false,
                detail: None,
            })
            .collect();

        // Moving a session off-machine is treated as the destructive/irreversible
        // variant regardless of count; a multi-session hibernate-only drain is printed
        // first either way.
        let requires_confirmation = to.is_some() || plan.len() > 1;
        if requires_confirmation && !yes {
            let sessions = plan
                .into_iter()
                .map(|mut o| {
                    o.detail = Some("planned; rerun with --yes to execute".into());
                    o
                })
                .collect();
            return Ok(DrainReport {
                location: location.to_string(),
                executed: false,
                sessions,
                all_safe: false,
            });
        }

        *self
            .draining
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(location.to_string());
        let mut outcomes = Vec::with_capacity(candidates.len());
        for row in candidates {
            if cancelled_now(&mut cancel).await {
                outcomes.push(DrainOutcome {
                    name: row.name.clone(),
                    action: action.to_string(),
                    ok: false,
                    detail: Some("drain interrupted before this session was reached".into()),
                });
                continue;
            }
            // `handoff_cancellable`/`hibernate_cancellable` each already acquire the
            // per-session lock themselves — a locked/blocked session surfaces here as
            // their own "operation already in progress" error, not a separate check.
            let result = if let Some(destination) = &to {
                self.handoff_cancellable(&row.id, destination, None, None)
                    .await
                    .map(|_| ())
            } else {
                self.hibernate_cancellable(&row.id, None).await.map(|_| ())
            };
            outcomes.push(DrainOutcome {
                name: row.name.clone(),
                action: action.to_string(),
                ok: result.is_ok(),
                detail: result.err().map(|e| e.to_string()),
            });
        }
        *self
            .draining
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;

        let all_safe = outcomes.iter().all(|o| o.ok);
        Ok(DrainReport {
            location: location.to_string(),
            executed: true,
            sessions: outcomes,
            all_safe,
        })
    }
}

/// `true` only once cancellation has actually fired — used between sessions, not
/// mid-session (each of `handoff_cancellable`/`hibernate_cancellable` already has its
/// own safe rollback for a cancellation reaching it directly).
async fn cancelled_now(cancel: &mut Option<oneshot::Receiver<()>>) -> bool {
    match cancel {
        Some(rx) => matches!(
            rx.try_recv(),
            Ok(()) | Err(oneshot::error::TryRecvError::Closed)
        ),
        None => false,
    }
}
