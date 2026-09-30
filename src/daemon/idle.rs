//! Basic idle policy: auto-hibernates sessions that have had no query activity for a
//! while. Deliberately not a scheduler — one fixed threshold, one poll loop, reusing the
//! same per-session lock every other operation already goes through, so a concurrent
//! pause/resume/query just makes this tick skip that session rather than needing its own
//! cancellation protocol.
use std::sync::Arc;
use std::time::Duration;

use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

use super::Daemon;
use crate::engine::Engine;
use crate::error::CliError;
use crate::state::SessionState;

const SWEEP_INTERVAL: Duration = Duration::from_secs(30);
/// Also the make-room eligibility threshold (`continuity.rs`) — a session isn't
/// "cheap to sacrifice" just because it's momentarily asleep, only once it's been
/// idle long enough that this same sweep would hibernate it anyway.
pub(super) const IDLE_HIBERNATE_AFTER: Duration = Duration::from_secs(900);

pub(super) fn idle_for(updated_at: &str) -> Option<Duration> {
    let then = OffsetDateTime::parse(updated_at, &Rfc3339).ok()?;
    Some((OffsetDateTime::now_utc() - then).unsigned_abs())
}

impl<E: Engine + 'static> Daemon<E> {
    pub fn spawn_idle_sweep(self: &Arc<Self>) {
        let daemon = self.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(SWEEP_INTERVAL).await;
                daemon.sweep_idle_sessions().await;
            }
        });
    }

    async fn sweep_idle_sessions(self: &Arc<Self>) {
        let Ok(sessions) = self.storage().list() else {
            return;
        };
        for row in sessions {
            if row.state != SessionState::Active {
                continue;
            }
            let idle = idle_for(&row.updated_at).unwrap_or_default();
            if idle < IDLE_HIBERNATE_AFTER {
                continue;
            }
            // A lock-contention InvalidState just means a concurrent op already has it.
            if let Err(error) = self.hibernate_cancellable(&row.id, None).await
                && !matches!(error, CliError::InvalidState(_))
            {
                eprintln!("idle sweep: could not hibernate {}: {error}", row.name);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idle_for_a_past_timestamp_is_positive() {
        let past = (OffsetDateTime::now_utc() - Duration::from_secs(120))
            .format(&Rfc3339)
            .unwrap();
        let idle = idle_for(&past).unwrap();
        assert!(idle >= Duration::from_secs(119));
    }

    #[test]
    fn idle_for_garbage_is_none() {
        assert!(idle_for("not a timestamp").is_none());
    }
}
