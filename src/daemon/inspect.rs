use super::{Daemon, to_session_info};
use crate::checkpoint::RestoreReadiness;
use crate::engine::Engine;
use crate::error::CliError;
use crate::ipc::{InspectInfo, SessionInfo};
use crate::state::SessionState;

impl<E: Engine + 'static> Daemon<E> {
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
        let readiness = self.restore_readiness(&row);
        let fast_restore = if readiness == RestoreReadiness::NativeAvailable {
            "available"
        } else {
            "unavailable"
        };
        let fingerprint_diff = if readiness == RestoreReadiness::NativeAvailable {
            None
        } else {
            self.fingerprint_diff(&row).filter(|d| !d.is_empty())
        };
        let moved_to = self
            .storage()
            .get_handoff(&row.id)?
            .map(|h| format!("{} as {}", h.destination, h.remote_name));
        Ok(InspectInfo {
            session: to_session_info(&row),
            recoverability: if complete { recoverability } else { "degraded" }.to_string(),
            fast_restore: fast_restore.to_string(),
            restore_readiness: readiness.as_str().to_string(),
            fingerprint_diff,
            last_failure: self.storage().last_worker_failure(&row.model)?,
            portable_state: if complete { "complete" } else { "incomplete" }.to_string(),
            moved_to,
            continuity_policy: row.continuity_policy.to_string(),
            continuity_target_ms: row.continuity_target_ms,
        })
    }
}
