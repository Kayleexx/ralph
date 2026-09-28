//! Session lifecycle state machine.
//!
//! The full lifecycle is defined now so later phases don't need to widen this enum (and
//! touch every match arm / storage serialization). This phase only ever constructs
//! `Created`, `Starting`, `Active`, `Recovering`, `Failed`, and `Stopped` — the rest have no valid
//! transition into them yet.
use std::fmt;
use std::str::FromStr;

use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    Created,
    Starting,
    Active,
    /// Reserved for Phase 3 (pause/resume). Not constructed in Phase 1.
    Pausing,
    /// Reserved for Phase 3 (pause/resume). Not constructed in Phase 1.
    Paused,
    /// Reserved for Phase 3 (pause/resume). Not constructed in Phase 1.
    Resuming,
    /// Reserved for Phase 4 (hibernation). Not constructed in Phase 1.
    Hibernated,
    /// Worker loss or an explicit recovery attempt.
    Recovering,
    /// Reserved for Phase 6 (handoff/drain). Not constructed in Phase 1.
    Moving,
    Failed,
    Stopped,
}

impl SessionState {
    const ALL: &'static [SessionState] = &[
        SessionState::Created,
        SessionState::Starting,
        SessionState::Active,
        SessionState::Pausing,
        SessionState::Paused,
        SessionState::Resuming,
        SessionState::Hibernated,
        SessionState::Recovering,
        SessionState::Moving,
        SessionState::Failed,
        SessionState::Stopped,
    ];

    fn as_str(self) -> &'static str {
        match self {
            SessionState::Created => "created",
            SessionState::Starting => "starting",
            SessionState::Active => "active",
            SessionState::Pausing => "pausing",
            SessionState::Paused => "paused",
            SessionState::Resuming => "resuming",
            SessionState::Hibernated => "hibernated",
            SessionState::Recovering => "recovering",
            SessionState::Moving => "moving",
            SessionState::Failed => "failed",
            SessionState::Stopped => "stopped",
        }
    }
}

impl fmt::Display for SessionState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Error)]
#[error("unrecognized session state: {0}")]
pub struct ParseStateError(String);

impl FromStr for SessionState {
    type Err = ParseStateError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        SessionState::ALL
            .iter()
            .copied()
            .find(|state| state.as_str() == s)
            .ok_or_else(|| ParseStateError(s.to_string()))
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
#[error("cannot transition session from {from} to {to}")]
pub struct StateError {
    pub from: SessionState,
    pub to: SessionState,
}

/// Validates lifecycle transitions through Phase 2;
/// everything else fails closed with a typed error rather than panicking, since several
/// `(from, to)` pairs are legitimately unreachable until later phases implement them.
pub fn validate_transition(from: SessionState, to: SessionState) -> Result<(), StateError> {
    use SessionState::*;
    match (from, to) {
        (Created, Starting) => Ok(()),
        (Starting, Active) => Ok(()),
        // vLLM never became healthy within the startup timeout.
        (Starting, Failed) | (Starting, Stopped) => Ok(()),
        (Stopped, Recovering) => Ok(()),
        // worker exited or was stopped cleanly; a crashed worker does not make the
        // session itself a failure, so this lands in Stopped, not Failed.
        (Active, Stopped) => Ok(()),
        // reserved for a genuinely unrecoverable session-level failure, distinct from
        // the ordinary worker-gone case above.
        (Active, Failed) => Ok(()),
        // Phase 2: worker lost mid-session (crash detected) or found orphaned after a
        // daemon restart — the logical session survives, it just needs `ralph recover`.
        (Active, Recovering) => Ok(()),
        (Starting, Recovering) => Ok(()),
        // `ralph recover` succeeded: context replayed onto a fresh/reused worker.
        (Recovering, Active) => Ok(()),
        // the bounded automatic restart policy gave up; per Invariant 5 this is still
        // not a session failure, so it lands in Stopped, not Failed.
        (Recovering, Stopped) => Ok(()),
        // reserved for a genuinely unrecoverable session found during recovery.
        (Recovering, Failed) => Ok(()),
        (from, to) => Err(StateError { from, to }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_phase1_transitions_succeed() {
        assert!(validate_transition(SessionState::Created, SessionState::Starting).is_ok());
        assert!(validate_transition(SessionState::Starting, SessionState::Active).is_ok());
        assert!(validate_transition(SessionState::Starting, SessionState::Failed).is_ok());
        assert!(validate_transition(SessionState::Active, SessionState::Stopped).is_ok());
        assert!(validate_transition(SessionState::Active, SessionState::Failed).is_ok());
    }

    #[test]
    fn valid_phase2_recovery_transitions_succeed() {
        assert!(validate_transition(SessionState::Active, SessionState::Recovering).is_ok());
        assert!(validate_transition(SessionState::Starting, SessionState::Recovering).is_ok());
        assert!(validate_transition(SessionState::Recovering, SessionState::Active).is_ok());
        assert!(validate_transition(SessionState::Recovering, SessionState::Stopped).is_ok());
        assert!(validate_transition(SessionState::Recovering, SessionState::Failed).is_ok());
    }

    #[test]
    fn explicit_stopped_recovery_uses_recovering_state() {
        // The daemon drives this pair only for an explicit recovery request.
        assert!(validate_transition(SessionState::Stopped, SessionState::Recovering).is_ok());
        assert!(validate_transition(SessionState::Recovering, SessionState::Paused).is_err());
    }

    #[test]
    fn skipping_starting_is_rejected() {
        let err = validate_transition(SessionState::Created, SessionState::Active).unwrap_err();
        assert_eq!(err.from, SessionState::Created);
        assert_eq!(err.to, SessionState::Active);
    }

    #[test]
    fn transitions_into_later_phase_states_are_rejected() {
        assert!(validate_transition(SessionState::Active, SessionState::Paused).is_err());
        assert!(validate_transition(SessionState::Paused, SessionState::Active).is_err());
        assert!(validate_transition(SessionState::Active, SessionState::Hibernated).is_err());
    }

    #[test]
    fn repeating_a_transition_into_the_same_state_is_not_defined() {
        assert!(validate_transition(SessionState::Active, SessionState::Active).is_err());
    }

    #[test]
    fn display_and_parse_round_trip_for_every_variant() {
        for state in SessionState::ALL {
            let parsed: SessionState = state.to_string().parse().unwrap();
            assert_eq!(parsed, *state);
        }
    }

    #[test]
    fn parse_rejects_unknown_string() {
        assert!("not-a-state".parse::<SessionState>().is_err());
    }
}
