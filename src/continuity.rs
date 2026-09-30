//! Deterministic residency cost model. Given a session's already-known resource state
//! (never anything new to measure), estimate what each existing demotion action would
//! cost, so a planner can pick the cheapest safe one under VRAM pressure. Every number
//! is either a real input already computed elsewhere (`checkpoint::RestoreReadiness`,
//! KV directory size, worker reservation) or a measured constant recorded from real
//! `tests/e2e_benchmarks.rs` runs — never a guessed weight or ML prediction.
use std::fmt;
use std::str::FromStr;

use thiserror::Error;

use crate::checkpoint::RestoreReadiness;

/// A session's own continuity preference (`ralph run --policy`), persisted per session
/// (`sessions.continuity_policy`). Governs `make_room`'s eviction preference — cheaper
/// policies get demoted before more valuable ones — not the demotion mechanism itself.
/// `clap::ValueEnum` lets the CLI validate `--policy` itself, matching `--shell`'s own
/// typed-enum pattern, rather than round-tripping an invalid string to the daemon.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Default,
    clap::ValueEnum,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum ContinuityPolicy {
    Ephemeral,
    #[default]
    Warm,
    Durable,
}

impl ContinuityPolicy {
    fn as_str(self) -> &'static str {
        match self {
            ContinuityPolicy::Ephemeral => "ephemeral",
            ContinuityPolicy::Warm => "warm",
            ContinuityPolicy::Durable => "durable",
        }
    }

    /// Lower sorts first: `make_room` prefers demoting an `Ephemeral` session over a
    /// `Warm` one, and a `Warm` one over a `Durable` one, all else equal.
    pub fn eviction_rank(self) -> u8 {
        match self {
            ContinuityPolicy::Ephemeral => 0,
            ContinuityPolicy::Warm => 1,
            ContinuityPolicy::Durable => 2,
        }
    }
}

impl fmt::Display for ContinuityPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Error)]
#[error("unrecognized continuity policy {0:?} (expected ephemeral, warm, or durable)")]
pub struct ParsePolicyError(String);

impl FromStr for ContinuityPolicy {
    type Err = ParsePolicyError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "ephemeral" => Ok(ContinuityPolicy::Ephemeral),
            "warm" => Ok(ContinuityPolicy::Warm),
            "durable" => Ok(ContinuityPolicy::Durable),
            other => Err(ParsePolicyError(other.to_string())),
        }
    }
}

/// Internal-only — never serialized or shown to users. CLI language stays
/// running/sleeping/hibernated/ready regardless of which action the planner picks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContinuityAction {
    KeepResident,
    Sleep,
    Hibernate,
    DurableOnly,
}

/// Measured constants for this profile (Qwen2.5-0.5B / RTX 5050 Laptop, vLLM 0.30.0),
/// the same spirit as `engine::profiles::measured`'s real per-model numbers — update
/// these from `tests/e2e_benchmarks.rs` results, never guess them.
#[derive(Debug, Clone, Copy)]
pub struct MeasuredCosts {
    pub wake_ms: f64,
    pub native_resume_ms: f64,
    pub worker_cold_start_ms: f64,
    pub portable_replay_ms_per_token: f64,
}

impl Default for MeasuredCosts {
    fn default() -> Self {
        Self {
            wake_ms: 50.0,
            native_resume_ms: 400.0,
            worker_cold_start_ms: 3000.0,
            portable_replay_ms_per_token: 8.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActionCost {
    pub action: ContinuityAction,
    pub estimated_latency_ms: f64,
    pub vram_freed_mib: u64,
    pub storage_bytes: u64,
}

/// Everything the cost model needs about one session, sourced from data the daemon
/// already has on hand (`SessionRow`, `admission::reserved_deficit`,
/// `checkpoint::dir_size`, `restore_readiness`) — this struct adds no new measurement.
#[derive(Debug, Clone, Copy)]
pub struct SessionResourceState {
    pub token_count: i64,
    pub worker_reservation_mib: u64,
    pub kv_bytes: u64,
    pub restore_readiness: RestoreReadiness,
}

fn portable_replay_ms(state: &SessionResourceState, costs: &MeasuredCosts) -> f64 {
    costs.worker_cold_start_ms + state.token_count as f64 * costs.portable_replay_ms_per_token
}

/// Every demotion action available for a resident session, cheapest-safe-first is left
/// to `cheapest`. `KeepResident` is always included as the zero-cost baseline so callers
/// can compare "do nothing" against every real alternative.
pub fn candidate_actions(state: &SessionResourceState, costs: &MeasuredCosts) -> Vec<ActionCost> {
    let hibernate_latency = if state.restore_readiness == RestoreReadiness::NativeAvailable {
        costs.native_resume_ms
    } else {
        portable_replay_ms(state, costs)
    };
    vec![
        ActionCost {
            action: ContinuityAction::KeepResident,
            estimated_latency_ms: 0.0,
            vram_freed_mib: 0,
            storage_bytes: 0,
        },
        ActionCost {
            action: ContinuityAction::Sleep,
            estimated_latency_ms: costs.wake_ms,
            vram_freed_mib: state.worker_reservation_mib,
            storage_bytes: 0,
        },
        ActionCost {
            action: ContinuityAction::Hibernate,
            estimated_latency_ms: hibernate_latency,
            vram_freed_mib: state.worker_reservation_mib,
            storage_bytes: state.kv_bytes,
        },
        ActionCost {
            action: ContinuityAction::DurableOnly,
            estimated_latency_ms: portable_replay_ms(state, costs),
            vram_freed_mib: state.worker_reservation_mib,
            storage_bytes: 0,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_round_trips_through_display_and_from_str() {
        for policy in [
            ContinuityPolicy::Ephemeral,
            ContinuityPolicy::Warm,
            ContinuityPolicy::Durable,
        ] {
            assert_eq!(
                policy.to_string().parse::<ContinuityPolicy>().unwrap(),
                policy
            );
        }
    }

    #[test]
    fn unrecognized_policy_string_is_rejected() {
        assert!("aggressive".parse::<ContinuityPolicy>().is_err());
    }

    #[test]
    fn eviction_rank_prefers_ephemeral_then_warm_then_durable() {
        assert!(
            ContinuityPolicy::Ephemeral.eviction_rank() < ContinuityPolicy::Warm.eviction_rank()
        );
        assert!(ContinuityPolicy::Warm.eviction_rank() < ContinuityPolicy::Durable.eviction_rank());
    }

    #[test]
    fn default_policy_is_warm() {
        assert_eq!(ContinuityPolicy::default(), ContinuityPolicy::Warm);
    }

    fn state(readiness: RestoreReadiness) -> SessionResourceState {
        SessionResourceState {
            token_count: 100,
            worker_reservation_mib: 2000,
            kv_bytes: 50_000_000,
            restore_readiness: readiness,
        }
    }

    #[test]
    fn native_available_makes_hibernate_cheaper_than_durable_only() {
        let costs = MeasuredCosts::default();
        let actions = candidate_actions(&state(RestoreReadiness::NativeAvailable), &costs);
        let hibernate = actions
            .iter()
            .find(|a| a.action == ContinuityAction::Hibernate)
            .unwrap();
        let durable_only = actions
            .iter()
            .find(|a| a.action == ContinuityAction::DurableOnly)
            .unwrap();
        assert!(hibernate.estimated_latency_ms < durable_only.estimated_latency_ms);
    }

    #[test]
    fn without_native_state_hibernate_and_durable_only_cost_the_same() {
        let costs = MeasuredCosts::default();
        let actions = candidate_actions(&state(RestoreReadiness::NoNativeState), &costs);
        let hibernate = actions
            .iter()
            .find(|a| a.action == ContinuityAction::Hibernate)
            .unwrap();
        let durable_only = actions
            .iter()
            .find(|a| a.action == ContinuityAction::DurableOnly)
            .unwrap();
        assert_eq!(
            hibernate.estimated_latency_ms,
            durable_only.estimated_latency_ms
        );
    }

    #[test]
    fn every_action_is_present_exactly_once() {
        let costs = MeasuredCosts::default();
        let actions = candidate_actions(&state(RestoreReadiness::NativeAvailable), &costs);
        for action in [
            ContinuityAction::KeepResident,
            ContinuityAction::Sleep,
            ContinuityAction::Hibernate,
            ContinuityAction::DurableOnly,
        ] {
            assert_eq!(actions.iter().filter(|a| a.action == action).count(), 1);
        }
    }

    #[test]
    fn longer_history_makes_durable_only_more_expensive() {
        let costs = MeasuredCosts::default();
        let short = state(RestoreReadiness::NoNativeState);
        let mut long = short;
        long.token_count = 10_000;
        let short_cost = candidate_actions(&short, &costs);
        let long_cost = candidate_actions(&long, &costs);
        let get = |actions: &[ActionCost]| {
            actions
                .iter()
                .find(|a| a.action == ContinuityAction::DurableOnly)
                .unwrap()
                .estimated_latency_ms
        };
        assert!(get(&long_cost) > get(&short_cost));
    }
}
