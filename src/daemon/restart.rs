//! Bounded backoff policy for automatic worker restart after a crash. Pure and
//! I/O-free so it's testable without real sleeps.
use std::time::Duration;

const MAX_ATTEMPTS: u32 = 3;

pub(super) struct RestartPolicy {
    max_attempts: u32,
    backoff: fn(u32) -> Duration,
}

fn default_backoff(attempt: u32) -> Duration {
    Duration::from_millis(200 * (1u64 << attempt.min(3)))
}

impl Default for RestartPolicy {
    fn default() -> Self {
        RestartPolicy {
            max_attempts: MAX_ATTEMPTS,
            backoff: default_backoff,
        }
    }
}

impl RestartPolicy {
    #[cfg(test)]
    fn with_backoff(max_attempts: u32, backoff: fn(u32) -> Duration) -> Self {
        RestartPolicy {
            max_attempts,
            backoff,
        }
    }

    pub(super) fn exhausted(&self, attempts: u32) -> bool {
        attempts >= self.max_attempts
    }

    pub(super) fn backoff_for(&self, attempt: u32) -> Duration {
        (self.backoff)(attempt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn not_exhausted_below_the_bound() {
        let policy = RestartPolicy::with_backoff(3, |_| Duration::ZERO);
        assert!(!policy.exhausted(0));
        assert!(!policy.exhausted(2));
    }

    #[test]
    fn exhausted_at_and_beyond_the_bound() {
        let policy = RestartPolicy::with_backoff(3, |_| Duration::ZERO);
        assert!(policy.exhausted(3));
        assert!(policy.exhausted(4));
    }

    #[test]
    fn backoff_grows_with_attempt_number() {
        let policy = RestartPolicy::default();
        assert!(policy.backoff_for(1) > policy.backoff_for(0));
        assert!(policy.backoff_for(2) > policy.backoff_for(1));
    }
}
