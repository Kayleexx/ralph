//! RALPH_E2E_VLLM=1 cargo test --test e2e_hibernate -- --ignored --test-threads=1
//!
//! Real, unmocked vLLM — validates `ralph hibernate` actually releases the GPU worker
//! and that `ralph resume` restores it, on top of the daemon-side plumbing already
//! covered by unit tests in `src/daemon/tests_lifecycle.rs`.
//!
//! The "RAM tier full -> fall through to NVMe" edge case is vLLM's own
//! `TieringOffloadingSpec` behavior, not Ralph's — it was validated during the real
//! feasibility experiment behind `engine/vllm/kv_offload.rs` (cross-process KV reuse
//! confirmed via `/metrics` after a SIGKILL). Re-simulating a host-memory-full
//! condition here would not exercise any Ralph code path, so it is not re-tested; this
//! is a documentation note, not a fake success path.
mod support;
use std::time::Duration;
use support::*;

fn enabled() -> bool {
    std::env::var("RALPH_E2E_VLLM").as_deref() == Ok("1")
}

#[test]
#[ignore]
fn hibernate_then_resume_round_trip_and_portable_fallback() {
    if !enabled() {
        return;
    }
    let mut h = Harness::new();
    let original = h.start();
    let session_id = original["id"].as_str().unwrap().to_string();
    h.json(&[
        "query",
        SESSION,
        "Remember the secret word papaya. Reply with that word only.",
    ]);
    h.checkpoint();

    let hibernated = h.hibernate();
    assert_eq!(hibernated["state"], "hibernated");
    wait_until(Duration::from_secs(10), || h.worker().is_none());

    let resumed = h.resume();
    assert_eq!(resumed["session"]["state"], "active");
    assert_eq!(h.inspect()["session"]["id"], session_id);

    let reply = h.json(&[
        "query",
        SESSION,
        "What secret word did I ask you to remember? One word only.",
    ]);
    assert!(
        reply["text"]
            .as_str()
            .unwrap()
            .to_lowercase()
            .contains("papaya"),
        "context lost across hibernate/resume: {reply}"
    );

    // Deleting the KV directory must still allow a portable resume out of Hibernated —
    // same Invariant 2 guarantee pause/resume already relies on.
    h.checkpoint();
    let hibernated = h.hibernate();
    assert_eq!(hibernated["state"], "hibernated");
    wait_until(Duration::from_secs(10), || h.worker().is_none());
    std::fs::remove_dir_all(h.kvcache_dir(&session_id)).unwrap();
    let resumed = h.resume();
    assert_eq!(resumed["session"]["state"], "active");
    assert_eq!(
        resumed["native"], false,
        "must not claim native restore from a deleted directory: {resumed}"
    );
    assert_eq!(h.inspect()["session"]["id"], session_id);
}
