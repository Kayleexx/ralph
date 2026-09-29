//! RALPH_E2E_VLLM=1 cargo test --test e2e_checkpoint_resume -- --ignored --test-threads=1
//!
//! Real, unmocked vLLM — validates the fast (native KV) vs. portable resume decision
//! against actual vLLM 0.30.0 behavior (`OffloadingConnector` + `TieringOffloadingSpec`),
//! not just the daemon-side plumbing already covered by unit tests in
//! `src/daemon/tests_lifecycle.rs`.
mod support;
use std::time::Duration;
use support::*;

fn enabled() -> bool {
    std::env::var("RALPH_E2E_VLLM").as_deref() == Ok("1")
}

#[test]
#[ignore]
fn checkpoint_pause_resume_fast_then_portable_fallback_paths() {
    if !enabled() {
        return;
    }
    let mut h = Harness::new();
    let original = h.start();
    let session_id = original["id"].as_str().unwrap().to_string();
    h.json(&[
        "query",
        SESSION,
        "Remember the secret word mango. Reply with that word only.",
    ]);

    // Explicit checkpoint, then pause: the GPU worker is freed, the session stays present.
    // The worker that was just paused ran *before* this checkpoint existed, so it never
    // had the KV connector attached — nothing is cached yet, so this first resume is
    // honestly portable, not a fake "native" claim.
    h.checkpoint();
    let paused = h.pause();
    assert_eq!(paused["state"], "paused");
    wait_until(Duration::from_secs(10), || h.worker().is_none());
    let resumed = h.resume();
    assert_eq!(resumed["session"]["state"], "active");
    assert_eq!(
        resumed["native"], false,
        "nothing was cached yet under the first checkpoint: {resumed}"
    );

    // The resumed worker now runs *with* the connector attached, so this query is the
    // first one to actually populate the KV directory.
    h.json(&[
        "query",
        SESSION,
        "What secret word did I ask you to remember? One word only.",
    ]);
    h.checkpoint();
    h.pause();
    wait_until(Duration::from_secs(10), || h.worker().is_none());

    // Now a real fast resume: fingerprint still matches and the KV directory holds
    // content from the previous worker's actual generation.
    let resumed = h.resume();
    assert_eq!(resumed["session"]["state"], "active");
    assert_eq!(
        resumed["native"], true,
        "expected a native resume: {resumed}"
    );
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
            .contains("mango"),
        "context lost across pause/resume: {reply}"
    );
    assert_eq!(h.inspect()["session"]["id"], session_id);

    // Deleting the KV directory must still allow a portable resume — no session loss,
    // and the daemon must not crash pointing a fresh worker at a missing directory.
    h.pause();
    wait_until(Duration::from_secs(10), || h.worker().is_none());
    std::fs::remove_dir_all(h.kvcache_dir(&session_id)).unwrap();
    let resumed = h.resume();
    assert_eq!(resumed["session"]["state"], "active");
    assert_eq!(
        resumed["native"], false,
        "must not claim native restore from a deleted directory: {resumed}"
    );
    assert_eq!(h.inspect()["session"]["id"], session_id);

    // An incompatible fingerprint (moved model revision) must never be trusted: --fast-only
    // fails clearly and the session is left exactly Paused, not corrupted or lost.
    h.checkpoint();
    h.pause();
    wait_until(Duration::from_secs(10), || h.worker().is_none());
    h.db()
        .execute(
            "UPDATE sessions SET model_revision = 'moved-for-test' WHERE name = ?1",
            [SESSION],
        )
        .unwrap();
    let fast_only = h.resume_json(&["--fast-only"]);
    assert_eq!(fast_only.status.code(), Some(6));
    assert_eq!(h.inspect()["session"]["state"], "paused");

    // Undo the tamper (restore the real resolved revision) and resume portable, which
    // must always succeed regardless of any checkpoint's validity.
    h.db()
        .execute(
            "UPDATE sessions SET model_revision = tokenizer_revision WHERE name = ?1",
            [SESSION],
        )
        .unwrap();
    let portable = h.resume_json(&["--portable"]);
    assert!(portable.status.success());
    let resumed: serde_json::Value = serde_json::from_slice(&portable.stdout).unwrap();
    assert_eq!(resumed["session"]["state"], "active");
    assert_eq!(
        resumed["native"], false,
        "--portable must always take the full-replay path: {resumed}"
    );
    assert_eq!(h.inspect()["session"]["id"], session_id);
}
