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

/// Session-continuity audit at real scale: existing tests all build 2-4 turns of
/// history — this proves replay/prefill still reproduces exact recall across a
/// checkpoint/pause/resume round trip with a genuinely deep history, not just a toy one.
#[test]
#[ignore]
fn deep_history_replays_correctly_across_native_and_portable_resume() {
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

    // The first worker never had the KV connector attached (nothing was checkpointed
    // before it started), so its first checkpoint/pause/resume is honestly portable —
    // same mechanic `checkpoint_pause_resume_fast_then_portable_fallback_paths` already
    // documents. Only the *resumed* worker (started with the connector from the outset)
    // can produce a real native resume, so the deep history is built after this warm-up.
    h.checkpoint();
    h.pause();
    wait_until(Duration::from_secs(10), || h.worker().is_none());
    let resumed = h.resume();
    assert_eq!(
        resumed["native"], false,
        "warm-up resume should be portable: {resumed}"
    );

    for i in 0..25 {
        h.json(&[
            "query",
            SESSION,
            &format!("Say the number {i}. Just the number."),
        ]);
    }

    h.checkpoint();
    h.pause();
    wait_until(Duration::from_secs(10), || h.worker().is_none());
    let resumed = h.resume();
    assert_eq!(resumed["native"], true, "expected native resume: {resumed}");
    let reply = h.json(&[
        "query",
        SESSION,
        "What secret word did I ask you to remember at the very start? One word only.",
    ]);
    assert!(
        reply["text"]
            .as_str()
            .unwrap()
            .to_lowercase()
            .contains("papaya"),
        "deep history lost the early turn on native resume: {reply}"
    );

    h.pause();
    wait_until(Duration::from_secs(10), || h.worker().is_none());
    std::fs::remove_dir_all(h.kvcache_dir(&session_id)).unwrap();
    let resumed = h.resume();
    assert_eq!(
        resumed["native"], false,
        "expected portable resume: {resumed}"
    );
    let reply = h.json(&[
        "query",
        SESSION,
        "What secret word did I ask you to remember at the very start? One word only.",
    ]);
    assert!(
        reply["text"]
            .as_str()
            .unwrap()
            .to_lowercase()
            .contains("papaya"),
        "deep history lost the early turn on portable replay: {reply}"
    );
}

/// A vLLM upgrade (same engine, different version) must be treated the same as any
/// other fingerprint mismatch: resume stays correct via the portable path, and
/// `inspect` reports the honest reason rather than a bare "unavailable".
#[test]
#[ignore]
fn engine_version_change_forces_portable_resume_and_is_reported_honestly() {
    if !enabled() {
        return;
    }
    let mut h = Harness::new();
    h.start();
    h.json(&[
        "query",
        SESSION,
        "Remember the secret word kiwi. Reply with that word only.",
    ]);
    h.checkpoint();
    h.pause();
    wait_until(Duration::from_secs(10), || h.worker().is_none());
    h.resume();
    h.json(&[
        "query",
        SESSION,
        "What secret word did I ask you to remember? One word only.",
    ]);
    h.checkpoint();
    h.pause();
    wait_until(Duration::from_secs(10), || h.worker().is_none());

    h.db()
        .execute(
            "UPDATE sessions SET engine_version = 'upgraded-for-test' WHERE name = ?1",
            [SESSION],
        )
        .unwrap();
    assert_eq!(
        h.inspect()["restore_readiness"],
        "portable only - engine mismatch"
    );

    let resumed = h.resume();
    assert_eq!(resumed["session"]["state"], "active");
    assert_eq!(
        resumed["native"], false,
        "an engine-version mismatch must never be trusted for native resume: {resumed}"
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
            .contains("kiwi"),
        "context lost across an engine-version change: {reply}"
    );
}
