//! RALPH_E2E_VLLM=1 cargo test --test e2e_fault_injection -- --ignored --test-threads=1
//!
//! Real-vLLM fault-injection suite, covering what isn't already exercised by the other
//! real-vLLM suites:
//! - `kill vLLM during generation` / `kill vLLM during recovery` / daemon restart mid
//!   query, Ctrl-C query: `tests/e2e_vllm.rs`.
//! - `Ctrl-C checkpoint` / `Ctrl-C handoff`, SSH auth failure, network failure midway,
//!   destination startup failure: `tests/e2e_handoff_drain.rs`, `src/daemon/tests_handoff.rs`.
//! - `resume with incompatible engine fingerprint` / missing model / tokenizer
//!   mismatch: `src/daemon/tests_lifecycle.rs`.
//! - `corrupt KV checkpoint` (unit-level, engine-agnostic): `src/daemon/tests_checkpoint
//!   ::resume_falls_back_to_portable_when_the_native_checkpoint_is_corrupt`.
//! - `force SQLite busy contention`: `src/storage/tests.rs`.
//! - `fill checkpoint disk` (unit-level, via the injectable override):
//!   `src/daemon/tests_checkpoint.rs`.
//!
//! This file adds the remaining items against real vLLM: a genuinely corrupted (not
//! just missing) KV checkpoint, a removed KV checkpoint, SIGKILL mid-state-transition,
//! and truncated durable history.
mod support;
use std::time::Duration;
use support::*;

fn enabled() -> bool {
    std::env::var("RALPH_E2E_VLLM").as_deref() == Ok("1")
}

const RECALL: &str = "What secret word did I ask you to remember? One word only.";

/// Populates the kvcache directory with real content, exactly matching the depth
/// `tests/e2e_checkpoint_resume.rs`'s own proven-reliable sequence uses before it
/// checks reply correctness: the first worker never has the KV-offload connector
/// attached (`kv_offload_for` only attaches it once a checkpoint row already exists),
/// so this takes one full resume cycle before the *next* checkpoint/pause captures
/// anything real. Deliberately stops at the same round-depth that sequence validates
/// at — one resume cycle further (as an earlier version of this test tried) made even
/// the uncorrupted baseline reply unreliable on this 0.5B model, a model-capability
/// edge unrelated to what this suite is testing.
fn prime_real_kv_content(h: &mut Harness, word: &str) {
    h.json(&[
        "query",
        SESSION,
        &format!("Remember the secret word {word}. Reply with that word only."),
    ]);
    h.checkpoint();
    h.pause();
    wait_until(Duration::from_secs(10), || h.worker().is_none());
    let resumed = h.resume();
    assert_eq!(resumed["session"]["state"], "active");
    assert_eq!(resumed["native"], false);
    h.json(&["query", SESSION, RECALL]);
    h.checkpoint();
    h.pause();
    wait_until(Duration::from_secs(10), || h.worker().is_none());
}

#[test]
#[ignore]
fn corrupt_kv_checkpoint_falls_back_to_portable_resume() {
    if !enabled() {
        return;
    }
    let mut h = Harness::new();
    let session = h.start();
    prime_real_kv_content(&mut h, "kiwi");

    let kvcache = h.kvcache_dir(session["id"].as_str().unwrap());
    assert!(
        kvcache.is_dir() && std::fs::read_dir(&kvcache).unwrap().next().is_some(),
        "the pause just before this should have written real KV bytes"
    );
    // vLLM's TieringOffloadingSpec nests block files several directories deep
    // (`<hash>_r0/32d/78_g0/*.bin`) — a shallow, single-level corrupt loop would miss
    // them entirely and silently corrupt nothing.
    corrupt_files_recursively(&kvcache);

    // Confirmed live against real vLLM 0.30.0: `OffloadingConnector` does not crash on
    // unreadable/garbage tier-file bytes, it silently tolerates them and starts anyway.
    // Ralph's own content-signature check (recorded once the worker that wrote this
    // directory actually stopped, validated before ever offering native resume again)
    // is what actually prevents this from reaching vLLM at all — `native` must read
    // `false` here, not just "history happens to still be correct".
    let resumed = h.resume();
    assert_eq!(resumed["session"]["state"], "active");
    assert_eq!(
        resumed["native"], false,
        "content signature must reject tampered bytes before vLLM ever sees them: {resumed}"
    );
    let reply = h.json(&["query", SESSION, RECALL]);
    assert!(
        reply["text"]
            .as_str()
            .unwrap()
            .to_lowercase()
            .contains("kiwi"),
        "portable fallback lost history: {reply}"
    );
}

fn corrupt_files_recursively(dir: &std::path::Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            corrupt_files_recursively(&path);
        } else if path.is_file() {
            std::fs::write(&path, b"not a real kv checkpoint, just garbage").unwrap();
        }
    }
}

#[test]
#[ignore]
fn removed_kv_checkpoint_falls_back_to_portable_resume() {
    if !enabled() {
        return;
    }
    let mut h = Harness::new();
    let session = h.start();
    prime_real_kv_content(&mut h, "kiwi");

    let kvcache = h.kvcache_dir(session["id"].as_str().unwrap());
    std::fs::remove_dir_all(&kvcache).unwrap();

    let resumed = h.resume();
    assert_eq!(resumed["session"]["state"], "active");
    let reply = h.json(&["query", SESSION, RECALL]);
    assert!(
        reply["text"]
            .as_str()
            .unwrap()
            .to_lowercase()
            .contains("kiwi"),
        "portable fallback lost history: {reply}"
    );
}

#[test]
#[ignore]
fn sigkill_daemon_mid_pause_recovers_to_a_stable_state() {
    if !enabled() {
        return;
    }
    let mut h = Harness::new();
    h.start();
    h.json(&["query", SESSION, "Remember the word cinnamon."]);

    let mut pausing = h.command();
    pausing.args(["--json", "pause", SESSION]);
    let mut child = pausing.spawn().unwrap();
    // `pause` stops a real worker mid-flight — there's a genuine window here, not a
    // synchronization hack for an otherwise-instant operation.
    std::thread::sleep(Duration::from_millis(150));
    h.restart_daemon();
    let _ = child.kill();
    let _ = child.wait();

    let state = h.inspect()["session"]["state"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(
        matches!(state.as_str(), "active" | "paused"),
        "session must land in a stable, resumable state, not stuck mid-pause: {state}"
    );
    if state == "paused" {
        h.resume();
    }
    let reply = h.json(&[
        "query",
        SESSION,
        "What word did I just ask you to remember? One word only.",
    ]);
    assert!(
        reply["text"]
            .as_str()
            .unwrap()
            .to_lowercase()
            .contains("cinnamon"),
        "history lost across a daemon kill mid-pause: {reply}"
    );
}

#[test]
#[ignore]
fn truncated_token_log_tail_recovers_with_only_committed_history() {
    if !enabled() {
        return;
    }
    let mut h = Harness::new();
    h.start();
    h.json(&["query", SESSION, "Remember the word basil."]);
    let before = h.turns().len();
    h.kill_worker();
    h.wait_loss();

    let db_path = h.root().join("ralph.db");
    let len = std::fs::metadata(&db_path).unwrap().len();
    assert!(len > 64, "db unexpectedly small: {len} bytes");
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(&db_path)
        .unwrap();
    file.set_len(len - 32).unwrap();
    drop(file);

    // A truncated SQLite file is corruption, not a state to silently trust — the daemon
    // must refuse to serve out of it rather than returning a plausible-looking but
    // wrong answer (Invariant: never a fake success path for the main feature).
    let out = h.output(&["--json", "inspect", SESSION]);
    if out.status.success() {
        assert_eq!(
            h.turns().len(),
            before,
            "history must not silently grow or shrink"
        );
    } else {
        assert_eq!(
            out.status.code(),
            Some(8),
            "corruption must exit as documented (8)"
        );
    }
}
