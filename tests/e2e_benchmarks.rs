//! RALPH_E2E_VLLM=1 cargo test --test e2e_benchmarks -- --ignored --test-threads=1
//!
//! Validation tooling, not a user-facing profiler: real, unmocked vLLM timing for the
//! operations this project's hardening pass asks to be measured. Prints one JSON
//! report per run rather than asserting thresholds, since absolute numbers depend on
//! this machine's GPU/disk and aren't a regression gate by themselves.
#[path = "support/benchmark.rs"]
#[allow(dead_code)]
mod benchmark;
mod support;
use std::time::{Duration, Instant};

use serde_json::json;
use support::*;

fn enabled() -> bool {
    std::env::var("RALPH_E2E_VLLM").as_deref() == Ok("1")
}

fn dir_size(dir: &std::path::Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .map(|entry| {
            let path = entry.path();
            if path.is_dir() {
                dir_size(&path)
            } else {
                entry.metadata().map(|m| m.len()).unwrap_or(0)
            }
        })
        .sum()
}

#[test]
#[ignore]
fn recovery_checkpoint_pause_resume_hibernate_latency_and_kv_size() {
    if !enabled() {
        return;
    }
    let mut h = Harness::new();
    let original = h.start();
    let session_id = original["id"].as_str().unwrap().to_string();
    h.json(&[
        "query",
        SESSION,
        "Remember the secret word tangerine. Reply with that word only.",
    ]);

    // The first worker never had the KV connector attached (nothing was checkpointed
    // before it started), so its first pause/resume is honestly portable — same
    // mechanic `checkpoint_pause_resume_fast_then_portable_fallback_paths` documents.
    // Only the *resumed* worker (started with the connector from the outset) can
    // produce a real native resume, so the timed measurements below happen after this
    // untimed warm-up round.
    h.checkpoint();
    h.pause();
    wait_until(Duration::from_secs(10), || h.worker().is_none());
    h.resume();

    let t = Instant::now();
    h.checkpoint();
    let checkpoint_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    h.pause();
    let pause_ms = t.elapsed().as_secs_f64() * 1000.0;
    wait_until(Duration::from_secs(10), || h.worker().is_none());

    let t = Instant::now();
    let resumed = h.resume();
    let native_resume_ms = t.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(
        resumed["native"], true,
        "expected a native resume: {resumed}"
    );

    let kv_bytes = dir_size(&h.kvcache_dir(&session_id));
    let history_bytes: u64 = h
        .turns()
        .iter()
        .map(|(_, _, content, _)| content.len() as u64)
        .sum();

    let t = Instant::now();
    h.pause();
    wait_until(Duration::from_secs(10), || h.worker().is_none());
    std::fs::remove_dir_all(h.kvcache_dir(&session_id)).unwrap();
    let resumed = h.resume();
    let portable_resume_ms = t.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(resumed["native"], false);

    let t = Instant::now();
    h.hibernate();
    let hibernate_ms = t.elapsed().as_secs_f64() * 1000.0;
    wait_until(Duration::from_secs(10), || h.worker().is_none());
    h.resume();

    let inspected = h.inspect();
    let report = json!({
        "hardware": benchmark::vram()["gpu"],
        "model": inspected["session"]["model"],
        "engine": "vllm",
        "engine_version": inspected["session"]["engine_version"],
        "kv_dtype": "bfloat16",
        "storage_medium": "local disk (session dir mount)",
        "checkpoint_ms": checkpoint_ms,
        "pause_ms": pause_ms,
        "native_resume_ms": native_resume_ms,
        "portable_resume_ms": portable_resume_ms,
        "hibernate_ms": hibernate_ms,
        "kv_bytes": kv_bytes,
        "history_bytes": history_bytes,
    });
    eprintln!("{}", serde_json::to_string_pretty(&report).unwrap());
}

#[test]
#[ignore]
fn handoff_bytes_and_downtime() {
    if !enabled() {
        return;
    }
    let Some(dest) = std::env::var("RALPH_E2E_HANDOFF_DEST").ok() else {
        eprintln!("skipping: set RALPH_E2E_HANDOFF_DEST to run this benchmark");
        return;
    };
    let mut h = Harness::new();
    h.start();
    h.json(&["query", SESSION, "hello"]);

    let t = Instant::now();
    let output = h.handoff_json(&dest, Some("bench-handoff"));
    let downtime_ms = t.elapsed().as_secs_f64() * 1000.0;
    assert!(output.status.success(), "{output:?}");

    let report = json!({"downtime_ms": downtime_ms});
    eprintln!("{}", serde_json::to_string_pretty(&report).unwrap());
}

/// Prefix-cache benefit: repeated queries sharing a long common prefix, cold (first
/// hit) vs. warm (prefix already cached). An explicit `--no-enable-prefix-caching`
/// control run isn't possible on this machine — real vLLM 0.30.0 takes a different
/// startup path without prefix caching that requires `nvcc`/a CUDA toolkit (confirmed
/// live: `RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda'
/// doesn't exist`), which isn't installed here. Cold-vs-warm within the (always-on,
/// per `spawn.rs`) session is still a real, honest before/after measurement.
#[test]
#[ignore]
fn prefix_cache_cold_vs_warm_latency() {
    if !enabled() {
        return;
    }
    let mut h = Harness::new();
    h.start();
    let prefix = "You are a helpful assistant. ".repeat(40);

    let t = Instant::now();
    h.json(&[
        "query",
        SESSION,
        &format!("{prefix}Question 0: what is 0+1?"),
    ]);
    let cold_ms = t.elapsed().as_secs_f64() * 1000.0;

    let mut warm_ms = Vec::new();
    for i in 1..6 {
        let t = Instant::now();
        h.json(&[
            "query",
            SESSION,
            &format!("{prefix}Question {i}: what is {i}+1?"),
        ]);
        warm_ms.push(t.elapsed().as_secs_f64() * 1000.0);
    }

    let report = json!({
        "hardware": benchmark::vram()["gpu"],
        "model": MODEL,
        "engine": "vllm",
        "prefix_chars": prefix.len(),
        "cold_ms": cold_ms,
        "warm_ms": warm_ms,
    });
    eprintln!("{}", serde_json::to_string_pretty(&report).unwrap());
}

// Sequential-vs-concurrent multi-model throughput numbers already come out of
// `e2e_vllm.rs`'s `multimodel::two_qwen_workers_admission_recovery_and_reconciliation`
// (via `support::benchmark::measure`) — not duplicated here.
