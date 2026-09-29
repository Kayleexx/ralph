//! Real two-model admission, crash recovery, reconciliation and performance evidence.
#[path = "benchmark.rs"]
mod benchmark;
use crate::support::*;
use serde_json::{Value, json};
use std::io::Write;
use std::process::{Child, Stdio};
use std::time::{Duration, Instant};

const Q3: &str = "Qwen/Qwen3-0.6B";
const Q3_SESSION: &str = "qwen3-demo";

fn profile(h: &Harness, id: &str) -> Value {
    let text: String = h
        .db()
        .query_row(
            "SELECT profile FROM session_profiles WHERE session_id=?1",
            [id],
            |r| r.get(0),
        )
        .unwrap();
    serde_json::from_str(&text).unwrap()
}

fn recover_ready(h: &Harness, session: &str) -> Value {
    let mut result = None;
    wait_until(Duration::from_secs(310), || {
        let output = h.output(&["--json", "recover", session]);
        let value: Value = serde_json::from_slice(&output.stdout).unwrap();
        if output.status.success() {
            result = Some(value);
            return true;
        }
        assert_eq!(output.status.code(), Some(6), "{value}");
        assert!(
            value["error"]["summary"]
                .as_str()
                .unwrap()
                .contains("starting or attaching"),
            "{value}"
        );
        false
    });
    result.unwrap()
}

struct Occupier(Child);
impl Drop for Occupier {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn occupy(h: &Harness) -> Occupier {
    // Leave 2000 MiB free; this tests admission refusal without approaching CUDA OOM.
    let ready = h.home.path().join("vram-ready.json");
    let python = std::env::var("VIRTUAL_ENV")
        .map(|p| format!("{p}/bin/python"))
        .unwrap_or_else(|_| ".venv/bin/python".into());
    let child = std::process::Command::new(python)
        .args([
            "-c",
            r#"
import json, pathlib, sys, time
import torch, pynvml as nv
nv.nvmlInit()
gpu = nv.nvmlDeviceGetHandleByIndex(0)
torch.cuda.init()
free = nv.nvmlDeviceGetMemoryInfo(gpu).free
budget = free - 2000 * 1024**2
assert budget > 0
allocation = torch.empty(budget, dtype=torch.uint8, device='cuda')
allocation.fill_(1)
torch.cuda.synchronize()
remaining = nv.nvmlDeviceGetMemoryInfo(gpu).free / 1024**2
assert remaining >= 1800, remaining
pathlib.Path(sys.argv[1]).write_text(json.dumps({'free_mib':remaining}))
while True: time.sleep(1)
"#,
            ready.to_str().unwrap(),
        ])
        .env("RALPH_TEST_VRAM_OWNER", h.home.path())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    let mut owned = Occupier(child);
    wait_until(Duration::from_secs(30), || {
        assert!(owned.0.try_wait().unwrap().is_none(), "VRAM helper exited");
        ready.exists()
    });
    owned
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn two_qwen_workers_admission_recovery_and_reconciliation() {
    if std::env::var("RALPH_E2E_VLLM").as_deref() != Ok("1") {
        return;
    }
    let mut h = Harness::new();
    let before = benchmark::vram();
    let first = h
        .command()
        .args(["--json", "run", MODEL, "--name", SESSION])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    // Synchronize on persisted spawn ownership, while the first startup holds the gate.
    wait_until(Duration::from_secs(15), || {
        h.remember_worker();
        h.worker().is_some()
    });
    let second = h.output(&["--json", "run", MODEL, "--name", "same-model"]);
    assert_eq!(second.status.code(), Some(6));
    assert_eq!(h.worker_records().len(), 1);
    let first_output = first.wait_with_output().unwrap();
    assert!(
        first_output.status.success(),
        "{}",
        String::from_utf8_lossy(&first_output.stdout)
    );
    let a: Value = serde_json::from_slice(&first_output.stdout).unwrap();
    h.json(&["recover", "same-model"]);
    assert_eq!(h.worker_records().len(), 1);
    let b = h.json(&["run", Q3, "--name", Q3_SESSION]);
    h.remember_worker();
    assert_eq!(h.worker_records().len(), 2);
    println!("two workers active; same-model reuse and startup exclusion passed");
    let profiles = [
        profile(&h, a["id"].as_str().unwrap()),
        profile(&h, b["id"].as_str().unwrap()),
    ];
    assert_eq!(profiles[0]["max_context"], 16384);
    assert_eq!(profiles[0]["kv_mib"], 448);
    assert_eq!(profiles[1]["max_context"], 4096);
    assert_eq!(profiles[1]["kv_mib"], 960);
    assert!(benchmark::vram()["free_mib"].as_u64().unwrap() >= 1024);
    h.json(&[
        "query",
        SESSION,
        "Remember my secret word cobalt. Reply cobalt only.",
    ]);
    h.json(&[
        "query",
        Q3_SESSION,
        "Remember my secret word amber. Reply amber only. /no_think",
    ]);

    // CLI generation exercises session locks, persistence and streamed token IDs.
    let prompt = "List 20 short ocean facts, one sentence each. /no_think";
    let start = Instant::now();
    let qa = h
        .command()
        .args(["--json", "query", SESSION, prompt])
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let qb = h
        .command()
        .args(["--json", "query", Q3_SESSION, prompt])
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let replies: Vec<Value> = [qa, qb]
        .into_iter()
        .map(|child| {
            let output = child.wait_with_output().unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stdout)
            );
            serde_json::from_slice(&output.stdout).unwrap()
        })
        .collect();
    assert!(replies.iter().all(|r| r["tokens"].as_u64().unwrap() > 0));
    let cli_concurrent = json!({"replies":replies,"wall_s":start.elapsed().as_secs_f64()});

    let committed = h.turns();
    let mut oversized = h
        .command()
        .args(["--json", "query", Q3_SESSION, "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    oversized
        .stdin
        .take()
        .unwrap()
        .write_all(" ocean".repeat(5000).as_bytes())
        .unwrap();
    let rejected = oversized.wait_with_output().unwrap();
    assert_eq!(rejected.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&rejected.stdout).contains("4096"));
    assert_eq!(h.turns(), committed);
    println!("independent/concurrent queries and context rejection passed");

    let survivor = h.worker().unwrap();
    let crashed = h.worker_for(Q3).unwrap();
    signal_owned(&crashed, true);
    wait_until(Duration::from_secs(10), || {
        h.json(&["inspect", Q3_SESSION])["session"]["state"] == "recovering"
    });
    assert!(healthy(&survivor));
    assert_eq!(h.inspect()["session"]["state"], "active");
    h.json(&[
        "query",
        SESSION,
        "What is my secret word? Reply one word only.",
    ]);
    wait_until(Duration::from_secs(310), || {
        h.remember_worker();
        h.worker_for(Q3)
            .is_some_and(|w| w["nonce"] != crashed["nonce"] && healthy(&w))
    });
    let recovered = recover_ready(&h, Q3_SESSION);
    assert_eq!(recovered["id"], b["id"]);
    assert_eq!(h.worker().unwrap()["nonce"], survivor["nonce"]);
    assert_eq!(h.worker_for(Q3).unwrap()["profile"], profiles[1]);
    let reply = h.json(&[
        "query",
        Q3_SESSION,
        "What is my secret word? One word only. /no_think",
    ]);
    assert!(
        reply["text"]
            .as_str()
            .unwrap()
            .to_lowercase()
            .contains("amber"),
        "{reply}"
    );
    println!("worker replacement preserved profile/history; other worker unchanged");

    let durable = h.turns();
    h.restart_daemon();
    assert_eq!(h.inspect()["session"]["state"], "recovering");
    assert_eq!(
        h.json(&["inspect", Q3_SESSION])["session"]["state"],
        "recovering"
    );
    assert!(h.worker_records().is_empty());
    assert_eq!(h.turns(), durable);
    println!("daemon reconciled both workers and preserved durable history");
    // Process exit precedes the driver's asynchronous VRAM release.
    wait_until(Duration::from_secs(15), || {
        benchmark::vram()["free_mib"].as_u64().unwrap() >= 3316
    });
    h.json(&["recover", SESSION]);
    h.remember_worker();
    let survivor = h.worker().unwrap();
    let occupier = occupy(&h);
    let low_memory = benchmark::vram();
    let rejected = h.output(&["--json", "recover", Q3_SESSION]);
    assert_eq!(rejected.status.code(), Some(6));
    assert!(String::from_utf8_lossy(&rejected.stdout).contains("MiB required"));
    assert_eq!(
        h.json(&["inspect", Q3_SESSION])["session"]["state"],
        "stopped"
    );
    assert_eq!(h.worker_records().len(), 1);
    assert_eq!(h.worker().unwrap()["nonce"], survivor["nonce"]);
    assert!(healthy(&survivor));
    assert_eq!(h.turns(), durable);
    drop(occupier);
    wait_until(Duration::from_secs(15), || {
        benchmark::vram()["free_mib"].as_u64().unwrap() >= 4188
    });
    h.json(&["recover", Q3_SESSION]);
    h.remember_worker();
    assert_eq!(h.turns(), durable);
    assert_eq!(h.worker().unwrap()["profile"], profiles[0]);
    assert_eq!(h.worker_for(Q3).unwrap()["profile"], profiles[1]);
    assert_eq!(h.worker_records().len(), 2);
    println!(
        "insufficient-VRAM recovery refused before spawn; both workers recovered with original budgets"
    );

    let measurements = benchmark::measure(&[h.worker().unwrap(), h.worker_for(Q3).unwrap()]).await;
    let report = json!({"before":before,"profiles":profiles,"cli_concurrent":cli_concurrent,
        "insufficient_vram":low_memory,"final_measurements":measurements,"acceptance_checks":7});
    println!("{report}");
    if let Ok(path) = std::env::var("RALPH_E2E_REPORT") {
        std::fs::write(path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    }
}
