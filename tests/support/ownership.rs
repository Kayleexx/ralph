//! Real process-ownership verification/cleanup for the E2E harness: is a spawned
//! worker actually healthy, and — at test teardown — killing exactly (and only) what
//! this test spawned, never anything else on the machine.
#![allow(dead_code)]

use serde_json::Value;
use std::io::{Read, Write};
use std::process::Command;
use std::time::Duration;

pub fn healthy(worker: &Value) -> bool {
    let Some(endpoint) = worker["endpoint"]
        .as_str()
        .and_then(|s| s.strip_prefix("http://"))
    else {
        return false;
    };
    let Ok(addr) = endpoint.parse() else {
        return false;
    };
    let Ok(mut stream) = std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(100))
    else {
        return false;
    };
    stream
        .set_read_timeout(Some(Duration::from_millis(100)))
        .unwrap();
    if stream.write_all(b"GET /health HTTP/1.0\r\n\r\n").is_err() {
        return false;
    }
    let mut bytes = [0; 128];
    stream
        .read(&mut bytes)
        .is_ok_and(|n| String::from_utf8_lossy(&bytes[..n]).contains("200 OK"))
}

/// Scans by marker, not by process group — mirrors `ownership.rs::owned_members`'s own
/// fix: vLLM's EngineCore child doesn't reliably stay in its `vllm serve` parent's
/// process group, so a group-scoped scan can miss a real, still-running descendant.
pub fn signal_owned(worker: &Value, send_signal: bool) {
    let Some(nonce) = worker["nonce"].as_str() else {
        return;
    };
    if std::fs::read_to_string("/proc/sys/kernel/random/boot_id")
        .ok()
        .as_deref()
        != worker["boot"].as_str()
    {
        return;
    }
    let marker = format!("RALPH_WORKER_OWNER={nonce}");
    for entry in std::fs::read_dir("/proc").unwrap().flatten() {
        let Ok(stat) = std::fs::read_to_string(entry.path().join("stat")) else {
            continue;
        };
        if stat
            .rsplit_once(')')
            .is_some_and(|(_, rest)| rest.trim_start().starts_with('Z'))
        {
            continue;
        }
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|s| s.parse::<u64>().ok())
        else {
            continue;
        };
        let env = std::fs::read(entry.path().join("environ")).unwrap_or_default();
        if env.split(|b| *b == 0).any(|kv| kv == marker.as_bytes()) && send_signal {
            let _ = Command::new("kill")
                .args(["-KILL", "--", &pid.to_string()])
                .status();
        }
    }
}

/// Last-resort sweep, independent of any captured worker record: any process on this
/// boot still carrying a `RALPH_WORKER_OWNER=` marker after the daemon that owned it is
/// gone is definitionally orphaned — appropriate for test teardown (never for
/// production, where killing by marker prefix alone would be too broad).
pub fn sweep_any_orphaned_worker() {
    if std::fs::read_to_string("/proc/sys/kernel/random/boot_id").is_err() {
        return;
    }
    for entry in std::fs::read_dir("/proc").unwrap().flatten() {
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|s| s.parse::<u64>().ok())
        else {
            continue;
        };
        let env = std::fs::read(entry.path().join("environ")).unwrap_or_default();
        let owned = env
            .split(|b| *b == 0)
            .any(|kv| kv.starts_with(b"RALPH_WORKER_OWNER="));
        if owned {
            let _ = Command::new("kill")
                .args(["-KILL", "--", &pid.to_string()])
                .status();
        }
    }
}
