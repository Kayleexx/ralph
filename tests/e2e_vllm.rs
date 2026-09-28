//! Real vLLM end-to-end test — no mocked engine. Gated behind `RALPH_E2E_VLLM=1` so
//! plain `cargo test` never depends on a GPU or a `.venv` with vLLM installed; on the
//! actual workstation, run with:
//!
//!   RALPH_E2E_VLLM=1 cargo test --test e2e_vllm -- --ignored --test-threads=1
//!
//! `--test-threads=1` matters here: this test finds "the" daemon process by scanning
//! `/proc`, which assumes it's the only ralph daemon running.
use std::process::{Command, Stdio};
use std::time::Duration;

use assert_cmd::cargo::cargo_bin;
use tempfile::TempDir;

const MODEL: &str = "Qwen/Qwen2.5-0.5B-Instruct";
const SESSION: &str = "e2e-test";

fn ralph() -> Command {
    Command::new(cargo_bin("ralph"))
}

fn env_home(cmd: &mut Command, home: &TempDir) {
    cmd.env("XDG_DATA_HOME", home.path());
}

/// Scans `/proc` for a `ralph __daemon` process whose environment names this test's
/// isolated `XDG_DATA_HOME`, so a stray daemon from an unrelated run can't be mistaken
/// for this test's.
///
/// `/proc` holds plenty of non-PID entries (`self`, `cpuinfo`, ...) — each candidate is
/// skipped with `continue`, never `?`, since bailing the whole scan on the first
/// non-numeric name would make this silently give up before it ever reaches a real PID.
fn find_daemon_pid(home_marker: &str) -> Option<u32> {
    let entries = std::fs::read_dir("/proc").ok()?;
    for entry in entries.flatten() {
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|s| s.parse::<u32>().ok())
        else {
            continue;
        };
        let cmdline = std::fs::read(entry.path().join("cmdline")).unwrap_or_default();
        let cmdline = String::from_utf8_lossy(&cmdline);
        if !cmdline.contains("ralph") || !cmdline.contains("__daemon") {
            continue;
        }
        let environ = std::fs::read(entry.path().join("environ")).unwrap_or_default();
        let environ = String::from_utf8_lossy(&environ);
        if environ
            .split('\0')
            .any(|kv| kv == format!("XDG_DATA_HOME={home_marker}"))
        {
            return Some(pid);
        }
    }
    None
}

fn wait_for_daemon_gone(home_marker: &str, timeout: Duration) {
    let deadline = std::time::Instant::now() + timeout;
    while std::time::Instant::now() < deadline {
        if find_daemon_pid(home_marker).is_none() {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

#[test]
#[ignore]
fn full_lifecycle_against_real_vllm() {
    if std::env::var("RALPH_E2E_VLLM").is_err() {
        eprintln!("skipping: set RALPH_E2E_VLLM=1 to run this test against real vLLM");
        return;
    }

    let home = tempfile::tempdir().unwrap();
    let home_marker = home.path().to_str().unwrap().to_string();

    // 1. Start a real vLLM-backed session.
    let mut run = ralph();
    env_home(&mut run, &home);
    let run_output = run
        .args(["run", MODEL, "--name", SESSION])
        .output()
        .unwrap();
    assert!(
        run_output.status.success(),
        "ralph run failed: {}",
        String::from_utf8_lossy(&run_output.stderr)
    );

    // 2. Query it and confirm real output comes back.
    let mut query = ralph();
    env_home(&mut query, &home);
    let query_output = query.args(["query", SESSION, "2+2="]).output().unwrap();
    assert!(query_output.status.success());
    assert!(
        !query_output.stdout.is_empty(),
        "expected non-empty generated output"
    );

    // 3. Ctrl-C mid-generation: cancel, but the session must stay usable afterward.
    let mut cancel_cmd = ralph();
    env_home(&mut cancel_cmd, &home);
    let mut child = cancel_cmd
        .args(["query", SESSION, "write a long story about the ocean"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    std::thread::sleep(Duration::from_millis(300));
    let pid = child.id();
    Command::new("kill")
        .args(["-INT", &pid.to_string()])
        .status()
        .unwrap();
    let status = child.wait().unwrap();
    assert_eq!(status.code(), Some(1), "cancelled query should exit 1");

    let mut inspect_after_cancel = ralph();
    env_home(&mut inspect_after_cancel, &home);
    let inspect_output = inspect_after_cancel
        .args(["--json", "inspect", SESSION])
        .output()
        .unwrap();
    let inspected: serde_json::Value = serde_json::from_slice(&inspect_output.stdout).unwrap();
    assert_eq!(
        inspected["session"]["state"], "active",
        "session must remain active after cancellation"
    );

    let mut query_again = ralph();
    env_home(&mut query_again, &home);
    let query_again_output = query_again
        .args(["query", SESSION, "2+2="])
        .output()
        .unwrap();
    assert!(
        query_again_output.status.success(),
        "session should still be queryable after a cancelled request"
    );

    // 4. `ralph ps` lists it.
    let mut ps = ralph();
    env_home(&mut ps, &home);
    let ps_output = ps.args(["--json", "ps"]).output().unwrap();
    let sessions: serde_json::Value = serde_json::from_slice(&ps_output.stdout).unwrap();
    assert!(
        sessions
            .as_array()
            .unwrap()
            .iter()
            .any(|s| s["name"] == SESSION)
    );

    // 5. Kill the daemon (simulating a crash) and confirm metadata survives a restart.
    let daemon_pid = find_daemon_pid(&home_marker).expect("daemon should be running by now");
    Command::new("kill")
        .args(["-KILL", &daemon_pid.to_string()])
        .status()
        .unwrap();
    wait_for_daemon_gone(&home_marker, Duration::from_secs(5));

    let mut ps_after_restart = ralph();
    env_home(&mut ps_after_restart, &home);
    let ps_after_output = ps_after_restart.args(["--json", "ps"]).output().unwrap();
    let sessions_after: serde_json::Value =
        serde_json::from_slice(&ps_after_output.stdout).unwrap();
    let session_after = sessions_after
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["name"] == SESSION)
        .cloned();
    assert!(
        session_after.is_some(),
        "session metadata must survive a daemon crash + restart"
    );
    assert_eq!(session_after.unwrap()["model"], MODEL);

    let mut inspect_after_restart = ralph();
    env_home(&mut inspect_after_restart, &home);
    let inspect_after_output = inspect_after_restart
        .args(["--json", "inspect", SESSION])
        .output()
        .unwrap();
    let inspected_after: serde_json::Value =
        serde_json::from_slice(&inspect_after_output.stdout).unwrap();
    // No recovery machinery exists yet: a crashed worker's session is demoted to
    // `stopped`, never silently left `active`, and never promoted to `failed` either.
    assert_eq!(inspected_after["session"]["state"], "stopped");

    // The restart's own auto-started daemon (and the orphaned vLLM from the SIGKILL
    // above) would otherwise sit on GPU memory the next test needs.
    kill_leftover_vllm();
    if let Some(pid) = find_daemon_pid(&home_marker) {
        let _ = Command::new("kill")
            .args(["-KILL", &pid.to_string()])
            .status();
    }
}

#[test]
#[ignore]
fn kill_vllm_mid_generation_reports_stopped_not_active() {
    if std::env::var("RALPH_E2E_VLLM").is_err() {
        eprintln!("skipping: set RALPH_E2E_VLLM=1 to run this test against real vLLM");
        return;
    }

    // Defensive: a prior test's vLLM left running would starve this one of GPU memory.
    kill_leftover_vllm();

    let home = tempfile::tempdir().unwrap();
    let home_marker = home.path().to_str().unwrap().to_string();
    let mut run = ralph();
    env_home(&mut run, &home);
    let run_output = run
        .args(["run", MODEL, "--name", "e2e-kill-worker"])
        .output()
        .unwrap();
    assert!(
        run_output.status.success(),
        "ralph run failed: {}",
        String::from_utf8_lossy(&run_output.stderr)
    );

    // Find the vLLM child (not the ralph daemon) and SIGKILL its whole process group —
    // vLLM's EngineCore worker is a separate OS process, not just the APIServer.
    let vllm_pid = find_child_process("vllm").expect("vllm worker should be running");
    Command::new("kill")
        .args(["-KILL", &format!("-{vllm_pid}")])
        .status()
        .unwrap();
    std::thread::sleep(Duration::from_secs(2));

    let mut inspect = ralph();
    env_home(&mut inspect, &home);
    let inspect_output = inspect
        .args(["--json", "inspect", "e2e-kill-worker"])
        .output()
        .unwrap();
    let inspected: serde_json::Value = serde_json::from_slice(&inspect_output.stdout).unwrap();
    assert_eq!(inspected["session"]["state"], "stopped");

    if let Some(pid) = find_daemon_pid(&home_marker) {
        let _ = Command::new("kill")
            .args(["-KILL", &pid.to_string()])
            .status();
    }
}

fn find_child_process(name_contains: &str) -> Option<u32> {
    let entries = std::fs::read_dir("/proc").ok()?;
    for entry in entries.flatten() {
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|s| s.parse::<u32>().ok())
        else {
            continue;
        };
        let cmdline = std::fs::read(entry.path().join("cmdline")).unwrap_or_default();
        let cmdline = String::from_utf8_lossy(&cmdline);
        if cmdline.contains(name_contains) && cmdline.contains("serve") {
            return Some(pid);
        }
    }
    None
}

/// Best-effort teardown so one test's vLLM subprocess doesn't sit on GPU memory the next
/// test needs — Phase 1 has no `ralph stop`, so tests reach for `kill` directly.
fn kill_leftover_vllm() {
    if let Some(pid) = find_child_process("vllm") {
        let _ = Command::new("kill")
            .args(["-KILL", &format!("-{pid}")])
            .status();
        std::thread::sleep(Duration::from_millis(500));
    }
}
