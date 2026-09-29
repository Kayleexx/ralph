// Shared across multiple integration-test binaries (e2e_vllm, e2e_phase3, ...), each of
// which recompiles this module as its own crate and only calls the subset it needs — that
// makes every helper "unused" in some binary's own compilation, not actually dead code.
#![allow(dead_code)]

use rusqlite::Connection;
use serde_json::Value;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant};

pub const MODEL: &str = "Qwen/Qwen2.5-0.5B-Instruct";
pub const SESSION: &str = "recovery-demo";

pub fn wait_until(timeout: Duration, mut condition: impl FnMut() -> bool) {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if condition() {
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    panic!("condition did not become true within {timeout:?}");
}

pub struct Harness {
    pub home: tempfile::TempDir,
    daemon: Child,
    workers: Vec<Value>,
}

impl Harness {
    pub fn new() -> Self {
        let home = tempfile::tempdir().unwrap();
        let daemon = Self::spawn_daemon(&home);
        let h = Self {
            home,
            daemon,
            workers: Vec::new(),
        };
        wait_until(Duration::from_secs(10), || {
            h.root().join("daemon.sock").exists()
        });
        h
    }
    fn spawn_daemon(home: &tempfile::TempDir) -> Child {
        let log = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(home.path().join("daemon.log"))
            .unwrap();
        Command::new(assert_cmd::cargo::cargo_bin("ralph"))
            .env("XDG_DATA_HOME", home.path())
            .arg("__daemon")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(log)
            .spawn()
            .unwrap()
    }
    pub fn root(&self) -> PathBuf {
        self.home.path().join("ralph")
    }
    pub fn command(&self) -> Command {
        let mut cmd = Command::new(assert_cmd::cargo::cargo_bin("ralph"));
        cmd.env("XDG_DATA_HOME", self.home.path());
        cmd
    }
    pub fn output(&self, args: &[&str]) -> Output {
        self.command().args(args).output().unwrap()
    }
    pub fn json(&self, args: &[&str]) -> Value {
        let out = self.command().arg("--json").args(args).output().unwrap();
        assert!(
            out.status.success(),
            "{}: {} {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr),
            String::from_utf8_lossy(&out.stdout)
        );
        serde_json::from_slice(&out.stdout).unwrap()
    }
    pub fn inspect(&self) -> Value {
        self.json(&["inspect", SESSION])
    }
    pub fn start(&mut self) -> Value {
        let session = self.json(&["run", MODEL, "--name", SESSION]);
        self.remember_worker();
        signal_owned(&self.worker().unwrap(), false);
        session
    }
    pub fn worker(&self) -> Option<Value> {
        self.worker_for(MODEL)
    }
    pub fn worker_records(&self) -> Vec<Value> {
        let sessions = self.root().join("sessions");
        std::fs::read_dir(sessions)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|entry| {
                let bytes = std::fs::read(entry.path().join("worker.json")).ok()?;
                serde_json::from_slice(&bytes).ok()
            })
            .collect()
    }
    pub fn worker_for(&self, model: &str) -> Option<Value> {
        self.worker_records()
            .into_iter()
            .find(|w| w["model"] == model)
    }
    pub fn remember_worker(&mut self) {
        for worker in self.worker_records() {
            if !self
                .workers
                .iter()
                .any(|old| old["nonce"] == worker["nonce"])
            {
                self.workers.push(worker);
            }
        }
    }
    pub fn kill_worker(&mut self) {
        self.remember_worker();
        let worker = self.worker().expect("worker ownership record");
        signal_owned(&worker, true);
    }
    pub fn recover(&mut self) -> Value {
        let mut recovered = None;
        wait_until(Duration::from_secs(310), || {
            self.remember_worker();
            if !self.worker().is_some_and(|worker| healthy(&worker)) {
                return false;
            }
            let output = self.output(&["--json", "recover", SESSION]);
            let value: Value = serde_json::from_slice(&output.stdout).unwrap();
            if output.status.success() {
                recovered = Some(value);
                return true;
            }
            assert_eq!(
                output.status.code(),
                Some(6),
                "unexpected recovery failure: {value}"
            );
            false
        });
        recovered.unwrap()
    }
    pub fn remember_and_has_replacement(&mut self, original: &Value) -> bool {
        self.remember_worker();
        self.worker()
            .is_some_and(|worker| worker["pid"] != original["pid"])
    }
    pub fn restart_daemon(&mut self) {
        self.remember_worker();
        self.daemon.kill().unwrap();
        self.daemon.wait().unwrap();
        let socket = self.root().join("daemon.sock");
        if socket.exists() {
            std::fs::remove_file(&socket).unwrap();
        }
        self.daemon = Self::spawn_daemon(&self.home);
        wait_until(Duration::from_secs(10), || socket.exists());
    }
    pub fn db(&self) -> Connection {
        Connection::open(self.root().join("ralph.db")).unwrap()
    }
    pub fn turns(&self) -> Vec<(i64, String, String, i64)> {
        self.db()
            .prepare("SELECT seq,role,content,complete FROM token_turns ORDER BY seq")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))
            .unwrap()
            .map(Result::unwrap)
            .collect()
    }
    pub fn wait_loss(&self) {
        wait_until(Duration::from_secs(10), || {
            self.inspect()["session"]["state"] != "active"
        });
    }
    pub fn checkpoint(&self) -> Value {
        self.json(&["checkpoint", SESSION])
    }
    pub fn pause(&self) -> Value {
        self.json(&["pause", SESSION])
    }
    pub fn hibernate(&self) -> Value {
        self.json(&["hibernate", SESSION])
    }
    pub fn resume_json(&self, args: &[&str]) -> Output {
        self.command()
            .arg("--json")
            .arg("resume")
            .arg(SESSION)
            .args(args)
            .output()
            .unwrap()
    }
    pub fn resume(&self) -> Value {
        let out = self.resume_json(&[]);
        assert!(
            out.status.success(),
            "stdout: {}\nstderr: {}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        serde_json::from_slice(&out.stdout).unwrap()
    }
    pub fn kvcache_dir(&self, session_id: &str) -> PathBuf {
        self.root()
            .join("sessions")
            .join(session_id)
            .join("kvcache")
    }
    pub fn export(&self, output: &std::path::Path, with_accel: bool) -> Value {
        let output = output.to_string_lossy().into_owned();
        let mut args = vec!["export", SESSION, "--output", &output];
        if with_accel {
            args.push("--with-accel");
        }
        self.json(&args)
    }
    pub fn import(&self, path: &std::path::Path, name: &str) -> Value {
        let path = path.to_string_lossy().into_owned();
        self.json(&["import", &path, "--name", name])
    }
    pub fn resume_named(&self, name: &str) -> Value {
        let out = self
            .command()
            .args(["--json", "resume", name])
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "stdout: {}\nstderr: {}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        serde_json::from_slice(&out.stdout).unwrap()
    }
    pub fn query_named(&self, name: &str, prompt: &str) -> Value {
        self.json(&["query", name, prompt])
    }
    pub fn spawn_query(&self, prompt: &str) -> Child {
        self.command()
            .args(["query", SESSION, prompt])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap()
    }
}

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

pub fn signal_owned(worker: &Value, send_signal: bool) {
    let Some(group) = worker["group"].as_u64() else {
        return;
    };
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
    let mut verified = false;
    for entry in std::fs::read_dir("/proc").unwrap().flatten() {
        let Ok(stat) = std::fs::read_to_string(entry.path().join("stat")) else {
            continue;
        };
        let Some((_, rest)) = stat.rsplit_once(')') else {
            continue;
        };
        let fields: Vec<_> = rest.split_whitespace().collect();
        if fields.get(2).and_then(|s| s.parse::<u64>().ok()) != Some(group)
            || fields.first() == Some(&"Z")
        {
            continue;
        }
        let env = std::fs::read(entry.path().join("environ")).unwrap_or_default();
        assert!(
            env.split(|b| *b == 0)
                .any(|kv| kv == format!("RALPH_WORKER_OWNER={nonce}").as_bytes()),
            "unowned process group member"
        );
        verified = true;
    }
    if verified && send_signal {
        let _ = Command::new("kill")
            .args(["-KILL", "--", &format!("-{group}")])
            .status();
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        if std::thread::panicking() {
            if let Some(worker) = self.worker() {
                eprintln!("last owned worker: {worker}");
            }
            if let Ok(entries) = std::fs::read_dir(self.root().join("sessions")) {
                for entry in entries.flatten() {
                    if let Ok(log) = std::fs::read_to_string(entry.path().join("vllm.log")) {
                        eprintln!(
                            "{}",
                            log.lines()
                                .rev()
                                .take(20)
                                .collect::<Vec<_>>()
                                .into_iter()
                                .rev()
                                .collect::<Vec<_>>()
                                .join("\n")
                        );
                    }
                }
            }
        }
        self.remember_worker();
        // Stop this direct child before reading the final worker record to close the
        // startup/teardown race; no installed/user daemon is ever selected.
        let _ = self.daemon.kill();
        let _ = self.daemon.wait();
        self.remember_worker();
        for worker in &self.workers {
            signal_owned(worker, true);
        }
    }
}
