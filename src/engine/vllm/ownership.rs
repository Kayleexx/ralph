//! Private process identity; a recycled PID never authorizes a signal.
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::process::Command;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub(super) struct Ownership {
    pub pid: u32,
    pub group: u32,
    pub start: String,
    pub boot: String,
    pub nonce: String,
    pub endpoint: String,
    pub model: String,
    pub revision: Option<String>,
    #[serde(default)]
    pub profile: Option<crate::engine::profiles::WorkerProfile>,
}

/// A process that exits mid-scan can make `/proc/<pid>/*` reads fail with ENOENT
/// (the usual case) or, during the kernel's brief teardown window, ESRCH — both mean
/// "this pid is gone," not a real I/O failure `owned_members`'s scan should abort on.
/// Reproduced live: a real "No such process" during `owned_members` aborted `cleanup`
/// entirely, leaking the actual worker process instead of just skipping the unrelated
/// vanished pid that triggered it.
fn process_vanished(error: &io::Error) -> bool {
    matches!(error.raw_os_error(), Some(2) | Some(3))
}

/// `process_vanished`, plus permission-denied — `owned_members` now scans by marker
/// across every pid on the system, so most candidates belong to another user and their
/// `environ`/`stat` are unreadable by design, not evidence of anything worth erroring on.
fn not_ours(error: &io::Error) -> bool {
    process_vanished(error) || error.kind() == io::ErrorKind::PermissionDenied
}

fn identity(pid: u32) -> io::Result<(u32, String)> {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat"))?;
    let fields: Vec<_> = stat
        .rsplit_once(')')
        .ok_or_else(|| io::Error::other("invalid proc stat"))?
        .1
        .split_whitespace()
        .collect();
    let group = fields
        .get(2)
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| io::Error::other("invalid process group"))?;
    let start = fields
        .get(19)
        .ok_or_else(|| io::Error::other("missing process start time"))?
        .to_string();
    Ok((group, start))
}

impl Ownership {
    pub fn new(
        pid: u32,
        nonce: String,
        endpoint: String,
        model: String,
        revision: Option<String>,
        profile: Option<crate::engine::profiles::WorkerProfile>,
    ) -> io::Result<Self> {
        let (group, start) = identity(pid)?;
        if group != pid {
            return Err(io::Error::other("worker has no private process group"));
        }
        Ok(Self {
            pid,
            group,
            start,
            boot: fs::read_to_string("/proc/sys/kernel/random/boot_id")?,
            nonce,
            endpoint,
            model,
            revision,
            profile,
        })
    }

    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)?;
        file.write_all(&serde_json::to_vec(self)?)?;
        file.sync_all()?;
        if let Some(parent) = path.parent() {
            fs::File::open(parent)?.sync_all()?;
        }
        Ok(())
    }

    /// Finds every process carrying this worker's exact `RALPH_WORKER_OWNER` marker —
    /// scans all of `/proc` by marker presence, not by process-group membership.
    /// Reproduced live: vLLM's own EngineCore child does not reliably stay in the
    /// process group its `vllm serve` parent was launched into (some vLLM startup
    /// paths detach it into a session/group of its own), so a group-only scan can
    /// silently miss a real, still-running descendant and report "nothing to kill"
    /// while it keeps holding VRAM — `process_group(0)` at spawn time still gives the
    /// leader a predictable group for `kill_unowned_spawn`'s early-failure fallback,
    /// but ongoing ownership tracking must not depend on every descendant staying in it.
    /// The nonce is a fresh random ULID per spawn, so a marker match alone is exactly
    /// as safe as the old group+marker check (`reused_identity_cannot_authorize_a_signal`
    /// covers the same non-goal: never signal a pid this worker doesn't actually own).
    fn owned_members(&self) -> io::Result<Vec<u32>> {
        if fs::read_to_string("/proc/sys/kernel/random/boot_id")? != self.boot {
            return Ok(vec![]);
        }
        if let Ok((group, start)) = identity(self.pid)
            && (group != self.group || start != self.start)
        {
            return Err(io::Error::other("worker PID was reused; refusing cleanup"));
        }
        let marker = format!("RALPH_WORKER_OWNER={}", self.nonce);
        let mut members = Vec::new();
        for entry in fs::read_dir("/proc")? {
            let entry = entry?;
            let Some(pid) = entry
                .file_name()
                .to_str()
                .and_then(|s| s.parse::<u32>().ok())
            else {
                continue;
            };
            let stat = match fs::read_to_string(entry.path().join("stat")) {
                Ok(stat) => stat,
                Err(error) if not_ours(&error) => continue,
                Err(error) => return Err(error),
            };
            if stat
                .rsplit_once(')')
                .is_some_and(|(_, rest)| rest.trim_start().starts_with('Z'))
            {
                continue;
            }
            // Scanning by marker (rather than pre-filtering by group first) means most
            // candidates now belong to other users — `environ` is only readable by the
            // owning uid, so `PermissionDenied` here just means "not ours," not an error.
            let env = match fs::read(entry.path().join("environ")) {
                Ok(env) => env,
                Err(error) if not_ours(&error) => continue,
                Err(error) => return Err(error),
            };
            if env.split(|b| *b == 0).any(|kv| kv == marker.as_bytes()) {
                members.push(pid);
            }
        }
        Ok(members)
    }

    pub fn cleanup(&self) -> io::Result<()> {
        // A single /proc scan can transiently miss a real, still-running member (the
        // same class of kernel-timing race `admission.rs`'s VRAM re-check already
        // tolerates) — recheck briefly before trusting one snapshot that says there's
        // nothing to kill. Reproduced live: a genuinely running worker group was
        // skipped here and leaked past this point.
        let mut members = self.owned_members()?;
        for _ in 0..5 {
            if !members.is_empty() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
            members = self.owned_members()?;
        }
        if members.is_empty() {
            return Ok(());
        }
        // Kill each verified member by its own pid rather than one blanket `-<group>`
        // signal — `owned_members` no longer guarantees every member shares `self.group`
        // (see its doc comment), so a group-wide kill alone could again miss a
        // descendant that detached into its own group.
        for pid in &members {
            let _ = Command::new("kill")
                .args(["-KILL", "--", &pid.to_string()])
                .status();
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
        while !self.owned_members()?.is_empty() {
            if std::time::Instant::now() >= deadline {
                return Err(io::Error::other("owned worker group did not exit"));
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        Ok(())
    }
}

pub(crate) fn reconcile(
    sessions: &Path,
    valid: impl Fn(&str, &str, Option<&str>, Option<crate::engine::profiles::WorkerProfile>) -> bool,
) -> io::Result<()> {
    if !sessions.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(sessions)? {
        let path = entry?.path().join("worker.json");
        if !path.exists() {
            continue;
        }
        let owned: Ownership = serde_json::from_slice(&fs::read(&path)?)?;
        let id = path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .ok_or_else(|| io::Error::other("invalid ownership directory"))?;
        if !valid(id, &owned.model, owned.revision.as_deref(), owned.profile)
            || !owned.endpoint.starts_with("http://127.0.0.1:")
        {
            return Err(io::Error::other(
                "worker record differs from durable model identity",
            ));
        }
        owned.cleanup()?;
        fs::remove_file(path)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reused_identity_cannot_authorize_a_signal() {
        let pid = std::process::id();
        let (group, _) = identity(pid).unwrap();
        let record = Ownership {
            pid,
            group,
            start: "wrong".into(),
            boot: fs::read_to_string("/proc/sys/kernel/random/boot_id").unwrap(),
            nonce: "test".into(),
            endpoint: "local".into(),
            model: "test".into(),
            revision: None,
            profile: None,
        };
        assert!(record.cleanup().is_err());
    }

    /// Regression: a real "No such process" (ESRCH) hit during `owned_members`'s /proc
    /// scan of an unrelated, concurrently-exiting pid used to propagate as a hard error
    /// and abort `cleanup` entirely, leaking the actual worker. Both ESRCH and the more
    /// common ENOENT must be treated as "this pid is already gone."
    #[test]
    fn process_vanished_recognizes_enoent_and_esrch_but_not_other_errors() {
        assert!(process_vanished(&io::Error::from_raw_os_error(2)));
        assert!(process_vanished(&io::Error::from_raw_os_error(3)));
        assert!(!process_vanished(&io::Error::from_raw_os_error(13)));
    }
}

#[cfg(test)]
mod process_tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::os::unix::process::CommandExt;

    #[test]
    fn dead_leader_orphan_is_cleaned_only_with_verified_marker_members() {
        let nonce = ulid::Ulid::new().to_string();
        let mut child = Command::new("sh")
            .args(["-c", "sleep 30 & wait"])
            .env("RALPH_WORKER_OWNER", &nonce)
            .process_group(0)
            .spawn()
            .unwrap();
        let record = Ownership::new(
            child.id(),
            nonce,
            "http://127.0.0.1:9000".into(),
            "test".into(),
            Some("rev".into()),
            None,
        )
        .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("worker.json");
        record.save(&path).unwrap();
        assert_eq!(
            fs::metadata(path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while record.owned_members().unwrap().len() < 2 {
            assert!(std::time::Instant::now() < deadline, "child did not start");
            std::thread::yield_now();
        }
        child.kill().unwrap();
        child.wait().unwrap();
        record.cleanup().unwrap();
        assert!(record.owned_members().unwrap().is_empty());
    }

    /// Regression: reproduced live with real vLLM — a descendant that detaches into a
    /// *different* process group than the one recorded at spawn time (here forced with
    /// `setsid`, matching what some vLLM startup paths do to their EngineCore child)
    /// used to be invisible to a group-scoped scan, so `cleanup` believed there was
    /// nothing left to kill while the real process kept running and holding VRAM.
    #[test]
    fn a_descendant_that_escapes_into_its_own_process_group_is_still_found_and_killed() {
        let nonce = ulid::Ulid::new().to_string();
        let mut child = Command::new("sh")
            .args(["-c", "setsid sleep 30 & wait"])
            .env("RALPH_WORKER_OWNER", &nonce)
            .process_group(0)
            .spawn()
            .unwrap();
        let record = Ownership::new(
            child.id(),
            nonce,
            "http://127.0.0.1:9001".into(),
            "test".into(),
            Some("rev".into()),
            None,
        )
        .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while record.owned_members().unwrap().len() < 2 {
            assert!(std::time::Instant::now() < deadline, "child did not start");
            std::thread::yield_now();
        }
        record.cleanup().unwrap();
        assert!(record.owned_members().unwrap().is_empty());
        let _ = child.wait();
    }
}
