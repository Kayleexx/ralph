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

    fn owned_members(&self) -> io::Result<Vec<u32>> {
        if fs::read_to_string("/proc/sys/kernel/random/boot_id")? != self.boot {
            return Ok(vec![]);
        }
        if let Ok((group, start)) = identity(self.pid)
            && (group != self.group || start != self.start)
        {
            return Err(io::Error::other("worker PID was reused; refusing cleanup"));
        }
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
            if !matches!(identity(pid), Ok((group, _)) if group == self.group) {
                continue;
            }
            let stat = match fs::read_to_string(entry.path().join("stat")) {
                Ok(stat) => stat,
                Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
                Err(error) => return Err(error),
            };
            if stat
                .rsplit_once(')')
                .is_some_and(|(_, rest)| rest.trim_start().starts_with('Z'))
            {
                continue;
            }
            let env = match fs::read(entry.path().join("environ")) {
                Ok(env) => env,
                Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
                Err(error) => return Err(error),
            };
            let marker = format!("RALPH_WORKER_OWNER={}", self.nonce);
            if !env.split(|b| *b == 0).any(|kv| kv == marker.as_bytes()) {
                let current = fs::read_to_string(entry.path().join("stat"));
                if current.as_ref().is_ok_and(|stat| {
                    stat.rsplit_once(')')
                        .is_some_and(|(_, rest)| rest.trim_start().starts_with('Z'))
                }) || current
                    .as_ref()
                    .is_err_and(|error| error.kind() == io::ErrorKind::NotFound)
                {
                    continue;
                }
                return Err(io::Error::other(
                    "process group contains an unverified member",
                ));
            }
            members.push(pid);
        }
        Ok(members)
    }

    pub fn cleanup(&self) -> io::Result<()> {
        if self.owned_members()?.is_empty() {
            return Ok(());
        }
        let status = Command::new("kill")
            .args(["-KILL", "--", &format!("-{}", self.group)])
            .status()?;
        if !status.success() && !self.owned_members()?.is_empty() {
            return Err(io::Error::other("cannot stop owned worker group"));
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
    valid: impl Fn(&str, &str, Option<&str>) -> bool,
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
        if !valid(id, &owned.model, owned.revision.as_deref())
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
        };
        assert!(record.cleanup().is_err());
    }
}

#[cfg(test)]
mod process_tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::os::unix::process::CommandExt;

    #[test]
    fn dead_leader_orphan_is_cleaned_only_with_verified_group_members() {
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
}
