//! Read-only environment checks shared by `ralph doctor` and `run`'s pre-flight checks.
//! Never mutates anything on disk or in the environment.
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::Command;

use serde::Serialize;

use crate::engine::vllm::resolve_vllm_binary;

const MIN_FREE_DISK_BYTES: u64 = 1024 * 1024 * 1024;

#[derive(Debug, Serialize)]
pub struct CheckResult {
    pub name: String,
    pub status: String,
    pub detail: String,
}

fn pass(name: &str, detail: impl Into<String>) -> CheckResult {
    CheckResult {
        name: name.to_string(),
        status: "pass".to_string(),
        detail: detail.into(),
    }
}

fn warn(name: &str, detail: impl Into<String>) -> CheckResult {
    CheckResult {
        name: name.to_string(),
        status: "warn".to_string(),
        detail: detail.into(),
    }
}

fn fail(name: &str, detail: impl Into<String>) -> CheckResult {
    CheckResult {
        name: name.to_string(),
        status: "fail".to_string(),
        detail: detail.into(),
    }
}

pub fn run_checks(ralph_home: &Path) -> Vec<CheckResult> {
    vec![
        check_data_dir(ralph_home),
        check_sqlite(ralph_home),
        check_disk_space(ralph_home),
        check_vllm_binary(),
        check_gpu(),
        check_model_cache(),
    ]
}

/// Read-only: checks existence/writability via metadata rather than actually creating
/// anything, so `doctor` never leaves a data directory behind on a machine that hasn't
/// run `ralph run` yet.
fn check_data_dir(ralph_home: &Path) -> CheckResult {
    match std::fs::metadata(ralph_home) {
        Ok(meta) if meta.is_dir() => {
            if meta.permissions().mode() & 0o200 != 0 {
                pass("data directory", ralph_home.display().to_string())
            } else {
                fail(
                    "data directory",
                    format!("{} is not writable", ralph_home.display()),
                )
            }
        }
        Ok(_) => fail(
            "data directory",
            format!("{} exists but is not a directory", ralph_home.display()),
        ),
        Err(_) => pass(
            "data directory",
            format!(
                "{} does not exist yet; created on first run",
                ralph_home.display()
            ),
        ),
    }
}

/// Read-only: opening with `SQLITE_OPEN_READ_ONLY` never creates the file the way a
/// normal `Connection::open` would.
fn check_sqlite(ralph_home: &Path) -> CheckResult {
    let path = ralph_home.join("ralph.db");
    if !path.exists() {
        return pass(
            "sqlite",
            format!(
                "{} does not exist yet; created on first run",
                path.display()
            ),
        );
    }
    match rusqlite::Connection::open_with_flags(&path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY) {
        Ok(conn) => {
            match conn.pragma_query_value(None, "quick_check", |row| row.get::<_, String>(0)) {
                Ok(result) if result == "ok" => pass("sqlite", path.display().to_string()),
                Ok(result) => fail("sqlite", format!("quick_check reported: {result}")),
                Err(e) => fail("sqlite", e.to_string()),
            }
        }
        Err(e) => fail("sqlite", format!("cannot open {}: {e}", path.display())),
    }
}

/// Walks up to the nearest existing ancestor — `ralph_home` itself doesn't exist before
/// the first `ralph run`, and free space on the filesystem it *would* live on is still a
/// meaningful, non-mutating check.
fn nearest_existing_ancestor(path: &Path) -> &Path {
    let mut candidate = path;
    loop {
        if candidate.exists() {
            return candidate;
        }
        match candidate.parent() {
            Some(parent) => candidate = parent,
            None => return candidate,
        }
    }
}

fn check_disk_space(ralph_home: &Path) -> CheckResult {
    let probe = nearest_existing_ancestor(ralph_home);
    match fs2::available_space(probe) {
        Ok(bytes) if bytes >= MIN_FREE_DISK_BYTES => {
            pass("disk space", format!("{} MB free", bytes / 1024 / 1024))
        }
        Ok(bytes) => warn(
            "disk space",
            format!("only {} MB free", bytes / 1024 / 1024),
        ),
        Err(e) => warn("disk space", format!("{}: {e}", probe.display())),
    }
}

fn check_vllm_binary() -> CheckResult {
    let bin = resolve_vllm_binary();
    match Command::new(&bin).arg("--version").output() {
        Ok(output) if output.status.success() => pass(
            "vllm",
            String::from_utf8_lossy(&output.stdout).trim().to_string(),
        ),
        Ok(output) => fail(
            "vllm",
            format!("{} --version exited with {}", bin.display(), output.status),
        ),
        Err(e) => fail("vllm", format!("cannot run {}: {e}", bin.display())),
    }
}

/// Number of GPUs `nvidia-smi` reports, or `None` if it couldn't be run at all. Shared by
/// the `doctor` check and `run`'s pre-flight check so both agree on what "no GPU" means.
pub(crate) fn gpu_count() -> Option<usize> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter(|l| !l.trim().is_empty())
            .count(),
    )
}

fn check_gpu() -> CheckResult {
    match gpu_count() {
        Some(count) if count > 0 => pass("gpu", format!("{count} gpu(s) visible")),
        Some(_) => fail("gpu", "nvidia-smi reported no gpus"),
        None => fail("gpu", "cannot run nvidia-smi"),
    }
}

fn check_model_cache() -> CheckResult {
    let dir = std::env::var("HF_HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| dirs_home().join(".cache").join("huggingface"));
    if dir.exists() {
        pass("model cache", dir.display().to_string())
    } else {
        warn(
            "model cache",
            format!("{} does not exist yet", dir.display()),
        )
    }
}

fn dirs_home() -> std::path::PathBuf {
    std::env::var("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("/"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_dir_check_passes_without_creating_anything() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("ralph_home");
        let result = check_data_dir(&target);
        assert_eq!(result.status, "pass");
        assert!(
            !target.exists(),
            "doctor must not create the data directory"
        );
    }

    #[test]
    fn data_dir_check_passes_when_existing_and_writable() {
        let dir = tempfile::tempdir().unwrap();
        let result = check_data_dir(dir.path());
        assert_eq!(result.status, "pass");
    }

    #[test]
    fn sqlite_check_passes_without_creating_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let result = check_sqlite(dir.path());
        assert_eq!(result.status, "pass");
        assert!(
            !dir.path().join("ralph.db").exists(),
            "doctor must not create ralph.db"
        );
    }

    #[test]
    fn disk_space_check_passes_when_ralph_home_does_not_exist_yet() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("ralph_home").join("nested");
        let result = check_disk_space(&missing);
        assert_ne!(result.status, "fail", "detail: {}", result.detail);
        assert!(!result.detail.contains("No such file or directory"));
    }

    #[test]
    fn run_checks_returns_all_six() {
        let dir = tempfile::tempdir().unwrap();
        let results = run_checks(dir.path());
        assert_eq!(results.len(), 6);
    }
}
