//! Builds vLLM's `--kv-transfer-config` for the filesystem-backed KV offload tier
//! (`OffloadingConnector` + `TieringOffloadingSpec`), confirmed working end-to-end against
//! this project's installed vLLM 0.30.0 (real cross-process KV reuse after SIGKILL,
//! verified via `/metrics`). Two gaps that vLLM itself doesn't cover, found the same way:
//! its own directory hash omits model/tokenizer revision and engine version (never trust
//! it as a compatibility fingerprint — see `crate::checkpoint`), and it doesn't unlink its
//! `/dev/shm/vllm_offload_<engine_id>.mmap` on a killed worker (`cleanup_shm` below closes
//! that leak using a pinned, not vLLM-randomized, `engine_id`).
use std::path::{Path, PathBuf};

/// Small enough to be a reasonable default on a 16 GB workstation RAM budget; real budgets
/// belong in a future per-model profile, not hardcoded further than this.
pub const DEFAULT_CPU_BYTES: u64 = 256 * 1024 * 1024;

pub fn engine_id_for_session(session_id: &str) -> String {
    format!("ralph-{session_id}")
}

pub(super) fn kv_transfer_config_json(root_dir: &Path, cpu_bytes: u64, engine_id: &str) -> String {
    serde_json::json!({
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "engine_id": engine_id,
        "kv_connector_extra_config": {
            "spec_name": "TieringOffloadingSpec",
            "cpu_bytes_to_use": cpu_bytes,
            "secondary_tiers": [{
                "type": "fs",
                "root_dir": root_dir.to_string_lossy(),
            }],
        },
    })
    .to_string()
}

fn shm_path_for(engine_id: &str) -> PathBuf {
    PathBuf::from("/dev/shm").join(format!("vllm_offload_{engine_id}.mmap"))
}

/// Best-effort: called after a worker exit is already confirmed, so a missing file (clean
/// shutdown already unlinked it) or a permission error are both fine to ignore here.
pub fn cleanup_shm(engine_id: &str) {
    let _ = std::fs::remove_file(shm_path_for(engine_id));
}

/// Coarse floor, not an exact accounting of what the fs tier will eventually use — good
/// enough to refuse the connector outright on a visibly full disk (§16.4 "KV checkpoint
/// larger than available disk") without pretending to predict real growth.
pub fn has_room(dir: &Path, needed_bytes: u64) -> bool {
    std::fs::create_dir_all(dir).is_ok()
        && fs2::available_space(dir).is_ok_and(|free| free >= needed_bytes)
}

const SIGNATURE_FILE: &str = ".ralph_kv_signature";

/// (total bytes, newest mtime, file count) across every file anywhere under `dir`
/// (recursive — vLLM's own `TieringOffloadingSpec` nests block files several
/// directories deep, e.g. `<hash>_r0/32d/78_g0/*.bin`), except the signature file
/// itself. A coarse fingerprint, not a cryptographic hash: good enough to detect the
/// realistic corruption shapes (truncation, a swapped-in file, a partial write) §16.15
/// "corrupt KV checkpoint" cares about, without reading gigabytes of KV content on
/// every checkpoint/resume.
fn content_fingerprint(dir: &Path) -> Option<(u64, u64, u64)> {
    fn walk(dir: &Path, total: &mut u64, newest: &mut u64, count: &mut u64) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            if entry.file_name() == SIGNATURE_FILE {
                continue;
            }
            let path = entry.path();
            let Ok(meta) = entry.metadata() else { continue };
            if meta.is_dir() {
                walk(&path, total, newest, count);
                continue;
            }
            if !meta.is_file() {
                continue;
            }
            *total += meta.len();
            let mtime = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            *newest = (*newest).max(mtime);
            *count += 1;
        }
    }
    let (mut total, mut newest, mut count) = (0u64, 0u64, 0u64);
    walk(dir, &mut total, &mut newest, &mut count);
    if count == 0 {
        return None;
    }
    Some((total, newest, count))
}

/// Called once the worker that owned this directory has actually stopped (content is
/// only stable once nothing is still writing to it) — never from `ralph checkpoint`
/// alone, whose worker stays running.
pub fn record_content_signature(dir: &Path) {
    if let Some((total, newest, count)) = content_fingerprint(dir) {
        let _ = std::fs::write(
            dir.join(SIGNATURE_FILE),
            format!("{total}:{newest}:{count}"),
        );
    }
}

/// `true` unless a signature was previously recorded and the directory's content no
/// longer matches it — the honest way to tell "never populated yet" (nothing recorded,
/// not a contradiction) apart from "populated, then something changed it outside ralph's
/// own write path" (recorded, and now different — never trust it).
pub fn content_signature_intact(dir: &Path) -> bool {
    let Ok(recorded) = std::fs::read_to_string(dir.join(SIGNATURE_FILE)) else {
        return true;
    };
    match content_fingerprint(dir) {
        Some((total, newest, count)) => recorded == format!("{total}:{newest}:{count}"),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_json_carries_engine_id_and_root_dir() {
        let json = kv_transfer_config_json(Path::new("/tmp/kv"), 123, "ralph-abc");
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["engine_id"], "ralph-abc");
        assert_eq!(value["kv_connector"], "OffloadingConnector");
        assert_eq!(
            value["kv_connector_extra_config"]["spec_name"],
            "TieringOffloadingSpec"
        );
        assert_eq!(
            value["kv_connector_extra_config"]["secondary_tiers"][0]["root_dir"],
            "/tmp/kv"
        );
    }

    #[test]
    fn engine_id_is_stable_per_session() {
        assert_eq!(
            engine_id_for_session("01AAA"),
            engine_id_for_session("01AAA")
        );
        assert_ne!(
            engine_id_for_session("01AAA"),
            engine_id_for_session("01BBB")
        );
    }

    #[test]
    fn has_room_is_false_against_an_absurd_requirement() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!has_room(dir.path(), u64::MAX));
        assert!(has_room(dir.path(), 1));
    }

    #[test]
    fn cleanup_shm_on_a_missing_file_does_not_panic() {
        cleanup_shm("no-such-engine-id-ever");
    }
}
