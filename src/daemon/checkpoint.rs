//! `ralph checkpoint`, and the compatibility-fingerprint/disk-quota bookkeeping that
//! `pause`/`hibernate`/`resume` (`lifecycle.rs`) share around it. The KV bytes themselves
//! are already written continuously and atomically by vLLM's own offload connector during
//! ordinary generation (see `engine::vllm::kv_offload`) — ralph only ever writes a small,
//! already-atomic (SQLite WAL) pointer+fingerprint row, never the KV content itself.
use std::sync::Arc;

use super::{Daemon, now_rfc3339, to_session_info};
use crate::checkpoint::Fingerprint;
use crate::engine::vllm::kv_offload;
use crate::engine::{Engine, KvOffloadSpec};
use crate::error::CliError;
use crate::ipc::SessionInfo;
use crate::session;
use crate::state::SessionState;
use crate::storage::SessionRow;
use crate::storage::checkpoints::CheckpointRow;

/// Coarse total across every session's KV directory combined — good enough to bound
/// disk growth from accumulating checkpoints without per-model/per-user accounting no
/// part of this codebase has today.
const KV_QUOTA_BYTES: u64 = 4 * 1024 * 1024 * 1024;

pub(super) fn dir_size(dir: &std::path::Path) -> u64 {
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

impl<E: Engine + 'static> Daemon<E> {
    pub(super) fn current_fingerprint(&self, row: &SessionRow) -> Fingerprint {
        Fingerprint {
            model_revision: row.model_revision.clone(),
            tokenizer_revision: row.tokenizer_revision.clone(),
            engine: row.engine.clone(),
            engine_version: row.engine_version.clone(),
            gpu_name: (self.gpu_memory)(crate::session::gpu_index_of(&row.location))
                .ok()
                .map(|g| g.name),
            tensor_parallel: 1,
            adapter: None,
        }
    }

    /// Refuses to point a checkpoint at a filesystem that's already too full for vLLM's
    /// offload connector to write into — checked against the session directory itself,
    /// not `kv_offload::has_room`'s target (which creates
    /// the kvcache dir as a side effect and would prematurely materialize it before
    /// vLLM's connector ever attaches, breaking `export`'s "dir may not exist yet" check).
    pub(super) fn has_checkpoint_room(&self, row: &SessionRow) -> bool {
        if *self
            .force_disk_full
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
        {
            return false;
        }
        let dir = session::session_dir(&self.sessions_root, &row.id);
        fs2::available_space(&dir).is_ok_and(|free| free >= kv_offload::DEFAULT_CPU_BYTES)
    }

    pub(super) fn checkpoint_row_for(&self, row: &SessionRow) -> CheckpointRow {
        let dir = session::kvcache_dir(&session::session_dir(&self.sessions_root, &row.id));
        CheckpointRow {
            session_id: row.id.clone(),
            fingerprint_json: self.current_fingerprint(row).to_json(),
            kv_dir: dir.to_string_lossy().into_owned(),
            engine_id: kv_offload::engine_id_for_session(&row.id),
            created_at: now_rfc3339(),
        }
    }

    /// Why fast (native KV) resume is or isn't available right now.
    pub(super) fn restore_readiness(
        &self,
        row: &SessionRow,
    ) -> crate::checkpoint::RestoreReadiness {
        use crate::checkpoint::{RestoreReadiness, classify};
        let Ok(Some(checkpoint)) = self.storage().get_checkpoint(&row.id) else {
            return RestoreReadiness::NoNativeState;
        };
        let Some(stored) = Fingerprint::from_json(&checkpoint.fingerprint_json) else {
            return RestoreReadiness::NoNativeState;
        };
        let dir = std::path::PathBuf::from(&checkpoint.kv_dir);
        let room_ok = kv_offload::has_room(&dir, kv_offload::DEFAULT_CPU_BYTES);
        let content_intact = kv_offload::content_signature_intact(&dir);
        classify(
            &self.current_fingerprint(row),
            Some(&stored),
            content_intact,
            room_ok,
        )
    }

    /// (field, current, stored) triples that differ, for `--verbose` inspect — never
    /// turn/prompt content, only fingerprint fields.
    pub(super) fn fingerprint_diff(
        &self,
        row: &SessionRow,
    ) -> Option<Vec<(String, String, String)>> {
        let checkpoint = self.storage().get_checkpoint(&row.id).ok().flatten()?;
        let stored = Fingerprint::from_json(&checkpoint.fingerprint_json)?;
        let current = self.current_fingerprint(row);
        let mut diff = Vec::new();
        macro_rules! field {
            ($name:literal, $accessor:ident) => {
                if current.$accessor != stored.$accessor {
                    diff.push((
                        $name.to_string(),
                        format!("{:?}", current.$accessor),
                        format!("{:?}", stored.$accessor),
                    ));
                }
            };
        }
        field!("model_revision", model_revision);
        field!("tokenizer_revision", tokenizer_revision);
        field!("engine", engine);
        field!("engine_version", engine_version);
        field!("gpu_name", gpu_name);
        Some(diff)
    }

    /// `None` unless `restore_readiness` reports the checkpoint is actually usable.
    pub(super) fn kv_offload_for(&self, row: &SessionRow) -> Option<KvOffloadSpec> {
        if self.restore_readiness(row) != crate::checkpoint::RestoreReadiness::NativeAvailable {
            return None;
        }
        let checkpoint = self.storage().get_checkpoint(&row.id).ok().flatten()?;
        Some(KvOffloadSpec {
            root_dir: std::path::PathBuf::from(&checkpoint.kv_dir),
            cpu_bytes: kv_offload::DEFAULT_CPU_BYTES,
            engine_id: checkpoint.engine_id,
        })
    }

    /// Deletes the oldest *other* sessions' KV directories and checkpoint rows until the
    /// combined total is back under budget. Only ever touches `checkpoints`/on-disk KV
    /// directories — never the durable `sessions`/`token_turns` tables, so eviction can
    /// never make a session unrecoverable.
    pub(super) fn enforce_kv_quota(&self, keep_session_id: &str) {
        self.enforce_kv_quota_within(keep_session_id, KV_QUOTA_BYTES);
    }

    /// Split out so tests can exercise real eviction against a small quota instead of
    /// writing multiple gigabytes of dummy data to disk.
    pub(super) fn enforce_kv_quota_within(&self, keep_session_id: &str, quota_bytes: u64) {
        let Ok(mut checkpoints) = self.storage().list_checkpoints() else {
            return;
        };
        let mut total: u64 = checkpoints
            .iter()
            .map(|c| dir_size(std::path::Path::new(&c.kv_dir)))
            .sum();
        checkpoints.retain(|c| c.session_id != keep_session_id);
        for checkpoint in checkpoints {
            if total <= quota_bytes {
                break;
            }
            let dir = std::path::Path::new(&checkpoint.kv_dir);
            let freed = dir_size(dir);
            let _ = std::fs::remove_dir_all(dir);
            let _ = self.storage().delete_checkpoint(&checkpoint.session_id);
            total = total.saturating_sub(freed);
        }
    }

    /// Graceful SIGTERM preemption: one best-effort checkpoint pass over every currently
    /// `Active` session before this process exits. Durable token history is already
    /// flushed continuously, so this only improves
    /// the odds of a *fast* resume elsewhere/after restart — the actual recovery path
    /// is the existing `reconcile_on_startup` → `ralph recover`/`ralph resume` story,
    /// unchanged. Never blocks on a fresh worker admission or state transition.
    pub(crate) fn checkpoint_active_sessions_best_effort(&self) {
        let Ok(rows) = self.storage().list() else {
            return;
        };
        for row in rows {
            if row.state != SessionState::Active {
                continue;
            }
            if let Err(error) = self
                .storage()
                .upsert_checkpoint(&self.checkpoint_row_for(&row))
            {
                eprintln!("SIGTERM checkpoint failed for {}: {error}", row.name);
            }
        }
    }

    pub async fn checkpoint(self: &Arc<Self>, identifier: &str) -> Result<SessionInfo, CliError> {
        let row = self.storage().resolve(identifier)?;
        let _guard = self.locks.try_acquire(&row.id).ok_or_else(|| {
            CliError::InvalidState("operation already in progress for this session".into())
        })?;
        let row = self.storage().resolve(&row.id)?;
        if row.state != SessionState::Active {
            return Err(CliError::InvalidState(format!(
                "session is {}, checkpoint requires an active session",
                row.state
            )));
        }
        self.enforce_kv_quota(&row.id);
        if !self.has_checkpoint_room(&row) {
            return Err(CliError::Resource(
                "not enough disk space to checkpoint; session is unaffected".into(),
            ));
        }
        self.storage()
            .upsert_checkpoint(&self.checkpoint_row_for(&row))?;
        Ok(to_session_info(&row))
    }
}
