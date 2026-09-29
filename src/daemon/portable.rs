//! `ralph export`/`ralph import` — see `crate::portable` for the `.ralph` archive
//! format itself; this module wires it to storage and the on-disk session layout.
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::{Daemon, now_rfc3339, to_session_info};
use crate::engine::Engine;
use crate::engine::vllm::kv_offload;
use crate::error::CliError;
use crate::ipc::SessionInfo;
use crate::portable::{self, ManifestInput, PortableError};
use crate::session;
use crate::state::SessionState;
use crate::storage::SessionRow;
use crate::storage::checkpoints::CheckpointRow;
use crate::storage::turn_export::TurnRow;

fn map_portable_error(e: PortableError) -> CliError {
    match e {
        PortableError::Corrupt(reason) => CliError::Corrupt(reason),
        PortableError::UnsupportedVersion { found, supported } => CliError::Usage(format!(
            "artifact format version {found} is not supported by this build (supports {supported})"
        )),
    }
}

/// Only the fingerprint travels — `kv_dir` is always a machine-local path, recomputed
/// fresh for whichever session ultimately imports it, never trusted from the archive.
#[derive(Serialize, Deserialize)]
struct ExportedCheckpoint {
    fingerprint_json: String,
}

impl<E: Engine + 'static> Daemon<E> {
    /// Assembles the `.ralph` archive bytes for an already-resolved, already-locked
    /// session — the one place archive bytes get built, shared by `export()` (writes
    /// them to a local file) and `handoff_cancellable()` (streams them over `ssh`).
    pub(super) fn build_export_archive(
        &self,
        row: &SessionRow,
        with_accel: bool,
    ) -> Result<Vec<u8>, CliError> {
        let turns = self.storage().export_turns(&row.id)?;
        let turns_json = serde_json::to_vec(&turns).map_err(|e| CliError::Other(e.into()))?;

        let checkpoint = if with_accel {
            self.storage().get_checkpoint(&row.id)?
        } else {
            None
        };
        // A checkpoint row can exist before its kvcache directory ever does: the very
        // first worker for a session (plain `ralph run`, no prior pause/resume) never
        // has the KV-offload connector attached — `kv_offload_for` only attaches it once
        // a checkpoint row already exists, so `ralph checkpoint` right after `run` points
        // at a directory vLLM never created. Falling back to logical-only here is the
        // same honest behavior `--with-accel` already gets against an empty directory
        // (see `Self::kv_dir_has_content`), not a raw I/O error.
        let checkpoint = checkpoint.filter(|c| Path::new(&c.kv_dir).is_dir());
        let checkpoint_json = checkpoint
            .as_ref()
            .map(|c| {
                serde_json::to_vec(&ExportedCheckpoint {
                    fingerprint_json: c.fingerprint_json.clone(),
                })
            })
            .transpose()
            .map_err(|e| CliError::Other(e.into()))?;
        let kvcache_dir = checkpoint
            .as_ref()
            .map(|_| session::kvcache_dir(&session::session_dir(&self.sessions_root, &row.id)));

        let fields = ManifestInput {
            id: row.id.clone(),
            name: row.name.clone(),
            model: row.model.clone(),
            model_revision: row.model_revision.clone(),
            tokenizer_revision: row.tokenizer_revision.clone(),
            engine: row.engine.clone(),
            engine_version: row.engine_version.clone(),
            created_at: row.created_at.clone(),
            exported_at: now_rfc3339(),
        };
        portable::build_archive(
            fields,
            &turns_json,
            checkpoint_json.as_deref(),
            kvcache_dir.as_deref(),
        )
        .map_err(|e| CliError::Other(e.into()))
    }

    pub async fn export(
        &self,
        identifier: &str,
        output_path: &Path,
        force: bool,
        with_accel: bool,
    ) -> Result<SessionInfo, CliError> {
        let row = self.storage().resolve(identifier)?;
        let _guard = self.locks.try_acquire(&row.id).ok_or_else(|| {
            CliError::InvalidState("operation already in progress for this session".into())
        })?;
        let row = self.storage().resolve(&row.id)?;

        if output_path.exists() && !force {
            return Err(CliError::Usage(format!(
                "{} already exists; use --force to overwrite",
                output_path.display()
            )));
        }

        let archive = self.build_export_archive(&row, with_accel)?;
        let dest_dir = output_path.parent().filter(|p| !p.as_os_str().is_empty());
        let dest_dir = dest_dir.unwrap_or_else(|| Path::new("."));
        if !kv_offload::has_room(dest_dir, archive.len() as u64) {
            return Err(CliError::Resource(
                "not enough disk space to write the export archive; session is unaffected".into(),
            ));
        }
        write_archive_atomically(output_path, &archive, force)?;
        Ok(to_session_info(&row))
    }

    pub async fn import(
        self: &Arc<Self>,
        archive_path: &Path,
        name: Option<String>,
    ) -> Result<SessionInfo, CliError> {
        let bytes = std::fs::read(archive_path).map_err(|e| CliError::Other(e.into()))?;

        let staging = self
            .sessions_root
            .join(format!("import-tmp-{}", session::new_session_id()));
        std::fs::create_dir_all(&staging).map_err(|e| CliError::Other(e.into()))?;
        let parsed = match portable::parse_archive(&bytes, &staging) {
            Ok(p) => p,
            Err(e) => {
                let _ = std::fs::remove_dir_all(&staging);
                return Err(map_portable_error(e));
            }
        };
        let cleanup = || {
            let _ = std::fs::remove_dir_all(&staging);
        };

        let name = name.unwrap_or_else(|| parsed.manifest.name.clone());
        if self.storage().name_taken(&name)? {
            cleanup();
            return Err(CliError::DuplicateName(name));
        }

        let turns: Vec<TurnRow> = match serde_json::from_slice(&parsed.turns_json) {
            Ok(t) => t,
            Err(e) => {
                cleanup();
                return Err(CliError::Corrupt(format!("malformed turn history: {e}")));
            }
        };
        if let Err(e) = crate::storage::turn_export::validate_turns(&turns) {
            cleanup();
            return Err(e.into());
        }
        let checkpoint: Option<ExportedCheckpoint> = match &parsed.checkpoint_json {
            Some(bytes) => match serde_json::from_slice(bytes) {
                Ok(c) => Some(c),
                Err(e) => {
                    cleanup();
                    return Err(CliError::Corrupt(format!(
                        "malformed checkpoint metadata: {e}"
                    )));
                }
            },
            None => None,
        };

        let id = session::new_session_id();
        let now = now_rfc3339();
        // `import_turns` runs before `insert` below (see comment), so its own
        // `updated_at`/`token_count` update would silently touch zero rows — the real
        // total is computed here instead and baked into the row `insert` writes.
        let token_count: i64 = turns.iter().map(|t| t.token_count).sum();
        let row = SessionRow {
            id: id.clone(),
            name,
            model: parsed.manifest.model.clone(),
            model_revision: parsed.manifest.model_revision.clone(),
            tokenizer_revision: parsed.manifest.tokenizer_revision.clone(),
            engine: parsed.manifest.engine.clone(),
            engine_version: parsed.manifest.engine_version.clone(),
            state: SessionState::Paused,
            pid: None,
            location: "local/gpu0".to_string(),
            token_count,
            created_at: parsed.manifest.created_at.clone(),
            updated_at: now.clone(),
        };

        // Everything below writes under the fresh id `id` before the session row itself
        // exists. No read path (`list`/`resolve`/`ps`) can ever see a session that has no
        // row in `sessions`, so a failure here leaves orphaned-but-invisible data rather
        // than a partially-visible session — `insert`, last, is what actually publishes
        // it (RALPH_SPEC.md §16.8 "fail without partial visible session"). There is
        // deliberately no delete-on-failure path: this codebase never deletes durable
        // session state (see `enforce_kv_quota`'s equivalent rule).
        if let Err(e) = session::create_session_dir(&self.sessions_root, &row) {
            cleanup();
            return Err(CliError::Other(e.into()));
        }
        if let Err(e) = self.storage().import_turns(&id, &turns, &now) {
            cleanup();
            return Err(e.into());
        }
        if let (Some(checkpoint), Some(kvcache_staged)) = (&checkpoint, &parsed.kvcache_dir) {
            let dest = session::kvcache_dir(&session::session_dir(&self.sessions_root, &id));
            if let Some(parent) = dest.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Err(e) = std::fs::rename(kvcache_staged, &dest) {
                cleanup();
                return Err(CliError::Other(e.into()));
            }
            let checkpoint_row = CheckpointRow {
                session_id: id.clone(),
                fingerprint_json: checkpoint.fingerprint_json.clone(),
                kv_dir: dest.to_string_lossy().into_owned(),
                engine_id: kv_offload::engine_id_for_session(&id),
                created_at: now,
            };
            if let Err(e) = self.storage().upsert_checkpoint(&checkpoint_row) {
                cleanup();
                return Err(e.into());
            }
        }

        if let Err(e) = self.storage().insert(&row) {
            cleanup();
            return Err(e.into());
        }
        cleanup();
        Ok(to_session_info(&self.storage().resolve(&id)?))
    }
}

fn write_archive_atomically(output_path: &Path, bytes: &[u8], force: bool) -> Result<(), CliError> {
    let tmp_path: PathBuf = {
        let mut name = output_path.as_os_str().to_owned();
        name.push(".tmp");
        name.into()
    };
    let _ = std::fs::remove_file(&tmp_path); // opportunistic cleanup of a stale temp file

    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(&tmp_path)
        .map_err(|e| CliError::Other(e.into()))?;
    file.write_all(bytes)
        .map_err(|e| CliError::Other(e.into()))?;
    file.sync_all().map_err(|e| CliError::Other(e.into()))?;
    drop(file);

    if output_path.exists() && !force {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(CliError::Usage(format!(
            "{} already exists; use --force to overwrite",
            output_path.display()
        )));
    }
    std::fs::rename(&tmp_path, output_path).map_err(|e| CliError::Other(e.into()))?;
    Ok(())
}
