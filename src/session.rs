//! Friendly session names, on-disk session directories, and the manifest file.
use std::fs;
use std::io;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use rand::Rng;
use serde::Serialize;
use ulid::Ulid;

use crate::storage::SessionRow;

const ADJECTIVES: &[&str] = &[
    "quiet", "brave", "calm", "swift", "bright", "gentle", "bold", "lucky", "steady", "keen",
    "wild", "tidy", "eager", "sharp", "cool", "warm", "lively", "silent", "rapid", "sunny",
];

const NOUNS: &[&str] = &[
    "otter", "falcon", "badger", "heron", "lynx", "sparrow", "beetle", "marten", "willow", "cedar",
    "comet", "harbor", "meadow", "ridge", "delta", "quartz", "ember", "pebble", "raven", "canyon",
];

pub fn generate_name() -> String {
    let mut rng = rand::thread_rng();
    let adjective = ADJECTIVES[rng.gen_range(0..ADJECTIVES.len())];
    let noun = NOUNS[rng.gen_range(0..NOUNS.len())];
    format!("{adjective}-{noun}")
}

pub fn new_session_id() -> String {
    Ulid::new().to_string()
}

#[derive(Debug, Serialize)]
pub struct Manifest {
    pub id: String,
    pub name: String,
    pub model: String,
    pub model_revision: Option<String>,
    pub tokenizer_revision: Option<String>,
    pub engine: String,
    pub engine_version: Option<String>,
    pub state: String,
    pub location: String,
    pub token_count: i64,
    pub created_at: String,
    pub updated_at: String,
}

impl From<&SessionRow> for Manifest {
    fn from(row: &SessionRow) -> Self {
        Manifest {
            id: row.id.clone(),
            name: row.name.clone(),
            model: row.model.clone(),
            model_revision: row.model_revision.clone(),
            tokenizer_revision: row.tokenizer_revision.clone(),
            engine: row.engine.clone(),
            engine_version: row.engine_version.clone(),
            state: row.state.to_string(),
            location: row.location.clone(),
            token_count: row.token_count,
            created_at: row.created_at.clone(),
            updated_at: row.updated_at.clone(),
        }
    }
}

pub fn session_dir(sessions_root: &Path, id: &str) -> PathBuf {
    sessions_root.join(id)
}

/// Creates the session directory (mode 0700) and writes manifest.json (mode 0600) inside
/// it. Session artifacts are private by default since prompts may end up in vllm.log.
pub fn create_session_dir(sessions_root: &Path, row: &SessionRow) -> io::Result<PathBuf> {
    let dir = session_dir(sessions_root, &row.id);
    fs::create_dir_all(&dir)?;
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o700))?;
    write_manifest(&dir, row)?;
    Ok(dir)
}

pub fn write_manifest(dir: &Path, row: &SessionRow) -> io::Result<()> {
    let manifest = Manifest::from(row);
    let path = dir.join("manifest.json");
    let contents = serde_json::to_string_pretty(&manifest)?;
    fs::write(&path, contents)?;
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600))?;
    Ok(())
}

pub fn vllm_log_path(dir: &Path) -> PathBuf {
    dir.join("vllm.log")
}

/// Per-session, never shared across sessions — this is what makes cross-session KV
/// contamination structurally impossible regardless of vLLM's own directory hashing.
pub fn kvcache_dir(dir: &Path) -> PathBuf {
    dir.join("kvcache")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::SessionState;
    use tempfile::tempdir;

    #[test]
    fn generated_names_are_well_formed() {
        for _ in 0..50 {
            let name = generate_name();
            assert!(name.contains('-'));
            assert!(!name.is_empty());
        }
    }

    #[test]
    fn session_ids_are_unique() {
        let a = new_session_id();
        let b = new_session_id();
        assert_ne!(a, b);
    }

    fn sample_row(id: &str) -> SessionRow {
        SessionRow {
            id: id.to_string(),
            name: "demo".to_string(),
            model: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
            model_revision: Some("abc123".to_string()),
            tokenizer_revision: None,
            engine: "vllm".to_string(),
            engine_version: None,
            state: SessionState::Created,
            pid: None,
            location: "local/gpu0".to_string(),
            token_count: 0,
            created_at: "2026-01-01T00:00:00Z".to_string(),
            updated_at: "2026-01-01T00:00:00Z".to_string(),
        }
    }

    #[test]
    fn creates_private_session_dir_and_manifest() {
        let root = tempdir().unwrap();
        let row = sample_row("01AAA");
        let dir = create_session_dir(root.path(), &row).unwrap();

        let dir_mode = fs::metadata(&dir).unwrap().permissions().mode() & 0o777;
        assert_eq!(dir_mode, 0o700);

        let manifest_path = dir.join("manifest.json");
        let manifest_mode = fs::metadata(&manifest_path).unwrap().permissions().mode() & 0o777;
        assert_eq!(manifest_mode, 0o600);

        let contents = fs::read_to_string(&manifest_path).unwrap();
        assert!(contents.contains("\"demo\""));
    }
}
