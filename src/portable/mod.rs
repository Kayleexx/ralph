//! The `.ralph` portable session artifact: a plain `tar` container with our own
//! path-safety checks on read (never `tar::Archive::unpack()`, which trusts
//! archive-supplied paths) and a whole-artifact checksum to catch corruption before
//! anything is imported. No compression, no crypto-hash dependency — see RALPH_SPEC.md
//! §17 Phase 5 and §16.8's export/import edge cases.
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const FORMAT_VERSION: u32 = 1;
/// Generous headroom over the KV quota (`daemon::lifecycle::KV_QUOTA_BYTES`) a single
/// session's checkpoint could plausibly hold — bounds how much a hostile or corrupt
/// archive can make an import read/write before being rejected.
const MAX_ARCHIVE_BYTES: u64 = 8 * 1024 * 1024 * 1024;

const MANIFEST_ENTRY: &str = "manifest.json";
const TURNS_ENTRY: &str = "turns.json";
const CHECKPOINT_ENTRY: &str = "checkpoint.json";
const KVCACHE_PREFIX: &str = "kvcache/";

#[derive(Debug, Error)]
pub enum PortableError {
    #[error("artifact is corrupt: {0}")]
    Corrupt(String),
    #[error("artifact format version {found} is not supported (this build supports {supported})")]
    UnsupportedVersion { found: u32, supported: u32 },
}

fn corrupt(msg: impl Into<String>) -> PortableError {
    PortableError::Corrupt(msg.into())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportManifest {
    pub format_version: u32,
    pub id: String,
    pub name: String,
    pub model: String,
    pub model_revision: Option<String>,
    pub tokenizer_revision: Option<String>,
    pub engine: String,
    pub engine_version: Option<String>,
    pub created_at: String,
    pub exported_at: String,
    pub has_accel: bool,
    pub checksum: String,
}

pub struct Parsed {
    pub manifest: ExportManifest,
    pub turns_json: Vec<u8>,
    pub checkpoint_json: Option<Vec<u8>>,
    /// Set only when `checkpoint_json` is `Some` and the archive actually contained
    /// `kvcache/` entries — extracted under the caller-supplied temp directory, never
    /// the session's real kvcache directory (the caller moves it into place only after
    /// every check here has passed).
    pub kvcache_dir: Option<PathBuf>,
}

// Mirrors `storage::token_log`'s per-row checksum scheme (stable FNV-1a) for the same
// "catch corruption" purpose — duplicated rather than shared since the two live in
// unrelated modules over unrelated data shapes.
fn checksum64(bytes: &[u8]) -> String {
    let hash = bytes.iter().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    });
    format!("{hash:016x}")
}

pub struct ManifestInput {
    pub id: String,
    pub name: String,
    pub model: String,
    pub model_revision: Option<String>,
    pub tokenizer_revision: Option<String>,
    pub engine: String,
    pub engine_version: Option<String>,
    pub created_at: String,
    pub exported_at: String,
}

pub fn build_archive(
    fields: ManifestInput,
    turns_json: &[u8],
    checkpoint_json: Option<&[u8]>,
    kvcache_dir: Option<&Path>,
) -> io::Result<Vec<u8>> {
    let checksum_input: Vec<u8> = [turns_json, checkpoint_json.unwrap_or(&[])].concat();
    let manifest = ExportManifest {
        format_version: FORMAT_VERSION,
        id: fields.id,
        name: fields.name,
        model: fields.model,
        model_revision: fields.model_revision,
        tokenizer_revision: fields.tokenizer_revision,
        engine: fields.engine,
        engine_version: fields.engine_version,
        created_at: fields.created_at,
        exported_at: fields.exported_at,
        has_accel: checkpoint_json.is_some(),
        checksum: checksum64(&checksum_input),
    };
    let manifest_json = serde_json::to_vec(&manifest)?;

    let mut builder = tar::Builder::new(Vec::new());
    append_file(&mut builder, MANIFEST_ENTRY, &manifest_json)?;
    append_file(&mut builder, TURNS_ENTRY, turns_json)?;
    if let Some(checkpoint_json) = checkpoint_json {
        append_file(&mut builder, CHECKPOINT_ENTRY, checkpoint_json)?;
    }
    if let Some(kvcache_dir) = kvcache_dir {
        append_dir_files(&mut builder, kvcache_dir, kvcache_dir)?;
    }
    builder.into_inner()
}

fn append_file<W: Write>(
    builder: &mut tar::Builder<W>,
    name: &str,
    bytes: &[u8],
) -> io::Result<()> {
    let mut header = tar::Header::new_gnu();
    header.set_size(bytes.len() as u64);
    header.set_mode(0o600);
    header.set_entry_type(tar::EntryType::Regular);
    header.set_cksum();
    builder.append_data(&mut header, name, bytes)
}

// Only regular files are walked in from our own kvcache directory; a symlink is skipped
// rather than followed — defense in depth even though we control this tree. A missing
// directory (the checkpoint pointer exists, but nothing was ever written into it — see
// `daemon::portable::export`) is treated as empty, not an error, mirroring
// `daemon::lifecycle::dir_size`'s existing tolerance for the same situation.
fn append_dir_files<W: Write>(
    builder: &mut tar::Builder<W>,
    root: &Path,
    dir: &Path,
) -> io::Result<()> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Ok(());
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            append_dir_files(builder, root, &path)?;
        } else if file_type.is_file() {
            let relative = path.strip_prefix(root).unwrap_or(&path);
            let name = format!("{KVCACHE_PREFIX}{}", relative.to_string_lossy());
            let bytes = std::fs::read(&path)?;
            append_file(builder, &name, &bytes)?;
        }
    }
    Ok(())
}

/// Rejects any path outside the fixed known set, any absolute path or `..` component,
/// and returns the safe relative suffix for a `kvcache/`-prefixed entry.
enum EntryKind<'a> {
    Manifest,
    Turns,
    Checkpoint,
    Kvcache(&'a Path),
    Reject,
}

fn classify(path: &Path) -> EntryKind<'_> {
    if path.components().any(|c| {
        matches!(
            c,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    }) {
        return EntryKind::Reject;
    }
    if path == Path::new(MANIFEST_ENTRY) {
        return EntryKind::Manifest;
    }
    if path == Path::new(TURNS_ENTRY) {
        return EntryKind::Turns;
    }
    if path == Path::new(CHECKPOINT_ENTRY) {
        return EntryKind::Checkpoint;
    }
    if let Ok(suffix) = path.strip_prefix("kvcache") {
        if suffix.as_os_str().is_empty() {
            return EntryKind::Reject; // the bare directory entry, nothing to extract
        }
        return EntryKind::Kvcache(suffix);
    }
    EntryKind::Reject
}

pub fn parse_archive(bytes: &[u8], extract_kvcache_into: &Path) -> Result<Parsed, PortableError> {
    let mut archive = tar::Archive::new(bytes);
    let entries = archive
        .entries()
        .map_err(|e| corrupt(format!("not a valid archive: {e}")))?;

    let mut manifest_json: Option<Vec<u8>> = None;
    let mut turns_json: Option<Vec<u8>> = None;
    let mut checkpoint_json: Option<Vec<u8>> = None;
    let mut has_kvcache = false;
    let mut total: u64 = 0;

    for entry in entries {
        let mut entry = entry.map_err(|e| corrupt(format!("unreadable entry: {e}")))?;
        let entry_type = entry.header().entry_type();
        if !matches!(
            entry_type,
            tar::EntryType::Regular | tar::EntryType::Directory
        ) {
            return Err(corrupt("archive contains an unsupported entry type"));
        }
        let path = entry
            .path()
            .map_err(|e| corrupt(format!("unreadable entry path: {e}")))?
            .into_owned();

        total = total.saturating_add(entry.header().size().unwrap_or(0));
        if total > MAX_ARCHIVE_BYTES {
            return Err(corrupt("artifact exceeds the maximum allowed size"));
        }

        match classify(&path) {
            EntryKind::Reject => {
                return Err(corrupt(format!(
                    "archive contains an unsafe or unrecognized path: {}",
                    path.display()
                )));
            }
            EntryKind::Manifest => manifest_json = Some(read_entry(&mut entry)?),
            EntryKind::Turns => turns_json = Some(read_entry(&mut entry)?),
            EntryKind::Checkpoint => checkpoint_json = Some(read_entry(&mut entry)?),
            EntryKind::Kvcache(suffix) => {
                if entry_type != tar::EntryType::Regular {
                    continue;
                }
                let dest = extract_kvcache_into.join(suffix);
                if let Some(parent) = dest.parent() {
                    std::fs::create_dir_all(parent)
                        .map_err(|e| corrupt(format!("cannot extract archive: {e}")))?;
                }
                let mut out = std::fs::File::create(&dest)
                    .map_err(|e| corrupt(format!("cannot extract archive: {e}")))?;
                std::io::copy(&mut entry, &mut out)
                    .map_err(|e| corrupt(format!("cannot extract archive: {e}")))?;
                has_kvcache = true;
            }
        }
    }

    let manifest_json = manifest_json.ok_or_else(|| corrupt("missing manifest.json"))?;
    let manifest: ExportManifest = serde_json::from_slice(&manifest_json)
        .map_err(|e| corrupt(format!("malformed manifest: {e}")))?;
    if manifest.format_version != FORMAT_VERSION {
        return Err(PortableError::UnsupportedVersion {
            found: manifest.format_version,
            supported: FORMAT_VERSION,
        });
    }
    let turns_json = turns_json.ok_or_else(|| corrupt("missing turns.json"))?;

    let checksum_input: Vec<u8> = [
        turns_json.as_slice(),
        checkpoint_json.as_deref().unwrap_or(&[]),
    ]
    .concat();
    if checksum64(&checksum_input) != manifest.checksum {
        return Err(corrupt("checksum mismatch"));
    }

    Ok(Parsed {
        manifest,
        turns_json,
        checkpoint_json,
        kvcache_dir: has_kvcache.then(|| extract_kvcache_into.to_path_buf()),
    })
}

fn read_entry<R: Read>(entry: &mut tar::Entry<'_, R>) -> Result<Vec<u8>, PortableError> {
    let mut buf = Vec::new();
    entry
        .read_to_end(&mut buf)
        .map_err(|e| corrupt(format!("cannot read entry: {e}")))?;
    Ok(buf)
}

#[cfg(test)]
mod tests;
