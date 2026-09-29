use super::*;
fn fields() -> ManifestInput {
    ManifestInput {
        id: "01AAA".into(),
        name: "demo".into(),
        model: "some/model".into(),
        model_revision: Some("rev".into()),
        tokenizer_revision: Some("rev".into()),
        engine: "vllm".into(),
        engine_version: Some("0.30.0".into()),
        created_at: "2026-01-01T00:00:00Z".into(),
        exported_at: "2026-01-02T00:00:00Z".into(),
    }
}

#[test]
fn round_trips_turns_only() {
    let archive = build_archive(fields(), b"[1,2,3]", None, None).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let parsed = parse_archive(&archive, dir.path()).unwrap();
    assert_eq!(parsed.turns_json, b"[1,2,3]");
    assert!(parsed.checkpoint_json.is_none());
    assert!(!parsed.manifest.has_accel);
    assert_eq!(parsed.manifest.name, "demo");
}

#[test]
fn round_trips_with_accel() {
    let kv = tempfile::tempdir().unwrap();
    std::fs::write(kv.path().join("block.bin"), b"kvbytes").unwrap();
    let archive = build_archive(fields(), b"[]", Some(b"{\"fp\":1}"), Some(kv.path())).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let parsed = parse_archive(&archive, dir.path()).unwrap();
    assert_eq!(
        parsed.checkpoint_json.as_deref(),
        Some(b"{\"fp\":1}".as_slice())
    );
    assert!(parsed.manifest.has_accel);
    let extracted = parsed.kvcache_dir.unwrap().join("block.bin");
    assert_eq!(std::fs::read(extracted).unwrap(), b"kvbytes");
}

#[test]
fn checksum_mismatch_is_rejected() {
    let mut archive = build_archive(fields(), b"[1,2,3]", None, None).unwrap();
    let pos = archive
        .windows(3)
        .position(|w| w == b"1,2")
        .expect("turns bytes present in raw tar");
    archive[pos] = b'9';
    let dir = tempfile::tempdir().unwrap();
    assert!(matches!(
        parse_archive(&archive, dir.path()),
        Err(PortableError::Corrupt(_))
    ));
}

#[test]
fn unsupported_version_is_rejected_clearly() {
    // build_archive always stamps the current FORMAT_VERSION, so a future version is
    // constructed by hand here to exercise the parse-side check.
    let manifest = ExportManifest {
        format_version: 99,
        id: "x".into(),
        name: "x".into(),
        model: "x".into(),
        model_revision: None,
        tokenizer_revision: None,
        engine: "vllm".into(),
        engine_version: None,
        created_at: "t".into(),
        exported_at: "t".into(),
        has_accel: false,
        checksum: checksum64(b"[]"),
    };
    let mut builder = tar::Builder::new(Vec::new());
    append_file(
        &mut builder,
        MANIFEST_ENTRY,
        &serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    append_file(&mut builder, TURNS_ENTRY, b"[]").unwrap();
    let archive = builder.into_inner().unwrap();
    let dir = tempfile::tempdir().unwrap();
    assert!(matches!(
        parse_archive(&archive, dir.path()),
        Err(PortableError::UnsupportedVersion {
            found: 99,
            supported: FORMAT_VERSION
        })
    ));
}

// The `tar` crate itself refuses to *write* an unsafe path via `set_path` — real
// attacker-crafted archives aren't built with this crate's safety rails, so these
// tests write the raw header bytes directly to simulate one.
fn set_raw_name(header: &mut tar::Header, name: &str) {
    let bytes = header.as_mut_bytes();
    bytes[0..100].fill(0);
    bytes[..name.len()].copy_from_slice(name.as_bytes());
}

#[test]
fn path_traversal_entry_is_rejected() {
    let mut builder = tar::Builder::new(Vec::new());
    let mut header = tar::Header::new_gnu();
    header.set_size(1);
    header.set_mode(0o600);
    header.set_entry_type(tar::EntryType::Regular);
    set_raw_name(&mut header, "../evil");
    header.set_cksum();
    builder.append(&header, &b"x"[..]).unwrap();
    let archive = builder.into_inner().unwrap();
    let dir = tempfile::tempdir().unwrap();
    assert!(matches!(
        parse_archive(&archive, dir.path()),
        Err(PortableError::Corrupt(_))
    ));
}

#[test]
fn absolute_path_entry_is_rejected() {
    let mut builder = tar::Builder::new(Vec::new());
    let mut header = tar::Header::new_gnu();
    header.set_size(1);
    header.set_mode(0o600);
    header.set_entry_type(tar::EntryType::Regular);
    set_raw_name(&mut header, "/etc/passwd");
    header.set_cksum();
    builder.append(&header, &b"x"[..]).unwrap();
    let archive = builder.into_inner().unwrap();
    let dir = tempfile::tempdir().unwrap();
    assert!(matches!(
        parse_archive(&archive, dir.path()),
        Err(PortableError::Corrupt(_))
    ));
}

#[test]
fn symlink_entry_is_rejected() {
    let mut builder = tar::Builder::new(Vec::new());
    let mut header = tar::Header::new_gnu();
    header.set_size(0);
    header.set_entry_type(tar::EntryType::Symlink);
    header.set_path("kvcache/link").unwrap();
    header.set_link_name("/etc/passwd").unwrap();
    header.set_cksum();
    builder.append(&header, &b""[..]).unwrap();
    let archive = builder.into_inner().unwrap();
    let dir = tempfile::tempdir().unwrap();
    assert!(matches!(
        parse_archive(&archive, dir.path()),
        Err(PortableError::Corrupt(_))
    ));
}

#[test]
fn missing_manifest_is_rejected() {
    let mut builder = tar::Builder::new(Vec::new());
    append_file(&mut builder, TURNS_ENTRY, b"[]").unwrap();
    let archive = builder.into_inner().unwrap();
    let dir = tempfile::tempdir().unwrap();
    assert!(matches!(
        parse_archive(&archive, dir.path()),
        Err(PortableError::Corrupt(_))
    ));
}
