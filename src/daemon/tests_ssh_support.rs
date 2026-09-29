//! A fake `ssh` for `tests_handoff.rs`/`tests_drain.rs`: branches purely on the
//! "destination" argument so a test never touches the network, real SSH, or a second
//! `ralph` binary.
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

pub(crate) fn fake_ssh_script(dir: &Path) -> PathBuf {
    let path = dir.join("fake-ssh");
    std::fs::write(
        &path,
        r#"#!/bin/sh
dest="$1"
sub="$3"
probe='{"ralph_version":"0.1.0","protocol_version":1,"disk_free_bytes":999999999999}'
case "$dest" in
  ok)
    if [ "$sub" = "__handoff-probe" ]; then echo "$probe"; else cat >/dev/null; echo '{"ok":true}'; fi
    ;;
  auth-fails)
    echo "permission denied (publickey)" >&2
    exit 255
    ;;
  bad-version)
    echo '{"ralph_version":"0.1.0","protocol_version":999,"disk_free_bytes":999999999999}'
    ;;
  low-disk)
    echo '{"ralph_version":"0.1.0","protocol_version":1,"disk_free_bytes":1}'
    ;;
  recv-rejects)
    if [ "$sub" = "__handoff-probe" ]; then echo "$probe"; else cat >/dev/null; echo '{"ok":false,"error":"name already exists"}'; fi
    ;;
  recv-hangs)
    if [ "$sub" = "__handoff-probe" ]; then echo "$probe"; else cat >/dev/null; sleep 100; fi
    ;;
  slow)
    sleep 0.3
    if [ "$sub" = "__handoff-probe" ]; then echo "$probe"; else cat >/dev/null; echo '{"ok":true}'; fi
    ;;
  *)
    echo "unknown fake destination: $dest" >&2
    exit 1
    ;;
esac
"#,
    )
    .unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
    path
}
