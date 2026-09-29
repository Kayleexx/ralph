//! `ralph __handoff-probe` / `ralph __handoff-recv` — the two hidden commands a source
//! machine's `ralph handoff` invokes over SSH on the destination. Neither is meant for a
//! human: both print one line of JSON to stdout and nothing else, so the source can
//! parse the result straight out of the SSH child's output (`daemon::handoff::run_ssh`).
use std::io::{Read, Write};
use std::path::Path;

use crate::client;
use crate::ipc::Request;
use crate::portable;
use crate::session;

fn disk_free_bytes(dir: &Path) -> u64 {
    let Ok(out) = std::process::Command::new("df")
        .arg("-B1")
        .arg("--output=avail")
        .arg(dir)
        .output()
    else {
        return 0;
    };
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .nth(1)
        .and_then(|line| line.trim().parse().ok())
        .unwrap_or(0)
}

pub async fn run_probe(home: &Path) -> i32 {
    if let Err(e) = std::fs::create_dir_all(home) {
        eprintln!("cannot prepare {}: {e}", home.display());
        return 1;
    }
    let probe = serde_json::json!({
        "ralph_version": env!("CARGO_PKG_VERSION"),
        "protocol_version": portable::FORMAT_VERSION,
        "disk_free_bytes": disk_free_bytes(home),
    });
    println!("{probe}");
    0
}

pub async fn run_recv(home: &Path, name: Option<String>) -> i32 {
    let sessions_root = home.join("sessions");
    if let Err(e) = std::fs::create_dir_all(&sessions_root) {
        eprintln!("cannot prepare {}: {e}", sessions_root.display());
        return 1;
    }

    let mut bytes = Vec::new();
    let read = std::io::stdin()
        .lock()
        .take(portable::MAX_ARCHIVE_BYTES + 1)
        .read_to_end(&mut bytes);
    if let Err(e) = read {
        eprintln!("cannot read handoff archive from stdin: {e}");
        return 1;
    }

    let staging_path =
        sessions_root.join(format!("handoff-recv-{}.ralph", session::new_session_id()));
    if let Err(e) = std::fs::write(&staging_path, &bytes) {
        eprintln!("cannot stage handoff archive: {e}");
        return 1;
    }

    let outcome = import_via_local_daemon(home, &staging_path, name).await;
    let _ = std::fs::remove_file(&staging_path);

    let response = match outcome {
        Ok(info) => serde_json::json!({ "ok": true, "session": info }),
        Err(message) => serde_json::json!({ "ok": false, "error": message }),
    };
    println!("{response}");
    let _ = std::io::stdout().flush();
    0
}

async fn import_via_local_daemon(
    home: &Path,
    staging_path: &Path,
    name: Option<String>,
) -> Result<crate::ipc::SessionInfo, String> {
    let mut stream = client::connect_or_start(home)
        .await
        .map_err(|e| format!("cannot reach local daemon: {e}"))?;
    let request = Request::Import {
        path: staging_path.to_string_lossy().into_owned(),
        name,
    };
    match client::send_request(&mut stream, &request).await {
        Ok(crate::ipc::Response::Run(info)) => Ok(info),
        Ok(crate::ipc::Response::Error(payload)) => Err(payload.summary),
        Ok(_) => Err("local daemon sent an unexpected response".to_string()),
        Err(e) => Err(e.to_string()),
    }
}
