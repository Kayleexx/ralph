//! Port selection, vLLM binary discovery, and the exact `vllm serve` argument list.
use std::net::TcpListener;
use std::path::{Path, PathBuf};

use crate::engine::{EngineError, KvOffloadSpec};

use super::GPU_MEMORY_UTILIZATION;
use super::kv_offload;

pub(super) fn pick_free_port() -> Result<u16, EngineError> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    Ok(listener.local_addr()?.port())
}

/// Checks, in order: an activated venv (`$VIRTUAL_ENV`, works from any directory once
/// `source .venv/bin/activate` has run), a `.venv` in the current directory (works when
/// `ralph` is run from inside the project without activating anything), then whatever
/// `vllm` resolves to on `PATH`. `ralph` is often installed globally and run from
/// wherever the user happens to be, so a CWD-relative check alone isn't enough.
pub fn resolve_vllm_binary() -> PathBuf {
    resolve_vllm_binary_from(
        std::env::var("VIRTUAL_ENV").ok(),
        Path::new(".venv/bin/vllm"),
    )
}

/// Split out so tests can control both inputs directly, rather than mutating the real
/// `VIRTUAL_ENV` process environment (`std::env::set_var` is `unsafe` on this toolchain).
pub(super) fn resolve_vllm_binary_from(virtual_env: Option<String>, cwd_venv: &Path) -> PathBuf {
    if let Some(venv) = virtual_env {
        let candidate = PathBuf::from(venv).join("bin/vllm");
        if candidate.exists() {
            return candidate;
        }
    }
    if cwd_venv.exists() {
        return cwd_venv.to_path_buf();
    }
    PathBuf::from("vllm")
}

/// Split out for testability: the exact flags passed to `vllm serve`.
pub(super) fn vllm_serve_args(
    model: &str,
    port: u16,
    revision: Option<&str>,
    profile: Option<super::super::profiles::WorkerProfile>,
    kv_offload: Option<&KvOffloadSpec>,
) -> Vec<String> {
    let mut args = vec![
        "serve".to_string(),
        model.to_string(),
        "--port".to_string(),
        port.to_string(),
        "--gpu-memory-utilization".to_string(),
        if profile.is_some() {
            "0.1"
        } else {
            GPU_MEMORY_UTILIZATION
        }
        .to_string(),
        // Eager execution keeps startup and runtime allocation predictable.
        "--enforce-eager".to_string(),
        // Preserve the existing idle sleep lifecycle.
        "--enable-sleep-mode".to_string(),
    ];
    // Explicit KV bypasses automatic sizing; utilization still gates startup in vLLM.
    if let Some(profile) = profile {
        args.extend([
            "--max-model-len".into(),
            profile.max_context.to_string(),
            "--kv-cache-memory-bytes".into(),
            (u64::from(profile.kv_mib) << 20).to_string(),
            "--dtype".into(),
            "bfloat16".into(),
        ]);
    }
    if let Some(revision) = revision {
        args.push("--revision".to_string());
        args.push(revision.to_string());
        args.push("--tokenizer-revision".to_string());
        args.push(revision.to_string());
    }
    if let Some(kv) = kv_offload {
        args.push("--kv-transfer-config".to_string());
        args.push(kv_offload::kv_transfer_config_json(
            &kv.root_dir,
            kv.cpu_bytes,
            &kv.engine_id,
        ));
    }
    args
}
