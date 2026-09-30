//! `ralph handoff`: moves a session's ownership to another Ralph installation over SSH.
//! Composes the existing archive format/import path with an explicit two-phase commit —
//! the destination's ACK (a successful `__handoff-recv`) is the only thing that ever
//! flips this session's state to `Moved` (`crate::state`), so a killed connection,
//! SSH auth failure, or destination rejection always leaves the source unchanged and
//! recoverable — never two machines both thinking they own the session.
use std::process::Stdio;
use std::sync::Arc;

use serde::Deserialize;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;
use tokio::sync::oneshot;

use super::{Daemon, cancelled, now_rfc3339, to_session_info};
use crate::engine::Engine;
use crate::error::CliError;
use crate::ipc::SessionInfo;
use crate::portable;
use crate::state::SessionState;
use crate::storage::SessionRow;
use crate::storage::handoff::HandoffRow;

/// Headroom over the archive's own byte length required on the destination before a
/// transfer is attempted — a coarse preflight, not a guarantee: a destination low on
/// disk space should fail before transfer, or at worst before commit, not partway
/// through leaving an orphaned partial archive.
const DISK_HEADROOM_FACTOR: u64 = 2;

#[derive(Debug, Deserialize)]
struct ProbeInfo {
    #[allow(dead_code)] // surfaced in error messages only, not compared today
    ralph_version: String,
    protocol_version: u32,
    disk_free_bytes: u64,
}

#[derive(Debug, Deserialize)]
struct RecvResult {
    ok: bool,
    #[serde(default)]
    error: Option<String>,
}

fn cancelled_error() -> CliError {
    CliError::InvalidState("handoff cancelled; source session unchanged and recoverable".into())
}

/// Runs `ssh <destination> ralph <remote_args...>`, optionally piping `stdin_bytes` in
/// first, cancellable at any point (kills the child and leaves the source untouched).
async fn run_ssh(
    ssh_program: &str,
    destination: &str,
    remote_args: &[&str],
    stdin_bytes: Option<&[u8]>,
    cancel: &mut Option<oneshot::Receiver<()>>,
) -> Result<std::process::Output, CliError> {
    let mut cmd = Command::new(ssh_program);
    cmd.arg(destination).arg("ralph");
    cmd.args(remote_args);
    cmd.stdin(if stdin_bytes.is_some() {
        Stdio::piped()
    } else {
        Stdio::null()
    });
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    let mut child = cmd
        .spawn()
        .map_err(|e| CliError::Other(anyhow::anyhow!("cannot run ssh: {e}")))?;

    if let Some(bytes) = stdin_bytes {
        let Some(mut stdin) = child.stdin.take() else {
            let _ = child.start_kill();
            let _ = child.wait().await;
            return Err(CliError::Other(anyhow::anyhow!(
                "ssh child has no stdin pipe; source session unchanged"
            )));
        };
        let write_and_close = async {
            stdin.write_all(bytes).await?;
            drop(stdin);
            Ok::<(), std::io::Error>(())
        };
        tokio::select! {
            biased;
            _ = cancelled(cancel) => {
                let _ = child.start_kill();
                let _ = child.wait().await;
                return Err(cancelled_error());
            }
            result = write_and_close => {
                result.map_err(|e| CliError::Other(e.into()))?;
            }
        }
    }

    let mut stdout_pipe = child.stdout.take();
    let mut stderr_pipe = child.stderr.take();
    let status = tokio::select! {
        biased;
        _ = cancelled(cancel) => {
            let _ = child.start_kill();
            let _ = child.wait().await;
            return Err(cancelled_error());
        }
        status = child.wait() => status.map_err(|e| CliError::Other(e.into()))?,
    };
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    if let Some(pipe) = stdout_pipe.as_mut() {
        let _ = pipe.read_to_end(&mut stdout).await;
    }
    if let Some(pipe) = stderr_pipe.as_mut() {
        let _ = pipe.read_to_end(&mut stderr).await;
    }
    Ok(std::process::Output {
        status,
        stdout,
        stderr,
    })
}

fn transport_error(destination: &str, output: &std::process::Output) -> CliError {
    let stderr = String::from_utf8_lossy(&output.stderr);
    CliError::Resource(format!(
        "could not reach {destination}: {}; source session unchanged",
        stderr.trim()
    ))
}

impl<E: Engine + 'static> Daemon<E> {
    #[cfg(test)]
    pub(crate) fn set_ssh_program(&self, ssh_program: String) {
        *self
            .ssh_program
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = ssh_program;
    }

    fn ssh_program(&self) -> String {
        self.ssh_program
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub async fn handoff_cancellable(
        self: &Arc<Self>,
        identifier: &str,
        destination: &str,
        name: Option<String>,
        mut cancel: Option<oneshot::Receiver<()>>,
    ) -> Result<SessionInfo, CliError> {
        let row = self.storage().resolve(identifier)?;
        let _guard = self.locks.try_acquire(&row.id).ok_or_else(|| {
            CliError::InvalidState("operation already in progress for this session".into())
        })?;
        let row = self.storage().resolve(&row.id)?;
        let origin = row.state;
        if !matches!(
            origin,
            SessionState::Active | SessionState::Paused | SessionState::Hibernated
        ) {
            return Err(CliError::InvalidState(format!(
                "session is {origin}, cannot hand it off"
            )));
        }
        // Checked before touching local state at all: an unreachable destination or a
        // protocol mismatch is knowable up front, so it should never cost the caller
        // their live worker — only the disk-space preflight has to wait for the worker
        // to actually be released (it needs the archive's real, post-release size).
        let probe = self.probe_destination(destination, &mut cancel).await?;

        self.transition(&row.id, origin, SessionState::Moving, None)?;

        let mut worker_released = false;
        if origin == SessionState::Active {
            self.enforce_kv_quota(&row.id);
            if let Err(error) = self
                .storage()
                .upsert_checkpoint(&self.checkpoint_row_for(&row))
            {
                eprintln!("handoff: checkpoint unavailable, continuing logical-only: {error}");
            }
            if let Err(error) = self.release_worker(&row).await {
                self.transition(&row.id, SessionState::Moving, SessionState::Active, None)?;
                return Err(error.for_session_state(&row.name, &row.model, "active"));
            }
            worker_released = true;
        }
        let rollback_to = if worker_released {
            SessionState::Paused
        } else {
            origin
        };

        let remote_name = name.unwrap_or_else(|| row.name.clone());
        if let Err(e) = self
            .attempt_handoff(&row, destination, &remote_name, &probe, &mut cancel)
            .await
        {
            self.transition(&row.id, SessionState::Moving, rollback_to, None)?;
            return Err(e);
        }

        self.storage().upsert_handoff(&HandoffRow {
            session_id: row.id.clone(),
            destination: destination.to_string(),
            remote_name,
            committed_at: now_rfc3339(),
        })?;
        self.transition(&row.id, SessionState::Moving, SessionState::Moved, None)?;
        Ok(to_session_info(&self.storage().resolve(&row.id)?))
    }

    /// Reachability + protocol-version check only — deliberately callable before the
    /// local worker is ever released, so a destination that's unreachable from the
    /// start never costs the caller their live session (see `handoff_cancellable`).
    /// The disk-space check is the one preflight step that must wait: it needs the
    /// archive's real byte length, which itself needs a stable (released) KV directory.
    async fn probe_destination(
        &self,
        destination: &str,
        cancel: &mut Option<oneshot::Receiver<()>>,
    ) -> Result<ProbeInfo, CliError> {
        let ssh_program = self.ssh_program();
        let probe_output = run_ssh(
            &ssh_program,
            destination,
            &["__handoff-probe"],
            None,
            cancel,
        )
        .await?;
        if !probe_output.status.success() {
            return Err(transport_error(destination, &probe_output));
        }
        let probe: ProbeInfo = serde_json::from_slice(&probe_output.stdout).map_err(|e| {
            CliError::Resource(format!(
                "{destination} did not answer the handoff probe correctly: {e}; source session unchanged"
            ))
        })?;
        if probe.protocol_version != portable::FORMAT_VERSION {
            return Err(CliError::Usage(format!(
                "{destination} supports artifact format version {}, this build produces {}; source session unchanged",
                probe.protocol_version,
                portable::FORMAT_VERSION
            )));
        }
        Ok(probe)
    }

    async fn attempt_handoff(
        &self,
        row: &SessionRow,
        destination: &str,
        remote_name: &str,
        probe: &ProbeInfo,
        cancel: &mut Option<oneshot::Receiver<()>>,
    ) -> Result<(), CliError> {
        let archive = self.build_export_archive(row, true)?;
        let ssh_program = self.ssh_program();
        let required = (archive.len() as u64).saturating_mul(DISK_HEADROOM_FACTOR);
        if probe.disk_free_bytes < required {
            return Err(CliError::Resource(format!(
                "{destination} has {} bytes free, at least {required} recommended; source session unchanged",
                probe.disk_free_bytes
            )));
        }

        let recv_output = run_ssh(
            &ssh_program,
            destination,
            &["__handoff-recv", "--name", remote_name],
            Some(&archive),
            cancel,
        )
        .await?;
        if !recv_output.status.success() {
            return Err(transport_error(destination, &recv_output));
        }
        let result: RecvResult = serde_json::from_slice(&recv_output.stdout).map_err(|e| {
            CliError::Resource(format!(
                "{destination} sent an unreadable handoff response: {e}; source session unchanged"
            ))
        })?;
        if !result.ok {
            return Err(CliError::Resource(format!(
                "{destination} rejected the handoff: {}; source session unchanged",
                result.error.unwrap_or_else(|| "unknown error".into())
            )));
        }
        Ok(())
    }
}
