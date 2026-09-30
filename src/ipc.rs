//! CLI <-> daemon wire protocol: one JSON value per frame, framed with a 4-byte
//! little-endian length prefix over a Unix domain socket. No RPC framework — the
//! command surface is small enough that this is simpler than pulling one in.
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use crate::continuity::ContinuityPolicy;

#[derive(Debug, Serialize, Deserialize)]
pub enum Request {
    Run {
        model: String,
        name: Option<String>,
        gpu: Option<u32>,
        policy: Option<ContinuityPolicy>,
        continuity_target_ms: Option<u64>,
    },
    Query {
        session: String,
        prompt: String,
    },
    Ps,
    Inspect {
        session: String,
    },
    Recover {
        session: String,
    },
    Checkpoint {
        session: String,
    },
    Pause {
        session: String,
    },
    Hibernate {
        session: String,
    },
    Resume {
        session: String,
        fast_only: bool,
        portable: bool,
    },
    Export {
        session: String,
        output_path: String,
        force: bool,
        with_accel: bool,
    },
    Import {
        path: String,
        name: Option<String>,
    },
    Handoff {
        session: String,
        destination: String,
        name: Option<String>,
    },
    Drain {
        location: String,
        to: Option<String>,
        yes: bool,
    },
    Migrate {
        session: String,
        to: u32,
    },
}

/// Sent by the client in place of a new request while a `Query` is streaming, to cancel
/// the in-flight generation (Ctrl-C).
#[derive(Debug, Serialize, Deserialize)]
pub struct Cancel;

#[derive(Debug, Serialize, Deserialize)]
pub struct SessionInfo {
    pub id: String,
    pub name: String,
    pub model: String,
    pub model_revision: Option<String>,
    pub engine_version: Option<String>,
    pub state: String,
    pub pid: Option<i64>,
    pub location: String,
    pub token_count: i64,
    /// The name of a session `ralph run` hibernated to free VRAM for this one, if
    /// pressure-aware admission had to (`daemon::continuity::make_room`) — `None` on
    /// every other response that reuses `SessionInfo` (`ps`, `inspect`, ...).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub made_room_for: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ResumeInfo {
    pub session: SessionInfo,
    /// Best-effort, honest-only: `false` unless a still-compatible KV directory with
    /// actual content was found — never claims a native restore that didn't happen.
    pub native: bool,
    /// Resume on an already-active session is an idempotent no-op, not a rebuild —
    /// distinguishes that from a genuine portable-fallback resume (both report
    /// `native: false`), so the CLI doesn't claim a KV rebuild that never happened.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub already_active: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InspectInfo {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_failure: Option<String>,
    pub session: SessionInfo,
    pub recoverability: String,
    pub fast_restore: String,
    /// Why, not just whether — `RestoreReadiness::as_str()`.
    pub restore_readiness: String,
    /// (field, current, stored) triples that differ, `--verbose` only. Never prompt content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fingerprint_diff: Option<Vec<(String, String, String)>>,
    pub portable_state: String,
    /// `"user@gpu-box as research"` once `ralph handoff` has committed this session
    /// elsewhere — `None` otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moved_to: Option<String>,
    pub continuity_policy: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub continuity_target_ms: Option<i64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorPayload {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostic: Option<String>,
    pub exit_code: i32,
    pub summary: String,
    pub detail: Vec<String>,
    pub next: Option<String>,
}

/// Non-streaming replies. `Query` is handled separately via `Chunk`/`Done` frames since
/// it needs to deliver tokens as they arrive rather than a single response.
#[derive(Debug, Serialize, Deserialize)]
pub enum Response {
    Run(SessionInfo),
    Ps(Vec<SessionInfo>),
    Inspect(InspectInfo),
    Resume(ResumeInfo),
    Drain(DrainReport),
    Migrate(MigrationReport),
    Error(ErrorPayload),
}

/// Everything Stage C's migration work asks to be measured, alongside the resulting
/// `SessionInfo`. The CLI only prints a terse progress/summary line in plain mode; this
/// whole struct is what `--json`/`--verbose` show.
#[derive(Debug, Serialize, Deserialize)]
pub struct MigrationReport {
    pub session: SessionInfo,
    pub source_gpu: u32,
    pub destination_gpu: u32,
    pub native: bool,
    pub total_ms: f64,
    /// Wall-clock time the session had no live worker at all — from the source
    /// worker's process actually stopping to the destination worker committing Active.
    pub interruption_ms: f64,
    pub replay_prefill_ms: f64,
    pub token_count: i64,
    pub source_vram_freed_mib: Option<u64>,
    pub destination_vram_used_mib: Option<u64>,
    /// Always `None` today: the first correct implementation is always portable
    /// reconstruction (replay from durable history), which moves no KV bytes at all —
    /// this field exists for when/if a proven-safe native cross-GPU KV transfer lands.
    pub bytes_transferred: Option<u64>,
    /// Not measured here: a real time-to-first-token needs an actual generation call,
    /// which a lifecycle op has no business making on its own. `None` in production;
    /// the dual-GPU acceptance harness measures it with a real post-migration query.
    pub ttft_ms: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrainOutcome {
    pub name: String,
    pub action: String,
    pub ok: bool,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrainReport {
    pub location: String,
    /// `false` while `ralph drain` is only printing its plan (no `--yes` yet, or more
    /// than one session affected) — nothing has been touched.
    pub executed: bool,
    pub sessions: Vec<DrainOutcome>,
    pub all_safe: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum ServerMessage {
    Chunk(String),
    Done { token_count: i64 },
    Result(Box<Response>),
}

pub async fn write_frame<W, T>(writer: &mut W, value: &T) -> std::io::Result<()>
where
    W: AsyncWrite + Unpin,
    T: Serialize,
{
    let bytes = serde_json::to_vec(value)?;
    writer
        .write_all(&(bytes.len() as u32).to_le_bytes())
        .await?;
    writer.write_all(&bytes).await?;
    writer.flush().await
}

/// Returns `Ok(None)` on a clean EOF between frames (the other side closed the
/// connection), and an error for anything else, including an EOF mid-frame.
pub async fn read_frame<R, T>(reader: &mut R) -> std::io::Result<Option<T>>
where
    R: AsyncRead + Unpin,
    T: for<'de> Deserialize<'de>,
{
    let mut len_buf = [0u8; 4];
    match reader.read_exact(&mut len_buf).await {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let len = u32::from_le_bytes(len_buf) as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf).await?;
    let value = serde_json::from_slice(&buf)?;
    Ok(Some(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn request_round_trips_through_framing() {
        let (mut client, mut server) = tokio::io::duplex(4096);
        let request = Request::Run {
            model: "demo-model".to_string(),
            name: Some("demo".to_string()),
            gpu: None,
            policy: None,
            continuity_target_ms: None,
        };
        write_frame(&mut client, &request).await.unwrap();
        let received: Request = read_frame(&mut server).await.unwrap().unwrap();
        match received {
            Request::Run {
                model,
                name,
                gpu,
                policy,
                continuity_target_ms,
            } => {
                assert_eq!(model, "demo-model");
                assert_eq!(name.as_deref(), Some("demo"));
                assert_eq!(gpu, None);
                assert_eq!(policy, None);
                assert_eq!(continuity_target_ms, None);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[tokio::test]
    async fn multiple_frames_are_read_in_order() {
        let (mut client, mut server) = tokio::io::duplex(4096);
        write_frame(&mut client, &ServerMessage::Chunk("hi".to_string()))
            .await
            .unwrap();
        write_frame(&mut client, &ServerMessage::Done { token_count: 1 })
            .await
            .unwrap();

        let first: ServerMessage = read_frame(&mut server).await.unwrap().unwrap();
        let second: ServerMessage = read_frame(&mut server).await.unwrap().unwrap();
        assert!(matches!(first, ServerMessage::Chunk(text) if text == "hi"));
        assert!(matches!(second, ServerMessage::Done { token_count: 1 }));
    }

    #[tokio::test]
    async fn clean_close_reads_as_none() {
        let (client, mut server) = tokio::io::duplex(4096);
        drop(client);
        let result: Option<Request> = read_frame(&mut server).await.unwrap();
        assert!(result.is_none());
    }
}
