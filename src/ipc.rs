//! CLI <-> daemon wire protocol: one JSON value per frame, framed with a 4-byte
//! little-endian length prefix over a Unix domain socket. No RPC framework — the
//! command surface is small enough that this is simpler than pulling one in.
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

#[derive(Debug, Serialize, Deserialize)]
pub enum Request {
    Run {
        model: String,
        name: Option<String>,
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
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ResumeInfo {
    pub session: SessionInfo,
    /// Best-effort, honest-only: `false` unless a still-compatible KV directory with
    /// actual content was found — never claims a native restore that didn't happen.
    pub native: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InspectInfo {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_failure: Option<String>,
    pub session: SessionInfo,
    pub recoverability: String,
    pub fast_restore: String,
    pub portable_state: String,
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
    Error(ErrorPayload),
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
        };
        write_frame(&mut client, &request).await.unwrap();
        let received: Request = read_frame(&mut server).await.unwrap().unwrap();
        match received {
            Request::Run { model, name } => {
                assert_eq!(model, "demo-model");
                assert_eq!(name.as_deref(), Some("demo"));
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
