//! Parses vLLM's chat-completion SSE stream into plain text token deltas.
use futures_util::StreamExt;
use tokio::sync::{mpsc, oneshot};

use super::EngineError;

pub(super) async fn stream_completion(
    client: reqwest::Client,
    url: String,
    body: serde_json::Value,
    tx: mpsc::Sender<Result<String, EngineError>>,
    mut cancel_rx: oneshot::Receiver<()>,
) {
    let resp = match client.post(&url).json(&body).send().await {
        Ok(r) => r,
        Err(e) => {
            let _ = tx.send(Err(EngineError::Request(e))).await;
            return;
        }
    };
    if !resp.status().is_success() {
        let status = resp.status();
        let body_text = resp.text().await.unwrap_or_default();
        let _ = tx
            .send(Err(classify_error_response(status, &body_text)))
            .await;
        return;
    }
    let mut stream = resp.bytes_stream();
    let mut buf = String::new();
    loop {
        tokio::select! {
            _ = &mut cancel_rx => return,
            chunk = stream.next() => {
                match chunk {
                    Some(Ok(bytes)) => {
                        buf.push_str(&String::from_utf8_lossy(&bytes));
                        if !forward_complete_events(&mut buf, &tx).await {
                            return;
                        }
                    }
                    Some(Err(e)) => {
                        let _ = tx.send(Err(EngineError::Request(e))).await;
                        return;
                    }
                    None => return,
                }
            }
        }
    }
}

/// Best-effort classification of a non-success completion response. Context-length wording
/// isn't standardized across vLLM versions, so this is a heuristic, not a guarantee — an
/// unrecognized error still reaches the user as `BadResponse` rather than being dropped.
fn classify_error_response(status: reqwest::StatusCode, body: &str) -> EngineError {
    let lower = body.to_lowercase();
    if lower.contains("maximum context length") || lower.contains("context_length_exceeded") {
        let limit = first_number(body).unwrap_or(0);
        return EngineError::ContextWindowExceeded { limit };
    }
    EngineError::BadResponse(format!(
        "{status}: {}",
        body.chars().take(300).collect::<String>()
    ))
}

fn first_number(text: &str) -> Option<u32> {
    let digits: String = text
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

/// Drains complete SSE events (`\n\n`-terminated) out of `buf`, forwarding each chat
/// completion chunk's text delta. Returns false once the receiver is gone or a `[DONE]`
/// marker is seen. The first chunk carries only a role with empty content, and the last
/// carries only a finish reason with no `content` key at all — both are silently skipped.
async fn forward_complete_events(
    buf: &mut String,
    tx: &mpsc::Sender<Result<String, EngineError>>,
) -> bool {
    while let Some(pos) = buf.find("\n\n") {
        let event = buf[..pos].to_string();
        buf.drain(..pos + 2);
        for line in event.lines() {
            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };
            if data == "[DONE]" {
                return false;
            }
            let Ok(json) = serde_json::from_str::<serde_json::Value>(data) else {
                continue;
            };
            let Some(text) = json["choices"][0]["delta"]["content"].as_str() else {
                continue;
            };
            if text.is_empty() {
                continue;
            }
            if tx.send(Ok(text.to_string())).await.is_err() {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn forwards_a_single_chat_delta() {
        let (tx, mut rx) = mpsc::channel(4);
        let mut buf = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n".to_string();
        assert!(forward_complete_events(&mut buf, &tx).await);
        assert_eq!(rx.recv().await.unwrap().unwrap(), "hi");
    }

    #[tokio::test]
    async fn stops_forwarding_on_done_marker() {
        let (tx, mut rx) = mpsc::channel(4);
        let mut buf = "data: [DONE]\n\n".to_string();
        assert!(!forward_complete_events(&mut buf, &tx).await);
        assert!(rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn leaves_incomplete_event_buffered() {
        let (tx, _rx) = mpsc::channel(4);
        let mut buf = "data: {\"choices\":[{\"delta\":{\"content\":\"partial".to_string();
        assert!(forward_complete_events(&mut buf, &tx).await);
        assert!(buf.contains("partial"));
    }

    #[tokio::test]
    async fn skips_role_only_opening_chunk() {
        // The first chat-completion chunk carries the role with empty content — it must
        // not be forwarded as a real (empty) token.
        let (tx, mut rx) = mpsc::channel(4);
        let mut buf =
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n"
                .to_string();
        assert!(forward_complete_events(&mut buf, &tx).await);
        assert!(rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn skips_finish_reason_only_closing_chunk() {
        // The last chunk before [DONE] typically has an empty delta and a finish_reason,
        // with no "content" key at all.
        let (tx, mut rx) = mpsc::channel(4);
        let mut buf =
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n".to_string();
        assert!(forward_complete_events(&mut buf, &tx).await);
        assert!(rx.try_recv().is_err());
    }
}
