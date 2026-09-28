//! Parses vLLM's chat-completion SSE stream into plain text token deltas.
use futures_util::StreamExt;
use tokio::sync::{mpsc, oneshot};

use super::EngineError;
use crate::engine::TokenChunk;

pub(super) async fn stream_completion(
    client: reqwest::Client,
    url: String,
    body: serde_json::Value,
    tx: mpsc::Sender<Result<TokenChunk, EngineError>>,
    mut cancel_rx: oneshot::Receiver<()>,
) {
    let request = client.post(&url).json(&body).send();
    let response = tokio::select! { _ = &mut cancel_rx => return, _ = tx.closed() => return, response = request => response };
    let resp = match response {
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
    let mut bytes_buf = Vec::new();
    let mut buf = String::new();
    loop {
        tokio::select! {
            _ = &mut cancel_rx => return,
            chunk = stream.next() => {
                match chunk {
                    Some(Ok(bytes)) => {
                        bytes_buf.extend_from_slice(&bytes);
                        let valid = match std::str::from_utf8(&bytes_buf) {
                            Ok(text) => text.len(),
                            Err(error) if error.error_len().is_none() => error.valid_up_to(),
                            Err(_) => { let _ = tx.send(Err(EngineError::BadResponse("invalid UTF-8 stream".into()))).await; return; }
                        };
                        if let Ok(text) = std::str::from_utf8(&bytes_buf[..valid]) { buf.push_str(text); }
                        bytes_buf.drain(..valid);
                        if !forward_complete_events(&mut buf, &tx).await {
                            return;
                        }
                    }
                    Some(Err(e)) => {
                        let _ = tx.send(Err(EngineError::Request(e))).await;
                        return;
                    }
                    None => { let _ = tx.send(Err(EngineError::BadResponse("stream ended before completion marker".into()))).await; return; },
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
    tx: &mpsc::Sender<Result<TokenChunk, EngineError>>,
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
            let json = match serde_json::from_str::<serde_json::Value>(data) {
                Ok(json) => json,
                Err(_) => {
                    let _ = tx
                        .send(Err(EngineError::BadResponse(
                            "malformed stream event".into(),
                        )))
                        .await;
                    return false;
                }
            };
            if json.get("error").is_some() {
                let _ = tx
                    .send(Err(EngineError::BadResponse(
                        "engine reported a stream error".into(),
                    )))
                    .await;
                return false;
            }
            let choice = &json["choices"][0];
            let text = choice["delta"]["content"].as_str().unwrap_or("");
            let ids: Vec<u32> = match choice.get("token_ids").filter(|v| !v.is_null()) {
                Some(ids) => match serde_json::from_value(ids.clone()) {
                    Ok(ids) => ids,
                    Err(_) => {
                        let _ = tx
                            .send(Err(EngineError::BadResponse(
                                "invalid streamed token IDs".into(),
                            )))
                            .await;
                        return false;
                    }
                },
                None if text.is_empty() => continue,
                None => {
                    let _ = tx
                        .send(Err(EngineError::BadResponse(
                            "stream omitted requested token IDs".into(),
                        )))
                        .await;
                    return false;
                }
            };
            if text.is_empty() && ids.is_empty() {
                continue;
            }
            if tx
                .send(Ok(TokenChunk {
                    text: text.into(),
                    ids,
                }))
                .await
                .is_err()
            {
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
        let mut buf =
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"token_ids\":[1]}]}\n\n"
                .to_string();
        assert!(forward_complete_events(&mut buf, &tx).await);
        assert_eq!(rx.recv().await.unwrap().unwrap().text, "hi");
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
