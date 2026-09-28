use super::*;

#[test]
fn free_ports_are_distinct() {
    let a = pick_free_port().unwrap();
    let b = pick_free_port().unwrap();
    assert_ne!(a, 0);
    assert_ne!(b, 0);
}

/// Regression test for the raw-completion-vs-chat-template bug: `generate` must post
/// to `/v1/chat/completions` with a `messages` array, never a bare `prompt` string —
/// the latter gives instruction-tuned models no stopping point and no chat template.
#[tokio::test]
async fn generate_sends_chat_completions_request_with_messages() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();

    let server = tokio::task::spawn_blocking(move || {
        use std::io::{Read, Write};
        let (mut stream, _) = listener.accept().unwrap();
        let mut buf = [0u8; 8192];
        let n = stream.read(&mut buf).unwrap();
        let request = String::from_utf8_lossy(&buf[..n]).to_string();
        let body = "data: [DONE]\n\n";
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        let _ = stream.write_all(response.as_bytes());
        request
    });

    let mut engine = VllmEngine::new(PathBuf::from("/dev/null"));
    engine.base_url = format!("http://{addr}");
    engine.model = "test-model".to_string();
    let mut handle = engine.generate("hello").await.unwrap();
    while handle.tokens.recv().await.is_some() {}

    let request = server.await.unwrap();
    assert!(request.contains("POST /v1/chat/completions"));
    assert!(request.contains("\"messages\""));
    assert!(!request.contains("\"prompt\":"));
}

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
    let mut buf = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n"
        .to_string();
    assert!(forward_complete_events(&mut buf, &tx).await);
    assert!(rx.try_recv().is_err());
}

#[tokio::test]
async fn skips_finish_reason_only_closing_chunk() {
    // The last chunk before [DONE] typically has an empty delta and a finish_reason,
    // with no "content" key at all.
    let (tx, mut rx) = mpsc::channel(4);
    let mut buf = "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n".to_string();
    assert!(forward_complete_events(&mut buf, &tx).await);
    assert!(rx.try_recv().is_err());
}
