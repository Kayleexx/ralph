use super::*;

#[test]
fn free_ports_are_distinct() {
    let a = pick_free_port().unwrap();
    let b = pick_free_port().unwrap();
    assert_ne!(a, 0);
    assert_ne!(b, 0);
}

/// Regression test: `ralph` is installed globally and run from wherever the user
/// happens to be, so vLLM discovery must not depend solely on the current directory
/// happening to contain a `.venv`.
#[test]
fn resolve_vllm_binary_prefers_activated_virtual_env_over_cwd() {
    let venv = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(venv.path().join("bin")).unwrap();
    std::fs::write(venv.path().join("bin/vllm"), "").unwrap();

    let missing_cwd_venv = Path::new("/nonexistent/for/this/test/.venv/bin/vllm");
    let resolved =
        resolve_vllm_binary_from(Some(venv.path().display().to_string()), missing_cwd_venv);
    assert_eq!(resolved, venv.path().join("bin/vllm"));
}

#[test]
fn resolve_vllm_binary_falls_back_to_cwd_venv_when_virtual_env_unset() {
    let dir = tempfile::tempdir().unwrap();
    let cwd_venv = dir.path().join(".venv/bin/vllm");
    std::fs::create_dir_all(cwd_venv.parent().unwrap()).unwrap();
    std::fs::write(&cwd_venv, "").unwrap();

    let resolved = resolve_vllm_binary_from(None, &cwd_venv);
    assert_eq!(resolved, cwd_venv);
}

#[test]
fn resolve_vllm_binary_falls_back_to_path_when_nothing_else_exists() {
    let missing = Path::new("/nonexistent/for/this/test/.venv/bin/vllm");
    let resolved = resolve_vllm_binary_from(None, missing);
    assert_eq!(resolved, PathBuf::from("vllm"));
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
