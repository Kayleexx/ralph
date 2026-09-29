use super::*;
use crate::engine::Role;

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

#[test]
fn serve_args_enable_sleep_mode() {
    let args = vllm_serve_args("Qwen/Qwen2.5-0.5B-Instruct", 8000, None, None);
    assert!(args.contains(&"--enable-sleep-mode".to_string()));
    assert!(!args.contains(&"--revision".to_string()));
}

#[test]
fn serve_args_include_revision_when_requested() {
    let args = vllm_serve_args("model", 8000, Some("abc123"), None);
    let pos = args.iter().position(|a| a == "--revision").unwrap();
    assert_eq!(args[pos + 1], "abc123");
    let tokenizer = args
        .iter()
        .position(|a| a == "--tokenizer-revision")
        .unwrap();
    assert_eq!(args[tokenizer + 1], "abc123");
}

#[test]
fn measured_launch_sets_context_and_kv_together_without_automatic_pool_sizing() {
    let profile = super::super::profiles::measured(super::super::profiles::QWEN3)
        .unwrap()
        .1[1];
    let args = vllm_serve_args("model", 8000, Some("revision"), Some(profile));
    for (flag, value) in [
        ("--max-model-len", "2048"),
        ("--kv-cache-memory-bytes", "536870912"),
        ("--gpu-memory-utilization", "0.1"),
        ("--dtype", "bfloat16"),
    ] {
        let at = args.iter().position(|a| a == flag).unwrap();
        assert_eq!(args[at + 1], value);
    }
    assert!(args.contains(&"--enforce-eager".into()));
}

#[tokio::test]
async fn is_sleeping_reflects_the_worker_response() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::task::spawn_blocking(move || {
        use std::io::{Read, Write};
        let (mut stream, _) = listener.accept().unwrap();
        let mut buf = [0u8; 8192];
        let _ = stream.read(&mut buf).unwrap();
        let body = "{\"is_sleeping\":true}";
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        let _ = stream.write_all(response.as_bytes());
    });

    let mut engine = VllmEngine::new(PathBuf::from("/dev/null"));
    engine.base_url = format!("http://{addr}");
    assert!(engine.is_sleeping().await);
    server.await.unwrap();
}

#[tokio::test]
async fn is_sleeping_is_false_when_the_worker_is_unreachable() {
    let mut engine = VllmEngine::new(PathBuf::from("/dev/null"));
    engine.base_url = "http://127.0.0.1:1".to_string(); // nothing listens here
    assert!(!engine.is_sleeping().await);
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
    let messages = [ChatMessage {
        role: Role::User,
        content: "hello".to_string(),
    }];
    let mut handle = engine.generate(&messages).await.unwrap();
    while handle.tokens.recv().await.is_some() {}

    let request = server.await.unwrap();
    assert!(request.contains("POST /v1/chat/completions"));
    assert!(request.contains("\"messages\""));
    assert!(!request.contains("\"prompt\":"));
}
