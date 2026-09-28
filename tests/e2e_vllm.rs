//! RALPH_E2E_VLLM=1 cargo test --test e2e_vllm -- --ignored --test-threads=1
mod support;
use std::time::Duration;
use support::*;

fn enabled() -> bool {
    std::env::var("RALPH_E2E_VLLM").as_deref() == Ok("1")
}

#[test]
#[ignore]
fn crash_before_query_after_exchange_and_daemon_restart() {
    if !enabled() {
        return;
    }
    let mut h = Harness::new();
    let original = h.start();
    h.kill_worker();
    h.wait_loss();
    assert!(h.turns().is_empty());
    let recovered = h.recover();
    assert_eq!(original["id"], recovered["id"]);
    h.json(&[
        "query",
        SESSION,
        "Remember the secret word pineapple. Reply with that word only.",
    ]);
    let committed = h.turns();
    assert_eq!(committed.len(), 2);
    h.kill_worker();
    h.wait_loss();
    h.recover();
    assert_eq!(h.turns(), committed);
    let reply = h.json(&[
        "query",
        SESSION,
        "What secret word did I ask you to remember? One word only.",
    ]);
    assert!(
        reply["text"]
            .as_str()
            .unwrap()
            .to_lowercase()
            .contains("pineapple"),
        "history was not reconstructed: {reply}"
    );
    h.restart_daemon();
    assert_eq!(h.inspect()["session"]["state"], "recovering");
    let recovered = h.json(&["recover", SESSION]);
    h.remember_worker();
    assert_eq!(recovered["id"], original["id"]);
    assert_eq!(h.turns().len(), 4);
    let mut query = h.spawn_query("Write a very long numbered list of 2000 facts about oceans.");
    wait_until(Duration::from_secs(15), || h.turns().len() == 6);
    std::process::Command::new("kill")
        .args(["-INT", &query.id().to_string()])
        .status()
        .unwrap();
    assert_eq!(query.wait().unwrap().code(), Some(1));
    assert_eq!(h.inspect()["session"]["state"], "active");
    h.json(&["query", SESSION, "Say hello in one word."]);
    let mut disconnected = h.spawn_query("Write another very long numbered list of ocean facts.");
    wait_until(Duration::from_secs(15), || h.turns().len() == 10);
    disconnected.kill().unwrap();
    disconnected.wait().unwrap();
    wait_until(Duration::from_secs(10), || {
        let out = h.output(&["query", SESSION, "Say goodbye in one word."]);
        out.status.success()
    });
    assert_eq!(h.inspect()["session"]["id"], original["id"]);
    assert_eq!(h.turns().len(), 12);
}

#[test]
#[ignore]
fn crash_before_first_token_partial_flush_and_restart_during_recovery() {
    if !enabled() {
        return;
    }
    let mut h = Harness::new();
    let original = h.start();
    let prompt = format!(
        "{}\nSummarize that in one word.",
        "The sky is blue. ".repeat(1800)
    );
    let output_path = h.home.path().join("before-first-token.out");
    let mut query = h
        .command()
        .args(["query", SESSION, &prompt])
        .stdout(std::fs::File::create(&output_path).unwrap())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    wait_until(Duration::from_secs(15), || h.turns().len() == 2);
    assert!(
        h.turns()[1].2.is_empty(),
        "crash must occur before the first durable output"
    );
    assert_eq!(std::fs::metadata(&output_path).unwrap().len(), 0);
    h.kill_worker();
    assert!(!query.wait().unwrap().success());
    h.wait_loss();
    wait_until(Duration::from_secs(10), || {
        h.remember_and_has_replacement(&original)
    });
    h.restart_daemon();
    assert_eq!(h.inspect()["session"]["state"], "recovering");
    h.json(&["recover", SESSION]);
    h.remember_worker();
    h.json(&["query", SESSION, "Say hello in one word."]);
    let mut query=h.spawn_query("Write a numbered list of 2000 short facts, with a full sentence for every item. Keep going without summarizing.");
    wait_until(Duration::from_secs(30), || {
        h.turns()
            .last()
            .is_some_and(|t| t.1 == "assistant" && !t.2.is_empty() && t.3 == 0)
    });
    let before = h.turns();
    h.kill_worker();
    assert!(!query.wait().unwrap().success());
    h.wait_loss();
    h.recover();
    let after = h.turns();
    assert_eq!(after.len(), before.len());
    assert!(
        after
            .last()
            .unwrap()
            .2
            .starts_with(&before.last().unwrap().2)
    );
    h.json(&[
        "query",
        SESSION,
        "The list is finished. What is 2+2? Reply with 4 only.",
    ]);
    assert_eq!(h.turns().len(), after.len() + 2);
    assert_eq!(h.inspect()["session"]["id"], original["id"]);
}

#[test]
#[ignore]
fn cancelled_recovery_and_bounded_startup_failure_preserve_sessions() {
    if !enabled() {
        return;
    }
    let mut h = Harness::new();
    let original = h.start();
    h.restart_daemon();
    let mut recover = h
        .command()
        .args(["recover", SESSION])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    wait_until(Duration::from_secs(15), || h.worker().is_some());
    h.remember_worker();
    std::process::Command::new("kill")
        .args(["-INT", &recover.id().to_string()])
        .status()
        .unwrap();
    assert!(!recover.wait().unwrap().success());
    wait_until(Duration::from_secs(10), || {
        h.inspect()["session"]["state"] == "stopped" && h.worker().is_none()
    });
    let started = std::time::Instant::now();
    let out = h.output(&[
        "--json",
        "run",
        "ralph-test/this-model-does-not-exist",
        "--name",
        "startup-failure",
    ]);
    assert!(!out.status.success());
    assert!(started.elapsed() < Duration::from_secs(300));
    let failure: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(
        failure["error"]["summary"]
            .as_str()
            .unwrap()
            .contains("ralph-test/this-model-does-not-exist")
    );
    assert!(
        failure["error"]["detail"]
            .as_array()
            .unwrap()
            .iter()
            .any(|line| line.as_str().unwrap().contains("stopped"))
    );
    assert_eq!(
        h.json(&["inspect", "startup-failure"])["session"]["state"],
        "stopped"
    );
    let recovered = h.json(&["recover", SESSION]);
    h.remember_worker();
    assert_eq!(recovered["id"], original["id"]);
    h.json(&["query", SESSION, "What is 2+2? Reply with 4 only."]);
    let conflicting = h.output(&[
        "--json",
        "run",
        "Qwen/Qwen3-0.6B",
        "--name",
        "conflicting-model",
    ]);
    assert_eq!(conflicting.status.code(), Some(6));
    assert_eq!(
        h.json(&["inspect", "conflicting-model"])["session"]["state"],
        "stopped"
    );
    assert_eq!(h.inspect()["session"]["state"], "active");
}
