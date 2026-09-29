//! RALPH_E2E_VLLM=1 cargo test --test e2e_handoff_drain -- --ignored --test-threads=1
//!
//! Real, unmocked vLLM — validates `ralph drain` (fully local, always runs) and
//! `ralph handoff` (needs a real SSH destination) on top of the daemon-side plumbing
//! already covered by unit tests in `src/daemon/tests_handoff.rs`/`tests_drain.rs`.
//!
//! The handoff test targets a real SSH destination rather than trying to smuggle a
//! distinct `XDG_DATA_HOME` through an unconfigured loopback SSH session (which most
//! sshd installs won't pass through anyway) — set RALPH_E2E_HANDOFF_DEST to an
//! "ssh"-reachable host with `ralph` on its PATH whose sshd accepts XDG_DATA_HOME
//! (e.g. `AcceptEnv XDG_DATA_HOME` for a loopback test account), or a real second
//! machine. Unset, the handoff test skips cleanly — the same spirit as the multi-GPU
//! tests skipping without a second GPU.
mod support;
use support::*;

fn enabled() -> bool {
    std::env::var("RALPH_E2E_VLLM").as_deref() == Ok("1")
}

fn handoff_dest() -> Option<String> {
    std::env::var("RALPH_E2E_HANDOFF_DEST").ok()
}

#[test]
#[ignore]
fn drain_hibernates_active_sessions_and_they_resume_normally() {
    if !enabled() {
        return;
    }
    let mut h = Harness::new();
    h.start();
    h.json(&[
        "query",
        SESSION,
        "Remember the secret word papaya. Reply with that word only.",
    ]);

    let report = h.drain("local/gpu0", None, true);
    assert_eq!(report["all_safe"], true);
    assert_eq!(report["sessions"][0]["action"], "hibernate");
    assert_eq!(h.inspect()["session"]["state"], "hibernated");

    h.resume();
    assert_eq!(h.inspect()["session"]["state"], "active");
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
            .contains("papaya"),
        "context lost across drain/resume: {reply}"
    );
}

#[test]
#[ignore]
fn drain_without_yes_only_prints_the_plan() {
    if !enabled() {
        return;
    }
    let mut h = Harness::new();
    h.start();

    let report = h.drain("local/gpu0", None, false);
    assert_eq!(report["executed"], false);
    assert_eq!(h.inspect()["session"]["state"], "active");
}

#[test]
#[ignore]
fn handoff_moves_a_session_and_it_resumes_on_the_destination() {
    if !enabled() {
        return;
    }
    let Some(dest) = handoff_dest() else {
        eprintln!(
            "skipping: set RALPH_E2E_HANDOFF_DEST to a reachable ssh destination to run this test"
        );
        return;
    };
    let mut h = Harness::new();
    h.start();
    h.json(&[
        "query",
        SESSION,
        "Remember the secret word cardamom. Reply with that word only.",
    ]);

    let output = h.handoff_json(&dest, Some("handoff-e2e"));
    assert!(
        output.status.success(),
        "handoff failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(h.inspect()["session"]["state"], "moved");

    // Source refuses to run the same logical session again — no dual ownership.
    let recover = h.output(&["recover", SESSION]);
    assert!(!recover.status.success());

    // On the destination, the arrived session resumes and recalls context normally —
    // an ordinary local CLI invocation against B's own data root, no SSH needed here
    // since verification happens on the machine the archive actually landed on.
    let dest_home = std::env::var("RALPH_E2E_HANDOFF_HOME").expect(
        "RALPH_E2E_HANDOFF_HOME must name the destination's XDG_DATA_HOME for verification",
    );
    let mut cmd = std::process::Command::new(assert_cmd::cargo::cargo_bin("ralph"));
    cmd.env("XDG_DATA_HOME", &dest_home);
    let resumed = cmd
        .args(["--json", "resume", "handoff-e2e"])
        .output()
        .unwrap();
    assert!(resumed.status.success());
    let mut cmd = std::process::Command::new(assert_cmd::cargo::cargo_bin("ralph"));
    cmd.env("XDG_DATA_HOME", &dest_home);
    let reply = cmd
        .args([
            "--json",
            "query",
            "handoff-e2e",
            "What secret word did I ask you to remember? One word only.",
        ])
        .output()
        .unwrap();
    let reply: serde_json::Value = serde_json::from_slice(&reply.stdout).unwrap();
    assert!(
        reply["text"]
            .as_str()
            .unwrap()
            .to_lowercase()
            .contains("cardamom"),
        "context lost across handoff: {reply}"
    );
}

#[test]
#[ignore]
fn handoff_to_an_unreachable_destination_leaves_the_source_active() {
    if !enabled() {
        return;
    }
    let mut h = Harness::new();
    h.start();

    let output = h.handoff_json("unreachable.invalid", None);
    assert!(!output.status.success());
    assert_eq!(h.inspect()["session"]["state"], "active");
    h.json(&["query", SESSION, "still usable"]);
}
