//! RALPH_E2E_VLLM=1 cargo test --test e2e_export_import -- --ignored --test-threads=1
//!
//! Real, unmocked vLLM — validates `ralph export`/`ralph import` actually reconstruct a
//! usable session, on top of the daemon-side plumbing already covered by unit tests in
//! `src/daemon/tests_portable.rs`.
mod support;
use support::*;

fn enabled() -> bool {
    std::env::var("RALPH_E2E_VLLM").as_deref() == Ok("1")
}

#[test]
#[ignore]
fn export_import_round_trip_with_and_without_accel() {
    if !enabled() {
        return;
    }
    let h = {
        let mut h = Harness::new();
        h.start();
        h
    };
    h.json(&[
        "query",
        SESSION,
        "Remember the secret word papaya. Reply with that word only.",
    ]);
    h.checkpoint();

    // With accel: the imported copy's fast-vs-portable decision depends on this same
    // machine's fingerprint still matching, which it does here.
    let with_accel = h.home.path().join("with-accel.ralph");
    let exported = h.export(&with_accel, true);
    assert_eq!(exported["session"]["name"], SESSION);
    let imported = h.import(&with_accel, "papaya-copy");
    assert_eq!(imported["state"], "paused");

    let resumed = h.resume_named("papaya-copy");
    assert_eq!(resumed["session"]["state"], "active");
    let reply = h.query_named(
        "papaya-copy",
        "What secret word did I ask you to remember? One word only.",
    );
    assert!(
        reply["text"]
            .as_str()
            .unwrap()
            .to_lowercase()
            .contains("papaya"),
        "context lost across export/import: {reply}"
    );

    // Without accel: nothing to carry natively, so resume must honestly report portable.
    let without_accel = h.home.path().join("without-accel.ralph");
    h.export(&without_accel, false);
    let imported2 = h.import(&without_accel, "papaya-copy-2");
    assert_eq!(imported2["state"], "paused");
    let resumed2 = h.resume_named("papaya-copy-2");
    assert_eq!(
        resumed2["native"], false,
        "no accel was exported: {resumed2}"
    );

    // A corrupted artifact must fail import cleanly, with no new session created.
    let mut bytes = std::fs::read(&without_accel).unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xff;
    let corrupt = h.home.path().join("corrupt.ralph");
    std::fs::write(&corrupt, &bytes).unwrap();
    let output = h
        .command()
        .args([
            "--json",
            "import",
            &corrupt.to_string_lossy(),
            "--name",
            "should-not-exist",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert_eq!(output.status.code(), Some(8));
    let sessions = h.json(&["ps"]);
    assert!(
        sessions
            .as_array()
            .unwrap()
            .iter()
            .all(|s| s["name"] != "should-not-exist")
    );
}
