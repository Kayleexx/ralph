//! CLI output-contract tests: run the built binary as a real subprocess and check the
//! `--json`/`--no-color`/human-mode contracts hold. These deliberately avoid `ralph run`
//! (which needs a real GPU) — the real-vLLM path is covered separately by the gated
//! end-to-end test.
//!
//! Each test gets its own isolated `XDG_DATA_HOME`, so it gets its own daemon and never
//! touches a real one. That daemon is auto-started and left running for the OS to reap
//! when the test's temp directory's socket path stops being reachable — there is no
//! `ralph daemon stop` in this phase to tear it down explicitly.
use std::process::Command as StdCommand;

use assert_cmd::Command;
use tempfile::TempDir;

fn isolated_home() -> TempDir {
    tempfile::tempdir().unwrap()
}

fn ralph(home: &TempDir) -> Command {
    let mut cmd = Command::cargo_bin("ralph").unwrap();
    cmd.env("XDG_DATA_HOME", home.path());
    cmd
}

#[test]
fn ps_json_on_empty_state_is_an_empty_array() {
    let home = isolated_home();
    let output = ralph(&home).args(["--json", "ps"]).output().unwrap();
    assert!(output.status.success());
    let parsed: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(parsed, serde_json::json!([]));
}

#[test]
fn ps_human_on_empty_state_prints_hint() {
    let home = isolated_home();
    let output = ralph(&home).arg("ps").output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("no sessions yet"));
    assert!(stdout.contains("ralph run"));
}

#[test]
fn no_color_never_emits_ansi_escapes() {
    let home = isolated_home();
    let output = ralph(&home).args(["--no-color", "ps"]).output().unwrap();
    assert!(!String::from_utf8_lossy(&output.stdout).contains('\u{1b}'));

    let home2 = isolated_home();
    let output2 = ralph(&home2)
        .args(["--no-color", "inspect", "missing"])
        .output()
        .unwrap();
    assert!(!String::from_utf8_lossy(&output2.stderr).contains('\u{1b}'));
}

#[test]
fn inspect_unknown_session_exits_3_with_json_error_on_stdout() {
    let home = isolated_home();
    let output = ralph(&home)
        .args(["--json", "inspect", "does-not-exist"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(3));
    let parsed: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(parsed["error"]["exit_code"], 3);
    // JSON mode: nothing but the JSON object on stdout.
    assert!(
        String::from_utf8_lossy(&output.stdout)
            .trim_end()
            .ends_with('}')
    );
}

#[test]
fn inspect_unknown_session_human_mode_prints_envelope_to_stderr() {
    let home = isolated_home();
    let output = ralph(&home)
        .args(["inspect", "does-not-exist"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(3));
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("no session found"));
}

#[test]
fn query_unknown_session_exits_3() {
    let home = isolated_home();
    let output = ralph(&home)
        .args(["query", "does-not-exist", "hello"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(3));
}

#[test]
fn unrecognized_subcommand_is_a_usage_error() {
    let home = isolated_home();
    let output = ralph(&home).arg("not-a-real-command").output().unwrap();
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn run_without_model_in_non_interactive_mode_is_a_concise_error() {
    // `.output()` gives the child piped (non-TTY) stdin/stdout, so this exercises the
    // same "not a real terminal" path a script or CI run would hit — it must fail fast
    // with a clear error rather than hang waiting for interactive picker input.
    let home = isolated_home();
    let output = ralph(&home).arg("run").output().unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("no model specified"));
}

#[test]
fn run_without_model_json_mode_is_a_concise_json_error() {
    let home = isolated_home();
    let output = ralph(&home).args(["--json", "run"]).output().unwrap();
    assert_eq!(output.status.code(), Some(2));
    let parsed: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(parsed["error"]["exit_code"], 2);
}

#[test]
fn doctor_json_is_an_array_of_check_results() {
    let home = isolated_home();
    let output = ralph(&home).args(["--json", "doctor"]).output().unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let checks = parsed.as_array().expect("doctor --json prints an array");
    assert!(!checks.is_empty());
    for check in checks {
        assert!(check["name"].is_string());
        assert!(check["status"].is_string());
    }
}

#[test]
fn doctor_never_mutates_the_data_directory_before_it_exists() {
    let home = isolated_home();
    ralph(&home).args(["--json", "doctor"]).output().unwrap();
    let ralph_dir = home.path().join("ralph");
    assert!(
        !ralph_dir.exists(),
        "doctor must not create the data directory on a fresh machine"
    );
}

// Sanity check that the binary under test actually exists and is executable directly,
// independent of assert_cmd's own resolution, to fail fast with a clear message if the
// build step that should have produced it didn't run.
#[test]
fn binary_is_built_and_runs() {
    let path = assert_cmd::cargo::cargo_bin("ralph");
    let status = StdCommand::new(path).arg("--help").status().unwrap();
    assert!(status.success());
}
