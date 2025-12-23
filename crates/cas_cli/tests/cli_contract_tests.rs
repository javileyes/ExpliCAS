//! CLI contract tests for the unified budget system and eval command.
//!
//! These tests validate the CLI behavior including:
//! - Help output shows correct commands
//! - JSON output parsing and schema version
//! - Budget presets and strict mode

use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::Value;

/// Get the CLI command
#[allow(deprecated)]
fn cli() -> Command {
    Command::cargo_bin("cas_cli").unwrap()
}

/// Test that help output shows expected commands (eval, repl) and hides aliases (eval-json).
#[test]
fn test_help_shows_correct_commands() {
    cli()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("eval"))
        .stdout(predicate::str::contains("repl"))
        .stdout(predicate::str::contains("help"))
        // eval-json should be hidden (not shown in help)
        .stdout(predicate::str::contains("eval-json").not());
}

/// Test that eval command help shows budget options.
#[test]
fn test_eval_help_shows_budget_options() {
    cli()
        .args(["eval", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--budget"))
        .stdout(predicate::str::contains("--strict"))
        .stdout(predicate::str::contains("--format"))
        .stdout(predicate::str::contains("small"))
        .stdout(predicate::str::contains("cli"))
        .stdout(predicate::str::contains("unlimited"));
}

/// Test that eval with --format json produces valid JSON with schema_version.
#[test]
fn test_eval_json_output_has_schema_version() {
    let output = cli()
        .args(["eval", "x+1", "--format", "json"])
        .output()
        .expect("Failed to run CLI");

    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).unwrap();
    let json: Value = serde_json::from_str(&stdout).expect("Invalid JSON output");

    assert_eq!(json["schema_version"], 1);
    assert_eq!(json["ok"], true);
    assert!(json["budget"].is_object());
    assert_eq!(json["budget"]["preset"], "cli");
    assert_eq!(json["budget"]["mode"], "best-effort");
}

/// Test that eval-json alias works (backward compatibility).
#[test]
fn test_eval_json_alias_works() {
    let output = cli()
        .args(["eval-json", "2+2"])
        .output()
        .expect("Failed to run CLI");

    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).unwrap();
    let json: Value = serde_json::from_str(&stdout).expect("Invalid JSON output");

    assert_eq!(json["ok"], true);
    assert_eq!(json["result"], "4");
}

/// Test that budget presets can be selected.
#[test]
fn test_eval_with_budget_preset() {
    let output = cli()
        .args(["eval", "x+1", "--format", "json", "--budget", "small"])
        .output()
        .expect("Failed to run CLI");

    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).unwrap();
    let json: Value = serde_json::from_str(&stdout).expect("Invalid JSON output");

    assert_eq!(json["budget"]["preset"], "small");
}

/// Test that --strict flag sets mode to strict.
#[test]
fn test_eval_strict_mode() {
    let output = cli()
        .args(["eval", "x+1", "--format", "json", "--strict"])
        .output()
        .expect("Failed to run CLI");

    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).unwrap();
    let json: Value = serde_json::from_str(&stdout).expect("Invalid JSON output");

    assert_eq!(json["budget"]["mode"], "strict");
}

/// Test that text format output works.
#[test]
fn test_eval_text_format() {
    cli()
        .args(["eval", "2+2"])
        .assert()
        .success()
        .stdout(predicate::str::contains("4"));
}
