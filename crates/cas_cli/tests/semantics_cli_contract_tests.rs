//! CLI contract tests for PR1.1 semantic flags.
//!
//! # Contract: Semantic Flags in JSON Output
//!
//! These tests verify that:
//! 1. New flags are reflected in JSON semantics block
//! 2. Defaults are correct (real/strict/principal)

use serde_json::Value;
use std::process::Command;

/// Run the CLI binary directly (not via `cargo run`) for stable test execution.
/// Uses CARGO_BIN_EXE_cas_cli set automatically by Cargo for integration tests.
fn run_cli(args: &[&str]) -> (String, i32) {
    // Get the binary path from the environment variable set by Cargo
    let bin_path = env!("CARGO_BIN_EXE_cas_cli");

    let output = Command::new(bin_path)
        .args(args)
        .output()
        .expect("Failed to execute binary");

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let code = output.status.code().unwrap_or(-1);
    (stdout, code)
}

fn parse_json(s: &str) -> Value {
    // Trim whitespace and find JSON content (in case of extra output)
    let trimmed = s.trim();
    serde_json::from_str(trimmed)
        .unwrap_or_else(|e| panic!("Failed to parse JSON: {} (error: {})", trimmed, e))
}

// =============================================================================
// Semantic Flags Tests
// =============================================================================

#[test]
fn semantics_block_present_in_json() {
    let (output, _code) = run_cli(&["eval-json", "1+1"]);
    let json = parse_json(&output);

    assert!(
        json.get("semantics").is_some(),
        "JSON should have 'semantics' field"
    );
}

#[test]
fn semantics_defaults_reflected() {
    let (output, _code) = run_cli(&["eval-json", "1+1"]);
    let json = parse_json(&output);

    let semantics = &json["semantics"];
    assert_eq!(semantics["domain_mode"], "generic");
    assert_eq!(semantics["value_domain"], "real");
    assert_eq!(semantics["inv_trig"], "strict");
    assert_eq!(semantics["branch"], "principal");
    assert_eq!(semantics["assume_scope"], "real");
}

#[test]
fn value_domain_complex_reflected() {
    let (output, _code) = run_cli(&["eval-json", "1+1", "--value-domain", "complex"]);
    let json = parse_json(&output);

    assert_eq!(json["semantics"]["value_domain"], "complex");
}

#[test]
fn inv_trig_principal_reflected() {
    let (output, _code) = run_cli(&["eval-json", "1+1", "--inv-trig", "principal"]);
    let json = parse_json(&output);

    assert_eq!(json["semantics"]["inv_trig"], "principal");
}

#[test]
fn domain_strict_with_semantics() {
    let (output, _code) = run_cli(&["eval-json", "1+1", "--domain", "strict"]);
    let json = parse_json(&output);

    assert_eq!(json["semantics"]["domain_mode"], "strict");
    assert_eq!(json["domain"]["mode"], "strict");
}

// =============================================================================
// AssumeScope Tests (PR-SCOPE-1)
// =============================================================================

#[test]
fn assume_scope_default_reflected() {
    let (output, _code) = run_cli(&["eval-json", "1+1"]);
    let json = parse_json(&output);

    assert_eq!(
        json["semantics"]["assume_scope"], "real",
        "assume_scope default should be 'real'"
    );
}

#[test]
fn assume_scope_wildcard_flag_reflected() {
    let (output, _code) = run_cli(&["eval-json", "1+1", "--assume-scope", "wildcard"]);
    let json = parse_json(&output);

    assert_eq!(
        json["semantics"]["assume_scope"], "wildcard",
        "--assume-scope wildcard should be reflected in JSON"
    );
}

#[test]
fn assume_scope_flag_does_not_change_result() {
    // Infrastructure-only: changing assume_scope should NOT change result
    // (behavior changes come in PR-SCOPE-3)
    let (output1, _) = run_cli(&["eval-json", "x/x", "--domain", "generic"]);
    let (output2, _) = run_cli(&[
        "eval-json",
        "x/x",
        "--domain",
        "generic",
        "--assume-scope",
        "wildcard",
    ]);

    let json1 = parse_json(&output1);
    let json2 = parse_json(&output2);

    assert_eq!(
        json1["result"], json2["result"],
        "assume_scope flag should not change result (infra only)"
    );
}
