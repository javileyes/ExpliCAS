//! CLI contract tests for PR1.1 semantic flags.
//!
//! # Contract: Semantic Flags in JSON Output
//!
//! These tests verify that:
//! 1. New flags are reflected in JSON semantics block
//! 2. Defaults are correct (real/strict/principal)

use serde_json::Value;
use std::process::Command;

fn run_cli(args: &[&str]) -> (String, i32) {
    let output = Command::new("cargo")
        .args(["run", "-p", "cas_cli", "--quiet", "--"])
        .args(args)
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let code = output.status.code().unwrap_or(-1);
    (stdout, code)
}

fn parse_json(s: &str) -> Value {
    serde_json::from_str(s).unwrap_or_else(|_| panic!("Failed to parse JSON: {}", s))
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
