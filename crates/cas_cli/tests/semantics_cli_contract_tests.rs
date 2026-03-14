//! CLI contract tests for PR1.1 semantic flags.
//!
//! # Contract: Semantic Flags in Wire Output
//!
//! These tests verify that:
//! 1. New flags are reflected in the wire semantics block
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

fn parse_wire(s: &str) -> Value {
    // Trim whitespace and find wire JSON content (in case of extra output)
    let trimmed = s.trim();
    serde_json::from_str(trimmed)
        .unwrap_or_else(|e| panic!("Failed to parse wire JSON: {} (error: {})", trimmed, e))
}

// =============================================================================
// Semantic Flags Tests
// =============================================================================

#[test]
fn semantics_block_present_in_json() {
    let (output, _code) = run_cli(&["eval", "1+1", "--format", "json"]);
    let wire = parse_wire(&output);

    assert!(
        wire.get("semantics").is_some(),
        "Wire output should have 'semantics' field"
    );
}

#[test]
fn semantics_defaults_reflected() {
    let (output, _code) = run_cli(&["eval", "1+1", "--format", "json"]);
    let wire = parse_wire(&output);

    let semantics = &wire["semantics"];
    assert_eq!(semantics["domain_mode"], "generic");
    assert_eq!(semantics["value_domain"], "real");
    assert_eq!(semantics["inv_trig"], "strict");
    assert_eq!(semantics["branch"], "principal");
    assert_eq!(semantics["assume_scope"], "real");
}

#[test]
fn value_domain_complex_reflected() {
    let (output, _code) = run_cli(&[
        "eval",
        "1+1",
        "--format",
        "json",
        "--value-domain",
        "complex",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["semantics"]["value_domain"], "complex");
}

#[test]
fn inv_trig_principal_reflected() {
    let (output, _code) = run_cli(&["eval", "1+1", "--format", "json", "--inv-trig", "principal"]);
    let wire = parse_wire(&output);

    assert_eq!(wire["semantics"]["inv_trig"], "principal");
}

#[test]
fn domain_strict_with_semantics() {
    let (output, _code) = run_cli(&["eval", "1+1", "--format", "json", "--domain", "strict"]);
    let wire = parse_wire(&output);

    assert_eq!(wire["semantics"]["domain_mode"], "strict");
    assert_eq!(wire["domain"]["mode"], "strict");
}

// =============================================================================
// AssumeScope Tests (PR-SCOPE-1)
// =============================================================================

#[test]
fn assume_scope_default_reflected() {
    let (output, _code) = run_cli(&["eval", "1+1", "--format", "json"]);
    let wire = parse_wire(&output);

    assert_eq!(
        wire["semantics"]["assume_scope"], "real",
        "assume_scope default should be 'real'"
    );
}

#[test]
fn assume_scope_wildcard_flag_reflected() {
    let (output, _code) = run_cli(&[
        "eval",
        "1+1",
        "--format",
        "json",
        "--assume-scope",
        "wildcard",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(
        wire["semantics"]["assume_scope"], "wildcard",
        "--assume-scope wildcard should be reflected in wire output"
    );
}

#[test]
fn assume_scope_flag_does_not_change_result() {
    // Infrastructure-only: changing assume_scope should NOT change result
    // (behavior changes come in PR-SCOPE-3)
    let (output1, _) = run_cli(&["eval", "x/x", "--format", "json", "--domain", "generic"]);
    let (output2, _) = run_cli(&[
        "eval",
        "x/x",
        "--format",
        "json",
        "--domain",
        "generic",
        "--assume-scope",
        "wildcard",
    ]);

    let wire1 = parse_wire(&output1);
    let wire2 = parse_wire(&output2);

    assert_eq!(
        wire1["result"], wire2["result"],
        "assume_scope flag should not change result (infra only)"
    );
}

// =============================================================================
// ConstFold + Complex semantics regression tests
// =============================================================================

#[test]
fn const_fold_safe_complex_sqrt_negative_reaches_cli_runtime() {
    let (output, _code) = run_cli(&[
        "eval",
        "sqrt(-1)",
        "--format",
        "json",
        "--value-domain",
        "complex",
        "--complex",
        "on",
        "--const-fold",
        "safe",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["options"]["const_fold"], "safe");
    assert_eq!(wire["semantics"]["value_domain"], "complex");
    assert_eq!(wire["result"], "i");
    assert_eq!(
        wire["warnings"].as_array().map(Vec::len),
        Some(0),
        "complex+safe should not emit imaginary-usage warning"
    );
}

#[test]
fn const_fold_safe_real_sqrt_negative_keeps_warning_contract() {
    let (output, _code) = run_cli(&[
        "eval",
        "sqrt(-1)",
        "--format",
        "json",
        "--value-domain",
        "real",
        "--complex",
        "on",
        "--const-fold",
        "safe",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["options"]["const_fold"], "safe");
    assert_eq!(wire["semantics"]["value_domain"], "real");
    assert_eq!(wire["result"], "undefined");

    let warnings = wire["warnings"].as_array().expect("warnings array");
    assert_eq!(warnings.len(), 1, "real+safe should emit one warning");
    assert_eq!(warnings[0]["rule"], "Imaginary Usage Warning");
}

#[test]
fn const_fold_safe_complex_i_squared_reaches_cli_runtime() {
    let (output, _code) = run_cli(&[
        "eval",
        "i^2",
        "--format",
        "json",
        "--value-domain",
        "complex",
        "--complex",
        "on",
        "--const-fold",
        "safe",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["options"]["const_fold"], "safe");
    assert_eq!(wire["semantics"]["value_domain"], "complex");
    assert_eq!(wire["result"], "-1");
}

#[test]
fn complex_principal_inv_trig_warning_surfaces_in_cli_wire() {
    let (output, _code) = run_cli(&[
        "eval",
        "arcsin(sin(x))",
        "--format",
        "json",
        "--value-domain",
        "complex",
        "--inv-trig",
        "principal",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["semantics"]["value_domain"], "complex");
    assert_eq!(wire["semantics"]["inv_trig"], "principal");
    assert_eq!(wire["result"], "x");

    let warnings = wire["warnings"].as_array().expect("warnings array");
    assert_eq!(
        warnings.len(),
        1,
        "principal inverse-trig should emit one warning"
    );
    assert_eq!(warnings[0]["rule"], "Principal Branch Inverse Trig");
}
