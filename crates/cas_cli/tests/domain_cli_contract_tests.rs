//! CLI domain mode contract tests.
//!
//! # Contract: --domain CLI Flag
//!
//! The CLI should support `--domain strict|generic|assume` to control
//! simplification behavior:
//!
//! - `--domain strict`: Only cancel provably nonzero factors
//! - `--domain generic` (default): Legacy behavior (`x/x → 1`)
//! - `--domain assume`: Simplify with warnings/assumptions
//!
//! Wire output should reflect the domain mode in a stable field.

use serde_json::Value;
use std::process::Command;

fn run_cli(args: &[&str]) -> (String, i32) {
    let output = Command::new(env!("CARGO_BIN_EXE_cas_cli"))
        .args(args)
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let code = output.status.code().unwrap_or(-1);
    (stdout, code)
}

fn parse_wire(s: &str) -> Value {
    serde_json::from_str(s).unwrap_or_else(|_| panic!("Failed to parse wire payload: {}", s))
}

// =============================================================================
// CLI --domain flag tests
// =============================================================================

#[test]
fn cli_domain_generic_x_div_x_simplifies_to_1() {
    // Generic (default) => x/x -> 1
    let (output, _code) = run_cli(&["eval", "x/x", "--format", "json"]);
    let wire = parse_wire(&output);

    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "1", "Generic mode should simplify x/x to 1");
}

#[test]
fn cli_domain_strict_x_div_x_stays_unchanged() {
    // Strict => x/x stays as x/x
    let (output, _code) = run_cli(&["eval", "x/x", "--format", "json", "--domain", "strict"]);
    let wire = parse_wire(&output);

    assert_eq!(wire["ok"], true);
    let result = wire["result"].as_str().unwrap_or("");
    assert!(
        result == "x/x" || result == "x / x",
        "Strict mode should NOT simplify x/x, got: {}",
        result
    );
}

#[test]
fn cli_domain_strict_partial_cancel_contract() {
    // Strict auto-eval preserves the residual numeric fraction after cancelling x.
    let (output, _code) = run_cli(&[
        "eval",
        "4*x/(2*x)",
        "--format",
        "json",
        "--domain",
        "strict",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["ok"], true);
    let result = wire["result"].as_str().unwrap_or("");
    assert!(
        result == "4 / 2" || result == "4/2",
        "Expected residual numeric fraction after x cancellation, got: {}",
        result
    );
    let required = wire["required_display"].as_array().unwrap();
    assert_eq!(
        required.len(),
        1,
        "Strict should preserve one domain condition"
    );
    assert_eq!(required[0], "2·x ≠ 0");
}

#[test]
fn cli_domain_assume_emits_warning() {
    // Assume => x/x -> 1 WITH warning
    let (output, _code) = run_cli(&["eval", "x/x", "--format", "json", "--domain", "assume"]);
    let wire = parse_wire(&output);

    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "1", "Assume mode should simplify x/x to 1");

    // Contract: warning present (or warnings[] if using list)
    let has_warning = wire.get("warning").is_some()
        || wire.get("warnings").is_some()
        || wire.get("assumptions").is_some();
    assert!(
        has_warning,
        "Assume mode must emit warning/assumptions field. Wire: {}",
        wire
    );
}

#[test]
fn cli_domain_strict_numeric_preserves_fraction_in_auto_eval() {
    // Strict auto-eval keeps the fraction shape; solver-specific solve simplification is separate.
    let (output, _code) = run_cli(&["eval", "2/2", "--format", "json", "--domain", "strict"]);
    let wire = parse_wire(&output);

    assert_eq!(wire["ok"], true);
    let result = wire["result"].as_str().unwrap_or("");
    assert!(
        result == "2 / 2" || result == "2/2",
        "Strict auto-eval should preserve 2/2, got: {}",
        result
    );
    assert!(
        wire["required_conditions"].as_array().unwrap().is_empty(),
        "Pure numeric fraction should not emit domain conditions"
    );
}

// =============================================================================
// Wire schema: domain field
// =============================================================================

#[test]
fn cli_wire_includes_domain_mode() {
    let (output, _code) = run_cli(&["eval", "x+x", "--format", "json", "--domain", "strict"]);
    let wire = parse_wire(&output);

    // Contract: wire output should include domain.mode field
    assert!(
        wire.get("domain").is_some(),
        "wire output should have 'domain' field"
    );
    assert_eq!(
        wire["domain"]["mode"], "strict",
        "domain.mode should reflect --domain flag"
    );
}

#[test]
fn cli_domain_default_is_generic() {
    // Without --domain flag, should use generic
    let (output, _code) = run_cli(&["eval", "x/x", "--format", "json"]);
    let wire = parse_wire(&output);

    // In generic mode, x/x simplifies to 1
    assert_eq!(wire["ok"], true);
    assert_eq!(
        wire["result"], "1",
        "Default (generic) should simplify x/x to 1"
    );
}
