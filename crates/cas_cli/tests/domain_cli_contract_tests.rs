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
//! JSON output should reflect the domain mode in a stable field.

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
// CLI --domain flag tests
// =============================================================================

#[test]
#[ignore = "pending: implement --domain CLI flag"]
fn cli_domain_generic_x_div_x_simplifies_to_1() {
    // Generic (default) => x/x -> 1
    let (output, _code) = run_cli(&["eval-json", "x/x"]);
    let json = parse_json(&output);

    assert_eq!(json["ok"], true);
    assert_eq!(json["result"], "1", "Generic mode should simplify x/x to 1");
}

#[test]
#[ignore = "pending: implement --domain CLI flag"]
fn cli_domain_strict_x_div_x_stays_unchanged() {
    // Strict => x/x stays as x/x
    let (output, _code) = run_cli(&["eval-json", "x/x", "--domain", "strict"]);
    let json = parse_json(&output);

    assert_eq!(json["ok"], true);
    let result = json["result"].as_str().unwrap_or("");
    assert!(
        result == "x/x" || result == "x / x",
        "Strict mode should NOT simplify x/x, got: {}",
        result
    );
}

#[test]
#[ignore = "pending: implement --domain CLI flag"]
fn cli_domain_strict_partial_cancel_contract() {
    // Strict with partial cancellation: 4x/(2x) → 2x/x (cancel only numeric content)
    let (output, _code) = run_cli(&["eval-json", "4*x/(2*x)", "--domain", "strict"]);
    let json = parse_json(&output);

    assert_eq!(json["ok"], true);
    let result = json["result"].as_str().unwrap_or("");
    // Should have cancelled the 2 but kept x in both num and den
    assert!(
        result.contains("x") && result.contains("/"),
        "Expected fraction with x, got: {}",
        result
    );
    assert_ne!(result, "2", "Strict should not fully simplify to 2");
}

#[test]
#[ignore = "pending: implement --domain CLI flag and assume mode warnings"]
fn cli_domain_assume_emits_warning() {
    // Assume => x/x -> 1 WITH warning
    let (output, _code) = run_cli(&["eval-json", "x/x", "--domain", "assume"]);
    let json = parse_json(&output);

    assert_eq!(json["ok"], true);
    assert_eq!(json["result"], "1", "Assume mode should simplify x/x to 1");

    // Contract: warning present (or warnings[] if using list)
    let has_warning = json.get("warning").is_some()
        || json.get("warnings").is_some()
        || json.get("assumptions").is_some();
    assert!(
        has_warning,
        "Assume mode must emit warning/assumptions field. JSON: {}",
        json
    );
}

#[test]
#[ignore = "pending: implement --domain CLI flag"]
fn cli_domain_strict_numeric_still_works() {
    // Strict: 2/2 -> 1 (numeric is provably nonzero)
    let (output, _code) = run_cli(&["eval-json", "2/2", "--domain", "strict"]);
    let json = parse_json(&output);

    assert_eq!(json["ok"], true);
    assert_eq!(json["result"], "1", "Strict should simplify 2/2 to 1");
}

// =============================================================================
// JSON schema: domain field
// =============================================================================

#[test]
#[ignore = "pending: implement --domain CLI flag with JSON reflection"]
fn cli_json_includes_domain_mode() {
    let (output, _code) = run_cli(&["eval-json", "x+x", "--domain", "strict"]);
    let json = parse_json(&output);

    // Contract: JSON should include domain.mode field
    assert!(
        json.get("domain").is_some(),
        "JSON should have 'domain' field"
    );
    assert_eq!(
        json["domain"]["mode"], "strict",
        "domain.mode should reflect --domain flag"
    );
}

#[test]
#[ignore = "pending: implement --domain CLI flag"]
fn cli_domain_default_is_generic() {
    // Without --domain flag, should use generic
    let (output, _code) = run_cli(&["eval-json", "x/x"]);
    let json = parse_json(&output);

    // In generic mode, x/x simplifies to 1
    assert_eq!(json["ok"], true);
    assert_eq!(
        json["result"], "1",
        "Default (generic) should simplify x/x to 1"
    );
}
