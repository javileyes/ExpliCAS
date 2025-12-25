//! CLI JSON API integration tests.
//!
//! These tests verify that the CLI returns proper JSON with stable kind/code.

use std::process::Command;

use serde_json::Value;

fn run_cli(args: &[&str]) -> (String, i32) {
    // Use cargo run to invoke the CLI
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
// Success cases
// =============================================================================

#[test]
fn test_eval_json_success() {
    let (output, _code) = run_cli(&["eval-json", "2+2"]);
    let json = parse_json(&output);

    assert_eq!(json["schema_version"], 1);
    assert_eq!(json["ok"], true);
    assert!(json["result"].is_string());
}

#[test]
fn test_eval_json_success_has_budget() {
    let (output, _code) = run_cli(&["eval-json", "x+x"]);
    let json = parse_json(&output);

    assert!(json["budget"].is_object(), "Should have budget object");
    assert!(json["budget"]["preset"].is_string());
    assert!(json["budget"]["mode"].is_string());
}

// =============================================================================
// Error cases with kind/code
// =============================================================================

#[test]
fn test_eval_json_parse_error_has_kind_code() {
    let (output, _code) = run_cli(&["eval-json", "("]);
    let json = parse_json(&output);

    assert_eq!(json["ok"], false);
    assert_eq!(json["kind"], "ParseError", "Should have kind=ParseError");
    assert_eq!(json["code"], "E_PARSE", "Should have code=E_PARSE");
    assert!(json["error"].is_string(), "Should have error message");
}

#[test]
fn test_eval_json_error_kind_in_known_set() {
    // Parse error
    let (output, _code) = run_cli(&["eval-json", "((("]);
    let json = parse_json(&output);

    let valid_kinds = [
        "ParseError",
        "DomainError",
        "SolverError",
        "BudgetExceeded",
        "NotImplemented",
        "InternalError",
    ];

    if !json["ok"].as_bool().unwrap_or(true) {
        let kind = json["kind"].as_str().unwrap_or("");
        assert!(
            valid_kinds.contains(&kind),
            "Kind '{}' should be in known set",
            kind
        );
    }
}

#[test]
fn test_eval_json_error_code_starts_with_e() {
    let (output, _code) = run_cli(&["eval-json", "((("]);
    let json = parse_json(&output);

    if !json["ok"].as_bool().unwrap_or(true) {
        let code = json["code"].as_str().unwrap_or("");
        assert!(
            code.starts_with("E_"),
            "Code '{}' should start with E_",
            code
        );
    }
}

// =============================================================================
// No __hold leaks
// =============================================================================

#[test]
fn test_eval_json_no_hold_in_result() {
    // Complex expression that might internally use __hold
    let (output, _code) = run_cli(&["eval-json", "(x+1)^2 - (x+1)^2"]);
    let json = parse_json(&output);

    if let Some(result) = json["result"].as_str() {
        assert!(
            !result.contains("__hold"),
            "Result should not contain __hold: {}",
            result
        );
    }
}

#[test]
fn test_eval_json_no_hold_in_error() {
    let (output, _code) = run_cli(&["eval-json", "((("]);
    let json = parse_json(&output);

    if let Some(error) = json["error"].as_str() {
        assert!(
            !error.contains("__hold"),
            "Error should not contain __hold: {}",
            error
        );
    }
}

// =============================================================================
// Schema version stability
// =============================================================================

#[test]
fn test_eval_json_schema_version_is_1() {
    let (output, _code) = run_cli(&["eval-json", "1+1"]);
    let json = parse_json(&output);

    assert_eq!(json["schema_version"], 1, "Schema version must be 1");
}
