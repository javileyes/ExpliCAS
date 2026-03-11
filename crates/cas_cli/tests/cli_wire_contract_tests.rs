//! CLI wire-contract integration tests.
//!
//! These tests verify that the CLI returns stable wire output with consistent
//! schema/kind/code fields.
//!
//! **NOTE**: These tests are IGNORED by default because they use `cargo run` internally,
//! which can cause deadlocks during parallel test execution (cargo lock contention).
//! Run manually with: `cargo test -p cas_cli --test cli_wire_contract_tests -- --ignored`

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

fn parse_wire(s: &str) -> Value {
    serde_json::from_str(s).unwrap_or_else(|_| panic!("Failed to parse wire payload: {}", s))
}

// =============================================================================
// Success cases
// =============================================================================

#[test]
#[ignore = "Uses cargo run internally, causing lock contention in CI"]
fn test_eval_wire_success() {
    let (output, _code) = run_cli(&["eval", "2+2", "--format", "json"]);
    let wire = parse_wire(&output);

    assert_eq!(wire["schema_version"], 1);
    assert_eq!(wire["ok"], true);
    assert!(wire["result"].is_string());
}

#[test]
#[ignore = "Uses cargo run internally, causing lock contention in CI"]
fn test_eval_wire_success_has_budget() {
    let (output, _code) = run_cli(&["eval", "x+x", "--format", "json"]);
    let wire = parse_wire(&output);

    assert!(wire["budget"].is_object(), "Should have budget object");
    assert!(wire["budget"]["preset"].is_string());
    assert!(wire["budget"]["mode"].is_string());
}

// =============================================================================
// Error cases with kind/code
// =============================================================================

#[test]
#[ignore = "Uses cargo run internally, causing lock contention in CI"]
fn test_eval_wire_parse_error_has_kind_code() {
    let (output, _code) = run_cli(&["eval", "(", "--format", "json"]);
    let wire = parse_wire(&output);

    assert_eq!(wire["ok"], false);
    assert_eq!(wire["kind"], "ParseError", "Should have kind=ParseError");
    assert_eq!(wire["code"], "E_PARSE", "Should have code=E_PARSE");
    assert!(wire["error"].is_string(), "Should have error message");
}

#[test]
#[ignore = "Uses cargo run internally, causing lock contention in CI"]
fn test_eval_wire_error_kind_in_known_set() {
    // Parse error
    let (output, _code) = run_cli(&["eval", "(((", "--format", "json"]);
    let wire = parse_wire(&output);

    let valid_kinds = [
        "ParseError",
        "DomainError",
        "SolverError",
        "BudgetExceeded",
        "NotImplemented",
        "InternalError",
    ];

    if !wire["ok"].as_bool().unwrap_or(true) {
        let kind = wire["kind"].as_str().unwrap_or("");
        assert!(
            valid_kinds.contains(&kind),
            "Kind '{}' should be in known set",
            kind
        );
    }
}

#[test]
#[ignore = "Uses cargo run internally, causing lock contention in CI"]
fn test_eval_wire_error_code_starts_with_e() {
    let (output, _code) = run_cli(&["eval", "(((", "--format", "json"]);
    let wire = parse_wire(&output);

    if !wire["ok"].as_bool().unwrap_or(true) {
        let code = wire["code"].as_str().unwrap_or("");
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
#[ignore = "Uses cargo run internally, causing lock contention in CI"]
fn test_eval_wire_no_hold_in_result() {
    // Complex expression that might internally use __hold
    let (output, _code) = run_cli(&["eval", "(x+1)^2 - (x+1)^2", "--format", "json"]);
    let wire = parse_wire(&output);

    if let Some(result) = wire["result"].as_str() {
        assert!(
            !result.contains("__hold"),
            "Result should not contain __hold: {}",
            result
        );
    }
}

#[test]
#[ignore = "Uses cargo run internally, causing lock contention in CI"]
fn test_eval_wire_no_hold_in_error() {
    let (output, _code) = run_cli(&["eval", "(((", "--format", "json"]);
    let wire = parse_wire(&output);

    if let Some(error) = wire["error"].as_str() {
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
#[ignore = "Uses cargo run internally, causing lock contention in CI"]
fn test_eval_wire_schema_version_is_1() {
    let (output, _code) = run_cli(&["eval", "1+1", "--format", "json"]);
    let wire = parse_wire(&output);

    assert_eq!(wire["schema_version"], 1, "Schema version must be 1");
}

// =============================================================================
// FFI Mock Tests - RequiredConditions Schema Validation
// =============================================================================

/// FFI MOCK: Validates required_conditions schema structure for Android/FFI consumers
#[test]
#[ignore = "Uses cargo run internally, causing lock contention in CI"]
fn test_ffi_mock_required_conditions_schema() {
    let (output, _code) = run_cli(&["eval", "sqrt(x)^2", "--format", "json"]);
    let wire = parse_wire(&output);

    // 1. schema_version present
    assert!(
        wire["schema_version"].is_u64(),
        "FFI contract: schema_version must be present and numeric"
    );

    // 2. required_conditions is array
    assert!(
        wire["required_conditions"].is_array(),
        "FFI contract: required_conditions must be an array"
    );

    // 3. At least one condition for sqrt(x)^2
    let conditions = wire["required_conditions"].as_array().unwrap();
    assert!(
        !conditions.is_empty(),
        "FFI contract: sqrt(x)^2 should have required conditions"
    );

    // 4. First condition has correct structure
    let first = &conditions[0];
    assert_eq!(
        first["kind"], "NonNegative",
        "FFI contract: kind must be 'NonNegative'"
    );
    assert!(
        first["expr_display"].is_string(),
        "FFI contract: expr_display must be string"
    );
    assert!(
        first["expr_canonical"].is_string(),
        "FFI contract: expr_canonical must be string"
    );

    // 5. required_display is array of strings
    assert!(
        wire["required_display"].is_array(),
        "FFI contract: required_display must be an array"
    );
}

/// FFI MOCK: Validates that domain requirements surface in required_conditions
#[test]
#[ignore = "Uses cargo run internally, causing lock contention in CI"]
fn test_ffi_mock_domain_requirements_are_preserved() {
    let (output, _code) = run_cli(&["eval", "(x-y)/(sqrt(x)-sqrt(y))", "--format", "json"]);
    let wire = parse_wire(&output);

    // Current contract: rationalization surfaces denominator/nonnegative requirements.
    let conditions = wire["required_conditions"].as_array().unwrap();
    assert!(
        conditions.len() >= 3,
        "FFI contract: rationalization should surface denominator and sqrt requirements"
    );

    let kinds: Vec<_> = conditions
        .iter()
        .filter_map(|c| c["kind"].as_str())
        .collect();
    assert!(kinds.contains(&"NonZero"));
    assert_eq!(
        kinds.iter().filter(|kind| **kind == "NonNegative").count(),
        2,
        "FFI contract: should preserve x >= 0 and y >= 0"
    );
}

// =============================================================================
// OutputEnvelope V1 wire-contract tests
// =============================================================================

/// Validates OutputEnvelope V1 root structure for Android/FFI
#[test]
#[ignore = "Uses cargo run internally, causing lock contention in CI"]
fn test_envelope_wire_v1_structure() {
    let (output, _code) = run_cli(&["envelope", "sqrt(x)^2"]);
    let wire = parse_wire(&output);

    // Root envelope fields
    assert_eq!(wire["schema_version"], 1, "V1: schema_version must be 1");
    assert!(wire["engine"].is_object(), "V1: engine must be object");
    assert_eq!(wire["engine"]["name"], "ExpliCAS");
    assert!(wire["request"].is_object(), "V1: request must be object");
    assert!(wire["result"].is_object(), "V1: result must be object");
    assert!(
        wire["transparency"].is_object(),
        "V1: transparency must be object"
    );
}

/// Validates result structure with kind discriminator
#[test]
#[ignore = "Uses cargo run internally, causing lock contention in CI"]
fn test_envelope_wire_result_eval() {
    let (output, _code) = run_cli(&["envelope", "2+2"]);
    let wire = parse_wire(&output);

    assert_eq!(
        wire["result"]["kind"], "eval_result",
        "V1: result.kind for eval"
    );
    assert!(
        wire["result"]["value"].is_object(),
        "V1: result.value must exist"
    );
    assert!(
        wire["result"]["value"]["display"].is_string(),
        "V1: value.display"
    );
    assert!(
        wire["result"]["value"]["canonical"].is_string(),
        "V1: value.canonical"
    );
}

/// Validates transparency section with requires vs assumed
#[test]
#[ignore = "Uses cargo run internally, causing lock contention in CI"]
fn test_envelope_wire_transparency_requires() {
    let (output, _code) = run_cli(&["envelope", "sqrt(x)^2"]);
    let wire = parse_wire(&output);

    let transparency = &wire["transparency"];
    assert!(transparency["required_conditions"].is_array());
    assert!(transparency["assumptions_used"].is_array());

    let required = transparency["required_conditions"].as_array().unwrap();
    assert!(
        !required.is_empty(),
        "V1: sqrt(x)^2 should have required_conditions"
    );

    let first = &required[0];
    assert_eq!(first["kind"], "NonNegative");
    assert!(first["display"].as_str().unwrap().contains("≥"));
    assert!(first["expr_canonical"].is_string());
}

/// Validates transparency requirements in envelope format
#[test]
#[ignore = "Uses cargo run internally, causing lock contention in CI"]
fn test_envelope_wire_transparency_preserves_domain_requirements() {
    let (output, _code) = run_cli(&["envelope", "(x-y)/(sqrt(x)-sqrt(y))"]);
    let wire = parse_wire(&output);

    let transparency = &wire["transparency"];
    let required = transparency["required_conditions"].as_array().unwrap();
    let assumed = transparency["assumptions_used"].as_array().unwrap();

    assert!(
        required.len() >= 3,
        "V1: transparency should preserve denominator and sqrt requirements"
    );
    let kinds: Vec<_> = required.iter().filter_map(|c| c["kind"].as_str()).collect();
    assert!(kinds.contains(&"NonZero"));
    assert_eq!(
        kinds.iter().filter(|kind| **kind == "NonNegative").count(),
        2,
        "V1: should preserve x >= 0 and y >= 0"
    );
    assert!(assumed.is_empty(), "V1: no assumptions_used expected here");
}
