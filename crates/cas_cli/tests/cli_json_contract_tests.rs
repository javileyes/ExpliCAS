//! CLI JSON API integration tests.
//!
//! These tests verify that the CLI returns proper JSON with stable kind/code.
//!
//! **NOTE**: These tests are IGNORED by default because they use `cargo run` internally,
//! which can cause deadlocks during parallel test execution (cargo lock contention).
//! Run manually with: `cargo test -p cas_cli --test cli_json_contract_tests -- --ignored`

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
#[ignore = "Uses cargo run internally, causing lock contention in CI"]
fn test_eval_json_success() {
    let (output, _code) = run_cli(&["eval-json", "2+2"]);
    let json = parse_json(&output);

    assert_eq!(json["schema_version"], 1);
    assert_eq!(json["ok"], true);
    assert!(json["result"].is_string());
}

#[test]
#[ignore = "Uses cargo run internally, causing lock contention in CI"]
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
#[ignore = "Uses cargo run internally, causing lock contention in CI"]
fn test_eval_json_parse_error_has_kind_code() {
    let (output, _code) = run_cli(&["eval-json", "("]);
    let json = parse_json(&output);

    assert_eq!(json["ok"], false);
    assert_eq!(json["kind"], "ParseError", "Should have kind=ParseError");
    assert_eq!(json["code"], "E_PARSE", "Should have code=E_PARSE");
    assert!(json["error"].is_string(), "Should have error message");
}

#[test]
#[ignore = "Uses cargo run internally, causing lock contention in CI"]
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
#[ignore = "Uses cargo run internally, causing lock contention in CI"]
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
#[ignore = "Uses cargo run internally, causing lock contention in CI"]
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
#[ignore = "Uses cargo run internally, causing lock contention in CI"]
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
#[ignore = "Uses cargo run internally, causing lock contention in CI"]
fn test_eval_json_schema_version_is_1() {
    let (output, _code) = run_cli(&["eval-json", "1+1"]);
    let json = parse_json(&output);

    assert_eq!(json["schema_version"], 1, "Schema version must be 1");
}

// =============================================================================
// FFI Mock Tests - RequiredConditions Schema Validation
// =============================================================================

/// FFI MOCK: Validates required_conditions schema structure for Android/FFI consumers
#[test]
#[ignore = "Uses cargo run internally, causing lock contention in CI"]
fn test_ffi_mock_required_conditions_schema() {
    let (output, _code) = run_cli(&["eval-json", "sqrt(x)^2"]);
    let json = parse_json(&output);

    // 1. schema_version present
    assert!(
        json["schema_version"].is_u64(),
        "FFI contract: schema_version must be present and numeric"
    );

    // 2. required_conditions is array
    assert!(
        json["required_conditions"].is_array(),
        "FFI contract: required_conditions must be an array"
    );

    // 3. At least one condition for sqrt(x)^2
    let conditions = json["required_conditions"].as_array().unwrap();
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
        json["required_display"].is_array(),
        "FFI contract: required_display must be an array"
    );
}

/// FFI MOCK: Validates witness survival - no false required conditions
#[test]
#[ignore = "Uses cargo run internally, causing lock contention in CI"]
fn test_ffi_mock_witness_survival_empty_required() {
    let (output, _code) = run_cli(&["eval-json", "(x-y)/(sqrt(x)-sqrt(y))"]);
    let json = parse_json(&output);

    // Witness survival: sqrt survives in result, no required_conditions emitted
    let conditions = json["required_conditions"].as_array().unwrap();
    assert!(
        conditions.is_empty(),
        "FFI contract: witness survival means no x≥0 or y≥0 requirements"
    );
}

// =============================================================================
// OutputEnvelope V1 Contract Tests (envelope-json command)
// =============================================================================

/// Validates OutputEnvelope V1 root structure for Android/FFI
#[test]
#[ignore = "Uses cargo run internally, causing lock contention in CI"]
fn test_envelope_json_v1_structure() {
    let (output, _code) = run_cli(&["envelope-json", "sqrt(x)^2"]);
    let json = parse_json(&output);

    // Root envelope fields
    assert_eq!(json["schema_version"], 1, "V1: schema_version must be 1");
    assert!(json["engine"].is_object(), "V1: engine must be object");
    assert_eq!(json["engine"]["name"], "ExpliCAS");
    assert!(json["request"].is_object(), "V1: request must be object");
    assert!(json["result"].is_object(), "V1: result must be object");
    assert!(
        json["transparency"].is_object(),
        "V1: transparency must be object"
    );
}

/// Validates result structure with kind discriminator
#[test]
#[ignore = "Uses cargo run internally, causing lock contention in CI"]
fn test_envelope_json_result_eval() {
    let (output, _code) = run_cli(&["envelope-json", "2+2"]);
    let json = parse_json(&output);

    assert_eq!(
        json["result"]["kind"], "eval_result",
        "V1: result.kind for eval"
    );
    assert!(
        json["result"]["value"].is_object(),
        "V1: result.value must exist"
    );
    assert!(
        json["result"]["value"]["display"].is_string(),
        "V1: value.display"
    );
    assert!(
        json["result"]["value"]["canonical"].is_string(),
        "V1: value.canonical"
    );
}

/// Validates transparency section with requires vs assumed
#[test]
#[ignore = "Uses cargo run internally, causing lock contention in CI"]
fn test_envelope_json_transparency_requires() {
    let (output, _code) = run_cli(&["envelope-json", "sqrt(x)^2"]);
    let json = parse_json(&output);

    let transparency = &json["transparency"];
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

/// Validates witness survival in envelope format
#[test]
#[ignore = "Uses cargo run internally, causing lock contention in CI"]
fn test_envelope_json_witness_survival() {
    let (output, _code) = run_cli(&["envelope-json", "(x-y)/(sqrt(x)-sqrt(y))"]);
    let json = parse_json(&output);

    let transparency = &json["transparency"];
    let required = transparency["required_conditions"].as_array().unwrap();
    let assumed = transparency["assumptions_used"].as_array().unwrap();

    assert!(
        required.is_empty(),
        "V1: witness survival → no required_conditions"
    );
    assert!(!assumed.is_empty(), "V1: should have assumptions_used");
    assert_eq!(assumed[0]["kind"], "NonZero");
}
