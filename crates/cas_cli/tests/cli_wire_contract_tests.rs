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

/// Validates value-domain gating in envelope format
#[test]
#[ignore = "Uses cargo run internally, causing lock contention in CI"]
fn test_envelope_wire_value_domain_gates_imaginary_warning() {
    let (complex_output, _code) = run_cli(&[
        "envelope",
        "sqrt(-1)",
        "--domain",
        "generic",
        "--value-domain",
        "complex",
    ]);
    let complex_wire = parse_wire(&complex_output);
    let complex_assumed = complex_wire["transparency"]["assumptions_used"]
        .as_array()
        .unwrap();

    assert_eq!(complex_wire["result"]["value"]["display"], "(-1)^(1/2)");
    assert!(
        complex_assumed.is_empty(),
        "Complex envelope should not emit imaginary warning"
    );

    let (real_output, _code) = run_cli(&[
        "envelope",
        "sqrt(-1)",
        "--domain",
        "generic",
        "--value-domain",
        "real",
    ]);
    let real_wire = parse_wire(&real_output);
    let real_assumed = real_wire["transparency"]["assumptions_used"]
        .as_array()
        .unwrap();

    assert_eq!(real_wire["result"]["value"]["display"], "(-1)^(1/2)");
    assert_eq!(
        real_assumed.len(),
        1,
        "Real envelope should emit one warning"
    );
    assert_eq!(real_assumed[0]["kind"], "domain_warning");
    assert_eq!(real_assumed[0]["rule"], "Imaginary Usage Warning");
}

/// Validates explicit-i family under real vs complex envelope value domains
#[test]
#[ignore = "Uses cargo run internally, causing lock contention in CI"]
fn test_envelope_wire_value_domain_gates_explicit_i_family() {
    let (complex_i2_output, _code) = run_cli(&[
        "envelope",
        "i^2",
        "--domain",
        "generic",
        "--value-domain",
        "complex",
    ]);
    let complex_i2_wire = parse_wire(&complex_i2_output);
    assert_eq!(complex_i2_wire["result"]["value"]["display"], "-1");
    assert!(complex_i2_wire["transparency"]["assumptions_used"]
        .as_array()
        .unwrap()
        .is_empty());

    let (real_i2_output, _code) = run_cli(&[
        "envelope",
        "i^2",
        "--domain",
        "generic",
        "--value-domain",
        "real",
    ]);
    let real_i2_wire = parse_wire(&real_i2_output);
    assert_eq!(real_i2_wire["result"]["value"]["display"], "i^2");
    let real_i2_assumed = real_i2_wire["transparency"]["assumptions_used"]
        .as_array()
        .unwrap();
    assert_eq!(real_i2_assumed.len(), 1);
    assert_eq!(real_i2_assumed[0]["rule"], "Imaginary Usage Warning");

    let (complex_recip_output, _code) = run_cli(&[
        "envelope",
        "1/i",
        "--domain",
        "generic",
        "--value-domain",
        "complex",
    ]);
    let complex_recip_wire = parse_wire(&complex_recip_output);
    assert_eq!(complex_recip_wire["result"]["value"]["display"], "-i");

    let (real_gauss_output, _code) = run_cli(&[
        "envelope",
        "(1+i)/(1-i)",
        "--domain",
        "generic",
        "--value-domain",
        "real",
    ]);
    let real_gauss_wire = parse_wire(&real_gauss_output);
    assert_eq!(
        real_gauss_wire["result"]["value"]["display"],
        "(1 + i) / (1 - i)"
    );
    let real_gauss_assumed = real_gauss_wire["transparency"]["assumptions_used"]
        .as_array()
        .unwrap();
    assert_eq!(real_gauss_assumed.len(), 1);
    assert_eq!(real_gauss_assumed[0]["rule"], "Imaginary Usage Warning");

    let (complex_gauss_output, _code) = run_cli(&[
        "envelope",
        "(1+i)/(1-i)",
        "--domain",
        "generic",
        "--value-domain",
        "complex",
    ]);
    let complex_gauss_wire = parse_wire(&complex_gauss_output);
    assert_eq!(complex_gauss_wire["result"]["value"]["display"], "i");
    assert!(complex_gauss_wire["transparency"]["assumptions_used"]
        .as_array()
        .unwrap()
        .is_empty());
}

/// Validates value-domain split for log-exp variable case, but stability for provable constants
#[test]
#[ignore = "Uses cargo run internally, causing lock contention in CI"]
fn test_envelope_wire_value_domain_log_exp_split_and_constant_stability() {
    let (real_variable_output, _code) = run_cli(&[
        "envelope",
        "ln(exp(x))",
        "--domain",
        "strict",
        "--value-domain",
        "real",
    ]);
    let real_variable_wire = parse_wire(&real_variable_output);
    assert_eq!(real_variable_wire["result"]["value"]["display"], "x");
    let real_variable_required = real_variable_wire["transparency"]["required_conditions"]
        .as_array()
        .unwrap();
    assert_eq!(real_variable_required.len(), 1);
    assert_eq!(real_variable_required[0]["expr_canonical"], "e^x");

    let (complex_variable_output, _code) = run_cli(&[
        "envelope",
        "ln(exp(x))",
        "--domain",
        "strict",
        "--value-domain",
        "complex",
    ]);
    let complex_variable_wire = parse_wire(&complex_variable_output);
    assert_eq!(
        complex_variable_wire["result"]["value"]["display"],
        "ln(e^x)"
    );
    let complex_variable_required = complex_variable_wire["transparency"]["required_conditions"]
        .as_array()
        .unwrap();
    assert_eq!(complex_variable_required.len(), 1);
    assert_eq!(complex_variable_required[0]["expr_canonical"], "e^x");

    let (real_constant_output, _code) = run_cli(&[
        "envelope",
        "exp(ln(5))",
        "--domain",
        "strict",
        "--value-domain",
        "real",
    ]);
    let real_constant_wire = parse_wire(&real_constant_output);
    assert_eq!(real_constant_wire["result"]["value"]["display"], "5");
    assert!(real_constant_wire["transparency"]["required_conditions"]
        .as_array()
        .unwrap()
        .is_empty());

    let (complex_constant_output, _code) = run_cli(&[
        "envelope",
        "exp(ln(5))",
        "--domain",
        "strict",
        "--value-domain",
        "complex",
    ]);
    let complex_constant_wire = parse_wire(&complex_constant_output);
    assert_eq!(complex_constant_wire["result"]["value"]["display"], "5");
    assert!(complex_constant_wire["transparency"]["required_conditions"]
        .as_array()
        .unwrap()
        .is_empty());
    assert!(complex_constant_wire["transparency"]["assumptions_used"]
        .as_array()
        .unwrap()
        .is_empty());
}

/// Validates assume-mode warning + guard structure in envelope format
#[test]
#[ignore = "Uses cargo run internally, causing lock contention in CI"]
fn test_envelope_wire_assume_log_exp_surfaces_warning_and_guard() {
    let (output, _code) = run_cli(&[
        "envelope",
        "log(b,b^x)",
        "--domain",
        "assume",
        "--value-domain",
        "real",
    ]);
    let wire = parse_wire(&output);

    let transparency = &wire["transparency"];
    let required = transparency["required_conditions"].as_array().unwrap();
    let assumed = transparency["assumptions_used"].as_array().unwrap();

    assert_eq!(wire["result"]["value"]["display"], "x");
    assert_eq!(required.len(), 1);
    assert_eq!(required[0]["kind"], "NonZero");
    assert_eq!(required[0]["expr_canonical"], "b - 1");
    assert_eq!(assumed.len(), 1);
    assert_eq!(assumed[0]["kind"], "domain_warning");
    assert_eq!(assumed[0]["rule"], "Log-Exp Inverse");
    assert_eq!(assumed[0]["display"], "b > 0");
}

/// Validates strict envelope preserves blocked educational hints
#[test]
#[ignore = "Uses cargo run internally, causing lock contention in CI"]
fn test_envelope_wire_strict_x_over_x_surfaces_blocked_hints() {
    let (output, _code) = run_cli(&[
        "envelope",
        "x/x",
        "--domain",
        "strict",
        "--value-domain",
        "real",
    ]);
    let wire = parse_wire(&output);

    let transparency = &wire["transparency"];
    let blocked = transparency["blocked_hints"].as_array().unwrap();
    let required = transparency["required_conditions"].as_array().unwrap();

    assert_eq!(wire["result"]["value"]["display"], "x / x");
    assert_eq!(required.len(), 1);
    assert_eq!(required[0]["kind"], "NonZero");
    assert!(
        !blocked.is_empty(),
        "Strict envelope should surface blocked hints"
    );
    assert!(
        blocked.iter().any(|hint| hint["tip"]
            .as_str()
            .is_some_and(|tip| tip.contains("domain generic"))),
        "Blocked hints should suggest domain generic"
    );
}

/// Validates generic vs assume split for log-exp inverse in envelope format
#[test]
#[ignore = "Uses cargo run internally, causing lock contention in CI"]
fn test_envelope_wire_generic_vs_assume_log_exp_split() {
    let (generic_output, _code) = run_cli(&[
        "envelope",
        "log(b,b^x)",
        "--domain",
        "generic",
        "--value-domain",
        "real",
    ]);
    let generic_wire = parse_wire(&generic_output);

    assert_eq!(generic_wire["result"]["value"]["display"], "log(b, b^x)");
    assert!(generic_wire["transparency"]["required_conditions"]
        .as_array()
        .unwrap()
        .is_empty());
    assert!(generic_wire["transparency"]["assumptions_used"]
        .as_array()
        .unwrap()
        .is_empty());

    let (assume_output, _code) = run_cli(&[
        "envelope",
        "log(b,b^x)",
        "--domain",
        "assume",
        "--value-domain",
        "real",
    ]);
    let assume_wire = parse_wire(&assume_output);

    assert_eq!(assume_wire["result"]["value"]["display"], "x");
    let required = assume_wire["transparency"]["required_conditions"]
        .as_array()
        .unwrap();
    let assumed = assume_wire["transparency"]["assumptions_used"]
        .as_array()
        .unwrap();
    assert_eq!(required.len(), 1);
    assert_eq!(required[0]["kind"], "NonZero");
    assert_eq!(required[0]["expr_canonical"], "b - 1");
    assert_eq!(assumed.len(), 1);
    assert_eq!(assumed[0]["rule"], "Log-Exp Inverse");
    assert_eq!(assumed[0]["display"], "b > 0");
}

/// Validates intrinsic-domain split for exp(ln(x)) in envelope format
#[test]
#[ignore = "Uses cargo run internally, causing lock contention in CI"]
fn test_envelope_wire_strict_vs_generic_exp_ln_x_split() {
    let (strict_output, _code) = run_cli(&[
        "envelope",
        "exp(ln(x))",
        "--domain",
        "strict",
        "--value-domain",
        "real",
    ]);
    let strict_wire = parse_wire(&strict_output);

    assert_eq!(strict_wire["result"]["value"]["display"], "e^ln(x)");
    let strict_required = strict_wire["transparency"]["required_conditions"]
        .as_array()
        .unwrap();
    assert_eq!(strict_required.len(), 1);
    assert_eq!(strict_required[0]["kind"], "Positive");
    assert_eq!(strict_required[0]["expr_canonical"], "x");
    assert!(strict_wire["transparency"]["assumptions_used"]
        .as_array()
        .unwrap()
        .is_empty());

    let (generic_output, _code) = run_cli(&[
        "envelope",
        "exp(ln(x))",
        "--domain",
        "generic",
        "--value-domain",
        "real",
    ]);
    let generic_wire = parse_wire(&generic_output);

    assert_eq!(generic_wire["result"]["value"]["display"], "x");
    let generic_required = generic_wire["transparency"]["required_conditions"]
        .as_array()
        .unwrap();
    assert_eq!(generic_required.len(), 1);
    assert_eq!(generic_required[0]["kind"], "Positive");
    assert_eq!(generic_required[0]["expr_canonical"], "x");
    assert!(generic_wire["transparency"]["assumptions_used"]
        .as_array()
        .unwrap()
        .is_empty());
}

/// Validates strict/generic/assume split for abs-radical family in envelope format
#[test]
#[ignore = "Uses cargo run internally, causing lock contention in CI"]
fn test_envelope_wire_strict_generic_assume_abs_radical_split() {
    let (strict_ln_output, _code) = run_cli(&[
        "envelope",
        "ln(a^2)",
        "--domain",
        "strict",
        "--value-domain",
        "real",
    ]);
    let strict_ln_wire = parse_wire(&strict_ln_output);
    let strict_ln_display = strict_ln_wire["result"]["value"]["display"]
        .as_str()
        .unwrap();
    assert!(matches!(strict_ln_display, "2·ln(|a|)" | "2 * ln(|a|)"));
    assert!(strict_ln_wire["transparency"]["assumptions_used"]
        .as_array()
        .unwrap()
        .is_empty());

    let (generic_sqrt_output, _code) = run_cli(&[
        "envelope",
        "sqrt(x^2)",
        "--domain",
        "generic",
        "--value-domain",
        "real",
    ]);
    let generic_sqrt_wire = parse_wire(&generic_sqrt_output);
    assert_eq!(generic_sqrt_wire["result"]["value"]["display"], "|x|");
    assert!(generic_sqrt_wire["transparency"]["assumptions_used"]
        .as_array()
        .unwrap()
        .is_empty());

    let (strict_sqrt_output, _code) = run_cli(&[
        "envelope",
        "sqrt(x^2)",
        "--domain",
        "strict",
        "--value-domain",
        "real",
    ]);
    let strict_sqrt_wire = parse_wire(&strict_sqrt_output);
    assert_eq!(strict_sqrt_wire["result"]["value"]["display"], "sqrt(x^2)");
    assert!(strict_sqrt_wire["transparency"]["assumptions_used"]
        .as_array()
        .unwrap()
        .is_empty());

    let (assume_sqrt_output, _code) = run_cli(&[
        "envelope",
        "sqrt(x^2)",
        "--domain",
        "assume",
        "--value-domain",
        "real",
    ]);
    let assume_sqrt_wire = parse_wire(&assume_sqrt_output);
    assert_eq!(assume_sqrt_wire["result"]["value"]["display"], "x");
    let assume_sqrt_assumed = assume_sqrt_wire["transparency"]["assumptions_used"]
        .as_array()
        .unwrap();
    assert_eq!(assume_sqrt_assumed.len(), 1);
    assert_eq!(assume_sqrt_assumed[0]["rule"], "Abs Under Positivity");
}

/// Validates generic vs assume split for even-power/log and abs collapse in envelope format
#[test]
#[ignore = "Uses cargo run internally, causing lock contention in CI"]
fn test_envelope_wire_generic_vs_assume_even_power_and_abs_split() {
    let (generic_ln_output, _code) = run_cli(&[
        "envelope",
        "ln(a^2)",
        "--domain",
        "generic",
        "--value-domain",
        "real",
    ]);
    let generic_ln_wire = parse_wire(&generic_ln_output);
    assert_eq!(generic_ln_wire["result"]["value"]["display"], "2·ln(|a|)");
    assert!(generic_ln_wire["transparency"]["assumptions_used"]
        .as_array()
        .unwrap()
        .is_empty());

    let (assume_ln_output, _code) = run_cli(&[
        "envelope",
        "ln(a^2)",
        "--domain",
        "assume",
        "--value-domain",
        "real",
    ]);
    let assume_ln_wire = parse_wire(&assume_ln_output);
    assert_eq!(assume_ln_wire["result"]["value"]["display"], "2·ln(a)");
    let assume_ln_assumed = assume_ln_wire["transparency"]["assumptions_used"]
        .as_array()
        .unwrap();
    assert_eq!(assume_ln_assumed.len(), 1);
    assert_eq!(assume_ln_assumed[0]["rule"], "Log Even Power");
    assert_eq!(assume_ln_assumed[0]["display"], "a > 0");

    let (generic_sqrt_output, _code) = run_cli(&[
        "envelope",
        "sqrt(x^2)",
        "--domain",
        "generic",
        "--value-domain",
        "real",
    ]);
    let generic_sqrt_wire = parse_wire(&generic_sqrt_output);
    assert_eq!(generic_sqrt_wire["result"]["value"]["display"], "|x|");
    assert!(generic_sqrt_wire["transparency"]["assumptions_used"]
        .as_array()
        .unwrap()
        .is_empty());

    let (assume_sqrt_output, _code) = run_cli(&[
        "envelope",
        "sqrt(x^2)",
        "--domain",
        "assume",
        "--value-domain",
        "real",
    ]);
    let assume_sqrt_wire = parse_wire(&assume_sqrt_output);
    assert_eq!(assume_sqrt_wire["result"]["value"]["display"], "x");
    let assume_sqrt_assumed = assume_sqrt_wire["transparency"]["assumptions_used"]
        .as_array()
        .unwrap();
    assert_eq!(assume_sqrt_assumed.len(), 1);
    assert_eq!(assume_sqrt_assumed[0]["rule"], "Abs Under Positivity");
    assert_eq!(assume_sqrt_assumed[0]["display"], "x > 0");
}
