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
    // Strict auto-eval preserves a residual fraction shape instead of collapsing to 2.
    // Depending on the active simplification path, this may appear as 4/2, 2*x/x,
    // or the equivalent reordered form (x*2)/x.
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
        result == "4 / 2"
            || result == "4/2"
            || result == "(x·2)/x"
            || result == "(x * 2)/x"
            || result == "2·x/x"
            || result == "2 * x/x"
            || result == "(2·x)/x"
            || result == "(2 * x)/x",
        "Expected strict residual fraction preserving x-domain, got: {}",
        result
    );
    let required = wire["required_display"].as_array().unwrap();
    assert_eq!(
        required.len(),
        1,
        "Strict should preserve one domain condition"
    );
    assert_eq!(required[0], "x ≠ 0");
}

#[test]
fn cli_domain_generic_pow_zero_emits_required_nonzero() {
    // x^0 -> 1 must surface the definability condition on the wire
    // (0^0 is undefined), mirroring the x/x cancellation contract.
    let (output, _code) = run_cli(&["eval", "x^0", "--format", "json", "--domain", "generic"]);
    let wire = parse_wire(&output);

    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "1");
    let required = wire["required_display"].as_array().unwrap();
    assert_eq!(
        required.len(),
        1,
        "expected exactly the x ≠ 0 condition, got: {required:?}"
    );
    assert_eq!(required[0], "x ≠ 0");
}

#[test]
fn cli_domain_assume_pow_zero_emits_required_nonzero() {
    let (output, _code) = run_cli(&["eval", "x^0", "--format", "json", "--domain", "assume"]);
    let wire = parse_wire(&output);

    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "1");
    let required = wire["required_display"].as_array().unwrap();
    assert_eq!(
        required.len(),
        1,
        "expected exactly the x ≠ 0 condition, got: {required:?}"
    );
    assert_eq!(required[0], "x ≠ 0");
}

#[test]
fn cli_domain_generic_pow_zero_proven_base_unconditional() {
    let (output, _code) = run_cli(&[
        "eval",
        "(x^2+1)^0",
        "--format",
        "json",
        "--domain",
        "generic",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "1");
    let required = wire["required_display"].as_array().unwrap();
    assert!(
        required.is_empty(),
        "provably nonzero base needs no condition, got: {required:?}"
    );
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

#[test]
fn cli_domain_generic_factor_cancellation_renders_atomic_requires_without_composite_duplicate() {
    let (output, _code) = run_cli(&[
        "eval",
        "(x^5 + x^4 - 2*x^2 - 2*x) / (x^3 - x)",
        "--format",
        "json",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "(x^3 - 2) / (x - 1)");

    let required = wire["required_display"]
        .as_array()
        .expect("required_display array");
    let required: Vec<_> = required.iter().filter_map(|item| item.as_str()).collect();

    assert!(
        !required.contains(&"x^3 - x ≠ 0"),
        "composite denominator guard should be expanded for display: {:?}",
        required
    );
    assert!(required.contains(&"x ≠ 0"));
    assert!(required.contains(&"x ≠ 1"));
    assert!(required.contains(&"x ≠ -1"));
}
