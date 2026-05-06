//! Non-regression tests for CLI limit command
//!
//! These tests verify the limit command works correctly for various cases.

use std::path::PathBuf;
use std::process::Command;

use serde_json::{json, Value};

/// Get path to the cas_cli binary
fn cas_cli_binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_cas_cli"))
}

/// Helper to run limit command and get stdout
fn run_limit(expr: &str, var: &str, to: &str, format: &str) -> (bool, String) {
    let binary = cas_cli_binary();

    // Use = format for --to to handle negative values like -infinity
    let to_arg = format!("--to={}", to);
    let format_arg = format!("--format={}", format);

    let output = Command::new(&binary)
        .args(["limit", expr, "--var", var, &to_arg, &format_arg])
        .output()
        .unwrap_or_else(|e| panic!("Failed to run {:?}: {}", binary, e));

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    (output.status.success(), stdout)
}

fn run_limit_with_stderr(expr: &str, var: &str, to: &str, format: &str) -> (bool, String, String) {
    let binary = cas_cli_binary();
    let to_arg = format!("--to={}", to);
    let format_arg = format!("--format={}", format);

    let output = Command::new(&binary)
        .args(["limit", expr, "--var", var, &to_arg, &format_arg])
        .output()
        .unwrap_or_else(|e| panic!("Failed to run {:?}: {}", binary, e));

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    (output.status.success(), stdout, stderr)
}

fn run_eval(expr: &str, format: &str) -> (bool, String) {
    let binary = cas_cli_binary();
    let format_arg = format!("--format={}", format);

    let output = Command::new(&binary)
        .args(["eval", expr, &format_arg])
        .output()
        .unwrap_or_else(|e| panic!("Failed to run {:?}: {}", binary, e));

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    (output.status.success(), stdout)
}

#[test]
fn test_limit_x_to_infinity_text() {
    let (success, stdout) = run_limit("x", "x", "infinity", "text");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("infinity"),
        "Result should be infinity, got: {}",
        stdout
    );
}

#[test]
fn test_limit_rational_poly_json() {
    let (success, stdout) = run_limit("(x^2+1)/(2*x^2-3)", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(stdout.contains("\"ok\":true"), "JSON should have ok:true");
    assert!(
        stdout.contains("\"result\":\"1/2\""),
        "Result should be 1/2, got: {}",
        stdout
    );
}

#[test]
fn test_limit_neg_infinity_parity() {
    let (success, stdout) = run_limit("x^3/x^2", "x", "-infinity", "text");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("-infinity"),
        "Result should be -infinity, got: {}",
        stdout
    );
}

#[test]
fn test_limit_residual_with_warning_json() {
    let (success, stdout) = run_limit("sin(x)/x", "x", "infinity", "json");
    assert!(success, "Command should succeed even for residual");
    assert!(stdout.contains("\"ok\":true"), "JSON should have ok:true");
    assert!(
        stdout.contains("\"warning\""),
        "Should have warning for unresolved limit"
    );
    assert!(
        stdout.contains("limit("),
        "Result should contain residual limit(...)"
    );
}

#[test]
fn test_limit_subcommand_rejects_finite_to_value_before_eval_json() {
    let (success, stdout, stderr) = run_limit_with_stderr("x", "x", "0", "json");
    assert!(!success, "Finite --to should be rejected by the CLI");
    assert!(
        stdout.trim().is_empty(),
        "Rejected CLI invocation should not emit a JSON result, got: {stdout}"
    );
    assert!(
        stderr.contains("invalid value '0'") && stderr.contains("infinity"),
        "Finite --to should advertise the infinity-only contract, got: {stderr}"
    );
}

#[test]
fn test_eval_finite_dependent_limit_stays_residual_with_warning_json() {
    let (success, stdout) = run_eval("limit(x + 1, x, 0)", "json");
    assert!(success, "Command should succeed even for finite residual");
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "limit(x + 1, x, 0)");
    assert!(
        wire["warnings"].as_array().is_some_and(|warnings| {
            warnings.iter().any(|warning| {
                warning["rule"] == "Limit Evaluation"
                    && warning["assumption"].as_str().is_some_and(|message| {
                        message.contains("Finite point limits are not supported safely yet")
                    })
            })
        }),
        "Dependent finite limit should keep explicit finite-limit warning, got: {wire:?}"
    );
}

#[test]
fn test_eval_finite_identity_limit_returns_point_json() {
    let (success, stdout) = run_eval("limit(x, x, 0)", "json");
    assert!(success, "Command should succeed for finite identity limit");
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "0");
    assert_eq!(wire["warnings"], json!([]));
}

#[test]
fn test_eval_finite_identity_limit_returns_independent_symbolic_point_json() {
    let (success, stdout) = run_eval("limit(x, x, y + 1)", "json");
    assert!(
        success,
        "Command should succeed for independent finite identity point"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "y + 1");
    assert_eq!(wire["warnings"], json!([]));
}

#[test]
fn test_eval_finite_independent_limit_preserves_domain_requirements_json() {
    let (success, stdout) = run_eval("limit(ln(y), x, -1)", "json");
    assert!(
        success,
        "Command should succeed for independent finite limit expression"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "ln(y)");
    assert_eq!(wire["warnings"], json!([]));
    assert_eq!(wire["required_display"], json!(["y > 0"]));
}

#[test]
fn test_eval_finite_independent_sqrt_limit_preserves_domain_requirements_json() {
    let (success, stdout) = run_eval("limit(sqrt(y), x, -1)", "json");
    assert!(
        success,
        "Command should succeed for independent finite sqrt limit expression"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "sqrt(y)");
    assert_eq!(wire["warnings"], json!([]));
    assert_eq!(wire["required_conditions"][0]["kind"], "NonNegative");
    assert_eq!(wire["required_conditions"][0]["expr_display"], "y");
}

#[test]
fn test_eval_finite_independent_reciprocal_limit_preserves_domain_requirements_json() {
    let (success, stdout) = run_eval("limit(1/y, x, -1)", "json");
    assert!(
        success,
        "Command should succeed for independent finite reciprocal limit expression"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "1 / y");
    assert_eq!(wire["warnings"], json!([]));
    assert_eq!(wire["required_conditions"][0]["kind"], "NonZero");
    assert_eq!(wire["required_conditions"][0]["expr_display"], "y");
}

#[test]
fn test_eval_finite_independent_composite_reciprocal_limit_preserves_domain_requirements_json() {
    let (success, stdout) = run_eval("limit(1/(y+1), x, -1)", "json");
    assert!(
        success,
        "Command should succeed for independent finite composite reciprocal limit expression"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "1 / (y + 1)");
    assert_eq!(wire["warnings"], json!([]));
    assert_eq!(wire["required_conditions"][0]["kind"], "NonZero");
    assert_eq!(wire["required_conditions"][0]["expr_display"], "y + 1");
}

#[test]
fn test_eval_finite_independent_limit_preserves_multiple_domain_requirements_json() {
    let (success, stdout) = run_eval("limit(ln(y)/(z+1), x, -1)", "json");
    assert!(
        success,
        "Command should succeed for independent finite limit with multiple domain requirements"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "ln(y) / (z + 1)");
    assert_eq!(wire["warnings"], json!([]));
    let required = wire["required_conditions"]
        .as_array()
        .expect("required conditions");
    assert!(
        required
            .iter()
            .any(|cond| cond["kind"] == "Positive" && cond["expr_display"] == "y"),
        "Expected Positive(y), got: {required:?}"
    );
    assert!(
        required
            .iter()
            .any(|cond| cond["kind"] == "NonZero" && cond["expr_display"] == "z + 1"),
        "Expected NonZero(z + 1), got: {required:?}"
    );
}

#[test]
fn test_eval_finite_identity_limit_rejects_dependent_point_json() {
    let (success, stdout) = run_eval("limit(x, x, x + 1)", "json");
    assert!(
        success,
        "Command should succeed with residual for dependent finite point"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "limit(x, x, x + 1)");
    assert!(
        wire["warnings"].as_array().is_some_and(|warnings| {
            warnings.iter().any(|warning| {
                warning["rule"] == "Limit Evaluation"
                    && warning["assumption"].as_str().is_some_and(|message| {
                        message.contains("Finite point limits are not supported safely yet")
                    })
            })
        }),
        "Dependent finite identity point should keep finite-limit warning, got: {wire:?}"
    );
}

#[test]
fn test_limit_deg_num_less_than_deg_den() {
    let (success, stdout) = run_limit("x^2/x^3", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"0\""),
        "Result should be 0, got: {}",
        stdout
    );
}

#[test]
fn test_limit_higher_num_degree_infinity() {
    let (success, stdout) = run_limit("x^3/x^2", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("infinity"),
        "Result should contain infinity, got: {}",
        stdout
    );
    assert!(
        !stdout.contains("-infinity"),
        "Should be positive infinity at +∞"
    );
}

#[test]
fn test_limit_polynomial_growth_pos_infinity() {
    let (success, stdout) = run_limit("x^2 + 1", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"infinity\""),
        "Polynomial growth should resolve to infinity, got: {}",
        stdout
    );
    assert!(
        !stdout.contains("\"warning\""),
        "Resolved polynomial limit should not warn, got: {}",
        stdout
    );
}

#[test]
fn test_limit_polynomial_growth_neg_infinity_uses_leading_sign_and_parity() {
    let (success, stdout) = run_limit("x - 2*x^3", "x", "-infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"infinity\""),
        "Negative odd leading term should resolve to +infinity at -∞, got: {}",
        stdout
    );
    assert!(
        !stdout.contains("-infinity"),
        "Result should not be negative infinity at -∞, got: {}",
        stdout
    );
}

#[test]
fn test_limit_sqrt_plus_constant_pos_infinity() {
    let (success, stdout) = run_limit("sqrt(x) + 1", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"infinity\""),
        "sqrt(x)+1 should resolve to infinity, got: {}",
        stdout
    );
    assert!(
        !stdout.contains("\"warning\""),
        "Resolved sqrt growth limit should not warn, got: {}",
        stdout
    );
}

#[test]
fn test_limit_exp_neg_infinity() {
    let (success, stdout) = run_limit("exp(x)", "x", "-infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"0\""),
        "exp(x) should tend to 0 at -∞, got: {}",
        stdout
    );
}

#[test]
fn test_limit_exp_negative_linear_pos_infinity() {
    let (success, stdout) = run_limit("exp(-x)", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"0\""),
        "exp(-x) should tend to 0 at +∞, got: {}",
        stdout
    );
    assert!(
        !stdout.contains("\"warning\""),
        "Resolved linear exponential limit should not warn, got: {}",
        stdout
    );
}

#[test]
fn test_limit_decaying_exp_plus_constant_pos_infinity() {
    let (success, stdout) = run_limit("exp(-x) + 1", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"1\""),
        "exp(-x)+1 should resolve to 1 at +∞, got: {}",
        stdout
    );
    assert!(
        !stdout.contains("\"warning\""),
        "Resolved linear exponential sum should not warn, got: {}",
        stdout
    );
}

#[test]
fn test_limit_exp_positive_linear_neg_infinity() {
    let (success, stdout) = run_limit("exp(2*x)", "x", "-infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"0\""),
        "exp(2*x) should tend to 0 at -∞, got: {}",
        stdout
    );
    assert!(
        !stdout.contains("\"warning\""),
        "Resolved linear exponential limit should not warn, got: {}",
        stdout
    );
}

#[test]
fn test_limit_ln_negative_linear_neg_infinity() {
    let (success, stdout) = run_limit("ln(-x + 1)", "x", "-infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"infinity\""),
        "ln(-x+1) should tend to infinity at -∞, got: {}",
        stdout
    );
    assert!(
        !stdout.contains("\"warning\""),
        "Resolved linear log tail should not warn, got: {}",
        stdout
    );
}

#[test]
fn test_limit_ln_neg_infinity_remains_residual() {
    let (success, stdout) = run_limit("ln(x)", "x", "-infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"warning\"") && stdout.contains("limit("),
        "ln(x) at -∞ should remain residual over the real domain, got: {}",
        stdout
    );
}

#[test]
fn test_limit_scaled_sqrt_pos_infinity() {
    let (success, stdout) = run_limit("2*sqrt(x)", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"infinity\""),
        "2*sqrt(x) should resolve to infinity, got: {}",
        stdout
    );
}

#[test]
fn test_limit_negative_sqrt_pos_infinity() {
    let (success, stdout) = run_limit("0 - sqrt(x)", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"-infinity\""),
        "-sqrt(x) should resolve to -infinity, got: {}",
        stdout
    );
}

#[test]
fn test_limit_reciprocal_exp_pos_infinity() {
    let (success, stdout) = run_limit("1/exp(x)", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"0\""),
        "1/exp(x) should resolve to 0, got: {}",
        stdout
    );
}

#[test]
fn test_limit_exp_dominates_polynomial_difference_pos_infinity() {
    let (success, stdout) = run_limit("exp(x) - x^2", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"infinity\""),
        "exp(x)-x^2 should resolve to infinity, got: {}",
        stdout
    );
    assert!(
        !stdout.contains("\"warning\""),
        "Resolved exponential dominance limit should not warn, got: {}",
        stdout
    );
}

#[test]
fn test_limit_polynomial_over_exp_pos_infinity() {
    let (success, stdout) = run_limit("x^2/exp(x)", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"0\""),
        "x^2/exp(x) should resolve to 0, got: {}",
        stdout
    );
}

#[test]
fn test_limit_exp_over_polynomial_pos_infinity() {
    let (success, stdout) = run_limit("exp(x)/x^2", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"infinity\""),
        "exp(x)/x^2 should resolve to infinity, got: {}",
        stdout
    );
}

#[test]
fn test_limit_polynomial_over_linear_exp_pos_infinity() {
    let (success, stdout) = run_limit("x^2/exp(2*x)", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"0\""),
        "x^2/exp(2*x) should resolve to 0, got: {}",
        stdout
    );
    assert!(
        !stdout.contains("\"warning\""),
        "Resolved linear exponential dominance limit should not warn, got: {}",
        stdout
    );
}

#[test]
fn test_limit_linear_exp_over_polynomial_pos_infinity() {
    let (success, stdout) = run_limit("exp(2*x)/x^2", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"infinity\""),
        "exp(2*x)/x^2 should resolve to infinity, got: {}",
        stdout
    );
    assert!(
        !stdout.contains("\"warning\""),
        "Resolved linear exponential dominance limit should not warn, got: {}",
        stdout
    );
}

#[test]
fn test_limit_constant_over_decaying_linear_exp_pos_infinity() {
    let (success, stdout) = run_limit("1/exp(-2*x)", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"infinity\""),
        "1/exp(-2*x) should resolve to infinity, got: {}",
        stdout
    );
    assert!(
        !stdout.contains("\"warning\""),
        "Resolved decaying linear exponential denominator should not warn, got: {}",
        stdout
    );
}

#[test]
fn test_limit_polynomial_times_exp_neg_infinity() {
    let (success, stdout) = run_limit("x*exp(x)", "x", "-infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"0\""),
        "x*exp(x) at -∞ should resolve to 0, got: {}",
        stdout
    );
}

#[test]
fn test_limit_polynomial_times_linear_exp_neg_infinity() {
    let (success, stdout) = run_limit("x^2*exp(2*x)", "x", "-infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"0\""),
        "x^2*exp(2*x) at -∞ should resolve to 0, got: {}",
        stdout
    );
    assert!(
        !stdout.contains("\"warning\""),
        "Resolved polynomial times decaying linear exp should not warn, got: {}",
        stdout
    );
}

#[test]
fn test_limit_polynomial_over_exp_neg_infinity_uses_polynomial_sign() {
    let (success, stdout) = run_limit("x/exp(x)", "x", "-infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"-infinity\""),
        "x/exp(x) at -∞ should resolve to -infinity, got: {}",
        stdout
    );
}

#[test]
fn test_limit_even_polynomial_over_exp_neg_infinity() {
    let (success, stdout) = run_limit("x^2/exp(x)", "x", "-infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"infinity\""),
        "x^2/exp(x) at -∞ should resolve to infinity, got: {}",
        stdout
    );
}

#[test]
fn test_limit_zero_scaled_exp_denominator_remains_residual() {
    let (success, stdout) = run_limit("x^2/(0*exp(x))", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"warning\"") && stdout.contains("limit("),
        "Identically zero exponential denominator should remain residual, got: {}",
        stdout
    );
}

#[test]
fn test_limit_zero_scaled_linear_exp_denominator_remains_residual() {
    let (success, stdout) = run_limit("x^2/(0*exp(2*x))", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"warning\"") && stdout.contains("limit("),
        "Identically zero linear exponential denominator should remain residual, got: {}",
        stdout
    );
}

#[test]
fn test_limit_composed_exp_dominance_gap_remains_residual() {
    let (success, stdout) = run_limit("exp(x^2)/x^2", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"warning\"") && stdout.contains("limit("),
        "Composed exp dominance remains out of scope and should be residual, got: {}",
        stdout
    );
}

#[test]
fn test_limit_ln_over_polynomial_pos_infinity() {
    let (success, stdout) = run_limit("ln(x)/x", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"0\""),
        "ln(x)/x should resolve to 0, got: {}",
        stdout
    );
    assert!(
        !stdout.contains("\"warning\""),
        "Resolved log-over-polynomial dominance should not warn, got: {}",
        stdout
    );
}

#[test]
fn test_limit_polynomial_over_ln_pos_infinity() {
    let (success, stdout) = run_limit("x/ln(x)", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"infinity\""),
        "x/ln(x) should resolve to infinity, got: {}",
        stdout
    );
    assert!(
        !stdout.contains("\"warning\""),
        "Resolved polynomial-over-log dominance should not warn, got: {}",
        stdout
    );
}

#[test]
fn test_limit_sqrt_over_polynomial_pos_infinity() {
    let (success, stdout) = run_limit("sqrt(x)/x", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"0\""),
        "sqrt(x)/x should resolve to 0, got: {}",
        stdout
    );
    assert!(
        !stdout.contains("\"warning\""),
        "Resolved root-over-polynomial dominance should not warn, got: {}",
        stdout
    );
}

#[test]
fn test_limit_polynomial_over_sqrt_pos_infinity() {
    let (success, stdout) = run_limit("x/sqrt(x)", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"infinity\""),
        "x/sqrt(x) should resolve to infinity, got: {}",
        stdout
    );
    assert!(
        !stdout.contains("\"warning\""),
        "Resolved polynomial-over-root dominance should not warn, got: {}",
        stdout
    );
}

#[test]
fn test_limit_polynomial_over_ln_negative_orientation() {
    let (success, stdout) = run_limit("x/ln(1 - x)", "x", "-infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"-infinity\""),
        "x/ln(1-x) should resolve to -infinity at -∞, got: {}",
        stdout
    );
    assert!(
        !stdout.contains("\"warning\""),
        "Resolved negative-orientation log dominance should not warn, got: {}",
        stdout
    );
}

#[test]
fn test_limit_log_dominance_bad_domain_remains_residual() {
    let (success, stdout) = run_limit("ln(x)/x", "x", "-infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"warning\"") && stdout.contains("limit("),
        "ln(x)/x at -∞ should remain residual over the real domain, got: {}",
        stdout
    );
}

#[test]
fn test_limit_base_log_over_polynomial_pos_infinity() {
    let (success, stdout) = run_limit("log(2, x)/x", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"0\""),
        "log(2,x)/x should resolve to 0, got: {}",
        stdout
    );
    assert!(
        !stdout.contains("\"warning\""),
        "Resolved general-base log dominance should not warn, got: {}",
        stdout
    );
}

#[test]
fn test_limit_polynomial_over_half_base_log_pos_infinity() {
    let (success, stdout) = run_limit("x/log(1/2, x)", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"-infinity\""),
        "x/log(1/2,x) should resolve to -infinity, got: {}",
        stdout
    );
    assert!(
        !stdout.contains("\"warning\""),
        "Resolved base < 1 log dominance should not warn, got: {}",
        stdout
    );
}

#[test]
fn test_limit_exp_over_half_base_log_pos_infinity() {
    let (success, stdout) = run_limit("exp(x)/log(1/2, x)", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"-infinity\""),
        "exp(x)/log(1/2,x) should resolve to -infinity, got: {}",
        stdout
    );
    assert!(
        !stdout.contains("\"warning\""),
        "Resolved exp over base < 1 log dominance should not warn, got: {}",
        stdout
    );
}

#[test]
fn test_limit_invalid_base_log_dominance_remains_residual() {
    let (success, stdout) = run_limit("log(1, x)/x", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"warning\"") && stdout.contains("limit("),
        "log base 1 should remain residual, got: {}",
        stdout
    );
}

#[test]
fn test_limit_base_log_bad_domain_remains_residual() {
    let (success, stdout) = run_limit("log(2, x)/x", "x", "-infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"warning\"") && stdout.contains("limit("),
        "log(2,x)/x at -∞ should remain residual over the real domain, got: {}",
        stdout
    );
}

#[test]
fn test_limit_e_base_log_over_polynomial_pos_infinity() {
    let (success, stdout) = run_limit("log(e, x)/x", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"0\""),
        "log(e,x)/x should resolve to 0, got: {}",
        stdout
    );
    assert!(
        !stdout.contains("\"warning\""),
        "Resolved e-base log dominance should not warn, got: {}",
        stdout
    );
}

#[test]
fn test_limit_polynomial_over_reciprocal_e_base_log_pos_infinity() {
    let (success, stdout) = run_limit("x/log(1/e, x)", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"-infinity\""),
        "x/log(1/e,x) should resolve to -infinity, got: {}",
        stdout
    );
    assert!(
        !stdout.contains("\"warning\""),
        "Resolved reciprocal e-base log dominance should not warn, got: {}",
        stdout
    );
}

#[test]
fn test_limit_negative_e_base_log_remains_residual() {
    let (success, stdout) = run_limit("log(-e, x)/x", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"warning\"") && stdout.contains("limit("),
        "negative e-base log should remain residual, got: {}",
        stdout
    );
}

#[test]
fn test_limit_powered_e_base_log_over_polynomial_pos_infinity() {
    let (success, stdout) = run_limit("log(e^2, x)/x", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"0\""),
        "log(e^2,x)/x should resolve to 0, got: {}",
        stdout
    );
    assert!(
        !stdout.contains("\"warning\""),
        "Resolved powered e-base log dominance should not warn, got: {}",
        stdout
    );
}

#[test]
fn test_limit_polynomial_over_negative_power_e_base_log_pos_infinity() {
    let (success, stdout) = run_limit("x/log(e^-2, x)", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"-infinity\""),
        "x/log(e^-2,x) should resolve to -infinity, got: {}",
        stdout
    );
    assert!(
        !stdout.contains("\"warning\""),
        "Resolved negative powered e-base log dominance should not warn, got: {}",
        stdout
    );
}

#[test]
fn test_limit_zero_power_e_base_log_remains_residual() {
    let (success, stdout) = run_limit("log(e^0, x)/x", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"warning\"") && stdout.contains("limit("),
        "e^0 is base 1 and should remain residual, got: {}",
        stdout
    );
}

#[test]
fn test_limit_ln_over_exp_pos_infinity() {
    let (success, stdout) = run_limit("ln(x)/exp(x)", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"0\""),
        "ln(x)/exp(x) should resolve to 0, got: {}",
        stdout
    );
    assert!(
        !stdout.contains("\"warning\""),
        "Resolved log-over-exp dominance should not warn, got: {}",
        stdout
    );
}

#[test]
fn test_limit_exp_over_ln_pos_infinity() {
    let (success, stdout) = run_limit("exp(x)/ln(x)", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"infinity\""),
        "exp(x)/ln(x) should resolve to infinity, got: {}",
        stdout
    );
    assert!(
        !stdout.contains("\"warning\""),
        "Resolved exp-over-log dominance should not warn, got: {}",
        stdout
    );
}

#[test]
fn test_limit_sqrt_times_decaying_exp_pos_infinity() {
    let (success, stdout) = run_limit("sqrt(x)*exp(-x)", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"0\""),
        "sqrt(x)*exp(-x) should resolve to 0, got: {}",
        stdout
    );
    assert!(
        !stdout.contains("\"warning\""),
        "Resolved root-times-decaying-exp dominance should not warn, got: {}",
        stdout
    );
}

#[test]
fn test_limit_ln_over_decaying_exp_pos_infinity() {
    let (success, stdout) = run_limit("ln(x)/exp(-x)", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"infinity\""),
        "ln(x)/exp(-x) should resolve to infinity, got: {}",
        stdout
    );
    assert!(
        !stdout.contains("\"warning\""),
        "Resolved log-over-decaying-exp dominance should not warn, got: {}",
        stdout
    );
}

#[test]
fn test_limit_negative_orientation_exp_over_sqrt() {
    let (success, stdout) = run_limit("exp(x)/(-sqrt(x))", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"-infinity\""),
        "exp(x)/(-sqrt(x)) should resolve to -infinity, got: {}",
        stdout
    );
    assert!(
        !stdout.contains("\"warning\""),
        "Resolved negative-orientation exp/root dominance should not warn, got: {}",
        stdout
    );
}

#[test]
fn test_limit_nonlinear_exp_over_log_remains_residual() {
    let (success, stdout) = run_limit("exp(x^2)/ln(x)", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"warning\"") && stdout.contains("limit("),
        "Nonlinear exponential dominance remains out of scope, got: {}",
        stdout
    );
}
