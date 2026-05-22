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
fn test_limit_bounded_trig_over_divergent_denominator_json() {
    let (success, stdout) = run_limit("sin(x)/x", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(stdout.contains("\"ok\":true"), "JSON should have ok:true");
    assert!(
        stdout.contains("\"result\":\"0\""),
        "Bounded trig over divergent denominator should resolve to 0, got: {stdout}"
    );
    assert!(
        !stdout.contains("\"warning\""),
        "Resolved bounded-over-divergent limit should not warn, got: {stdout}"
    );
}

#[test]
fn test_limit_bounded_arctan_over_divergent_denominator_json() {
    let (success, stdout) = run_limit("arctan(x)/x", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(stdout.contains("\"ok\":true"), "JSON should have ok:true");
    assert!(
        stdout.contains("\"result\":\"0\""),
        "Bounded arctan over divergent denominator should resolve to 0, got: {stdout}"
    );
    assert!(
        !stdout.contains("\"warning\""),
        "Resolved bounded-over-divergent limit should not warn, got: {stdout}"
    );
}

#[test]
fn test_limit_bounded_tanh_over_divergent_denominator_json() {
    let (success, stdout) = run_limit("tanh(x)/x", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(stdout.contains("\"ok\":true"), "JSON should have ok:true");
    assert!(
        stdout.contains("\"result\":\"0\""),
        "Bounded tanh over divergent denominator should resolve to 0, got: {stdout}"
    );
    assert!(
        !stdout.contains("\"warning\""),
        "Resolved bounded-over-divergent limit should not warn, got: {stdout}"
    );
}

#[test]
fn test_limit_bounded_cross_family_arithmetic_over_divergent_denominator_json() {
    let (success, stdout) = run_limit("(sin(x)+cos(x)*arctan(x))/(x^2+1)", "x", "infinity", "json");
    assert!(success, "Command should succeed");
    assert!(stdout.contains("\"ok\":true"), "JSON should have ok:true");
    assert!(
        stdout.contains("\"result\":\"0\""),
        "Bounded cross-family arithmetic over divergent denominator should resolve to 0, got: {stdout}"
    );
    assert!(
        !stdout.contains("\"warning\""),
        "Resolved bounded cross-family arithmetic should not warn, got: {stdout}"
    );
}

#[test]
fn test_limit_trig_over_nondivergent_denominator_remains_residual_json() {
    let (success, stdout) = run_limit("sin(x)/cos(x)", "x", "infinity", "json");
    assert!(success, "Command should succeed even for residual");
    assert!(stdout.contains("\"ok\":true"), "JSON should have ok:true");
    assert!(
        stdout.contains("\"warning\"") && stdout.contains("limit("),
        "Nondivergent trig denominator should stay residual, got: {stdout}"
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
    let (success, stdout) = run_eval("limit(ln(x), x, -1)", "json");
    assert!(success, "Command should succeed even for finite residual");
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "limit(ln(x), x, -1)");
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
fn test_eval_finite_limit_residual_suppresses_impossible_zero_nonzero_requirement_json() {
    let (success, stdout) = run_eval("limit(1/(sqrt(x^2+1)-sqrt(x^2+1)), x, -2)", "json");
    assert!(
        success,
        "Command should succeed even when the finite limit remains residual"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(
        wire["result"],
        "limit(1 / (sqrt(x^2 + 1) - sqrt(x^2 + 1)), x, -2)"
    );
    assert!(
        wire["warnings"].as_array().is_some_and(|warnings| {
            warnings.iter().any(|warning| {
                warning["rule"] == "Limit Evaluation"
                    && warning["assumption"].as_str().is_some_and(|message| {
                        message.contains("Finite point limits are not supported safely yet")
                    })
            })
        }),
        "Finite zero-denominator residual should keep explicit warning, got: {wire:?}"
    );
    assert_eq!(wire["required_conditions"], json!([]));
    assert_eq!(wire["required_display"], json!([]));
}

#[test]
fn test_eval_finite_polynomial_limit_returns_substituted_value_json() {
    let (success, stdout) = run_eval("limit(x^2 + x + 1, x, -2)", "json");
    assert!(
        success,
        "Command should succeed for finite polynomial limit"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "3");
    assert_eq!(wire["warnings"], json!([]));
}

#[test]
fn test_eval_finite_rational_polynomial_limit_requires_nonzero_denominator_at_point_json() {
    let (success, stdout) = run_eval("limit((x^2+3*x+2)/(x+2), x, 0)", "json");
    assert!(
        success,
        "Command should succeed for nonsingular finite rational-polynomial limit"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "1");
    assert_eq!(wire["warnings"], json!([]));
    assert_eq!(wire["required_display"], json!(["x ≠ -2"]));

    let (success, stdout) = run_eval("limit((x^2+3*x+2)/(x+2), x, -2)", "json");
    assert!(
        success,
        "Command should keep singular finite rational-polynomial limit residual"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "limit((x^2 + 3·x + 2) / (x + 2), x, -2)");
    assert!(
        wire["warnings"].as_array().is_some_and(|warnings| {
            warnings.iter().any(|warning| {
                warning["rule"] == "Limit Evaluation"
                    && warning["assumption"].as_str().is_some_and(|message| {
                        message.contains("Finite point limits are not supported safely yet")
                    })
            })
        }),
        "Singular rational finite limit should remain residual with warning, got: {wire:?}"
    );
}

#[test]
fn test_eval_finite_elementary_polynomial_limit_requires_positive_argument_at_point_json() {
    let (success, stdout) = run_eval("limit(sqrt(x^2 + 1), x, -2)", "json");
    assert!(
        success,
        "Command should succeed for positive-argument finite sqrt polynomial limit"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "sqrt(5)");
    assert_eq!(wire["warnings"], json!([]));

    let (success, stdout) = run_eval("limit(ln(x + 3), x, 0)", "json");
    assert!(
        success,
        "Command should succeed for positive-argument finite ln polynomial limit"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "ln(3)");
    assert_eq!(wire["warnings"], json!([]));
    assert_eq!(wire["required_display"], json!(["x > -3"]));

    let (success, stdout) = run_eval("limit(ln(x + 3), x, -4)", "json");
    assert!(
        success,
        "Command should keep out-of-domain finite ln polynomial limit residual"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "limit(ln(x + 3), x, -4)");
    assert!(
        wire["warnings"].as_array().is_some_and(|warnings| {
            warnings.iter().any(|warning| {
                warning["rule"] == "Limit Evaluation"
                    && warning["assumption"].as_str().is_some_and(|message| {
                        message.contains("Finite point limits are not supported safely yet")
                    })
            })
        }),
        "Out-of-domain finite ln limit should remain residual with warning, got: {wire:?}"
    );
}

#[test]
fn test_eval_finite_total_real_elementary_polynomial_limits_json() {
    let cases = [
        ("limit(exp(x), x, 0)", "1"),
        ("limit(sin(x), x, 0)", "0"),
        ("limit(cos(x), x, 0)", "1"),
        ("limit(sin(x^2 + 1), x, -1)", "sin(2)"),
        ("limit(sinh(x), x, 0)", "0"),
        ("limit(cosh(x), x, 0)", "1"),
        ("limit(tanh(x), x, 0)", "0"),
        ("limit(tanh(x^2 + 1), x, -1)", "tanh(2)"),
        ("limit(atan(x), x, 0)", "0"),
        ("limit(arctan(x), x, 0)", "0"),
        ("limit(atan(x^2 + 1), x, -1)", "atan(2)"),
        ("limit(arctan(x^2 + 1), x, -1)", "arctan(2)"),
        ("limit(asinh(x), x, 0)", "0"),
        ("limit(asinh(x^2 + 1), x, -1)", "asinh(2)"),
        ("limit(cbrt(x), x, -8)", "-2"),
        ("limit(cbrt(x), x, 0)", "0"),
        ("limit(cbrt(x^2 + 1), x, -1)", "cbrt(2)"),
        ("limit((x^2 - 9)^(1/3), x, 1)", "-2"),
        ("limit(sin(cbrt(x)), x, 8)", "sin(2)"),
        ("limit(abs(x), x, -2)", "2"),
        ("limit(abs(x), x, 0)", "0"),
        ("limit(abs(x^2 - 1), x, 0)", "1"),
        ("limit(cos(sin(x)), x, 0)", "1"),
        ("limit(sin(sqrt(x^2 + 1)), x, -2)", "sin(sqrt(5))"),
        ("limit(exp(abs(x)), x, -2)", "exp(2)"),
    ];

    for (input, expected) in cases {
        let (success, stdout) = run_eval(input, "json");
        assert!(success, "Command should succeed for {input}");
        let wire: Value = serde_json::from_str(&stdout).expect("eval json");
        assert_eq!(wire["ok"], true, "input: {input}, wire: {wire:?}");
        assert_eq!(wire["result"], expected, "input: {input}");
        assert!(
            !wire["result"]
                .as_str()
                .unwrap_or_default()
                .contains("limit("),
            "finite elementary limit should not remain residual for {input}: {wire:?}"
        );
        assert_eq!(wire["warnings"], json!([]), "input: {input}");
        assert_eq!(wire["required_display"], json!([]), "input: {input}");
    }
}

#[test]
fn test_eval_finite_partial_domain_inverse_elementary_limits_stay_residual_json() {
    let (success, stdout) = run_eval("limit(arcsin(x), x, 2)", "json");
    assert!(
        success,
        "Command should keep out-of-domain finite arcsin limit residual"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "limit(arcsin(x), x, 2)");
    assert!(
        wire["warnings"].as_array().is_some_and(|warnings| {
            warnings.iter().any(|warning| {
                warning["rule"] == "Limit Evaluation"
                    && warning["assumption"].as_str().is_some_and(|message| {
                        message.contains("Finite point limits are not supported safely yet")
                    })
            })
        }),
        "Partial-domain inverse limit should remain residual with warning, got: {wire:?}"
    );
}

#[test]
fn test_eval_finite_discontinuous_elementary_limits_stay_residual_json() {
    for input in ["limit(sign(x), x, 0)", "limit(sin(sign(x)), x, 0)"] {
        let (success, stdout) = run_eval(input, "json");
        assert!(
            success,
            "Command should keep discontinuous finite sign limit residual for {input}"
        );
        let wire: Value = serde_json::from_str(&stdout).expect("eval json");
        assert_eq!(wire["ok"], true, "input: {input}");
        assert_eq!(wire["result"], input, "input: {input}");
        assert!(
            wire["warnings"].as_array().is_some_and(|warnings| {
                warnings.iter().any(|warning| {
                    warning["rule"] == "Limit Evaluation"
                        && warning["assumption"].as_str().is_some_and(|message| {
                            message.contains("Finite point limits are not supported safely yet")
                        })
                })
            }),
            "Discontinuous finite limit should remain residual with warning for {input}, got: {wire:?}"
        );
    }
}

#[test]
fn test_eval_finite_positive_domain_unary_composition_limits_json() {
    let cases = [
        ("limit(ln(sqrt(x^2 + 1)), x, -2)", "ln(sqrt(5))", json!([])),
        ("limit(sqrt(abs(x) + 1), x, -2)", "sqrt(3)", json!([])),
        ("limit(ln(abs(x)), x, -2)", "ln(2)", json!(["x ≠ 0"])),
        ("limit(log2(x^2 + 1), x, -2)", "log2(5)", json!([])),
        (
            "limit(log10(sqrt(x^2 + 1)), x, -2)",
            "log10(sqrt(5))",
            json!([]),
        ),
        ("limit(log2(abs(x)), x, -2)", "log2(2)", json!(["x ≠ 0"])),
    ];

    for (input, expected, expected_required) in cases {
        let (success, stdout) = run_eval(input, "json");
        assert!(success, "Command should succeed for {input}");
        let wire: Value = serde_json::from_str(&stdout).expect("eval json");
        assert_eq!(wire["ok"], true, "input: {input}, wire: {wire:?}");
        assert_eq!(wire["result"], expected, "input: {input}");
        assert!(
            !wire["result"]
                .as_str()
                .unwrap_or_default()
                .contains("limit("),
            "finite positive-domain composition should not remain residual for {input}: {wire:?}"
        );
        assert_eq!(wire["warnings"], json!([]), "input: {input}");
        assert_eq!(
            wire["required_display"], expected_required,
            "input: {input}"
        );
    }
}

#[test]
fn test_eval_finite_positive_domain_exact_value_limits_fold_locally_json() {
    let cases = [
        ("limit(sqrt(x^2 + 1), x, 0)", "1", json!([])),
        ("limit(sqrt(x^2 + 4*x + 4), x, 0)", "2", json!([])),
        ("limit(ln(x^2 + 1), x, 0)", "0", json!([])),
        ("limit(log2(x^2 + 1), x, 0)", "0", json!([])),
        ("limit(log10(x^2 + 1), x, 0)", "0", json!([])),
        ("limit(log(2, x^2 + 1), x, 0)", "0", json!([])),
        ("limit(log(x^2 + 3, x^2 + 3), x, -1)", "1", json!([])),
    ];

    for (input, expected, expected_required) in cases {
        let (success, stdout) = run_eval(input, "json");
        assert!(success, "Command should succeed for {input}");
        let wire: Value = serde_json::from_str(&stdout).expect("eval json");
        assert_eq!(wire["ok"], true, "input: {input}, wire: {wire:?}");
        assert_eq!(wire["result"], expected, "input: {input}");
        assert_eq!(wire["warnings"], json!([]), "input: {input}");
        assert_eq!(
            wire["required_display"], expected_required,
            "input: {input}"
        );
    }
}

#[test]
fn test_eval_finite_unary_inverse_and_abs_limits_fold_locally_json() {
    let cases = [
        ("limit(exp(ln(abs(x))), x, -2)", "2", json!(["x ≠ 0"])),
        ("limit(ln(exp(abs(x))), x, -2)", "2", json!([])),
        ("limit(ln(exp(x^2 + 1)), x, 0)", "1", json!([])),
        ("limit(abs(sqrt(x^2 + 1)), x, -2)", "sqrt(5)", json!([])),
        ("limit(abs(-sqrt(x^2 + 1)), x, -2)", "sqrt(5)", json!([])),
    ];

    for (input, expected, expected_required) in cases {
        let (success, stdout) = run_eval(input, "json");
        assert!(success, "Command should succeed for {input}");
        let wire: Value = serde_json::from_str(&stdout).expect("eval json");
        assert_eq!(wire["ok"], true, "input: {input}, wire: {wire:?}");
        assert_eq!(wire["result"], expected, "input: {input}");
        assert_eq!(wire["warnings"], json!([]), "input: {input}");
        assert_eq!(
            wire["required_display"], expected_required,
            "input: {input}"
        );
    }
}

#[test]
fn test_eval_finite_unary_inverse_limits_keep_unsafe_boundaries_residual_json() {
    for input in ["limit(exp(ln(abs(x))), x, 0)", "limit(exp(ln(x)), x, 0)"] {
        let (success, stdout) = run_eval(input, "json");
        assert!(
            success,
            "Command should keep unsafe finite inverse-wrapper residual for {input}"
        );
        let wire: Value = serde_json::from_str(&stdout).expect("eval json");
        assert_eq!(wire["ok"], true, "input: {input}");
        assert!(
            wire["result"]
                .as_str()
                .is_some_and(|result| result.contains("limit(")),
            "unsafe finite inverse-wrapper should remain residual for {input}: {wire:?}"
        );
        assert!(
            wire["warnings"].as_array().is_some_and(|warnings| {
                warnings.iter().any(|warning| {
                    warning["rule"] == "Limit Evaluation"
                        && warning["assumption"].as_str().is_some_and(|message| {
                            message.contains("Finite point limits are not supported safely yet")
                        })
                })
            }),
            "Unsafe finite inverse-wrapper should remain residual with warning for {input}, got: {wire:?}"
        );
    }
}

#[test]
fn test_eval_finite_positive_domain_unary_composition_rejects_nonpositive_sublimits_json() {
    for input in [
        "limit(sqrt(abs(x)), x, 0)",
        "limit(ln(sin(x)), x, 0)",
        "limit(log10(abs(x)), x, 0)",
    ] {
        let (success, stdout) = run_eval(input, "json");
        assert!(
            success,
            "Command should keep nonpositive finite {input} residual"
        );
        let wire: Value = serde_json::from_str(&stdout).expect("eval json");
        assert_eq!(wire["ok"], true, "input: {input}");
        assert!(
            wire["result"]
                .as_str()
                .is_some_and(|result| result.contains("limit(")),
            "nonpositive positive-domain composition should remain residual for {input}: {wire:?}"
        );
        assert!(
            wire["warnings"].as_array().is_some_and(|warnings| {
                warnings.iter().any(|warning| {
                    warning["rule"] == "Limit Evaluation"
                        && warning["assumption"].as_str().is_some_and(|message| {
                            message.contains("Finite point limits are not supported safely yet")
                        })
                })
            }),
            "Nonpositive finite limit should remain residual with warning for {input}, got: {wire:?}"
        );
    }
}

#[test]
fn test_eval_finite_binary_log_composition_limits_json() {
    let cases = [
        ("limit(log(2, x^2 + 1), x, -2)", "log(2, 5)", json!([])),
        (
            "limit(log(1/2, sqrt(x^2 + 1)), x, -2)",
            "log(1 / 2, sqrt(5))",
            json!([]),
        ),
        ("limit(log(2, abs(x)), x, -2)", "1", json!(["x ≠ 0"])),
        (
            "limit(log(x^2 + 3, x^2 + 1), x, -2)",
            "log(7, 5)",
            json!([]),
        ),
        (
            "limit(log(x^2 + 3, sqrt(x^2 + 1)), x, -2)",
            "log(7, sqrt(5))",
            json!([]),
        ),
    ];

    for (input, expected, expected_required) in cases {
        let (success, stdout) = run_eval(input, "json");
        assert!(success, "Command should succeed for {input}");
        let wire: Value = serde_json::from_str(&stdout).expect("eval json");
        assert_eq!(wire["ok"], true, "input: {input}, wire: {wire:?}");
        assert_eq!(wire["result"], expected, "input: {input}");
        assert!(
            !wire["result"]
                .as_str()
                .unwrap_or_default()
                .contains("limit("),
            "finite binary log should not remain residual for {input}: {wire:?}"
        );
        assert_eq!(wire["warnings"], json!([]), "input: {input}");
        assert_eq!(
            wire["required_display"], expected_required,
            "input: {input}"
        );
    }
}

#[test]
fn test_eval_finite_binary_log_composition_rejects_unsafe_cases_json() {
    for input in [
        "limit(log(2, abs(x)), x, 0)",
        "limit(log(1, x^2 + 1), x, -2)",
        "limit(log(-2, x^2 + 1), x, -2)",
        "limit(log(x^2 - 3, x^2 + 1), x, -2)",
        "limit(log(x^2 - 4, x^2 + 1), x, -2)",
    ] {
        let (success, stdout) = run_eval(input, "json");
        assert!(
            success,
            "Command should keep unsafe finite binary log residual for {input}"
        );
        let wire: Value = serde_json::from_str(&stdout).expect("eval json");
        assert_eq!(wire["ok"], true, "input: {input}");
        assert!(
            wire["result"]
                .as_str()
                .is_some_and(|result| result.contains("limit(")),
            "unsafe binary log should remain residual for {input}: {wire:?}"
        );
        assert!(
            wire["warnings"].as_array().is_some_and(|warnings| {
                warnings.iter().any(|warning| {
                    warning["rule"] == "Limit Evaluation"
                        && warning["assumption"].as_str().is_some_and(|message| {
                            message.contains("Finite point limits are not supported safely yet")
                        })
                })
            }),
            "Unsafe finite binary log should remain residual with warning for {input}, got: {wire:?}"
        );
    }
}

#[test]
fn test_eval_finite_integer_power_composition_limits_json() {
    let cases = [
        ("limit((abs(x)+1)^2, x, -2)", "9"),
        ("limit((sqrt(x^2 + 1))^2, x, -2)", "5"),
        ("limit((sqrt(x^2 + 1))^3, x, -2)", "sqrt(5)^3"),
        ("limit((abs(x)+1)^(-2), x, -2)", "1/9"),
        ("limit((sqrt(x^2 + 1))^(-1), x, -2)", "1 / sqrt(5)"),
        ("limit((sqrt(x^2 + 1))^(-2), x, -2)", "1/5"),
        ("limit((cbrt(x^2 + 1))^3, x, -1)", "2"),
        ("limit((cbrt(x^2 + 1))^2, x, -1)", "cbrt(2)^2"),
        ("limit((cbrt(x^2 + 1))^(-3), x, -1)", "1/2"),
        ("limit((cbrt(x^2 + 1))^0, x, -1)", "1"),
        ("limit((abs(x)+1)^0, x, -2)", "1"),
    ];

    for (input, expected) in cases {
        let (success, stdout) = run_eval(input, "json");
        assert!(success, "Command should succeed for {input}");
        let wire: Value = serde_json::from_str(&stdout).expect("eval json");
        assert_eq!(wire["ok"], true, "input: {input}, wire: {wire:?}");
        assert_eq!(wire["result"], expected, "input: {input}");
        assert!(
            !wire["result"]
                .as_str()
                .unwrap_or_default()
                .contains("limit("),
            "finite integer power should not remain residual for {input}: {wire:?}"
        );
        assert_eq!(wire["warnings"], json!([]), "input: {input}");
    }
}

#[test]
fn test_eval_finite_integer_power_composition_rejects_unsafe_cases_json() {
    for input in [
        "limit((abs(x)-2)^(-1), x, -2)",
        "limit(abs(x)^0, x, 0)",
        "limit((sqrt(x))^2, x, 0)",
        "limit((cbrt(x))^(-3), x, 0)",
    ] {
        let (success, stdout) = run_eval(input, "json");
        assert!(
            success,
            "Command should keep unsafe finite integer power residual for {input}"
        );
        let wire: Value = serde_json::from_str(&stdout).expect("eval json");
        assert_eq!(wire["ok"], true, "input: {input}");
        assert!(
            wire["result"]
                .as_str()
                .is_some_and(|result| result.contains("limit(")),
            "unsafe integer power should remain residual for {input}: {wire:?}"
        );
        assert!(
            wire["warnings"].as_array().is_some_and(|warnings| {
                warnings.iter().any(|warning| {
                    warning["rule"] == "Limit Evaluation"
                        && warning["assumption"].as_str().is_some_and(|message| {
                            message.contains("Finite point limits are not supported safely yet")
                        })
                })
            }),
            "Unsafe finite integer power should remain residual with warning for {input}, got: {wire:?}"
        );
    }
}

#[test]
fn test_eval_finite_arithmetic_composition_folds_safe_results_json() {
    let cases = [
        ("limit(abs(x)+1, x, -2)", "3"),
        ("limit(sqrt(x^2 + 1) - sqrt(x^2 + 1), x, -2)", "0"),
        (
            "limit((sqrt(x^2 + 1) - sqrt(x^2 + 1))/(abs(x)+1), x, -2)",
            "0",
        ),
        ("limit(sqrt(x^2 + 1) + ln(x + 5), x, -2)", "ln(3) + sqrt(5)"),
    ];

    for (input, expected) in cases {
        let (success, stdout) = run_eval(input, "json");
        assert!(success, "Command should succeed for {input}");
        let wire: Value = serde_json::from_str(&stdout).expect("eval json");
        assert_eq!(wire["ok"], true, "input: {input}, wire: {wire:?}");
        assert_eq!(wire["result"], expected, "input: {input}");
        assert_eq!(wire["warnings"], json!([]), "input: {input}");
    }
}

#[test]
fn test_eval_finite_composite_limit_requires_all_subterms_safe_json() {
    let (success, stdout) = run_eval("limit(sqrt(x^2 + 1) + ln(x + 5), x, -2)", "json");
    assert!(
        success,
        "Command should succeed for finite composite limit with safe subterms"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "ln(3) + sqrt(5)");
    assert_eq!(wire["warnings"], json!([]));
    assert_eq!(wire["required_display"], json!(["x > -5"]));

    let (success, stdout) = run_eval("limit(2*sqrt(x^2 + 1), x, -2)", "json");
    assert!(
        success,
        "Command should succeed for finite product limit with safe subterm"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "2·sqrt(5)");
    assert_eq!(wire["warnings"], json!([]));

    let (success, stdout) = run_eval("limit(sqrt(x) + 1, x, 0)", "json");
    assert!(
        success,
        "Command should keep finite composite limit residual when a sublimit is unsafe"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "limit(sqrt(x) + 1, x, 0)");
    assert!(
        wire["warnings"].as_array().is_some_and(|warnings| {
            warnings.iter().any(|warning| {
                warning["rule"] == "Limit Evaluation"
                    && warning["assumption"].as_str().is_some_and(|message| {
                        message.contains("Finite point limits are not supported safely yet")
                    })
            })
        }),
        "Composite finite limit with unsafe sublimit should remain residual, got: {wire:?}"
    );

    let (success, stdout) = run_eval("limit(0*sqrt(x), x, 0)", "json");
    assert!(
        success,
        "Command should keep finite zero product residual when a sublimit is unsafe"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert!(
        wire["result"]
            .as_str()
            .is_some_and(|result| result.contains("limit(")),
        "Finite zero product with unsafe sublimit should remain residual, got: {wire:?}"
    );
}

#[test]
fn test_eval_finite_division_limit_accepts_structurally_nonzero_denominator_json() {
    let (success, stdout) = run_eval("limit(1/sqrt(x^2 + 1), x, -2)", "json");
    assert!(
        success,
        "Command should succeed for finite division with structurally positive denominator"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "1 / sqrt(5)");
    assert_eq!(wire["warnings"], json!([]));

    let (success, stdout) = run_eval("limit(1/sqrt(x), x, 0)", "json");
    assert!(
        success,
        "Command should keep finite division residual when denominator is not proven nonzero"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "limit(1 / sqrt(x), x, 0)");
    assert!(
        wire["warnings"].as_array().is_some_and(|warnings| {
            warnings.iter().any(|warning| {
                warning["rule"] == "Limit Evaluation"
                    && warning["assumption"].as_str().is_some_and(|message| {
                        message.contains("Finite point limits are not supported safely yet")
                    })
            })
        }),
        "Finite division with unproven nonzero denominator should remain residual, got: {wire:?}"
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
fn test_limit_sqrt_quadratic_over_linear_pos_and_neg_infinity() {
    let (success_pos, stdout_pos) = run_limit("sqrt(x^2+1)/x", "x", "infinity", "json");
    assert!(success_pos, "Command should succeed");
    assert!(
        stdout_pos.contains("\"result\":\"1\""),
        "sqrt(x^2+1)/x should resolve to 1 at +∞, got: {}",
        stdout_pos
    );
    assert!(
        !stdout_pos.contains("\"warning\""),
        "Resolved sqrt-polynomial ratio should not warn at +∞, got: {}",
        stdout_pos
    );

    let (success_neg, stdout_neg) = run_limit("sqrt(x^2+1)/x", "x", "-infinity", "json");
    assert!(success_neg, "Command should succeed");
    assert!(
        stdout_neg.contains("\"result\":\"-1\""),
        "sqrt(x^2+1)/x should resolve to -1 at -∞, got: {}",
        stdout_neg
    );
    assert!(
        !stdout_neg.contains("\"warning\""),
        "Resolved sqrt-polynomial ratio should not warn at -∞, got: {}",
        stdout_neg
    );
}

#[test]
fn test_limit_sqrt_quadratic_with_surd_leading_coeff_over_linear() {
    let (success_pos, stdout_pos) = run_limit("sqrt(2*x^2+1)/x", "x", "infinity", "json");
    assert!(success_pos, "Command should succeed");
    assert!(
        stdout_pos.contains("\"result\":\"sqrt(2)\""),
        "sqrt(2*x^2+1)/x should resolve to sqrt(2) at +∞, got: {}",
        stdout_pos
    );
    assert!(
        !stdout_pos.contains("\"warning\""),
        "Resolved surd sqrt-polynomial ratio should not warn at +∞, got: {}",
        stdout_pos
    );

    let (success_neg, stdout_neg) = run_limit("sqrt(2*x^2+1)/x", "x", "-infinity", "json");
    assert!(success_neg, "Command should succeed");
    assert!(
        stdout_neg.contains("\"result\":\"-sqrt(2)\""),
        "sqrt(2*x^2+1)/x should resolve to -sqrt(2) at -∞, got: {}",
        stdout_neg
    );
    assert!(
        !stdout_neg.contains("\"warning\""),
        "Resolved surd sqrt-polynomial ratio should not warn at -∞, got: {}",
        stdout_neg
    );
}

#[test]
fn test_limit_sqrt_quadratic_with_surd_over_scaled_linear_denominator() {
    let (success_pos, stdout_pos) = run_limit("sqrt(2*x^2+1)/(3*x)", "x", "infinity", "json");
    assert!(success_pos, "Command should succeed");
    assert!(
        stdout_pos.contains("\"result\":\"sqrt(2) / 3\""),
        "sqrt(2*x^2+1)/(3*x) should resolve to sqrt(2)/3 at +∞, got: {}",
        stdout_pos
    );
    assert!(
        !stdout_pos.contains("\"warning\""),
        "Resolved scaled-denominator sqrt-polynomial ratio should not warn at +∞, got: {}",
        stdout_pos
    );

    let (success_neg, stdout_neg) = run_limit("sqrt(2*x^2+1)/(-3*x)", "x", "-infinity", "json");
    assert!(success_neg, "Command should succeed");
    assert!(
        stdout_neg.contains("\"result\":\"sqrt(2) / 3\""),
        "sqrt(2*x^2+1)/(-3*x) should resolve to sqrt(2)/3 at -∞, got: {}",
        stdout_neg
    );
    assert!(
        !stdout_neg.contains("\"warning\""),
        "Resolved negative scaled-denominator sqrt-polynomial ratio should not warn at -∞, got: {}",
        stdout_neg
    );
}

#[test]
fn test_limit_sqrt_quadratic_over_noisy_scaled_linear_denominator() {
    let (success_pos, stdout_pos) = run_limit("sqrt(2*x^2+x+1)/(3*x+1)", "x", "infinity", "json");
    assert!(success_pos, "Command should succeed");
    assert!(
        stdout_pos.contains("\"result\":\"sqrt(2) / 3\""),
        "sqrt(2*x^2+x+1)/(3*x+1) should resolve to sqrt(2)/3 at +∞, got: {}",
        stdout_pos
    );
    assert!(
        !stdout_pos.contains("\"warning\""),
        "Resolved noisy scaled-denominator sqrt-polynomial ratio should not warn at +∞, got: {}",
        stdout_pos
    );

    let (success_neg, stdout_neg) = run_limit("sqrt(2*x^2+x+1)/(3*x+1)", "x", "-infinity", "json");
    assert!(success_neg, "Command should succeed");
    assert!(
        stdout_neg.contains("\"result\":\"-sqrt(2) / 3\""),
        "sqrt(2*x^2+x+1)/(3*x+1) should resolve to -sqrt(2)/3 at -∞, got: {}",
        stdout_neg
    );
    assert!(
        !stdout_neg.contains("\"warning\""),
        "Resolved noisy scaled-denominator negative-orientation sqrt-polynomial ratio should not warn at -∞, got: {}",
        stdout_neg
    );
}

#[test]
fn test_limit_sqrt_quadratic_with_bounded_radicand_noise() {
    let (success_pos, stdout_pos) =
        run_limit("sqrt((3*x+1)^2+sin(x))/(2*x+1)", "x", "infinity", "json");
    assert!(success_pos, "Command should succeed");
    assert!(
        stdout_pos.contains("\"result\":\"3/2\""),
        "sqrt((3*x+1)^2+sin(x))/(2*x+1) should resolve to 3/2 at +∞, got: {}",
        stdout_pos
    );
    assert!(
        !stdout_pos.contains("\"warning\""),
        "Resolved bounded-noise sqrt-polynomial ratio should not warn at +∞, got: {}",
        stdout_pos
    );

    let (success_neg, stdout_neg) =
        run_limit("sqrt((3*x+1)^2+sin(x))/(2*x+1)", "x", "-infinity", "json");
    assert!(success_neg, "Command should succeed");
    assert!(
        stdout_neg.contains("\"result\":\"-3/2\""),
        "sqrt((3*x+1)^2+sin(x))/(2*x+1) should resolve to -3/2 at -∞, got: {}",
        stdout_neg
    );
    assert!(
        !stdout_neg.contains("\"warning\""),
        "Resolved bounded-noise negative-orientation sqrt-polynomial ratio should not warn at -∞, got: {}",
        stdout_neg
    );
}

#[test]
fn test_limit_sqrt_quadratic_with_bounded_external_noise() {
    let (success_pos, stdout_pos) = run_limit(
        "sqrt((3*x+1)^2+sin(x))/(2*x+1+cos(x))",
        "x",
        "infinity",
        "json",
    );
    assert!(success_pos, "Command should succeed");
    assert!(
        stdout_pos.contains("\"result\":\"3/2\""),
        "sqrt((3*x+1)^2+sin(x))/(2*x+1+cos(x)) should resolve to 3/2 at +∞, got: {}",
        stdout_pos
    );
    assert!(
        !stdout_pos.contains("\"warning\""),
        "Resolved bounded external denominator noise should not warn at +∞, got: {}",
        stdout_pos
    );

    let (success_neg, stdout_neg) = run_limit(
        "sqrt((3*x+1)^2+sin(x))/(2*x+1+cos(x))",
        "x",
        "-infinity",
        "json",
    );
    assert!(success_neg, "Command should succeed");
    assert!(
        stdout_neg.contains("\"result\":\"-3/2\""),
        "sqrt((3*x+1)^2+sin(x))/(2*x+1+cos(x)) should resolve to -3/2 at -∞, got: {}",
        stdout_neg
    );
    assert!(
        !stdout_neg.contains("\"warning\""),
        "Resolved bounded external denominator noise should not warn at -∞, got: {}",
        stdout_neg
    );

    let (success_inverse_pos, stdout_inverse_pos) = run_limit(
        "(2*x+1+cos(x))/sqrt((3*x+1)^2+sin(x))",
        "x",
        "infinity",
        "json",
    );
    assert!(success_inverse_pos, "Command should succeed");
    assert!(
        stdout_inverse_pos.contains("\"result\":\"2/3\""),
        "(2*x+1+cos(x))/sqrt((3*x+1)^2+sin(x)) should resolve to 2/3 at +∞, got: {}",
        stdout_inverse_pos
    );
    assert!(
        !stdout_inverse_pos.contains("\"warning\""),
        "Resolved bounded external numerator noise should not warn at +∞, got: {}",
        stdout_inverse_pos
    );

    let (success_inverse_neg, stdout_inverse_neg) = run_limit(
        "(2*x+1+cos(x))/sqrt((3*x+1)^2+sin(x))",
        "x",
        "-infinity",
        "json",
    );
    assert!(success_inverse_neg, "Command should succeed");
    assert!(
        stdout_inverse_neg.contains("\"result\":\"-2/3\""),
        "(2*x+1+cos(x))/sqrt((3*x+1)^2+sin(x)) should resolve to -2/3 at -∞, got: {}",
        stdout_inverse_neg
    );
    assert!(
        !stdout_inverse_neg.contains("\"warning\""),
        "Resolved bounded external numerator noise should not warn at -∞, got: {}",
        stdout_inverse_neg
    );
}

#[test]
fn test_limit_sqrt_quadratic_with_negative_bounded_external_noise() {
    let (success_pos, stdout_pos) = run_limit(
        "sqrt((3*x+1)^2+sin(x))/(1-2*x+cos(x))",
        "x",
        "infinity",
        "json",
    );
    assert!(success_pos, "Command should succeed");
    assert!(
        stdout_pos.contains("\"result\":\"-3/2\""),
        "sqrt((3*x+1)^2+sin(x))/(1-2*x+cos(x)) should resolve to -3/2 at +∞, got: {}",
        stdout_pos
    );
    assert!(
        !stdout_pos.contains("\"warning\""),
        "Resolved negative bounded external denominator noise should not warn at +∞, got: {}",
        stdout_pos
    );

    let (success_neg, stdout_neg) = run_limit(
        "sqrt((3*x+1)^2+sin(x))/(1-2*x+cos(x))",
        "x",
        "-infinity",
        "json",
    );
    assert!(success_neg, "Command should succeed");
    assert!(
        stdout_neg.contains("\"result\":\"3/2\""),
        "sqrt((3*x+1)^2+sin(x))/(1-2*x+cos(x)) should resolve to 3/2 at -∞, got: {}",
        stdout_neg
    );
    assert!(
        !stdout_neg.contains("\"warning\""),
        "Resolved negative bounded external denominator noise should not warn at -∞, got: {}",
        stdout_neg
    );

    let (success_inverse_pos, stdout_inverse_pos) = run_limit(
        "(1-2*x+cos(x))/sqrt((3*x+1)^2+sin(x))",
        "x",
        "infinity",
        "json",
    );
    assert!(success_inverse_pos, "Command should succeed");
    assert!(
        stdout_inverse_pos.contains("\"result\":\"-2/3\""),
        "(1-2*x+cos(x))/sqrt((3*x+1)^2+sin(x)) should resolve to -2/3 at +∞, got: {}",
        stdout_inverse_pos
    );
    assert!(
        !stdout_inverse_pos.contains("\"warning\""),
        "Resolved negative bounded external numerator noise should not warn at +∞, got: {}",
        stdout_inverse_pos
    );

    let (success_inverse_neg, stdout_inverse_neg) = run_limit(
        "(1-2*x+cos(x))/sqrt((3*x+1)^2+sin(x))",
        "x",
        "-infinity",
        "json",
    );
    assert!(success_inverse_neg, "Command should succeed");
    assert!(
        stdout_inverse_neg.contains("\"result\":\"2/3\""),
        "(1-2*x+cos(x))/sqrt((3*x+1)^2+sin(x)) should resolve to 2/3 at -∞, got: {}",
        stdout_inverse_neg
    );
    assert!(
        !stdout_inverse_neg.contains("\"warning\""),
        "Resolved negative bounded external numerator noise should not warn at -∞, got: {}",
        stdout_inverse_neg
    );
}

#[test]
fn test_limit_sqrt_quadratic_with_constant_scaled_bounded_external_noise() {
    let (success_scaled_num, stdout_scaled_num) = run_limit(
        "5*sqrt((3*x+1)^2+sin(x))/(2*x+1+cos(x))",
        "x",
        "infinity",
        "json",
    );
    assert!(success_scaled_num, "Command should succeed");
    assert!(
        stdout_scaled_num.contains("\"result\":\"15/2\""),
        "5*sqrt((3*x+1)^2+sin(x))/(2*x+1+cos(x)) should resolve to 15/2 at +∞, got: {}",
        stdout_scaled_num
    );
    assert!(
        !stdout_scaled_num.contains("\"warning\""),
        "Resolved scaled bounded numerator noise should not warn at +∞, got: {}",
        stdout_scaled_num
    );

    let (success_scaled_den, stdout_scaled_den) = run_limit(
        "sqrt((3*x+1)^2+sin(x))/(2*(2*x+1+cos(x)))",
        "x",
        "infinity",
        "json",
    );
    assert!(success_scaled_den, "Command should succeed");
    assert!(
        stdout_scaled_den.contains("\"result\":\"3/4\""),
        "sqrt((3*x+1)^2+sin(x))/(2*(2*x+1+cos(x))) should resolve to 3/4 at +∞, got: {}",
        stdout_scaled_den
    );
    assert!(
        !stdout_scaled_den.contains("\"warning\""),
        "Resolved scaled bounded denominator noise should not warn at +∞, got: {}",
        stdout_scaled_den
    );
}

#[test]
fn test_limit_sqrt_quadratic_composes_with_explicit_rational_constants() {
    let (success_sum, stdout_sum) = run_limit(
        "sqrt((3*x+1)^2+sin(x))/(2*x+1+cos(x))+1/6",
        "x",
        "infinity",
        "json",
    );
    assert!(success_sum, "Command should succeed");
    assert!(
        stdout_sum.contains("\"result\":\"5/3\""),
        "resolved bounded sqrt ratio plus 1/6 should fold to 5/3, got: {}",
        stdout_sum
    );
    assert!(
        !stdout_sum.contains("\"warning\""),
        "resolved bounded sqrt ratio plus rational constant should not warn, got: {}",
        stdout_sum
    );

    let (success_diff, stdout_diff) = run_limit(
        "sqrt((3*x+1)^2+sin(x))/(2*x+1+cos(x))-1/2",
        "x",
        "infinity",
        "json",
    );
    assert!(success_diff, "Command should succeed");
    assert!(
        stdout_diff.contains("\"result\":\"1\""),
        "resolved bounded sqrt ratio minus 1/2 should fold to 1, got: {}",
        stdout_diff
    );
    assert!(
        !stdout_diff.contains("\"warning\""),
        "resolved bounded sqrt ratio minus rational constant should not warn, got: {}",
        stdout_diff
    );

    let (success_product, stdout_product) = run_limit(
        "sqrt((3*x+1)^2+sin(x))/(2*x+1+cos(x))*(2/3)",
        "x",
        "infinity",
        "json",
    );
    assert!(success_product, "Command should succeed");
    assert!(
        stdout_product.contains("\"result\":\"1\""),
        "resolved bounded sqrt ratio times 2/3 should fold to 1, got: {}",
        stdout_product
    );
    assert!(
        !stdout_product.contains("\"warning\""),
        "resolved bounded sqrt ratio times rational constant should not warn, got: {}",
        stdout_product
    );

    let (success_quotient, stdout_quotient) = run_limit(
        "sqrt((3*x+1)^2+sin(x))/(2*x+1+cos(x))/(3/2)",
        "x",
        "infinity",
        "json",
    );
    assert!(success_quotient, "Command should succeed");
    assert!(
        stdout_quotient.contains("\"result\":\"1\""),
        "resolved bounded sqrt ratio divided by 3/2 should fold to 1, got: {}",
        stdout_quotient
    );
    assert!(
        !stdout_quotient.contains("\"warning\""),
        "resolved bounded sqrt ratio divided by rational constant should not warn, got: {}",
        stdout_quotient
    );
}

#[test]
fn test_limit_sqrt_quadratic_does_not_treat_symbolic_addend_as_numeric_constant() {
    let (success, stdout) = run_limit(
        "sqrt((3*x+1)^2+sin(x))/(2*x+1+cos(x))+y",
        "x",
        "infinity",
        "json",
    );
    assert!(success, "Command should succeed");
    assert!(
        stdout.contains("\"result\":\"y + 3/2\""),
        "symbolic addend should remain symbolic after resolved limit, got: {}",
        stdout
    );
    assert!(
        !stdout.contains("\"warning\""),
        "resolved bounded sqrt ratio plus symbolic addend should not warn, got: {}",
        stdout
    );
}

#[test]
fn test_limit_linear_over_sqrt_quadratic_with_surd_coefficient() {
    let (success_pos, stdout_pos) = run_limit("x/sqrt(2*x^2+1)", "x", "infinity", "json");
    assert!(success_pos, "Command should succeed");
    assert!(
        stdout_pos.contains("\"result\":\"sqrt(2) / 2\""),
        "x/sqrt(2*x^2+1) should resolve to sqrt(2)/2 at +∞, got: {}",
        stdout_pos
    );
    assert!(
        !stdout_pos.contains("\"warning\""),
        "Resolved polynomial-over-sqrt ratio should not warn at +∞, got: {}",
        stdout_pos
    );

    let (success_neg, stdout_neg) = run_limit("x/sqrt(2*x^2+1)", "x", "-infinity", "json");
    assert!(success_neg, "Command should succeed");
    assert!(
        stdout_neg.contains("\"result\":\"-sqrt(2) / 2\""),
        "x/sqrt(2*x^2+1) should resolve to -sqrt(2)/2 at -∞, got: {}",
        stdout_neg
    );
    assert!(
        !stdout_neg.contains("\"warning\""),
        "Resolved negative-orientation polynomial-over-sqrt ratio should not warn at -∞, got: {}",
        stdout_neg
    );
}

#[test]
fn test_limit_polynomial_over_sqrt_quadratic_ignores_lower_order_noise() {
    let (success_pos, stdout_pos) = run_limit("(3*x+1)/sqrt(2*x^2+x+1)", "x", "infinity", "json");
    assert!(success_pos, "Command should succeed");
    assert!(
        stdout_pos.contains("\"result\":\"3·sqrt(2) / 2\""),
        "(3*x+1)/sqrt(2*x^2+x+1) should resolve to 3*sqrt(2)/2 at +∞, got: {}",
        stdout_pos
    );
    assert!(
        !stdout_pos.contains("\"warning\""),
        "Resolved noisy polynomial-over-sqrt ratio should not warn at +∞, got: {}",
        stdout_pos
    );

    let (success_neg, stdout_neg) = run_limit("(3*x+1)/sqrt(2*x^2+x+1)", "x", "-infinity", "json");
    assert!(success_neg, "Command should succeed");
    assert!(
        stdout_neg.contains("\"result\":\"-3·sqrt(2) / 2\""),
        "(3*x+1)/sqrt(2*x^2+x+1) should resolve to -3*sqrt(2)/2 at -∞, got: {}",
        stdout_neg
    );
    assert!(
        !stdout_neg.contains("\"warning\""),
        "Resolved noisy negative-orientation polynomial-over-sqrt ratio should not warn at -∞, got: {}",
        stdout_neg
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
fn test_limit_bounded_trig_times_decaying_exp_pos_infinity() {
    for expr in ["sin(x)*exp(-x)", "exp(-2*x)*cos(x)"] {
        let (success, stdout) = run_limit(expr, "x", "infinity", "json");
        assert!(success, "Command should succeed for {expr}");
        assert!(
            stdout.contains("\"result\":\"0\""),
            "{expr} should resolve to 0, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved bounded-trig-times-decaying-exp dominance should not warn for {expr}, got: {stdout}"
        );
    }
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
