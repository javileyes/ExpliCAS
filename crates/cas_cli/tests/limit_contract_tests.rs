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

fn run_eval_with_steps(expr: &str) -> (bool, String) {
    let binary = cas_cli_binary();

    let output = Command::new(&binary)
        .args(["eval", expr, "--format=json", "--steps=on"])
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
fn test_limit_bounded_composed_over_divergent_domain_policy_json() {
    for (input, to) in [
        ("sin(sqrt(x))/x", "infinity"),
        ("sin(ln(x))/x", "infinity"),
        ("sin(sqrt(-x))/x", "-infinity"),
        ("sin(ln(-x))/x", "-infinity"),
    ] {
        let (success, stdout) = run_limit(input, "x", to, "json");
        assert!(success, "Command should succeed for {input}");
        let wire: Value = serde_json::from_str(&stdout).expect("limit json");
        assert_eq!(wire["result"], json!("0"), "input: {input}");
        assert_eq!(wire["warning"], Value::Null, "input: {input}");
    }

    for (input, to, residual) in [
        (
            "sin(sqrt(1 - x))/x",
            "infinity",
            "limit(sin(sqrt(1 - x)) / x, x, infinity)",
        ),
        (
            "sin(ln(1 - x))/x",
            "infinity",
            "limit(sin(ln(1 - x)) / x, x, infinity)",
        ),
        (
            "sin(sqrt(x))/x",
            "-infinity",
            "limit(sin(sqrt(x)) / x, x, -infinity)",
        ),
        (
            "sin(ln(x))/x",
            "-infinity",
            "limit(sin(ln(x)) / x, x, -infinity)",
        ),
    ] {
        let (success, stdout) = run_limit(input, "x", to, "json");
        assert!(success, "Command should succeed for residual {input}");
        let wire: Value = serde_json::from_str(&stdout).expect("limit json");
        assert_eq!(wire["result"], json!(residual), "input: {input}");
        assert!(
            wire["warning"].as_str().is_some_and(|warning| {
                warning.contains("Could not determine limit safely")
            }),
            "domain-conflict bounded numerator should remain residual with warning for {input}, got: {wire:?}"
        );
    }
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
fn test_eval_one_sided_limit_orientation_policy_json() {
    for (input, expected_result) in [
        ("limit(abs(x)/x, x, 0+)", "1"),
        ("limit(abs(x)/x, x, 0-)", "-1"),
        ("limit(abs(x)/x, x, 0, right)", "1"),
        ("limit(1/x, x, 0+)", "infinity"),
        ("limit(1/x, x, 0-)", "-infinity"),
    ] {
        let (_success, stdout) = run_eval(input, "json");
        let wire: Value = serde_json::from_str(&stdout).expect("eval json");
        assert_eq!(wire["ok"], true, "input: {input}");
        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(wire["warnings"], json!([]), "input: {input}");
        assert!(
            wire["required_display"].as_array().is_some_and(|conditions| {
                conditions.iter().any(|condition| condition == "x ≠ 0")
            }),
            "One-sided orientation limit should preserve denominator domain for {input}, got: {wire:?}"
        );
    }

    for input in [
        "limit(ln(x), x, 0+)",
        "limit(log2(x), x, 0+)",
        "limit(log10(x), x, 0+)",
        "limit(ln((x-1)/(x+3)), x, 1+)",
        "limit(log2((x-1)/(x+3)), x, 1+)",
    ] {
        let (_success, stdout) = run_eval(input, "json");
        let wire: Value = serde_json::from_str(&stdout).expect("eval json");
        assert_eq!(wire["ok"], true, "input: {input}");
        assert_eq!(wire["result"], "-infinity", "input: {input}");
        assert_eq!(wire["warnings"], json!([]), "input: {input}");
        let expected_condition = if input.contains("(x-1)/(x+3)") {
            "x < -3 or x > 1"
        } else {
            "x > 0"
        };
        assert!(
            wire["required_display"].as_array().is_some_and(|conditions| {
                conditions
                    .iter()
                    .any(|condition| condition == expected_condition)
            }),
            "One-sided log endpoint should preserve positive-domain condition for {input}, got: {wire:?}"
        );
    }

    for (input, expected_result, expected_conditions) in [
        (
            "limit(log(x-1/2, (x-1)/(x+3)), x, 1+)",
            "infinity",
            &["x < -3 or x > 1", "x > 1/2", "x ≠ 3/2"][..],
        ),
        (
            "limit(log(x, (x-1)/(x+3)), x, 1+)",
            "-infinity",
            &["x < -3 or x > 1", "x > 0"][..],
        ),
        (
            "limit(log(2-x, (x-1)/(x+3)), x, 1+)",
            "infinity",
            &["x < -3 or x > 1", "x < 2"][..],
        ),
        (
            "limit(log((x+2)/(2*x+1), (x-1)/(x+3)), x, 1+)",
            "infinity",
            &["x < -2 or x > -1/2", "x < -3 or x > 1"][..],
        ),
    ] {
        let (_success, stdout) = run_eval(input, "json");
        let wire: Value = serde_json::from_str(&stdout).expect("eval json");
        assert_eq!(wire["ok"], true, "input: {input}");
        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(wire["warnings"], json!([]), "input: {input}");
        for expected_condition in expected_conditions {
            assert!(
                wire["required_display"].as_array().is_some_and(|conditions| {
                    conditions
                        .iter()
                        .any(|condition| condition == expected_condition)
                }),
                "One-sided variable-base log endpoint should preserve domain condition {expected_condition} for {input}, got: {wire:?}"
            );
        }
    }

    for (input, expected_condition) in [
        ("limit(sqrt(x), x, 0+)", "x ≥ 0"),
        ("limit(sqrt(-x), x, 0-)", "x ≤ 0"),
        ("limit(sqrt((x-1)/(x+3)), x, 1+)", "(x - 1) / (x + 3) ≥ 0"),
    ] {
        let (_success, stdout) = run_eval(input, "json");
        let wire: Value = serde_json::from_str(&stdout).expect("eval json");
        assert_eq!(wire["ok"], true, "input: {input}");
        assert_eq!(wire["result"], "0", "input: {input}");
        assert_eq!(wire["warnings"], json!([]), "input: {input}");
        assert!(
            wire["required_display"].as_array().is_some_and(|conditions| {
                conditions
                    .iter()
                    .any(|condition| condition == expected_condition)
            }),
            "One-sided sqrt endpoint should preserve nonnegative-domain condition for {input}, got: {wire:?}"
        );
        if input.contains("(x-1)/(x+3)") {
            assert!(
                wire["required_display"].as_array().is_some_and(|conditions| {
                    conditions.iter().any(|condition| condition == "x ≠ -3")
                }),
                "One-sided rational sqrt endpoint should preserve denominator condition for {input}, got: {wire:?}"
            );
        }
    }

    for (input, expected_condition) in [
        ("limit(acosh(x), x, 1+)", "x ≥ 1"),
        ("limit(acosh(x+2), x, -1+)", "x ≥ -1"),
        ("limit(acosh(2-x), x, 1-)", "x ≤ 1"),
    ] {
        let (_success, stdout) = run_eval(input, "json");
        let wire: Value = serde_json::from_str(&stdout).expect("eval json");
        assert_eq!(wire["ok"], true, "input: {input}");
        assert_eq!(wire["result"], "0", "input: {input}");
        assert_eq!(wire["warnings"], json!([]), "input: {input}");
        assert!(
            wire["required_display"].as_array().is_some_and(|conditions| {
                conditions
                    .iter()
                    .any(|condition| condition == expected_condition)
            }),
            "One-sided acosh endpoint should preserve lower-bound-domain condition for {input}, got: {wire:?}"
        );
    }

    for (input, expected_result, expected_condition) in [
        ("limit(acos(x), x, 1-)", "0", "-1 ≤ x ≤ 1"),
        ("limit(asin(x), x, 1-)", "pi / 2", "-1 ≤ x ≤ 1"),
        ("limit(acos(x), x, -1+)", "pi", "-1 ≤ x ≤ 1"),
        ("limit(asin(x), x, -1+)", "-pi / 2", "-1 ≤ x ≤ 1"),
        ("limit(acos(2-x), x, 1+)", "0", "1 ≤ x ≤ 3"),
    ] {
        let (_success, stdout) = run_eval(input, "json");
        let wire: Value = serde_json::from_str(&stdout).expect("eval json");
        assert_eq!(wire["ok"], true, "input: {input}");
        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(wire["warnings"], json!([]), "input: {input}");
        assert!(
            wire["required_display"].as_array().is_some_and(|conditions| {
                conditions
                    .iter()
                    .any(|condition| condition == expected_condition)
            }),
            "One-sided inverse-trig endpoint should preserve interval-domain condition for {input}, got: {wire:?}"
        );
    }

    for (input, expected_result, expected_condition) in [
        ("limit(atanh(x), x, 1-)", "infinity", "-1 < x < 1"),
        ("limit(atanh(x), x, -1+)", "-infinity", "-1 < x < 1"),
        ("limit(atanh(2-x), x, 1+)", "infinity", "1 < x < 3"),
    ] {
        let (_success, stdout) = run_eval(input, "json");
        let wire: Value = serde_json::from_str(&stdout).expect("eval json");
        assert_eq!(wire["ok"], true, "input: {input}");
        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(wire["warnings"], json!([]), "input: {input}");
        assert!(
            wire["required_display"].as_array().is_some_and(|conditions| {
                conditions
                    .iter()
                    .any(|condition| condition == expected_condition)
            }),
            "One-sided atanh endpoint should preserve open-interval-domain condition for {input}, got: {wire:?}"
        );
    }

    for (input, expected_result, expected_condition) in [
        (
            "limit(acos(x), x, 1+)",
            "limit(acos(x), x, 1, right)",
            "-1 ≤ x ≤ 1",
        ),
        (
            "limit(asin(x), x, -1-)",
            "limit(asin(x), x, -1, left)",
            "-1 ≤ x ≤ 1",
        ),
        (
            "limit(acos(1+x^2), x, 0+)",
            "limit(acos(x^2 + 1), x, 0, right)",
            "-1 ≤ x^2 + 1 ≤ 1",
        ),
        (
            "limit(atanh(x), x, 1+)",
            "limit(atanh(x), x, 1, right)",
            "-1 < x < 1",
        ),
        (
            "limit(atanh(x), x, -1-)",
            "limit(atanh(x), x, -1, left)",
            "-1 < x < 1",
        ),
        (
            "limit(atanh(1+x^2), x, 0+)",
            "limit(atanh(x^2 + 1), x, 0, right)",
            "1 - (x^2 + 1)^2 > 0",
        ),
    ] {
        let (_success, stdout) = run_eval(input, "json");
        let wire: Value = serde_json::from_str(&stdout).expect("eval json");
        assert_eq!(wire["ok"], true, "input: {input}");
        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert!(
            wire["required_display"].as_array().is_some_and(|conditions| {
                conditions
                    .iter()
                    .any(|condition| condition == expected_condition)
            }),
            "Unsupported one-sided inverse interval endpoint should preserve required condition for {input}, got: {wire:?}"
        );
        assert!(
            wire["warnings"].as_array().is_some_and(|warnings| {
                warnings.iter().any(|warning| {
                    warning["rule"] == "Limit Evaluation"
                        && warning["assumption"].as_str().is_some_and(|message| {
                            message.contains(
                                "One-sided finite point limits are not supported safely for this expression yet",
                            )
                        })
                }) && warnings.iter().any(|warning| {
                    warning["rule"] == "Limit Domain Path"
                        && warning["assumption"].as_str().is_some_and(|message| {
                            message.contains("Limit path conflicts with the input domain")
                                && message.contains(expected_condition)
                    })
                })
            }),
            "Unsupported one-sided inverse interval endpoint should stay residual with explicit domain-path warning for {input}, got: {wire:?}"
        );
    }

    for (input, expected_result, expected_required_condition, expected_warning_condition) in [
        (
            "limit(ln(x), x, 0-)",
            "limit(ln(x), x, 0, left)",
            "x > 0",
            "x > 0",
        ),
        (
            "limit(sqrt(x), x, 0-)",
            "limit(sqrt(x), x, 0, left)",
            "x ≥ 0",
            "x ≥ 0",
        ),
        (
            "limit(sqrt(x+1), x, -1-)",
            "limit(sqrt(x + 1), x, -1, left)",
            "x ≥ -1",
            "x ≥ -1",
        ),
        (
            "limit(sqrt(1-x), x, 1+)",
            "limit(sqrt(1 - x), x, 1, right)",
            "x ≤ 1",
            "x ≤ 1",
        ),
        (
            "limit(ln((x-1)/(x+3)), x, 1-)",
            "limit(ln((x - 1) / (x + 3)), x, 1, left)",
            "x < -3 or x > 1",
            "(x - 1) / (x + 3) > 0",
        ),
        (
            "limit(sqrt((x-1)/(x+3)), x, 1-)",
            "limit(sqrt((x - 1) / (x + 3)), x, 1, left)",
            "(x - 1) / (x + 3) ≥ 0",
            "(x - 1) / (x + 3) ≥ 0",
        ),
        (
            "limit(acosh(x), x, 1-)",
            "limit(acosh(x), x, 1, left)",
            "x ≥ 1",
            "x ≥ 1",
        ),
    ] {
        let (_success, stdout) = run_eval(input, "json");
        let wire: Value = serde_json::from_str(&stdout).expect("eval json");
        assert_eq!(wire["ok"], true, "input: {input}");
        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert!(
            wire["required_display"].as_array().is_some_and(|conditions| {
                conditions
                    .iter()
                    .any(|condition| condition == expected_required_condition)
            }),
            "Domain-conflicting one-sided endpoint should preserve required condition for {input}, got: {wire:?}"
        );
        assert!(
            wire["warnings"].as_array().is_some_and(|warnings| {
                warnings.iter().any(|warning| {
                    warning["rule"] == "Limit Evaluation"
                        && warning["assumption"].as_str().is_some_and(|message| {
                            message.contains(
                                "One-sided finite point limits are not supported safely for this expression yet",
                            )
                        })
                }) && warnings.iter().any(|warning| {
                    warning["rule"] == "Limit Domain Path"
                        && warning["assumption"].as_str().is_some_and(|message| {
                            message.contains("Limit path conflicts with the input domain")
                                && message.contains(expected_warning_condition)
                        })
                })
            }),
            "Unsupported one-sided endpoint should stay residual with explicit domain-path warning for {input}, got: {wire:?}"
        );
    }
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
fn test_eval_limit_steps_on_emits_limit_trace_json() {
    let (success, stdout) = run_eval_with_steps("limit(x^2 + x + 1, x, -2)");
    assert!(success, "Command should succeed with steps enabled");
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");

    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "3");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps[0]["rule"], "Evaluar límite finito");
    assert!(
        steps[0]["after"].as_str().is_some_and(|after| after == "3"),
        "Limit trace should show the evaluated result, got: {wire:?}"
    );
}

#[test]
fn test_eval_residual_limit_result_matches_display_cleanup_trace_json() {
    let (success, stdout) = run_eval_with_steps("limit(sqrt(x + 0), x, 0)");
    assert!(
        success,
        "Command should keep residual finite sqrt endpoint limit"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");

    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "limit(sqrt(x), x, 0)");
    assert_eq!(wire["required_display"], json!(["x ≥ 0"]));
    assert!(
        wire["warnings"].as_array().is_some_and(|warnings| {
            warnings.iter().any(|warning| {
                warning["rule"] == "Limit Evaluation"
                    && warning["assumption"].as_str().is_some_and(|message| {
                        message.contains("Finite point limits are not supported safely yet")
                    })
            })
        }),
        "Residual finite limit should keep the safety warning, got: {wire:?}"
    );

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps[0]["rule"], "Conservar límite residual");
    assert_eq!(steps[0]["after"], "limit(sqrt(x), x, 0)");

    let (success, stdout) = run_eval_with_steps("limit(0*sqrt(x), x, 0)");
    assert!(
        success,
        "Command should keep domain-sensitive residual products visible"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "limit(0·sqrt(x), x, 0)");
    assert_eq!(wire["required_display"], json!(["x ≥ 0"]));
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps[0]["rule"], "Conservar límite residual");
    assert_eq!(steps[0]["before"], "0·sqrt(x)");
    assert_eq!(steps[0]["after"], "limit(0·sqrt(x), x, 0)");
}

#[test]
fn test_eval_finite_rational_polynomial_limit_handles_removable_holes_json() {
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
        "Command should resolve exact removable finite rational-polynomial limit"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "-1");
    assert_eq!(wire["warnings"], json!([]));
    assert_eq!(wire["required_display"], json!(["x ≠ -2"]));

    let (success, stdout) = run_eval("limit((x^2-1)/(x-1), x, 1)", "json");
    assert!(
        success,
        "Command should resolve simple exact removable finite rational-polynomial limit"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "2");
    assert_eq!(wire["warnings"], json!([]));
    assert_eq!(wire["required_display"], json!(["x ≠ 1"]));

    let (success, stdout) = run_eval("limit((x-1)/(x-1)^2, x, 1)", "json");
    assert!(
        success,
        "Command should keep finite rational-polynomial poles residual"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert!(
        wire["result"]
            .as_str()
            .is_some_and(|result| result.contains("limit(")),
        "Finite rational pole should remain residual, got: {wire:?}"
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
fn test_eval_static_empty_real_domain_limit_returns_undefined_json() {
    for input in [
        "limit(log(1,2), x, 0)",
        "limit(log(-2,2), x, 0)",
        "limit(ln(0), x, 0)",
        "limit(sqrt(-1), x, 0)",
        "limit(log(1,2), x, infinity)",
    ] {
        let (success, stdout) = run_eval_with_steps(input);
        assert!(
            success,
            "Command should resolve static empty-domain limit for {input}"
        );
        let wire: Value = serde_json::from_str(&stdout).expect("eval json");
        assert_eq!(wire["ok"], true, "input: {input}");
        assert_eq!(
            wire["result"], "undefined",
            "static empty-domain limit should be undefined for {input}: {wire:?}"
        );
        assert_eq!(
            wire["required_display"],
            json!([]),
            "static empty-domain limit should not add assumptions for {input}"
        );
        assert_eq!(
            wire["warnings"],
            json!([]),
            "static empty-domain limit should not fall through to residual warning for {input}"
        );
        assert!(
            wire["steps"].to_string().contains("undefined"),
            "limit trace should expose the undefined result for {input}: {wire:?}"
        );
    }

    for input in ["limit(log(y,2), x, 0)", "limit(sqrt(y), x, 0)"] {
        let (success, stdout) = run_eval(input, "json");
        assert!(success, "Command should preserve symbolic static domains");
        let wire: Value = serde_json::from_str(&stdout).expect("eval json");
        assert!(
            wire["result"]
                .as_str()
                .is_some_and(|result| !result.contains("undefined")),
            "symbolic static domain should remain explicit, not undefined for {input}: {wire:?}"
        );
    }
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
        ("limit(arctan(x + 1), x, 0)", "pi / 4"),
        ("limit(atan(x^2 + 1), x, -1)", "atan(2)"),
        ("limit(arctan(x^2 + 1), x, -1)", "arctan(2)"),
        ("limit(asinh(x), x, 0)", "0"),
        ("limit(asinh(x^2 + 1), x, -1)", "asinh(2)"),
        ("limit(cbrt(x), x, -8)", "-2"),
        ("limit(cbrt(x), x, 0)", "0"),
        ("limit(cbrt(x^2 + 1), x, -1)", "cbrt(2)"),
        ("limit((x^2 - 9)^(1/3), x, 1)", "-2"),
        ("limit(sin(cbrt(x)), x, 8)", "sin(2)"),
        ("limit(sin(x + pi/6), x, 0)", "1 / 2"),
        ("limit(cos(x + pi/3), x, 0)", "1 / 2"),
        ("limit(arctan(sqrt(x^2 + 3)), x, 0)", "pi / 3"),
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
    for input in [
        "limit(arcsin(x), x, 2)",
        "limit(arcsin(x), x, 1)",
        "limit(arcsin(x), x, -1)",
        "limit(acos(x), x, 1)",
        "limit(acos(x), x, -1)",
        "limit(acos(1+x^2), x, 0)",
        "limit(acos(1-x^3), x, 0)",
        "limit(acos(-1-x^2), x, 0)",
        "limit(acos(-1+x^3), x, 0)",
        "limit(atanh(x), x, 1)",
        "limit(acosh(x), x, 1)",
        "limit(acosh(sqrt(x)), x, 1)",
        "limit(acosh(1+x^3), x, 0)",
        "limit(acosh(1-x^2), x, 0)",
    ] {
        let (success, stdout) = run_eval(input, "json");
        assert!(
            success,
            "Command should keep unsafe finite partial-domain inverse limit residual for {input}"
        );
        let wire: Value = serde_json::from_str(&stdout).expect("eval json");
        assert_eq!(wire["ok"], true, "input: {input}");
        assert!(
            wire["result"]
                .as_str()
                .is_some_and(|result| result.contains("limit(")),
            "Partial-domain inverse limit should remain residual for {input}: {wire:?}"
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
            "Partial-domain inverse limit should remain residual with warning for {input}, got: {wire:?}"
        );
    }
}

#[test]
fn test_eval_finite_inverse_trig_bilateral_endpoint_policy_json() {
    for (input, expected_result, expected_required) in [
        ("limit(acos(1-x^2), x, 0)", "0", "-1 ≤ x^2 - 1 ≤ 1"),
        ("limit(asin(1-x^2), x, 0)", "pi / 2", "-1 ≤ x^2 - 1 ≤ 1"),
        ("limit(acos(-1+x^2), x, 0)", "pi", "-1 ≤ x^2 - 1 ≤ 1"),
        ("limit(asin(-1+x^2), x, 0)", "-pi / 2", "-1 ≤ x^2 - 1 ≤ 1"),
    ] {
        let (success, stdout) = run_eval(input, "json");
        assert!(
            success,
            "Command should resolve bilateral inverse-trig endpoint for {input}"
        );
        let wire: Value = serde_json::from_str(&stdout).expect("eval json");
        assert_eq!(wire["ok"], true, "input: {input}, wire: {wire:?}");
        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(wire["warnings"], json!([]), "input: {input}");
        assert!(
            wire["required_display"].as_array().is_some_and(|conditions| {
                conditions
                    .iter()
                    .any(|condition| condition == expected_required)
            }),
            "Bilateral inverse-trig endpoint should preserve interval-domain condition for {input}, got: {wire:?}"
        );
    }
}

#[test]
fn test_eval_finite_acosh_bilateral_lower_bound_endpoint_policy_json() {
    for (input, expected_required) in [
        ("limit(acosh(1+x^2), x, 0)", "x^2 + 1 ≥ 1"),
        ("limit(acosh(1+(x-2)^2), x, 2)", "(x - 2)^2 + 1 ≥ 1"),
    ] {
        let (success, stdout) = run_eval(input, "json");
        assert!(
            success,
            "Command should resolve bilateral acosh lower-bound endpoint for {input}"
        );
        let wire: Value = serde_json::from_str(&stdout).expect("eval json");
        assert_eq!(wire["ok"], true, "input: {input}, wire: {wire:?}");
        assert_eq!(wire["result"], "0", "input: {input}");
        assert_eq!(wire["warnings"], json!([]), "input: {input}");
        assert!(
            wire["required_display"].as_array().is_some_and(|conditions| {
                conditions
                    .iter()
                    .any(|condition| condition == expected_required)
            }),
            "Bilateral acosh endpoint should preserve lower-bound-domain condition for {input}, got: {wire:?}"
        );
    }
}

#[test]
fn test_eval_finite_partial_domain_inverse_elementary_limits_inside_domain_json() {
    let cases = [
        ("limit(arcsin(x/2), x, 0)", "0", json!(["-2 ≤ x ≤ 2"])),
        ("limit(atanh(x/2), x, 0)", "0", json!(["-2 < x < 2"])),
        ("limit(acos(x/2), x, 0)", "pi / 2", json!(["-2 ≤ x ≤ 2"])),
        (
            "limit(arcsin(x/2 + 1/2), x, 0)",
            "pi / 6",
            json!(["-3 ≤ x ≤ 1"]),
        ),
        (
            "limit(arcsin(1/2-x/2), x, 0)",
            "pi / 6",
            json!(["-1 ≤ x ≤ 3"]),
        ),
        (
            "limit(arccos(x/2 + 1/2), x, 0)",
            "pi / 3",
            json!(["-3 ≤ x ≤ 1"]),
        ),
        ("limit(acosh(x+2), x, 0)", "acosh(2)", json!(["x ≥ -1"])),
    ];

    for (input, expected, expected_required) in cases {
        let (success, stdout) = run_eval(input, "json");
        assert!(
            success,
            "Command should resolve strict-interior finite partial-domain inverse limit for {input}"
        );
        let wire: Value = serde_json::from_str(&stdout).expect("eval json");
        assert_eq!(wire["ok"], true, "input: {input}, wire: {wire:?}");
        assert_eq!(wire["result"], expected, "input: {input}");
        assert!(
            !wire["result"]
                .as_str()
                .unwrap_or_default()
                .contains("limit("),
            "strict-interior finite partial-domain inverse limit should not remain residual for {input}: {wire:?}"
        );
        assert_eq!(wire["warnings"], json!([]), "input: {input}");
        assert_eq!(
            wire["required_display"], expected_required,
            "input: {input}"
        );
    }
}

#[test]
fn test_eval_finite_shifted_atanh_limit_inside_domain_json() {
    let (success, stdout) = run_eval("limit(atanh(x/3+1/3), x, 0)", "json");
    assert!(
        success,
        "Command should resolve shifted affine atanh limit inside the real domain"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "atanh(1/3)");
    assert_eq!(wire["warnings"], json!([]));
    assert_eq!(wire["required_display"], json!(["-4 < x < 2"]));
}

#[test]
fn test_eval_finite_atanh_sqrt_limit_inside_domain_uses_radicand_domain_json() {
    let (success, stdout) = run_eval("limit(atanh(sqrt(x)), x, 1/4)", "json");
    assert!(
        success,
        "Command should resolve atanh sqrt limit inside the real domain"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "atanh(1/2)");
    assert_eq!(wire["warnings"], json!([]));
    assert_eq!(wire["required_display"], json!(["x < 1", "x ≥ 0"]));
}

#[test]
fn test_eval_finite_closed_inverse_sqrt_limits_use_radicand_domain_json() {
    let cases = [
        ("limit(arcsin(sqrt(x)), x, 1/4)", "pi / 6"),
        ("limit(acos(sqrt(x)), x, 1/4)", "pi / 3"),
    ];

    for (input, expected) in cases {
        let (success, stdout) = run_eval(input, "json");
        assert!(
            success,
            "Command should resolve closed inverse sqrt limit inside the real domain for {input}"
        );
        let wire: Value = serde_json::from_str(&stdout).expect("eval json");
        assert_eq!(wire["ok"], true, "input: {input}");
        assert_eq!(wire["result"], expected, "input: {input}");
        assert_eq!(wire["warnings"], json!([]), "input: {input}");
        assert_eq!(wire["required_display"], json!(["x ≤ 1", "x ≥ 0"]));
    }
}

#[test]
fn test_eval_finite_partial_inverse_sqrt_limits_accept_non_square_interior_json() {
    let cases = [
        (
            "limit(atanh(sqrt(2*x+3)), x, -5/4)",
            "atanh(sqrt(1/2))",
            json!(["x < -1", "x ≥ -3/2"]),
        ),
        (
            "limit(arcsin(sqrt(2*x+3)), x, -5/4)",
            "pi / 4",
            json!(["x ≤ -1", "x ≥ -3/2"]),
        ),
        (
            "limit(acos(sqrt(2*x+3)), x, -5/4)",
            "pi / 4",
            json!(["x ≤ -1", "x ≥ -3/2"]),
        ),
        (
            "limit(arcsin(sqrt(3*x+3)), x, -3/4)",
            "pi / 3",
            json!(["x ≤ -2/3", "x ≥ -1"]),
        ),
        (
            "limit(acos(sqrt(3*x+3)), x, -3/4)",
            "pi / 6",
            json!(["x ≤ -2/3", "x ≥ -1"]),
        ),
        ("limit(arctan(sqrt(x/3)), x, 1)", "pi / 6", json!(["x ≥ 0"])),
    ];

    for (input, expected, expected_required) in cases {
        let (success, stdout) = run_eval(input, "json");
        assert!(
            success,
            "Command should resolve inverse sqrt limit at non-square interior value for {input}"
        );
        let wire: Value = serde_json::from_str(&stdout).expect("eval json");
        assert_eq!(wire["ok"], true, "input: {input}");
        assert_eq!(wire["result"], expected, "input: {input}");
        assert_eq!(wire["warnings"], json!([]), "input: {input}");
        assert_eq!(
            wire["required_display"], expected_required,
            "input: {input}"
        );
    }
}

#[test]
fn test_eval_finite_scaled_acosh_limit_preserves_lower_bound_domain_json() {
    let (success, stdout) = run_eval("limit(acosh(x/2+2), x, 0)", "json");
    assert!(
        success,
        "Command should resolve scaled affine acosh limit inside the real domain"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "acosh(2)");
    assert_eq!(wire["warnings"], json!([]));
    assert_eq!(wire["required_display"], json!(["x ≥ -2"]));
}

#[test]
fn test_eval_finite_negatively_scaled_acosh_limit_preserves_oriented_domain_json() {
    let (success, stdout) = run_eval("limit(acosh(2-x/2), x, 0)", "json");
    assert!(
        success,
        "Command should resolve negatively scaled affine acosh limit inside the real domain"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "acosh(2)");
    assert_eq!(wire["warnings"], json!([]));
    assert_eq!(wire["required_display"], json!(["x ≤ 2"]));
}

#[test]
fn test_eval_finite_acosh_sqrt_limit_uses_radicand_lower_bound_json() {
    let cases = [
        ("limit(acosh(sqrt(x)), x, 4)", "acosh(2)", json!(["x ≥ 1"])),
        (
            "limit(acosh(sqrt(2*x+3)), x, 1)",
            "acosh(sqrt(5))",
            json!(["x ≥ -1"]),
        ),
        (
            "limit(acosh(sqrt(5-3*x)), x, 0)",
            "acosh(sqrt(5))",
            json!(["x ≤ 4/3"]),
        ),
        (
            "limit(acosh((2*x+3)^(1/2)), x, 1)",
            "acosh(sqrt(5))",
            json!(["x ≥ -1"]),
        ),
    ];

    for (input, expected, expected_required) in cases {
        let (success, stdout) = run_eval(input, "json");
        assert!(
            success,
            "Command should resolve acosh sqrt limit inside the real domain for {input}"
        );
        let wire: Value = serde_json::from_str(&stdout).expect("eval json");
        assert_eq!(wire["ok"], true, "input: {input}");
        assert_eq!(wire["result"], expected, "input: {input}");
        assert_eq!(wire["warnings"], json!([]), "input: {input}");
        assert_eq!(
            wire["required_display"], expected_required,
            "input: {input}"
        );
    }
}

#[test]
fn test_eval_finite_square_root_power_limit_reuses_sqrt_policy_json() {
    let (success, stdout) = run_eval("limit((2*x+3)^(1/2), x, 1)", "json");
    assert!(
        success,
        "Command should resolve finite square-root power limit at positive radicand"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "sqrt(5)");
    assert_eq!(wire["warnings"], json!([]));
    assert_eq!(wire["required_display"], json!(["x ≥ -3/2"]));

    let (success, stdout) = run_eval("limit(x^(1/2), x, 0)", "json");
    assert!(
        success,
        "Command should keep finite square-root power endpoint residual"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert!(
        wire["result"]
            .as_str()
            .is_some_and(|result| result.contains("limit(")),
        "square-root power endpoint must remain residual, got: {wire:?}"
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
        "square-root power endpoint should keep finite-limit warning, got: {wire:?}"
    );
    assert_eq!(wire["required_display"], json!(["x ≥ 0"]));
}

#[test]
fn test_eval_finite_negative_integer_power_limit_requires_nonzero_base_json() {
    let (success, stdout) = run_eval("limit((sqrt(x+4))^(-2), x, 0)", "json");
    assert!(
        success,
        "Command should resolve negative integer power over nonzero finite root sublimit"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "1/4");
    assert_eq!(wire["warnings"], json!([]));
    assert_eq!(wire["required_display"], json!(["x > -4"]));

    let (success, stdout) = run_eval("limit((sqrt(x))^(-2), x, 0)", "json");
    assert!(
        success,
        "Command should keep negative integer power over zero root sublimit residual"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "limit(sqrt(x)^(-2), x, 0)");
    assert!(
        wire["warnings"].as_array().is_some_and(|warnings| {
            warnings.iter().any(|warning| {
                warning["rule"] == "Limit Evaluation"
                    && warning["assumption"].as_str().is_some_and(|message| {
                        message.contains("Finite point limits are not supported safely yet")
                    })
            })
        }),
        "negative integer power over zero root sublimit should keep finite-limit warning, got: {wire:?}"
    );
    assert_eq!(wire["required_display"], json!(["x > 0"]));
}

#[test]
fn test_eval_finite_domain_checked_trig_limits_json() {
    let cases = [
        ("limit(tan(x), x, 0)", "0", json!(["cos(x) ≠ 0"])),
        ("limit(tan(x/2), x, 0)", "0", json!(["cos(x / 2) ≠ 0"])),
        (
            "limit(tan(x + pi/4), x, 0)",
            "1",
            json!(["cos(pi / 4 + x) ≠ 0"]),
        ),
        ("limit(sec(x), x, 0)", "1", json!(["cos(x) ≠ 0"])),
        ("limit(sec(x^2), x, 0)", "1", json!(["cos(x^2) ≠ 0"])),
        (
            "limit(sec(x + pi/3), x, 0)",
            "2",
            json!(["cos(pi / 3 + x) ≠ 0"]),
        ),
        (
            "limit(csc(x + pi/6), x, 0)",
            "2",
            json!(["sin(pi / 6 + x) ≠ 0"]),
        ),
        (
            "limit(cot(x + pi/4), x, 0)",
            "1",
            json!(["sin(pi / 4 + x) ≠ 0"]),
        ),
    ];

    for (input, expected, expected_required) in cases {
        let (success, stdout) = run_eval(input, "json");
        assert!(
            success,
            "Command should resolve domain-checked finite trig limit for {input}"
        );
        let wire: Value = serde_json::from_str(&stdout).expect("eval json");
        assert_eq!(wire["ok"], true, "input: {input}, wire: {wire:?}");
        assert_eq!(wire["result"], expected, "input: {input}");
        assert!(
            !wire["result"]
                .as_str()
                .unwrap_or_default()
                .contains("limit("),
            "domain-checked finite trig limit should not remain residual for {input}: {wire:?}"
        );
        assert_eq!(wire["warnings"], json!([]), "input: {input}");
        assert_eq!(
            wire["required_display"], expected_required,
            "input: {input}"
        );
    }
}

#[test]
fn test_eval_finite_domain_checked_trig_limits_reject_unsafe_cases_json() {
    for input in [
        "limit(tan(x), x, 1)",
        "limit(tan(x + pi/2), x, 0)",
        "limit(sec(x), x, 1)",
        "limit(sec(x + pi/2), x, 0)",
        "limit(csc(x), x, 0)",
        "limit(csc(x + pi), x, 0)",
        "limit(cot(x), x, 0)",
        "limit(cot(x + pi), x, 0)",
    ] {
        let (success, stdout) = run_eval(input, "json");
        assert!(
            success,
            "Command should keep unsafe finite trig limit residual for {input}"
        );
        let wire: Value = serde_json::from_str(&stdout).expect("eval json");
        assert_eq!(wire["ok"], true, "input: {input}");
        assert!(
            wire["result"]
                .as_str()
                .is_some_and(|result| result.contains("limit(")),
            "unsafe finite trig limit should remain residual for {input}: {wire:?}"
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
            "unsafe finite trig limit should remain residual with warning for {input}, got: {wire:?}"
        );
    }
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
        ("limit(log2(abs(x)), x, -2)", "1", json!(["x ≠ 0"])),
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
        ("limit(log2(x^2 + 4), x, 2)", "3", json!([])),
        ("limit(log10(x^2 + 96), x, 2)", "2", json!([])),
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
        ("limit(log(2, x^2 + 4), x, 2)", "3", json!([])),
        ("limit(log(1/2, x^2 + 4), x, 2)", "-3", json!([])),
        ("limit(log(4, x^2 + 4), x, 2)", "3/2", json!([])),
        ("limit(log(1/4, x^2 + 4), x, 2)", "-3/2", json!([])),
        ("limit(log(27, x^2 + 5), x, 2)", "2/3", json!([])),
        ("limit(log(2, x^2 + 1), x, 2)", "log(2, 5)", json!([])),
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
fn test_limit_exp_polynomial_argument_tails_at_infinity() {
    for (expr, approach, expected) in [
        ("exp(x^2)", "infinity", "infinity"),
        ("exp(x^2)", "-infinity", "infinity"),
        ("exp(2 - x^4)", "infinity", "0"),
        ("exp(x^3 - 2*x)", "-infinity", "0"),
    ] {
        let (success, stdout) = run_limit(expr, "x", approach, "json");
        assert!(success, "Command should succeed for {expr} at {approach}");
        assert!(
            stdout.contains(&format!("\"result\":\"{expected}\"")),
            "{expr} at {approach} should resolve to {expected}, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved polynomial exponential tail should not warn for {expr} at {approach}, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_exp_unsupported_argument_tails_remain_residual() {
    for expr in ["exp(a*x^2 + 1)", "exp(exp(x^2))"] {
        let (success, stdout) = run_limit(expr, "x", "infinity", "json");
        assert!(success, "Command should succeed for {expr}");
        assert!(
            stdout.contains("\"warning\"") && stdout.contains("limit("),
            "Unsupported exponential argument tail should remain residual for {expr}, got: {stdout}"
        );
    }
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
fn test_limit_polynomial_exp_dominance_at_infinity() {
    for (expr, to, expected) in [
        ("x^2/exp(x^2)", "infinity", "0"),
        ("exp(x^2)/x^2", "infinity", "infinity"),
        ("x^2*exp(2 - x^4)", "infinity", "0"),
        ("x/exp(-x^2)", "infinity", "infinity"),
        ("x/exp(-x^2)", "-infinity", "-infinity"),
        ("0 - 2*exp(x^2)/x^2", "infinity", "-infinity"),
    ] {
        let (success, stdout) = run_limit(expr, "x", to, "json");
        assert!(success, "Command should succeed for {expr} at {to}");
        assert!(
            stdout.contains(&format!("\"result\":\"{expected}\"")),
            "{expr} at {to} should resolve to {expected}, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved polynomial exponential dominance should not warn for {expr} at {to}, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_unsupported_polynomial_exp_dominance_remains_residual() {
    for expr in ["exp(a*x^2)/x", "exp(exp(x^2))/x"] {
        let (success, stdout) = run_limit(expr, "x", "infinity", "json");
        assert!(success, "Command should succeed for {expr}");
        assert!(
            stdout.contains("\"warning\"") && stdout.contains("limit("),
            "Unsupported polynomial exponential dominance should remain residual for {expr}, got: {stdout}"
        );
    }
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
fn test_eval_resolved_infinity_limits_suppress_eventual_tail_conditions_json() {
    for (input, expected_result) in [
        ("limit(x/ln(x), x, infinity)", "infinity"),
        ("limit(x/ln(1 - x), x, -infinity)", "-infinity"),
        ("limit(sqrt(x)/x, x, infinity)", "0"),
    ] {
        let (success, stdout) = run_eval(input, "json");
        assert!(success, "Command should succeed for {input}");
        let wire: Value = serde_json::from_str(&stdout).expect("eval json");
        assert_eq!(wire["ok"], true, "{input}");
        assert_eq!(wire["result"], expected_result, "{input}");
        assert_eq!(wire["warnings"], json!([]), "{input}");
        assert_eq!(wire["required_conditions"], json!([]), "{input}");
        assert_eq!(wire["required_display"], json!([]), "{input}");
    }
}

#[test]
fn test_eval_resolved_infinity_limits_suppress_eventual_polynomial_nonzero_conditions_json() {
    for input in [
        "limit((x^2+1)/(2*x^2-3), x, infinity)",
        "limit((x^2+1)/(2*x^2-3), x, -infinity)",
    ] {
        let (success, stdout) = run_eval(input, "json");
        assert!(success, "Command should succeed for {input}");
        let wire: Value = serde_json::from_str(&stdout).expect("eval json");
        assert_eq!(wire["ok"], true, "{input}");
        assert_eq!(wire["result"], "1/2", "{input}");
        assert_eq!(wire["warnings"], json!([]), "{input}");
        assert_eq!(wire["required_conditions"], json!([]), "{input}");
        assert_eq!(wire["required_display"], json!([]), "{input}");
    }
}

#[test]
fn test_eval_resolved_infinity_limits_suppress_eventual_polynomial_sign_conditions_json() {
    for (input, expected_result) in [
        ("limit(sqrt(x^2 - 3)/x, x, infinity)", "1"),
        ("limit(sqrt(x^2 - 3)/x, x, -infinity)", "-1"),
        ("limit(sqrt(x^2 + x - 3)/x, x, -infinity)", "-1"),
    ] {
        let (success, stdout) = run_eval(input, "json");
        assert!(success, "Command should succeed for {input}");
        let wire: Value = serde_json::from_str(&stdout).expect("eval json");
        assert_eq!(wire["ok"], true, "{input}");
        assert_eq!(wire["result"], expected_result, "{input}");
        assert_eq!(wire["warnings"], json!([]), "{input}");
        assert_eq!(wire["required_conditions"], json!([]), "{input}");
        assert_eq!(wire["required_display"], json!([]), "{input}");
    }
}

#[test]
fn test_eval_resolved_infinity_limits_keep_parametric_polynomial_sign_conditions_json() {
    let (success, stdout) = run_eval("limit(sqrt(x^2 - a)/x, x, infinity)", "json");
    assert!(success, "Command should succeed");
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "1");
    assert_eq!(wire["warnings"], json!([]));
    assert_eq!(wire["required_display"], json!(["x^2 - a ≥ 0"]));
}

#[test]
fn test_eval_residual_infinity_limits_keep_incompatible_polynomial_sign_conditions_json() {
    let (success, stdout) = run_eval("limit(sqrt(3 - x^2)/x, x, infinity)", "json");
    assert!(success, "Command should succeed even for residual");
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "limit(sqrt(3 - x^2) / x, x, infinity)");
    assert_eq!(
        wire["required_display"],
        json!(["-sqrt(3) ≤ x ≤ sqrt(3)", "x ≠ 0"])
    );
    assert!(
        wire["warnings"].as_array().is_some_and(|warnings| warnings
            .iter()
            .any(|warning| warning["rule"] == "Limit Evaluation")),
        "residual incompatible polynomial domain should keep its warning, got: {wire:?}"
    );
}

#[test]
fn test_eval_residual_infinity_limits_keep_parametric_nonzero_conditions_json() {
    let (success, stdout) = run_eval("limit((x^2+1)/(a*x^2-3), x, infinity)", "json");
    assert!(success, "Command should succeed even for residual");
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(
        wire["result"],
        "limit((x^2 + 1) / (a·x^2 - 3), x, infinity)"
    );
    assert_eq!(wire["required_display"], json!(["a·x^2 - 3 ≠ 0"]));
    assert!(
        wire["warnings"].as_array().is_some_and(|warnings| warnings
            .iter()
            .any(|warning| warning["rule"] == "Limit Evaluation")),
        "residual parametric limit should keep its warning, got: {wire:?}"
    );
}

#[test]
fn test_eval_conflicting_infinity_limit_keeps_domain_path_requirement_json() {
    let (success, stdout) = run_eval("limit(ln(x), x, -infinity)", "json");
    assert!(success, "Command should succeed even for residual");
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "limit(ln(x), x, -infinity)");
    assert_eq!(wire["required_display"], json!(["x > 0"]));
    assert!(
        wire["warnings"].as_array().is_some_and(|warnings| {
            warnings
                .iter()
                .any(|warning| warning["rule"] == "Limit Domain Path")
        }),
        "conflicting domain path should stay explicit, got: {wire:?}"
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
fn test_limit_abs_polynomial_ratio_orientation_at_infinity() {
    for (expr, to, expected) in [
        ("abs(2*x+1)/(2*x+1)", "infinity", "1"),
        ("abs(2*x+1)/(2*x+1)", "-infinity", "-1"),
        ("(2*x+1)/abs(2*x+1)", "infinity", "1"),
        ("(2*x+1)/abs(2*x+1)", "-infinity", "-1"),
        ("abs(1-2*x)/(1-2*x)", "infinity", "-1"),
        ("abs(1-2*x)/(1-2*x)", "-infinity", "1"),
        ("(1-2*x)/abs(1-2*x)", "infinity", "-1"),
        ("(1-2*x)/abs(1-2*x)", "-infinity", "1"),
        ("abs(x^2+1)/x^2", "-infinity", "1"),
    ] {
        let (success, stdout) = run_limit(expr, "x", to, "json");
        assert!(success, "Command should succeed for {expr} at {to}");
        assert!(
            stdout.contains(&format!("\"result\":\"{expected}\"")),
            "{expr} at {to} should resolve to {expected}, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved abs-polynomial ratio should not warn for {expr} at {to}, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_abs_polynomial_ratio_degree_mismatch_at_infinity() {
    for (expr, to, expected) in [
        ("abs(1-2*x)/x^2", "infinity", "0"),
        ("abs(1-2*x)/x^2", "-infinity", "0"),
        ("x/abs(x^2+1)", "infinity", "0"),
        ("x/abs(x^2+1)", "-infinity", "0"),
        ("abs(x^2+1)/x", "infinity", "infinity"),
        ("abs(x^2+1)/x", "-infinity", "-infinity"),
        ("x^2/abs(1-2*x)", "infinity", "infinity"),
        ("(-x^2)/abs(1-2*x)", "infinity", "-infinity"),
    ] {
        let (success, stdout) = run_limit(expr, "x", to, "json");
        assert!(success, "Command should succeed for {expr} at {to}");
        assert!(
            stdout.contains(&format!("\"result\":\"{expected}\"")),
            "{expr} at {to} should resolve to {expected}, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved abs-polynomial dominance should not warn for {expr} at {to}, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_abs_linear_tail_at_infinity() {
    for (expr, to) in [
        ("abs(2*x+1)", "infinity"),
        ("abs(2*x+1)", "-infinity"),
        ("abs(1-2*x)", "infinity"),
        ("abs(1-2*x)", "-infinity"),
    ] {
        let (success, stdout) = run_limit(expr, "x", to, "json");
        assert!(success, "Command should succeed for {expr} at {to}");
        assert!(
            stdout.contains("\"result\":\"infinity\""),
            "{expr} at {to} should resolve to infinity, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved abs linear tail should not warn for {expr} at {to}, got: {stdout}"
        );
    }
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
fn test_limit_direct_log_functions_pos_infinity() {
    for (expr, expected) in [
        ("log2(x)", "infinity"),
        ("log10(x)", "infinity"),
        ("log(2, x)", "infinity"),
        ("log(e^2, x)", "infinity"),
        ("log(1/2, x)", "-infinity"),
        ("log(e^-2, x)", "-infinity"),
    ] {
        let (success, stdout) = run_limit(expr, "x", "infinity", "json");
        assert!(success, "Command should succeed for {expr}");
        assert!(
            stdout.contains(&format!("\"result\":\"{expected}\"")),
            "{expr} should resolve to {expected}, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved direct log limit should not warn for {expr}, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_direct_log_polynomial_argument_at_infinity() {
    for (expr, to, expected) in [
        ("ln(x^2)", "infinity", "infinity"),
        ("ln(x^2)", "-infinity", "infinity"),
        ("ln(x^2 - 3)", "infinity", "infinity"),
        ("log2(x^2)", "infinity", "infinity"),
        ("log(2, x^2)", "infinity", "infinity"),
        ("log(1/2, x^2)", "infinity", "-infinity"),
    ] {
        let (success, stdout) = run_limit(expr, "x", to, "json");
        assert!(success, "Command should succeed for {expr} at {to}");
        assert!(
            stdout.contains(&format!("\"result\":\"{expected}\"")),
            "{expr} at {to} should resolve to {expected}, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved direct log-polynomial limit should not warn for {expr} at {to}, got: {stdout}"
        );
    }

    let (success, stdout) = run_eval("limit(ln(x^2 - 3), x, infinity)", "json");
    assert!(success, "Eval should succeed for shifted polynomial log");
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "infinity");
    assert_eq!(wire["warnings"], json!([]));
    assert_eq!(wire["required_conditions"], json!([]));
    assert_eq!(wire["required_display"], json!([]));
}

#[test]
fn test_limit_direct_log_rational_argument_positive_unbounded_at_infinity() {
    for (expr, expected) in [
        ("ln((x^2 + 1)/(x + 1))", "infinity"),
        ("log2((x^2 + 1)/(x + 1))", "infinity"),
        ("log(2, (x^2 + 1)/(x + 1))", "infinity"),
        ("log(1/2, (x^2 + 1)/(x + 1))", "-infinity"),
    ] {
        let (success, stdout) = run_limit(expr, "x", "infinity", "json");
        assert!(success, "Command should succeed for {expr}");
        assert!(
            stdout.contains(&format!("\"result\":\"{expected}\"")),
            "{expr} should resolve to {expected}, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved log-rational positive-tail limit should not warn for {expr}, got: {stdout}"
        );
    }

    let (success, stdout) = run_eval("limit(ln((x^2 + 1)/(x + 1)), x, infinity)", "json");
    assert!(success, "Eval should succeed for rational log");
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "infinity");
    assert_eq!(wire["warnings"], json!([]));
    assert_eq!(wire["required_conditions"], json!([]));
    assert_eq!(wire["required_display"], json!([]));
}

#[test]
fn test_limit_direct_log_rational_argument_positive_zero_tail_at_infinity() {
    for (expr, expected) in [
        ("ln((x + 1)/(x^2 + 1))", "-infinity"),
        ("log2((x + 1)/(x^2 + 1))", "-infinity"),
        ("log10((x + 1)/(x^2 + 1))", "-infinity"),
        ("log(2, (x + 1)/(x^2 + 1))", "-infinity"),
        ("log(1/2, (x + 1)/(x^2 + 1))", "infinity"),
        ("log2(1/(x^2 + 1))", "-infinity"),
    ] {
        let (success, stdout) = run_limit(expr, "x", "infinity", "json");
        assert!(success, "Command should succeed for {expr}");
        assert!(
            stdout.contains(&format!("\"result\":\"{expected}\"")),
            "{expr} should resolve to {expected}, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved log-rational positive zero-tail limit should not warn for {expr}, got: {stdout}"
        );
    }

    let (success, stdout) = run_eval("limit(ln((x + 1)/(x^2 + 1)), x, infinity)", "json");
    assert!(success, "Eval should succeed for rational zero-tail log");
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "-infinity");
    assert_eq!(wire["warnings"], json!([]));
    assert_eq!(wire["required_conditions"], json!([]));
    assert_eq!(wire["required_display"], json!([]));
}

#[test]
fn test_limit_direct_log_rational_argument_positive_finite_tail_at_infinity() {
    for (expr, expected) in [
        ("ln((2*x^2 + 1)/(x^2 + 1))", "ln(2)"),
        ("log2((2*x^2 + 1)/(x^2 + 1))", "1"),
        ("log10((100*x^2 + 1)/(x^2 + 1))", "2"),
        ("log(2, (2*x^2 + 1)/(x^2 + 1))", "1"),
        ("log(1/2, (2*x^2 + 1)/(x^2 + 1))", "-1"),
        ("ln((x^2 + 1)/(x^2 + 1))", "0"),
    ] {
        let (success, stdout) = run_limit(expr, "x", "infinity", "json");
        assert!(success, "Command should succeed for {expr}");
        assert!(
            stdout.contains(&format!("\"result\":\"{expected}\"")),
            "{expr} should resolve to {expected}, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved log-rational positive finite-tail limit should not warn for {expr}, got: {stdout}"
        );
    }

    let (success, stdout) = run_eval("limit(log(2, (2*x^2 + 1)/(x^2 + 1)), x, infinity)", "json");
    assert!(success, "Eval should succeed for rational finite-tail log");
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "1");
    assert_eq!(wire["warnings"], json!([]));
    assert_eq!(wire["required_conditions"], json!([]));
    assert_eq!(wire["required_display"], json!([]));
}

#[test]
fn test_limit_direct_base_log_domain_and_base_guards_remain_residual() {
    for expr in ["log(2, x)", "log2(x)", "log10(x)"] {
        let (success, stdout) = run_limit(expr, "x", "-infinity", "json");
        assert!(success, "Command should succeed for residual {expr}");
        assert!(
            stdout.contains("\"warning\"") && stdout.contains("limit("),
            "{expr} at -∞ should remain residual over the real domain, got: {stdout}"
        );
    }

    for expr in ["log(1, x)", "log(-2, x)"] {
        let (success, stdout) = run_limit(expr, "x", "infinity", "json");
        assert!(success, "Command should succeed for residual {expr}");
        assert!(
            stdout.contains("\"warning\"") && stdout.contains("limit("),
            "{expr} should keep invalid-base residual, got: {stdout}"
        );
    }

    for expr in ["ln(3 - x^2)", "log(2, 3 - x^2)", "log(1, x^2)"] {
        let (success, stdout) = run_limit(expr, "x", "infinity", "json");
        assert!(success, "Command should succeed for residual {expr}");
        assert!(
            stdout.contains("\"warning\"") && stdout.contains("limit("),
            "{expr} should keep unsafe log-polynomial residual, got: {stdout}"
        );
    }

    for expr in [
        "ln((x^2 + 1)/(1 - x))",
        "log(2, (x^2 + 1)/(1 - x))",
        "ln((a*x^2 + 1)/(x + 1))",
        "ln((1 - x)/(x^2 + 1))",
        "ln((a*x + 1)/(x^2 + 1))",
        "ln((1 - 2*x^2)/(x^2 + 1))",
        "ln((a*x^2 + 1)/(x^2 + 1))",
        "log(1, (2*x^2 + 1)/(x^2 + 1))",
    ] {
        let (success, stdout) = run_limit(expr, "x", "infinity", "json");
        assert!(success, "Command should succeed for residual {expr}");
        assert!(
            stdout.contains("\"warning\"") && stdout.contains("limit("),
            "{expr} should keep unsafe or unsupported log-rational residual, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_log_of_linear_exp_argument_pos_infinity() {
    for (expr, expected) in [
        ("ln(exp(x))", "infinity"),
        ("ln(exp(-x))", "-infinity"),
        ("log2(exp(x))", "infinity"),
        ("log10(exp(-x))", "-infinity"),
        ("log(2, exp(x))", "infinity"),
        ("log(2, exp(-x))", "-infinity"),
        ("log(1/2, exp(x))", "-infinity"),
        ("log(1/2, exp(-x))", "infinity"),
    ] {
        let (success, stdout) = run_limit(expr, "x", "infinity", "json");
        assert!(success, "Command should succeed for {expr}");
        assert!(
            stdout.contains(&format!("\"result\":\"{expected}\"")),
            "{expr} should resolve to {expected}, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved log(exp(linear)) limit should not warn for {expr}, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_log_of_exp_argument_guards_remain_residual() {
    for expr in ["ln(exp(x^2))", "log(2, exp(x^2))"] {
        let (success, stdout) = run_limit(expr, "x", "infinity", "json");
        assert!(success, "Command should succeed for residual {expr}");
        assert!(
            stdout.contains("\"warning\"") && stdout.contains("limit("),
            "{expr} should keep nonlinear exponent residual, got: {stdout}"
        );
    }

    for expr in ["log(1, exp(x))", "log(-2, exp(x))"] {
        let (success, stdout) = run_limit(expr, "x", "infinity", "json");
        assert!(success, "Command should succeed for residual {expr}");
        assert!(
            stdout.contains("\"warning\"") && stdout.contains("limit("),
            "{expr} should keep invalid-base residual, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_sqrt_of_linear_exp_argument_at_infinity() {
    for (expr, to, expected) in [
        ("sqrt(exp(x))", "infinity", "infinity"),
        ("sqrt(exp(-x))", "infinity", "0"),
        ("sqrt(exp(x))", "-infinity", "0"),
        ("sqrt(exp(-x))", "-infinity", "infinity"),
    ] {
        let (success, stdout) = run_limit(expr, "x", to, "json");
        assert!(success, "Command should succeed for {expr} at {to}");
        assert!(
            stdout.contains(&format!("\"result\":\"{expected}\"")),
            "{expr} at {to} should resolve to {expected}, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved sqrt(exp(linear)) limit should not warn for {expr} at {to}, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_sqrt_of_rational_finite_tail_argument_at_infinity() {
    for (expr, expected) in [
        ("sqrt((2*x^2 + 1)/(x^2 + 1))", "sqrt(2)"),
        ("sqrt((4*x^2 + 1)/(x^2 + 1))", "2"),
        ("sqrt((x^2 + 1)/(x^2 + 1))", "1"),
    ] {
        let (success, stdout) = run_limit(expr, "x", "infinity", "json");
        assert!(success, "Command should succeed for {expr}");
        assert!(
            stdout.contains(&format!("\"result\":\"{expected}\"")),
            "{expr} should resolve to {expected}, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved sqrt(rational finite tail) limit should not warn for {expr}, got: {stdout}"
        );
    }

    let (success, stdout) = run_eval("limit(sqrt((2*x^2 + 1)/(x^2 + 1)), x, infinity)", "json");
    assert!(success, "Eval should succeed for finite-tail rational sqrt");
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "sqrt(2)");
    assert_eq!(wire["warnings"], json!([]));
    assert_eq!(wire["required_conditions"], json!([]));
    assert_eq!(wire["required_display"], json!([]));
}

#[test]
fn test_limit_sqrt_of_rational_positive_zero_tail_argument_at_infinity() {
    for (expr, to) in [
        ("sqrt((x + 1)/(x^2 + 1))", "infinity"),
        ("sqrt((1 - x)/(x^2 + 1))", "-infinity"),
    ] {
        let (success, stdout) = run_limit(expr, "x", to, "json");
        assert!(success, "Command should succeed for {expr} at {to}");
        assert!(
            stdout.contains("\"result\":\"0\""),
            "{expr} at {to} should resolve to 0, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved sqrt(rational positive zero tail) limit should not warn for {expr} at {to}, got: {stdout}"
        );
    }

    let (success, stdout) = run_eval("limit(sqrt((x + 1)/(x^2 + 1)), x, infinity)", "json");
    assert!(
        success,
        "Eval should succeed for positive-zero-tail rational sqrt"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "0");
    assert_eq!(wire["warnings"], json!([]));
    assert_eq!(wire["required_conditions"], json!([]));
    assert_eq!(wire["required_display"], json!([]));
}

#[test]
fn test_limit_sqrt_of_unsafe_rational_or_nonlinear_exp_argument_remains_residual() {
    for expr in [
        "sqrt(exp(x^2))",
        "sqrt((1 - x)/(x^2 + 1))",
        "sqrt((a*x + 1)/(x^2 + 1))",
        "sqrt((1 - 2*x^2)/(x^2 + 1))",
        "sqrt((a*x^2 + 1)/(x^2 + 1))",
    ] {
        let (success, stdout) = run_limit(expr, "x", "infinity", "json");
        assert!(success, "Command should succeed for residual {expr}");
        assert!(
            stdout.contains("\"warning\"") && stdout.contains("limit("),
            "{expr} should keep unsupported sqrt residual, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_cbrt_of_linear_argument_at_infinity() {
    for (expr, to, expected) in [
        ("cbrt(x)", "infinity", "infinity"),
        ("cbrt(x)", "-infinity", "-infinity"),
        ("cbrt(1 - x)", "infinity", "-infinity"),
        ("cbrt(1 - x)", "-infinity", "infinity"),
    ] {
        let (success, stdout) = run_limit(expr, "x", to, "json");
        assert!(success, "Command should succeed for {expr} at {to}");
        assert!(
            stdout.contains(&format!("\"result\":\"{expected}\"")),
            "{expr} at {to} should resolve to {expected}, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved cbrt(linear) limit should not warn for {expr} at {to}, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_cbrt_of_linear_exp_argument_at_infinity() {
    for (expr, to, expected) in [
        ("cbrt(exp(x))", "infinity", "infinity"),
        ("cbrt(exp(-x))", "infinity", "0"),
        ("cbrt(exp(x))", "-infinity", "0"),
        ("cbrt(exp(-x))", "-infinity", "infinity"),
    ] {
        let (success, stdout) = run_limit(expr, "x", to, "json");
        assert!(success, "Command should succeed for {expr} at {to}");
        assert!(
            stdout.contains(&format!("\"result\":\"{expected}\"")),
            "{expr} at {to} should resolve to {expected}, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved cbrt(exp(linear)) limit should not warn for {expr} at {to}, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_cbrt_of_polynomial_argument_tails_at_infinity() {
    for (expr, to, expected) in [
        ("cbrt(x^2)", "infinity", "infinity"),
        ("cbrt(x^2)", "-infinity", "infinity"),
        ("cbrt(x^3 - 2*x)", "-infinity", "-infinity"),
        ("cbrt(2 - x^4)", "infinity", "-infinity"),
    ] {
        let (success, stdout) = run_limit(expr, "x", to, "json");
        assert!(success, "Command should succeed for {expr} at {to}");
        assert!(
            stdout.contains(&format!("\"result\":\"{expected}\"")),
            "{expr} at {to} should resolve to {expected}, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved cbrt(polynomial tail) limit should not warn for {expr} at {to}, got: {stdout}"
        );
    }

    let (success, stdout) = run_eval("limit(cbrt(2 - x^4), x, infinity)", "json");
    assert!(
        success,
        "Eval should succeed for negative polynomial-tail cbrt"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "-infinity");
    assert_eq!(wire["warnings"], json!([]));
    assert_eq!(wire["required_conditions"], json!([]));
    assert_eq!(wire["required_display"], json!([]));
}

#[test]
fn test_limit_cbrt_of_rational_argument_tails_at_infinity() {
    for (expr, to, expected) in [
        ("cbrt((x^2 + 1)/(x + 1))", "infinity", "infinity"),
        ("cbrt((x^2 + 1)/(1 - x))", "infinity", "-infinity"),
        ("cbrt((8*x^2 + 1)/(x^2 + 1))", "infinity", "2"),
        ("cbrt((1 - 8*x^2)/(x^2 + 1))", "infinity", "-2"),
        ("cbrt((2*x^2 + 1)/(x^2 + 1))", "infinity", "cbrt(2)"),
        ("cbrt((x + 1)/(x^2 + 1))", "infinity", "0"),
        ("cbrt((1 - x)/(x^2 + 1))", "infinity", "0"),
        ("cbrt((1 - x)/(x^2 + 1))", "-infinity", "0"),
    ] {
        let (success, stdout) = run_limit(expr, "x", to, "json");
        assert!(success, "Command should succeed for {expr} at {to}");
        assert!(
            stdout.contains(&format!("\"result\":\"{expected}\"")),
            "{expr} at {to} should resolve to {expected}, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved cbrt(rational tail) limit should not warn for {expr} at {to}, got: {stdout}"
        );
    }

    let (success, stdout) = run_eval("limit(cbrt((1 - 8*x^2)/(x^2 + 1)), x, infinity)", "json");
    assert!(
        success,
        "Eval should succeed for signed finite-tail rational cbrt"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "-2");
    assert_eq!(wire["warnings"], json!([]));
    assert_eq!(wire["required_conditions"], json!([]));
    assert_eq!(wire["required_display"], json!([]));
}

#[test]
fn test_limit_cbrt_of_nonlinear_argument_remains_residual() {
    for expr in [
        "cbrt(exp(x^2))",
        "cbrt(a*x^2 + 1)",
        "cbrt((a*x^2 + 1)/(x^2 + 1))",
    ] {
        let (success, stdout) = run_limit(expr, "x", "infinity", "json");
        assert!(success, "Command should succeed for residual {expr}");
        assert!(
            stdout.contains("\"warning\"") && stdout.contains("limit("),
            "{expr} should keep nonlinear argument residual, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_cbrt_polynomial_dominance_at_infinity() {
    for (expr, to, expected) in [
        ("cbrt(x)/x", "infinity", "0"),
        ("x/cbrt(x)", "infinity", "infinity"),
        ("x/cbrt(x)", "-infinity", "infinity"),
        ("x/cbrt(1 - x)", "infinity", "-infinity"),
    ] {
        let (success, stdout) = run_limit(expr, "x", to, "json");
        assert!(success, "Command should succeed for {expr} at {to}");
        assert!(
            stdout.contains(&format!("\"result\":\"{expected}\"")),
            "{expr} at {to} should resolve to {expected}, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved cbrt-polynomial dominance should not warn for {expr} at {to}, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_cbrt_exp_dominance_at_infinity() {
    for (expr, expected) in [
        ("cbrt(x)*exp(-x)", "0"),
        ("cbrt(1 - x)/exp(-x)", "-infinity"),
        ("exp(x)/cbrt(1 - x)", "-infinity"),
    ] {
        let (success, stdout) = run_limit(expr, "x", "infinity", "json");
        assert!(success, "Command should succeed for {expr}");
        assert!(
            stdout.contains(&format!("\"result\":\"{expected}\"")),
            "{expr} should resolve to {expected}, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved cbrt-exp dominance should not warn for {expr}, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_cbrt_dominance_nonlinear_argument_remains_residual() {
    for expr in ["cbrt(x^2)/x", "x/cbrt(x^2)", "exp(x)/cbrt(x^2)"] {
        let (success, stdout) = run_limit(expr, "x", "infinity", "json");
        assert!(success, "Command should succeed for residual {expr}");
        assert!(
            stdout.contains("\"warning\"") && stdout.contains("limit("),
            "{expr} should keep nonlinear cube-root dominance residual, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_asinh_of_linear_argument_at_infinity() {
    for (expr, to, expected) in [
        ("asinh(x)", "infinity", "infinity"),
        ("asinh(x)", "-infinity", "-infinity"),
        ("asinh(1 - x)", "infinity", "-infinity"),
        ("asinh(1 - x)", "-infinity", "infinity"),
    ] {
        let (success, stdout) = run_limit(expr, "x", to, "json");
        assert!(success, "Command should succeed for {expr} at {to}");
        assert!(
            stdout.contains(&format!("\"result\":\"{expected}\"")),
            "{expr} at {to} should resolve to {expected}, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved asinh(linear) limit should not warn for {expr} at {to}, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_asinh_of_linear_exp_argument_at_infinity() {
    for (expr, to, expected) in [
        ("asinh(exp(x))", "infinity", "infinity"),
        ("asinh(exp(-x))", "infinity", "0"),
        ("asinh(exp(x))", "-infinity", "0"),
        ("asinh(exp(-x))", "-infinity", "infinity"),
    ] {
        let (success, stdout) = run_limit(expr, "x", to, "json");
        assert!(success, "Command should succeed for {expr} at {to}");
        assert!(
            stdout.contains(&format!("\"result\":\"{expected}\"")),
            "{expr} at {to} should resolve to {expected}, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved asinh(exp(linear)) limit should not warn for {expr} at {to}, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_asinh_of_polynomial_argument_tails_at_infinity() {
    for (expr, to, expected) in [
        ("asinh(x^2)", "infinity", "infinity"),
        ("asinh(x^2)", "-infinity", "infinity"),
        ("asinh(x^3 - 2*x)", "-infinity", "-infinity"),
        ("asinh(2 - x^4)", "infinity", "-infinity"),
    ] {
        let (success, stdout) = run_limit(expr, "x", to, "json");
        assert!(success, "Command should succeed for {expr} at {to}");
        assert!(
            stdout.contains(&format!("\"result\":\"{expected}\"")),
            "{expr} at {to} should resolve to {expected}, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved asinh(polynomial tail) limit should not warn for {expr} at {to}, got: {stdout}"
        );
    }

    let (success, stdout) = run_eval("limit(asinh(2 - x^4), x, infinity)", "json");
    assert!(
        success,
        "Eval should succeed for negative polynomial-tail asinh"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "-infinity");
    assert_eq!(wire["warnings"], json!([]));
    assert_eq!(wire["required_conditions"], json!([]));
    assert_eq!(wire["required_display"], json!([]));
}

#[test]
fn test_limit_asinh_of_unsupported_argument_remains_residual() {
    for expr in ["asinh(exp(x^2))", "asinh(a*x^2 + 1)"] {
        let (success, stdout) = run_limit(expr, "x", "infinity", "json");
        assert!(success, "Command should succeed for residual {expr}");
        assert!(
            stdout.contains("\"warning\"") && stdout.contains("limit("),
            "{expr} should keep unsupported argument residual, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_asinh_polynomial_dominance_at_infinity() {
    for (expr, to, expected) in [
        ("asinh(x)/x", "infinity", "0"),
        ("x/asinh(x)", "infinity", "infinity"),
        ("x/asinh(x)", "-infinity", "infinity"),
        ("x/asinh(1 - x)", "infinity", "-infinity"),
    ] {
        let (success, stdout) = run_limit(expr, "x", to, "json");
        assert!(success, "Command should succeed for {expr} at {to}");
        assert!(
            stdout.contains(&format!("\"result\":\"{expected}\"")),
            "{expr} at {to} should resolve to {expected}, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved asinh-polynomial dominance should not warn for {expr} at {to}, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_asinh_exp_dominance_at_infinity() {
    for (expr, expected) in [
        ("asinh(x)*exp(-x)", "0"),
        ("asinh(1 - x)/exp(-x)", "-infinity"),
        ("exp(x)/asinh(1 - x)", "-infinity"),
    ] {
        let (success, stdout) = run_limit(expr, "x", "infinity", "json");
        assert!(success, "Command should succeed for {expr}");
        assert!(
            stdout.contains(&format!("\"result\":\"{expected}\"")),
            "{expr} should resolve to {expected}, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved asinh-exp dominance should not warn for {expr}, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_asinh_dominance_nonlinear_argument_remains_residual() {
    for expr in ["asinh(x^2)/x", "x/asinh(x^2)", "exp(x)/asinh(x^2)"] {
        let (success, stdout) = run_limit(expr, "x", "infinity", "json");
        assert!(success, "Command should succeed for residual {expr}");
        assert!(
            stdout.contains("\"warning\"") && stdout.contains("limit("),
            "{expr} should keep nonlinear asinh dominance residual, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_atan_of_linear_argument_at_infinity() {
    for (expr, to, expected) in [
        ("atan(x)", "infinity", "pi / 2"),
        ("atan(x)", "-infinity", "-pi / 2"),
        ("arctan(1 - x)", "infinity", "-pi / 2"),
        ("arctan(1 - x)", "-infinity", "pi / 2"),
    ] {
        let (success, stdout) = run_limit(expr, "x", to, "json");
        assert!(success, "Command should succeed for {expr} at {to}");
        assert!(
            stdout.contains(&format!("\"result\":\"{expected}\"")),
            "{expr} at {to} should resolve to {expected}, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved atan(linear) limit should not warn for {expr} at {to}, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_atan_of_linear_exp_argument_at_infinity() {
    for (expr, to, expected) in [
        ("atan(exp(x))", "infinity", "pi / 2"),
        ("atan(exp(-x))", "infinity", "0"),
        ("arctan(exp(x))", "-infinity", "0"),
        ("arctan(exp(-x))", "-infinity", "pi / 2"),
    ] {
        let (success, stdout) = run_limit(expr, "x", to, "json");
        assert!(success, "Command should succeed for {expr} at {to}");
        assert!(
            stdout.contains(&format!("\"result\":\"{expected}\"")),
            "{expr} at {to} should resolve to {expected}, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved atan(exp(linear)) limit should not warn for {expr} at {to}, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_atan_of_polynomial_argument_tails_at_infinity() {
    for (expr, to, expected) in [
        ("atan(x^2)", "infinity", "pi / 2"),
        ("atan(x^2)", "-infinity", "pi / 2"),
        ("arctan(x^3 - 2*x)", "-infinity", "-pi / 2"),
        ("atan(2 - x^4)", "infinity", "-pi / 2"),
    ] {
        let (success, stdout) = run_limit(expr, "x", to, "json");
        assert!(success, "Command should succeed for {expr} at {to}");
        assert!(
            stdout.contains(&format!("\"result\":\"{expected}\"")),
            "{expr} at {to} should resolve to {expected}, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved atan(polynomial tail) limit should not warn for {expr} at {to}, got: {stdout}"
        );
    }

    let (success, stdout) = run_eval("limit(atan(2 - x^4), x, infinity)", "json");
    assert!(
        success,
        "Eval should succeed for negative polynomial-tail atan"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "-pi / 2");
    assert_eq!(wire["warnings"], json!([]));
    assert_eq!(wire["required_conditions"], json!([]));
    assert_eq!(wire["required_display"], json!([]));
}

#[test]
fn test_limit_atan_of_unsupported_argument_remains_residual() {
    for expr in ["arctan(exp(x^2))", "atan(a*x^2 + 1)"] {
        let (success, stdout) = run_limit(expr, "x", "infinity", "json");
        assert!(success, "Command should succeed for residual {expr}");
        assert!(
            stdout.contains("\"warning\"") && stdout.contains("limit("),
            "{expr} should keep nonlinear argument residual, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_tanh_of_linear_argument_at_infinity() {
    for (expr, to, expected) in [
        ("tanh(x)", "infinity", "1"),
        ("tanh(x)", "-infinity", "-1"),
        ("tanh(1 - x)", "infinity", "-1"),
        ("tanh(1 - x)", "-infinity", "1"),
    ] {
        let (success, stdout) = run_limit(expr, "x", to, "json");
        assert!(success, "Command should succeed for {expr} at {to}");
        assert!(
            stdout.contains(&format!("\"result\":\"{expected}\"")),
            "{expr} at {to} should resolve to {expected}, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved tanh(linear) limit should not warn for {expr} at {to}, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_tanh_of_linear_exp_argument_at_infinity() {
    for (expr, to, expected) in [
        ("tanh(exp(x))", "infinity", "1"),
        ("tanh(exp(-x))", "infinity", "0"),
        ("tanh(exp(x))", "-infinity", "0"),
        ("tanh(exp(-x))", "-infinity", "1"),
    ] {
        let (success, stdout) = run_limit(expr, "x", to, "json");
        assert!(success, "Command should succeed for {expr} at {to}");
        assert!(
            stdout.contains(&format!("\"result\":\"{expected}\"")),
            "{expr} at {to} should resolve to {expected}, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved tanh(exp(linear)) limit should not warn for {expr} at {to}, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_tanh_of_polynomial_argument_tails_at_infinity() {
    for (expr, to, expected) in [
        ("tanh(x^2)", "infinity", "1"),
        ("tanh(x^2)", "-infinity", "1"),
        ("tanh(x^3 - 2*x)", "-infinity", "-1"),
        ("tanh(2 - x^4)", "infinity", "-1"),
    ] {
        let (success, stdout) = run_limit(expr, "x", to, "json");
        assert!(success, "Command should succeed for {expr} at {to}");
        assert!(
            stdout.contains(&format!("\"result\":\"{expected}\"")),
            "{expr} at {to} should resolve to {expected}, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved tanh(polynomial tail) limit should not warn for {expr} at {to}, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_tanh_of_unsupported_argument_remains_residual() {
    for expr in ["tanh(exp(x^2))", "tanh(a*x^2 + 1)"] {
        let (success, stdout) = run_limit(expr, "x", "infinity", "json");
        assert!(success, "Command should succeed for residual {expr}");
        assert!(
            stdout.contains("\"warning\"") && stdout.contains("limit("),
            "{expr} should keep nonlinear argument residual, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_sinh_cosh_of_linear_argument_at_infinity() {
    for (expr, to, expected) in [
        ("sinh(x)", "infinity", "infinity"),
        ("sinh(x)", "-infinity", "-infinity"),
        ("sinh(1 - x)", "infinity", "-infinity"),
        ("sinh(1 - x)", "-infinity", "infinity"),
        ("cosh(x)", "infinity", "infinity"),
        ("cosh(x)", "-infinity", "infinity"),
        ("cosh(1 - x)", "infinity", "infinity"),
        ("cosh(1 - x)", "-infinity", "infinity"),
    ] {
        let (success, stdout) = run_limit(expr, "x", to, "json");
        assert!(success, "Command should succeed for {expr} at {to}");
        assert!(
            stdout.contains(&format!("\"result\":\"{expected}\"")),
            "{expr} at {to} should resolve to {expected}, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved sinh/cosh(linear) limit should not warn for {expr} at {to}, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_sinh_cosh_of_linear_exp_argument_at_infinity() {
    for (expr, to, expected) in [
        ("sinh(exp(x))", "infinity", "infinity"),
        ("sinh(exp(-x))", "infinity", "0"),
        ("sinh(exp(x))", "-infinity", "0"),
        ("sinh(exp(-x))", "-infinity", "infinity"),
        ("cosh(exp(x))", "infinity", "infinity"),
        ("cosh(exp(-x))", "infinity", "1"),
        ("cosh(exp(x))", "-infinity", "1"),
        ("cosh(exp(-x))", "-infinity", "infinity"),
    ] {
        let (success, stdout) = run_limit(expr, "x", to, "json");
        assert!(success, "Command should succeed for {expr} at {to}");
        assert!(
            stdout.contains(&format!("\"result\":\"{expected}\"")),
            "{expr} at {to} should resolve to {expected}, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved sinh/cosh(exp(linear)) limit should not warn for {expr} at {to}, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_sinh_cosh_of_polynomial_argument_tails_at_infinity() {
    for (expr, to, expected) in [
        ("sinh(x^2)", "infinity", "infinity"),
        ("sinh(x^2)", "-infinity", "infinity"),
        ("sinh(x^3 - 2*x)", "-infinity", "-infinity"),
        ("sinh(2 - x^4)", "infinity", "-infinity"),
        ("cosh(x^2)", "infinity", "infinity"),
        ("cosh(x^2)", "-infinity", "infinity"),
        ("cosh(2 - x^4)", "infinity", "infinity"),
    ] {
        let (success, stdout) = run_limit(expr, "x", to, "json");
        assert!(success, "Command should succeed for {expr} at {to}");
        assert!(
            stdout.contains(&format!("\"result\":\"{expected}\"")),
            "{expr} at {to} should resolve to {expected}, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved sinh/cosh(polynomial tail) limit should not warn for {expr} at {to}, got: {stdout}"
        );
    }

    let (success, stdout) = run_eval("limit(sinh(2 - x^4), x, infinity)", "json");
    assert!(
        success,
        "Eval should succeed for negative polynomial-tail sinh"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "-infinity");
    assert_eq!(wire["warnings"], json!([]));
    assert_eq!(wire["required_conditions"], json!([]));
    assert_eq!(wire["required_display"], json!([]));
}

#[test]
fn test_limit_sinh_cosh_of_unsupported_argument_remains_residual() {
    for expr in [
        "sinh(exp(x^2))",
        "cosh(exp(x^2))",
        "sinh(a*x^2 + 1)",
        "cosh(a*x^2 + 1)",
    ] {
        let (success, stdout) = run_limit(expr, "x", "infinity", "json");
        assert!(success, "Command should succeed for residual {expr}");
        assert!(
            stdout.contains("\"warning\"") && stdout.contains("limit("),
            "{expr} should keep nonlinear argument residual, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_acosh_of_domain_safe_linear_argument_at_infinity() {
    for (expr, to) in [("acosh(x)", "infinity"), ("acosh(1 - x)", "-infinity")] {
        let (success, stdout) = run_limit(expr, "x", to, "json");
        assert!(success, "Command should succeed for {expr} at {to}");
        assert!(
            stdout.contains("\"result\":\"infinity\""),
            "{expr} at {to} should resolve to infinity, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved acosh(linear) limit should not warn for {expr} at {to}, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_acosh_of_domain_unsafe_linear_argument_remains_residual() {
    for (expr, to) in [("acosh(x)", "-infinity"), ("acosh(1 - x)", "infinity")] {
        let (success, stdout) = run_limit(expr, "x", to, "json");
        assert!(
            success,
            "Command should succeed for residual {expr} at {to}"
        );
        assert!(
            stdout.contains("\"warning\"") && stdout.contains("limit("),
            "{expr} at {to} should keep domain-unsafe residual, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_acosh_of_domain_safe_linear_exp_argument_at_infinity() {
    for (expr, to) in [
        ("acosh(exp(x))", "infinity"),
        ("acosh(exp(-x))", "-infinity"),
    ] {
        let (success, stdout) = run_limit(expr, "x", to, "json");
        assert!(success, "Command should succeed for {expr} at {to}");
        assert!(
            stdout.contains("\"result\":\"infinity\""),
            "{expr} at {to} should resolve to infinity, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved acosh(exp(linear)) limit should not warn for {expr} at {to}, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_acosh_of_domain_safe_polynomial_argument_at_infinity() {
    for (expr, to) in [
        ("acosh(x^2)", "infinity"),
        ("acosh(x^2)", "-infinity"),
        ("acosh(x^2 - 3)", "infinity"),
    ] {
        let (success, stdout) = run_limit(expr, "x", to, "json");
        assert!(success, "Command should succeed for {expr} at {to}");
        assert!(
            stdout.contains("\"result\":\"infinity\""),
            "{expr} at {to} should resolve to infinity, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved acosh(polynomial positive tail) limit should not warn for {expr} at {to}, got: {stdout}"
        );
    }

    let (success, stdout) = run_eval("limit(acosh(x^2 - 3), x, infinity)", "json");
    assert!(success, "Eval should succeed for shifted polynomial acosh");
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "infinity");
    assert_eq!(wire["warnings"], json!([]));
    assert_eq!(wire["required_conditions"], json!([]));
    assert_eq!(wire["required_display"], json!([]));
}

#[test]
fn test_limit_acosh_of_domain_safe_rational_argument_at_infinity() {
    let expr = "acosh((x^2 + 1)/(x + 1))";
    let (success, stdout) = run_limit(expr, "x", "infinity", "json");
    assert!(success, "Command should succeed for {expr}");
    assert!(
        stdout.contains("\"result\":\"infinity\""),
        "{expr} should resolve to infinity, got: {stdout}"
    );
    assert!(
        !stdout.contains("\"warning\""),
        "Resolved acosh(rational positive tail) limit should not warn, got: {stdout}"
    );

    let (success, stdout) = run_eval("limit(acosh((x^2 + 1)/(x + 1)), x, infinity)", "json");
    assert!(success, "Eval should succeed for rational acosh");
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "infinity");
    assert_eq!(wire["warnings"], json!([]));
    assert_eq!(wire["required_conditions"], json!([]));
    assert_eq!(wire["required_display"], json!([]));
}

#[test]
fn test_limit_acosh_of_domain_safe_rational_finite_tail_argument_at_infinity() {
    let expr = "acosh((2*x^2 + 1)/(x^2 + 1))";
    let (success, stdout) = run_limit(expr, "x", "infinity", "json");
    assert!(success, "Command should succeed for {expr}");
    assert!(
        stdout.contains("\"result\":\"acosh(2)\""),
        "{expr} should resolve to acosh(2), got: {stdout}"
    );
    assert!(
        !stdout.contains("\"warning\""),
        "Resolved acosh(rational finite tail) limit should not warn, got: {stdout}"
    );

    let (success, stdout) = run_eval("limit(acosh((2*x^2 + 1)/(x^2 + 1)), x, infinity)", "json");
    assert!(
        success,
        "Eval should succeed for finite-tail rational acosh"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "acosh(2)");
    assert_eq!(wire["warnings"], json!([]));
    assert_eq!(wire["required_conditions"], json!([]));
    assert_eq!(wire["required_display"], json!([]));
}

#[test]
fn test_limit_acosh_of_domain_unsafe_or_nonlinear_argument_remains_residual() {
    for (expr, to) in [
        ("acosh(exp(x))", "-infinity"),
        ("acosh(exp(-x))", "infinity"),
        ("acosh(3 - x^2)", "infinity"),
        ("acosh(a*x^2 + 1)", "infinity"),
        ("acosh((x^2 + 1)/(1 - x))", "infinity"),
        ("acosh((x + 1)/(x^2 + 1))", "infinity"),
        ("acosh((x^2 + 1)/(x^2 + 1))", "infinity"),
        ("acosh((x^2 + 1)/(2*x^2 + 1))", "infinity"),
        ("acosh((1 - 2*x^2)/(x^2 + 1))", "infinity"),
        ("acosh((a*x^2 + 1)/(x^2 + 1))", "infinity"),
        ("acosh((a*x^2 + 1)/(x + 1))", "infinity"),
        ("acosh(exp(x^2))", "infinity"),
    ] {
        let (success, stdout) = run_limit(expr, "x", to, "json");
        assert!(
            success,
            "Command should succeed for residual {expr} at {to}"
        );
        assert!(
            stdout.contains("\"warning\"") && stdout.contains("limit("),
            "{expr} at {to} should keep unsupported acosh residual, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_acosh_polynomial_dominance_at_infinity() {
    for (expr, to, expected) in [
        ("acosh(x)/x", "infinity", "0"),
        ("x/acosh(x)", "infinity", "infinity"),
        ("acosh(x^2)/x", "infinity", "0"),
        ("acosh(x^2 - 3)/x", "infinity", "0"),
        ("x/acosh(x^2)", "infinity", "infinity"),
        ("x/acosh(x^2)", "-infinity", "-infinity"),
        ("acosh((x^2 + 1)/(x + 1))/x", "infinity", "0"),
        ("x/acosh((x^2 + 1)/(x + 1))", "infinity", "infinity"),
        ("acosh(1 - x)/x^2", "-infinity", "0"),
        ("x/acosh(1 - x)", "-infinity", "-infinity"),
    ] {
        let (success, stdout) = run_limit(expr, "x", to, "json");
        assert!(success, "Command should succeed for {expr} at {to}");
        assert!(
            stdout.contains(&format!("\"result\":\"{expected}\"")),
            "{expr} at {to} should resolve to {expected}, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved acosh-polynomial dominance should not warn for {expr} at {to}, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_acosh_exp_dominance_at_infinity() {
    for (expr, to, expected) in [
        ("acosh(x)*exp(-x)", "infinity", "0"),
        ("acosh(x^2)*exp(-x)", "infinity", "0"),
        ("acosh((x^2 + 1)/(x + 1))*exp(-x)", "infinity", "0"),
        ("acosh(1 - x)*exp(x)", "-infinity", "0"),
        ("exp(x)/acosh(x)", "infinity", "infinity"),
        ("exp(x)/acosh(x^2)", "infinity", "infinity"),
        ("exp(x)/acosh((x^2 + 1)/(x + 1))", "infinity", "infinity"),
    ] {
        let (success, stdout) = run_limit(expr, "x", to, "json");
        assert!(success, "Command should succeed for {expr} at {to}");
        assert!(
            stdout.contains(&format!("\"result\":\"{expected}\"")),
            "{expr} at {to} should resolve to {expected}, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved acosh-exp dominance should not warn for {expr} at {to}, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_acosh_dominance_bad_domain_or_nonlinear_argument_remains_residual() {
    for (expr, to) in [
        ("acosh(x)/x", "-infinity"),
        ("x/acosh(1 - x)", "infinity"),
        ("acosh(3 - x^2)/x", "infinity"),
        ("x/acosh(3 - x^2)", "infinity"),
        ("acosh(a*x^2 + 1)/x", "infinity"),
        ("acosh((x^2 + 1)/(1 - x))/x", "infinity"),
        ("x/acosh((x + 1)/(x^2 + 1))", "infinity"),
        ("acosh((a*x^2 + 1)/(x + 1))/x", "infinity"),
        ("exp(x)/acosh(1 - x)", "infinity"),
        ("exp(x)/acosh(3 - x^2)", "infinity"),
        ("exp(x)/acosh((x^2 + 1)/(1 - x))", "infinity"),
    ] {
        let (success, stdout) = run_limit(expr, "x", to, "json");
        assert!(
            success,
            "Command should succeed for residual {expr} at {to}"
        );
        assert!(
            stdout.contains("\"warning\"") && stdout.contains("limit("),
            "{expr} at {to} should keep acosh dominance residual, got: {stdout}"
        );
    }
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
fn test_limit_log_polynomial_argument_dominance_at_infinity() {
    for (expr, expected) in [
        ("ln(x^2)/x", "0"),
        ("ln(x^2 - 3)/x", "0"),
        ("x/ln(x^2)", "infinity"),
        ("log(2, x^2)/x", "0"),
        ("ln((x^2 + 1)/(x + 1))/x", "0"),
        ("x/ln((x^2 + 1)/(x + 1))", "infinity"),
        ("log(2, (x^2 + 1)/(x + 1))/x", "0"),
        ("ln((x + 1)/(x^2 + 1))/x", "0"),
        ("x/ln((x + 1)/(x^2 + 1))", "-infinity"),
        ("log(1/2, (x + 1)/(x^2 + 1))/x", "0"),
    ] {
        let (success, stdout) = run_limit(expr, "x", "infinity", "json");
        assert!(success, "Command should succeed for {expr}");
        assert!(
            stdout.contains(&format!("\"result\":\"{expected}\"")),
            "{expr} should resolve to {expected}, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved log-polynomial dominance should not warn for {expr}, got: {stdout}"
        );
    }

    let (success, stdout) = run_eval("limit(ln(x^2 - 3)/x, x, infinity)", "json");
    assert!(success, "Eval should succeed for shifted polynomial log");
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "0");
    assert_eq!(wire["warnings"], json!([]));
    assert_eq!(wire["required_conditions"], json!([]));
    assert_eq!(wire["required_display"], json!([]));

    let (success, stdout) = run_eval("limit(x/ln((x + 1)/(x^2 + 1)), x, infinity)", "json");
    assert!(
        success,
        "Eval should succeed for rational zero-tail log denominator"
    );
    let wire: Value = serde_json::from_str(&stdout).expect("eval json");
    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "-infinity");
    assert_eq!(wire["warnings"], json!([]));
    assert_eq!(wire["required_conditions"], json!([]));
    assert_eq!(wire["required_display"], json!([]));
}

#[test]
fn test_limit_log_polynomial_argument_bad_tail_remains_residual() {
    for expr in [
        "ln(3 - x^2)/x",
        "ln((x^2 + 1)/(1 - x))/x",
        "ln((a*x^2 + 1)/(x + 1))/x",
        "ln((1 - x)/(x^2 + 1))/x",
        "ln((a*x + 1)/(x^2 + 1))/x",
    ] {
        let (success, stdout) = run_limit(expr, "x", "infinity", "json");
        assert!(
            success,
            "Command should succeed for residual log polynomial"
        );
        assert!(
            stdout.contains("\"warning\"") && stdout.contains("limit("),
            "Unsupported log argument should remain residual for {expr}, got: {stdout}"
        );
    }
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
    for expr in ["ln(x)/exp(x)", "ln((x + 1)/(x^2 + 1))/exp(x)"] {
        let (success, stdout) = run_limit(expr, "x", "infinity", "json");
        assert!(success, "Command should succeed for {expr}");
        assert!(
            stdout.contains("\"result\":\"0\""),
            "{expr} should resolve to 0, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved log-over-exp dominance should not warn for {expr}, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_exp_over_ln_pos_infinity() {
    for (expr, expected) in [
        ("exp(x)/ln(x)", "infinity"),
        ("exp(x)/ln((x + 1)/(x^2 + 1))", "-infinity"),
    ] {
        let (success, stdout) = run_limit(expr, "x", "infinity", "json");
        assert!(success, "Command should succeed for {expr}");
        assert!(
            stdout.contains(&format!("\"result\":\"{expected}\"")),
            "{expr} should resolve to {expected}, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved exp-over-log dominance should not warn for {expr}, got: {stdout}"
        );
    }
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
    for expr in [
        "sin(x)*exp(-x)",
        "exp(-2*x)*cos(x)",
        "sin(x)*exp(2 - x^4)",
        "cos(x)*exp(-x^2)",
        "sin(sqrt(x))*exp(-x)",
        "sin(ln(x))*exp(-x)",
    ] {
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
fn test_limit_bounded_trig_times_decaying_exp_domain_conflicts_remain_residual() {
    for expr in [
        "sin(sqrt(1 - x))*exp(-x)",
        "sin(ln(1 - x))*exp(-x)",
        "sin(sqrt(1 - x))*exp(2 - x^4)",
        "sin(x)*exp(a*x^2)",
        "sin(x)*exp(exp(0 - x))",
    ] {
        let (success, stdout) = run_limit(expr, "x", "infinity", "json");
        assert!(success, "Command should succeed for {expr}");
        assert!(
            stdout.contains("\"warning\"") && stdout.contains("limit("),
            "Unsupported or domain-conflicting bounded-exp shape should remain residual for {expr}, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_ln_over_decaying_exp_pos_infinity() {
    for (expr, expected) in [
        ("ln(x)/exp(-x)", "infinity"),
        ("ln((x + 1)/(x^2 + 1))/exp(-x)", "-infinity"),
    ] {
        let (success, stdout) = run_limit(expr, "x", "infinity", "json");
        assert!(success, "Command should succeed for {expr}");
        assert!(
            stdout.contains(&format!("\"result\":\"{expected}\"")),
            "{expr} should resolve to {expected}, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved log-over-decaying-exp dominance should not warn for {expr}, got: {stdout}"
        );
    }
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
fn test_limit_polynomial_exp_subpolynomial_dominance_pos_infinity() {
    for (expr, expected) in [
        ("exp(x^2)/ln(x)", "infinity"),
        ("ln(x)/exp(x^2)", "0"),
        ("sqrt(x)*exp(2 - x^4)", "0"),
        ("exp(x^2)/(-sqrt(x))", "-infinity"),
    ] {
        let (success, stdout) = run_limit(expr, "x", "infinity", "json");
        assert!(success, "Command should succeed for {expr}");
        assert!(
            stdout.contains(&format!("\"result\":\"{expected}\"")),
            "{expr} should resolve to {expected}, got: {stdout}"
        );
        assert!(
            !stdout.contains("\"warning\""),
            "Resolved polynomial-exp/subpolynomial dominance should not warn for {expr}, got: {stdout}"
        );
    }
}

#[test]
fn test_limit_unsupported_polynomial_exp_subpolynomial_dominance_remains_residual() {
    for expr in ["exp(a*x^2)/ln(x)", "exp(exp(x^2))/ln(x)"] {
        let (success, stdout) = run_limit(expr, "x", "infinity", "json");
        assert!(success, "Command should succeed for {expr}");
        assert!(
            stdout.contains("\"warning\"") && stdout.contains("limit("),
            "Unsupported polynomial-exp/subpolynomial shape should remain residual for {expr}, got: {stdout}"
        );
    }
}
