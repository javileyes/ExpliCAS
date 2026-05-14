//! Smoke tests for the eval wire model.
//!
//! Verifies that the wire field is present and correctly structured
//! in JSON output from eval commands.

use assert_cmd::cargo;
use serde_json::Value;
use std::process::Command;

/// Helper to run eval and parse the wire field from the JSON output.
fn eval_wire(expr: &str) -> Value {
    let output = Command::new(cargo::cargo_bin!("cas_cli"))
        .arg("eval")
        .arg(expr)
        .arg("--format")
        .arg("json")
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: Value = serde_json::from_str(&stdout).expect("Failed to parse JSON");
    json.get("wire").cloned().unwrap_or(Value::Null)
}

/// Helper to run eval and parse the full JSON output.
fn eval_json_with_args(expr: &str, args: &[&str]) -> Value {
    let mut command = Command::new(cargo::cargo_bin!("cas_cli"));
    command.arg("eval").arg(expr).arg("--format").arg("json");
    command.args(args);

    let output = command.output().expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&stdout).expect("Failed to parse JSON")
}

/// Helper to run eval and parse the full JSON output.
fn eval_json(expr: &str) -> Value {
    eval_json_with_args(expr, &[])
}

#[test]
fn test_wire_present_on_success() {
    let wire = eval_wire("2+2");

    // Wire should exist
    assert!(!wire.is_null(), "wire field should be present");

    // Schema version should be 1
    assert_eq!(
        wire.get("schema_version"),
        Some(&Value::Number(1.into())),
        "schema_version should be 1"
    );

    // Messages should be an array
    let messages = wire.get("messages").expect("messages should exist");
    assert!(messages.is_array(), "messages should be an array");
    assert!(
        !messages.as_array().unwrap().is_empty(),
        "messages should not be empty"
    );
}

#[test]
fn test_eval_negative_power_denominator_preserves_required_conditions() {
    let json = eval_json("ln(y)*(z+1)^(-2)");

    assert_eq!(json["ok"], true);
    assert_eq!(json["result"], "ln(y) / (z + 1)^2");

    let required = json["required_conditions"]
        .as_array()
        .expect("required_conditions should be an array");

    assert!(
        required
            .iter()
            .any(|condition| condition["kind"] == "Positive" && condition["expr_display"] == "y"),
        "expected Positive(y), got: {required:?}"
    );
    assert!(
        required
            .iter()
            .any(|condition| condition["kind"] == "NonZero"
                && condition["expr_display"] == "z + 1"),
        "expected NonZero(z + 1), got: {required:?}"
    );
}

#[test]
fn test_eval_calculus_residual_domain_modes_preserve_required_conditions() {
    let expr = "diff(integrate(ln(y)*(z+1)^(-2), x), x) - ln(y)*(z+1)^(-2)";

    for mode in ["generic", "assume"] {
        let json = eval_json_with_args(expr, &["--domain", mode]);

        assert_eq!(json["ok"], true, "eval should succeed in {mode}");
        assert_eq!(json["result"], "0", "residual should verify in {mode}");
        assert_eq!(json["domain"]["mode"], mode);
        assert_eq!(json["semantics"]["domain_mode"], mode);

        let required = json["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array");
        assert_eq!(
            required.len(),
            2,
            "expected only the intrinsic input-domain guards in {mode}: {required:?}"
        );
        assert!(
            required.iter().any(|condition| {
                condition["kind"] == "Positive" && condition["expr_canonical"] == "y"
            }),
            "expected Positive(y) in {mode}, got: {required:?}"
        );
        assert!(
            required.iter().any(|condition| {
                condition["kind"] == "NonZero" && condition["expr_canonical"] == "z + 1"
            }),
            "expected NonZero(z + 1) in {mode}, got: {required:?}"
        );

        let messages = json["wire"]["messages"]
            .as_array()
            .expect("wire messages should be an array");
        assert!(
            messages.iter().any(|message| message["text"]
                .as_str()
                .is_some_and(|text| text.contains("Requires:"))),
            "intrinsic calculus-domain guards should render as Requires in {mode}: {messages:?}"
        );
        assert!(
            !messages.iter().any(|message| message["text"]
                .as_str()
                .is_some_and(|text| text.contains("Assume:"))),
            "intrinsic calculus-domain guards should not render as Assume in {mode}: {messages:?}"
        );
    }
}

#[test]
fn test_eval_json_inverse_trig_alias_sqrt_derivative_conditions_dedupe() {
    let cases = [
        (
            "diff(sqrt(asin(2*x+1)), x)",
            ["asin(2·x + 1)", "arcsin(2·x + 1)"],
            1,
        ),
        (
            "diff(sqrt(acos(2*x+1)), x)",
            ["acos(2·x + 1)", "arccos(2·x + 1)"],
            0,
        ),
    ];

    for (expr, aliases, min_alias_guards) in cases {
        let json = eval_json(expr);

        assert_eq!(json["ok"], true, "expr: {expr}");

        let required = json["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array");
        let alias_count = required
            .iter()
            .filter(|condition| {
                condition["kind"] == "Positive"
                    && aliases
                        .iter()
                        .any(|alias| condition["expr_canonical"] == *alias)
            })
            .count();

        assert!(
            (min_alias_guards..=1).contains(&alias_count),
            "expected {min_alias_guards}..=1 alias positivity guards for {expr}: {required:?}"
        );
        if expr.contains("asin(") {
            assert!(
                required.iter().any(|condition| {
                    condition["kind"] == "Positive"
                        && condition["expr_canonical"] == "asin(2·x + 1)"
                }),
                "expected asin alias display to match the public result for {expr}: {required:?}"
            );
        }
        assert!(
            required.iter().any(|condition| {
                condition["kind"] == "Positive" && condition["expr_canonical"] == "-x^2 - x"
            }),
            "expected the inverse-trig interval guard for {expr}: {required:?}"
        );
    }
}

#[test]
fn test_eval_json_steps_preserve_post_calculus_presentation_step() {
    let cases = [
        (
            "diff(integrate(3/(2*sqrt(3*x+1)*cosh(sqrt(3*x+1))^2), x), x)",
            "3 / (2·sqrt(3·x + 1)·cosh(sqrt(3·x + 1))^2)",
            "((3 · x + 1))^(-1/2)",
            ["sqrt(3 · x + 1)", "cosh(sqrt(3 · x + 1))"],
            Some("3·x + 1"),
        ),
        (
            "diff(arctan(sqrt(x)), x)",
            "1 / (2·sqrt(x)·(x + 1))",
            "x^(-1/2)",
            ["sqrt(x)", "x + 1"],
            Some("x"),
        ),
        (
            "diff(arctan(sqrt(3*x)), x)",
            "3 / (2·sqrt(3·x)·(3·x + 1))",
            "((3 · x))^(-1/2)",
            ["sqrt(3 · x)", "3 · x + 1"],
            Some("x"),
        ),
        (
            "diff(arctan(sqrt(x+1)), x)",
            "1 / (2·sqrt(x + 1)·(x + 2))",
            "((x + 1))^(-1/2)",
            ["sqrt(x + 1)", "x + 2"],
            Some("x + 1"),
        ),
        (
            "diff(arctan(sqrt(3-2*x)), x)",
            "-1 / (2·sqrt(3 - 2·x)·(2 - x))",
            "((3 - 2 · x))^(-1/2)",
            ["sqrt(3 - 2 · x)", "2 - x"],
            Some("3 - 2·x"),
        ),
        (
            "diff(arctan(sqrt(x^2+1)), x)",
            "x / (sqrt(x^2 + 1)·(x^2 + 2))",
            "((x^2 + 1))^(-1/2)",
            ["sqrt(x^2 + 1)", "x^2 + 2"],
            None,
        ),
        (
            "diff(sqrt(sin(x)), x)",
            "cos(x) / (2·sqrt(sin(x)))",
            "sin(x)^(-1/2)",
            ["cos(x)", "sqrt(sin(x))"],
            Some("sin(x)"),
        ),
        (
            "diff(sqrt(ln(x)), x)",
            "1 / (2·x·sqrt(ln(x)))",
            "ln(x)^(-1/2)",
            ["x", "sqrt(ln(x))"],
            Some("ln(x)"),
        ),
        (
            "diff(sqrt(log10(x)), x)",
            "1 / (2·x·ln(10)·sqrt(log10(x)))",
            "log10(x)^(-1/2)",
            ["ln(10)", "sqrt(log_10(x))"],
            Some("log10(x)"),
        ),
        (
            "diff(sqrt(tan(x)), x)",
            "1 / (2·cos(x)^2·sqrt(tan(x)))",
            "tan(x)^(-1/2)",
            ["sqrt(tan(x))", "cos(x)^2"],
            Some("sin(x) / cos(x)"),
        ),
        (
            "diff(sqrt(cot(x)), x)",
            "-1 / (2·sin(x)^2·sqrt(cot(x)))",
            "cot(x)^(-1/2)",
            ["sqrt(cot(x))", "sin(x)^2"],
            Some("cos(x) / sin(x)"),
        ),
        (
            "diff(sqrt(tanh(x)), x)",
            "1 / (2·cosh(x)^2·sqrt(tanh(x)))",
            "tanh(x)^(-1/2)",
            ["sqrt(tanh(x))", "cosh(x)^2"],
            Some("tanh(x)"),
        ),
        (
            "diff(sqrt(sinh(x)), x)",
            "cosh(x) / (2·sqrt(sinh(x)))",
            "sinh(x)^(-1/2)",
            ["cosh(x)", "sqrt(sinh(x))"],
            Some("sinh(x)"),
        ),
        (
            "diff(sqrt(cosh(x)), x)",
            "sinh(x) / (2·sqrt(cosh(x)))",
            "cosh(x)^(-1/2)",
            ["sinh(x)", "sqrt(cosh(x))"],
            None,
        ),
        (
            "diff(sqrt(sec(x)), x)",
            "tan(x)·sqrt(sec(x)) / 2",
            "|cos(x)^(-1/2)|",
            ["tan(x)", "sqrt(sec(x))"],
            Some("cos(x)"),
        ),
        (
            "diff(sqrt(csc(x)), x)",
            "-cot(x)·sqrt(csc(x)) / 2",
            "|sin(x)^(-1/2)|",
            ["cot(x)", "sqrt(csc(x))"],
            Some("sin(x)"),
        ),
        (
            "diff(sqrt(sec(3-2*x)), x)",
            "-tan(3 - 2·x)·sqrt(sec(3 - 2·x))",
            "|(cos(3 - 2 · x))^(-1/2)|",
            ["tan(3 - 2 · x)", "sqrt(sec(3 - 2 · x))"],
            Some("cos(3 - 2·x)"),
        ),
        (
            "diff(sqrt(csc(3-2*x)), x)",
            "cot(3 - 2·x)·sqrt(csc(3 - 2·x))",
            "|(sin(3 - 2 · x))^(-1/2)|",
            ["cot(3 - 2 · x)", "sqrt(csc(3 - 2 · x))"],
            Some("sin(3 - 2·x)"),
        ),
        (
            "diff(sqrt(arctan(x)), x)",
            "1 / (2·(x^2 + 1)·sqrt(arctan(x)))",
            "arctan(x)^(-1/2)",
            ["sqrt(arctan(x))", "x^2 + 1"],
            Some("arctan(x)"),
        ),
        (
            "diff(sqrt(arctan(2*x+1)), x)",
            "1 / (((2·x + 1)^2 + 1)·sqrt(arctan(2·x + 1)))",
            "arctan(2 · x + 1)",
            ["sqrt(arctan(2 · x + 1))", "((2 · x + 1))^2 + 1"],
            Some("arctan(2·x + 1)"),
        ),
        (
            "diff(sqrt(atanh(x)), x)",
            "1 / (2·(1 - x^2)·sqrt(atanh(x)))",
            "atanh(x)^(-1/2)",
            ["1 - x^2", "sqrt(atanh(x))"],
            Some("1 - x^2"),
        ),
        (
            "diff(sqrt(arcsin(x)), x)",
            "1 / (2·sqrt(1 - x^2)·sqrt(arcsin(x)))",
            "arcsin(x)^(-1/2)",
            ["sqrt(arcsin(x))", "sqrt(1 - x^2)"],
            Some("arcsin(x)"),
        ),
        (
            "diff(sqrt(arccos(x)), x)",
            "-1 / (2·sqrt(1 - x^2)·sqrt(arccos(x)))",
            "arccos(x)^(-1/2)",
            ["sqrt(arccos(x))", "sqrt(1 - x^2)"],
            Some("arccos(x)"),
        ),
        (
            "diff(sqrt(asinh(x)), x)",
            "1 / (2·sqrt(x^2 + 1)·sqrt(asinh(x)))",
            "asinh(x)^(-1/2)",
            ["sqrt(asinh(x))", "sqrt(x^2 + 1)"],
            Some("asinh(x)"),
        ),
        (
            "diff(sqrt(asinh(2*x+1)), x)",
            "1 / (sqrt((2·x + 1)^2 + 1)·sqrt(asinh(2·x + 1)))",
            "asinh(2 · x + 1)",
            ["sqrt(asinh(2 · x + 1))", "sqrt(((2 · x + 1))^2 + 1)"],
            Some("asinh(2·x + 1)"),
        ),
        (
            "diff(sqrt(acosh(x)), x)",
            "1 / (2·sqrt(x - 1)·sqrt(x + 1)·sqrt(acosh(x)))",
            "acosh(x)^(-1/2)",
            ["sqrt(x - 1)", "sqrt(x + 1)"],
            Some("acosh(x)"),
        ),
        (
            "diff(sqrt(acosh(2*x+3)), x)",
            "1 / (sqrt(2·x + 2)·sqrt(2·x + 4)·sqrt(acosh(2·x + 3)))",
            "acosh(2 · x + 3)",
            ["sqrt(2 · x + 2)", "sqrt(2 · x + 4)"],
            Some("acosh(2·x + 3)"),
        ),
    ];

    for (expr, expected_result, expected_before, expected_after, required_expr) in cases {
        let json = eval_json_with_args(expr, &["--steps", "on"]);

        assert_eq!(json["ok"], true, "expr: {expr}");
        assert_eq!(json["result"], expected_result, "expr: {expr}");

        let steps = json["steps"].as_array().expect("steps should be an array");
        assert!(
            steps.iter().any(|step| {
                step["rule"] == "Presentar resultado de cálculo en forma compacta"
                    && step["before"]
                        .as_str()
                        .is_some_and(|before| before.contains(expected_before))
                    && step["after"].as_str().is_some_and(|after| {
                        expected_after.iter().all(|needle| after.contains(needle))
                    })
            }) || steps.iter().any(|step| {
                step["rule"] == "Calcular la derivada"
                    && step["after"].as_str().is_some_and(|after| {
                        expected_after.iter().all(|needle| after.contains(needle))
                    })
            }),
            "expected public JSON steps to include compact calculus presentation for {expr}: {steps:?}"
        );

        let required = json["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array");
        if let Some(required_expr) = required_expr {
            assert!(
                required.iter().any(|condition| {
                    condition["kind"] == "Positive" && condition["expr_canonical"] == required_expr
                }),
                "post-calculus presentation should preserve the sqrt-domain guard for {expr}: {required:?}"
            );
        } else {
            assert!(
                required.is_empty(),
                "post-calculus presentation should not add redundant sqrt-domain guards for {expr}: {required:?}"
            );
        }
    }
}

#[test]
fn test_eval_json_post_calculus_required_conditions_follow_input_inverse_trig_alias() {
    let short = eval_json_with_args("diff(sqrt(atan(2*x+1)), x)", &[]);
    assert_eq!(short["ok"], true);
    assert_eq!(
        short["result"],
        "1 / (((2·x + 1)^2 + 1)·sqrt(atan(2·x + 1)))"
    );

    let short_required = short["required_conditions"]
        .as_array()
        .expect("required_conditions should be an array");
    let short_positive = short_required
        .iter()
        .find(|condition| condition["kind"] == "Positive")
        .expect("expected positive sqrt-domain guard");
    assert_eq!(short_positive["expr_display"], "atan(2·x + 1)");
    assert_eq!(short_positive["expr_canonical"], "atan(2·x + 1)");

    let parenthesized = eval_json_with_args("diff(sqrt(atan((2*x+1))), x)", &[]);
    assert_eq!(parenthesized["ok"], true);
    assert_eq!(
        parenthesized["result"],
        "1 / (((2·x + 1)^2 + 1)·sqrt(atan(2·x + 1)))"
    );
    let parenthesized_required = parenthesized["required_conditions"]
        .as_array()
        .expect("required_conditions should be an array");
    let parenthesized_positive = parenthesized_required
        .iter()
        .find(|condition| condition["kind"] == "Positive")
        .expect("expected positive sqrt-domain guard");
    assert_eq!(parenthesized_positive["expr_display"], "atan(2·x + 1)");
    assert_eq!(parenthesized_positive["expr_canonical"], "atan(2·x + 1)");

    let long = eval_json_with_args("diff(sqrt(arctan(2*x+1)), x)", &[]);
    assert_eq!(long["ok"], true);
    let long_required = long["required_conditions"]
        .as_array()
        .expect("required_conditions should be an array");
    let long_positive = long_required
        .iter()
        .find(|condition| condition["kind"] == "Positive")
        .expect("expected positive sqrt-domain guard");
    assert_eq!(long_positive["expr_display"], "arctan(2·x + 1)");
    assert_eq!(long_positive["expr_canonical"], "arctan(2·x + 1)");

    let mixed = eval_json_with_args("diff(sqrt(arctan(2*x+1)), x) + atan(y)-atan(y)", &[]);
    assert_eq!(mixed["ok"], true);
    let mixed_required = mixed["required_conditions"]
        .as_array()
        .expect("required_conditions should be an array");
    let mixed_positive = mixed_required
        .iter()
        .find(|condition| condition["kind"] == "Positive")
        .expect("expected positive sqrt-domain guard");
    assert_eq!(mixed_positive["expr_display"], "arctan(2·x + 1)");
    assert_eq!(mixed_positive["expr_canonical"], "arctan(2·x + 1)");

    let same_arg_noise = eval_json_with_args(
        "diff(sqrt(arctan(2*x+1)), x) + atan(2*x+1)-atan(2*x+1)",
        &[],
    );
    assert_eq!(same_arg_noise["ok"], true);
    let same_arg_noise_required = same_arg_noise["required_conditions"]
        .as_array()
        .expect("required_conditions should be an array");
    let same_arg_noise_positive = same_arg_noise_required
        .iter()
        .find(|condition| condition["kind"] == "Positive")
        .expect("expected positive sqrt-domain guard");
    assert_eq!(same_arg_noise_positive["expr_display"], "arctan(2·x + 1)");
    assert_eq!(same_arg_noise_positive["expr_canonical"], "arctan(2·x + 1)");
}

#[test]
fn test_eval_json_post_calculus_sqrt_elementary_derivative_verifies() {
    let cases = [
        (
            "diff(sqrt(sin(x)), x) - cos(x)/(2*sqrt(sin(x)))",
            ["sin(x)", ""],
        ),
        ("diff(sqrt(ln(x)), x) - 1/(2*x*sqrt(ln(x)))", ["ln(x)", "x"]),
        (
            "diff(sqrt(log10(x)), x) - 1/(2*x*ln(10)*sqrt(log10(x)))",
            ["log10(x)", "x"],
        ),
        (
            "diff(sqrt(tan(x)), x) - 1/(2*cos(x)^2*sqrt(tan(x)))",
            ["sin(x) / cos(x)", ""],
        ),
        (
            "diff(sqrt(cot(x)), x) + 1/(2*sin(x)^2*sqrt(cot(x)))",
            ["cos(x) / sin(x)", ""],
        ),
        (
            "diff(sqrt(tanh(x)), x) - 1/(2*cosh(x)^2*sqrt(tanh(x)))",
            ["tanh(x)", ""],
        ),
        (
            "diff(sqrt(sinh(x)), x) - cosh(x)/(2*sqrt(sinh(x)))",
            ["sinh(x)", ""],
        ),
        (
            "diff(sqrt(cosh(x)), x) - sinh(x)/(2*sqrt(cosh(x)))",
            ["", ""],
        ),
        (
            "diff(sqrt(sec(x)), x) - sqrt(sec(x))*tan(x)/2",
            ["cos(x)", ""],
        ),
        (
            "diff(sqrt(csc(x)), x) + sqrt(csc(x))*cot(x)/2",
            ["sin(x)", ""],
        ),
        (
            "diff(sqrt(arctan(x)), x) - 1/(2*(x^2+1)*sqrt(arctan(x)))",
            ["arctan(x)", ""],
        ),
        (
            "diff(sqrt(arctan(2*x+1)), x) - 1/(((2*x+1)^2+1)*sqrt(arctan(2*x+1)))",
            ["arctan(2·x + 1)", ""],
        ),
        (
            "diff(sqrt(atanh(x)), x) - 1/(2*(1-x^2)*sqrt(atanh(x)))",
            ["1 - x^2", "atanh(x)"],
        ),
        (
            "diff(sqrt(arcsin(x)), x) - 1/(2*sqrt(1-x^2)*sqrt(arcsin(x)))",
            ["1 - x^2", "arcsin(x)"],
        ),
        (
            "diff(sqrt(arccos(x)), x) + 1/(2*sqrt(1-x^2)*sqrt(arccos(x)))",
            ["1 - x^2", "arccos(x)"],
        ),
        (
            "diff(sqrt(asinh(x)), x) - 1/(2*sqrt(x^2+1)*sqrt(asinh(x)))",
            ["asinh(x)", ""],
        ),
        (
            "diff(sqrt(asinh(2*x+1)), x) - 1/(sqrt((2*x+1)^2+1)*sqrt(asinh(2*x+1)))",
            ["asinh(2·x + 1)", ""],
        ),
        (
            "diff(sqrt(acosh(x)), x) - 1/(2*sqrt(x-1)*sqrt(x+1)*sqrt(acosh(x)))",
            ["acosh(x)", "x - 1"],
        ),
        (
            "diff(sqrt(acosh(2*x+3)), x) - 1/(sqrt(2*x+2)*sqrt(2*x+4)*sqrt(acosh(2*x+3)))",
            ["acosh(2·x + 3)", "x + 1"],
        ),
    ];

    for (expr, required_positive) in cases {
        let json = eval_json(expr);

        assert_eq!(json["ok"], true, "expr: {expr}");
        assert_eq!(json["result"], "0", "expr: {expr}");

        let required = json["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array");
        for required_expr in required_positive
            .into_iter()
            .filter(|expr| !expr.is_empty())
        {
            assert!(
                required.iter().any(|condition| {
                    condition["kind"] == "Positive"
                        && condition["expr_canonical"] == required_expr
                }),
                "residual verification should preserve Positive({required_expr}) for {expr}: {required:?}"
            );
        }
    }
}

#[test]
fn test_eval_json_reciprocal_half_power_shared_denominator_verifies() {
    let cases = [
        "tan(x)^(-1/2)/(2*cos(x)^2) - 1/(2*cos(x)^2*sqrt(tan(x)))",
        "equiv(diff(sqrt(tan(x)), x), 1/(2*cos(x)^2*sqrt(tan(x))))",
        "(-cot(x)^(-1/2))/(2*sin(x)^2) + 1/(2*sin(x)^2*sqrt(cot(x)))",
        "equiv(diff(sqrt(cot(x)), x), -1/(2*sin(x)^2*sqrt(cot(x))))",
    ];

    for expr in cases {
        let json = eval_json(expr);

        assert_eq!(json["ok"], true, "expr: {expr}");
        assert!(
            json["result"] == "0" || json["result"] == "true",
            "reciprocal half-power equivalence should verify for {expr}: {}",
            json["result"]
        );

        let required = json["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array");
        let expected_positive = if expr.contains("cot") {
            "cos(x) / sin(x)"
        } else {
            "sin(x) / cos(x)"
        };
        assert!(
            required.iter().any(|condition| {
                condition["kind"] == "Positive" && condition["expr_canonical"] == expected_positive
            }),
            "expected Positive({expected_positive}) for {expr}: {required:?}"
        );
        assert!(
            json["blocked_hints"]
                .as_array()
                .is_none_or(|blocked| blocked.is_empty()),
            "direct cancellation should avoid blocked hints for {expr}: {:?}",
            json["blocked_hints"]
        );
    }
}

#[test]
fn test_eval_json_reciprocal_half_power_scaled_product_content_residual_verifies() {
    let expr = "1/2*(acosh(x)*(x^2-1))^(-1/2) - sqrt(acosh(x)*(x^2-1))/(2*acosh(x)*(x^2-1))";
    let json = eval_json(expr);

    assert_eq!(json["ok"], true, "expr: {expr}");
    assert_eq!(json["result"], "0", "expr: {expr}");
    assert!(
        json["blocked_hints"]
            .as_array()
            .is_none_or(|blocked| blocked.is_empty()),
        "scaled product-content half-power residual should avoid blocked hints for {expr}: {:?}",
        json["blocked_hints"]
    );
}

#[test]
fn test_eval_json_reciprocal_trig_sqrt_derivative_equiv_verifies() {
    let cases = [
        (
            "equiv(diff(sqrt(sec(x)), x), sqrt(sec(x))*tan(x)/2)",
            "sec(x)",
        ),
        (
            "equiv(diff(sqrt(csc(x)), x), -sqrt(csc(x))*cot(x)/2)",
            "csc(x)",
        ),
        (
            "equiv(diff(sqrt(sec(3-2*x)), x), -sqrt(sec(3-2*x))*tan(3-2*x))",
            "sec(3 - 2·x)",
        ),
        (
            "equiv(diff(sqrt(csc(3-2*x)), x), sqrt(csc(3-2*x))*cot(3-2*x))",
            "csc(3 - 2·x)",
        ),
    ];

    for (expr, required_nonnegative) in cases {
        let json = eval_json(expr);

        assert_eq!(json["ok"], true, "expr: {expr}");
        assert_eq!(json["result"], "true", "expr: {expr}");

        let required = json["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array");
        assert!(
            required.iter().any(|condition| {
                condition["kind"] == "NonNegative"
                    && condition["expr_canonical"] == required_nonnegative
            }),
            "expected NonNegative({required_nonnegative}) for {expr}: {required:?}"
        );
    }
}

#[test]
fn test_eval_json_reciprocal_trig_sqrt_derivative_residuals_collapse() {
    let cases = [
        (
            "diff(sqrt(sec(3-2*x)), x) + tan(3-2*x)*sqrt(sec(3-2*x))",
            "cos(3 - 2·x)",
            "sec(3 - 2·x)",
        ),
        (
            "diff(sqrt(csc(3-2*x)), x) - cot(3-2*x)*sqrt(csc(3-2*x))",
            "sin(3 - 2·x)",
            "csc(3 - 2·x)",
        ),
    ];

    for (expr, required_positive, required_nonnegative) in cases {
        let json = eval_json(expr);

        assert_eq!(json["ok"], true, "expr: {expr}");
        assert_eq!(json["result"], "0", "expr: {expr}");

        let required = json["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array");
        assert!(
            required.iter().any(|condition| {
                condition["kind"] == "Positive" && condition["expr_canonical"] == required_positive
            }),
            "expected Positive({required_positive}) for {expr}: {required:?}"
        );
        assert!(
            required.iter().any(|condition| {
                condition["kind"] == "NonNegative"
                    && condition["expr_canonical"] == required_nonnegative
            }),
            "expected NonNegative({required_nonnegative}) for {expr}: {required:?}"
        );
    }
}

#[test]
fn test_eval_json_omits_noop_post_calculus_presentation_step() {
    let expr = "diff(arctan(1/sqrt(x)), x)";
    let json = eval_json_with_args(expr, &["--steps", "on"]);

    assert_eq!(json["ok"], true, "expr: {expr}");
    assert_eq!(json["result"], "-1 / (2·sqrt(x)·(x + 1))");

    let steps = json["steps"].as_array().expect("steps should be an array");
    assert!(
        steps.iter().all(|step| {
            step["rule"] != "Presentar resultado de cálculo en forma compacta"
                || step["before"] != step["after"]
        }),
        "post-calculus presentation should not emit a visible no-op step for {expr}: {steps:?}"
    );

    let required = json["required_conditions"]
        .as_array()
        .expect("required_conditions should be an array");
    assert!(
        required.iter().any(|condition| {
            condition["kind"] == "Positive" && condition["expr_canonical"] == "x"
        }),
        "no-op presentation filtering should preserve the sqrt-domain guard for {expr}: {required:?}"
    );
}

#[test]
fn test_eval_diff_arcsin_sqrt_affine_residual_collapses_reciprocal_root_product() {
    let cases = [
        "diff(arcsin(sqrt(2*x-1)), x) - 1/(sqrt(2*x-1)*sqrt(2-2*x))",
        "(2-2*x)^(-1/2)*(2*x-1)^(-1/2) - ((2-2*x)*(2*x-1))^(-1/2)",
    ];

    for expr in cases {
        let json = eval_json(expr);
        assert_eq!(json["ok"], true, "expr: {expr}");
        assert_eq!(json["result"], "0", "expr: {expr}");

        let required = json["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array");
        assert!(
            required.iter().any(|condition| {
                condition["kind"] == "Positive" && condition["expr_canonical"] == "1 - x"
            }),
            "expected Positive(1 - x) for {expr}: {required:?}"
        );
        assert!(
            required.iter().any(|condition| {
                condition["kind"] == "Positive" && condition["expr_canonical"] == "2·x - 1"
            }),
            "expected Positive(2*x - 1) for {expr}: {required:?}"
        );
        assert!(
            !required.iter().any(|condition| {
                condition["kind"] == "Positive"
                    && condition["expr_canonical"] == "3·x - 2·x^2 - 1"
            }),
            "expanded product guard should be dominated by its positive factors for {expr}: {required:?}"
        );
    }
}

#[test]
fn test_eval_integral_assume_mode_separates_assumptions_from_requires() {
    let generic = eval_json_with_args("integrate(ln(a^2), x)", &["--domain", "generic"]);

    assert_eq!(generic["ok"], true);
    assert_eq!(generic["result"], "2·x·ln(|a|)");
    assert_eq!(generic["domain"]["mode"], "generic");
    assert!(generic["assumptions_used"].is_null());

    let generic_required = generic["required_conditions"]
        .as_array()
        .expect("generic required_conditions should be an array");
    assert_eq!(
        generic_required.len(),
        1,
        "expected only the absolute-log domain guard: {generic_required:?}"
    );
    assert_eq!(generic_required[0]["kind"], "NonZero");
    assert_eq!(generic_required[0]["expr_canonical"], "a");

    let generic_messages = generic["wire"]["messages"]
        .as_array()
        .expect("generic wire messages should be an array");
    assert!(
        generic_messages.iter().any(|message| message["text"]
            .as_str()
            .is_some_and(|text| text.contains("Requires:"))),
        "generic mode should render the log-domain guard as Requires: {generic_messages:?}"
    );
    assert!(
        !generic_messages.iter().any(|message| message["text"]
            .as_str()
            .is_some_and(|text| text.contains("Assume:"))),
        "generic mode should not render an assumption: {generic_messages:?}"
    );

    let assume = eval_json_with_args("integrate(ln(a^2), x)", &["--domain", "assume"]);

    assert_eq!(assume["ok"], true);
    assert_eq!(assume["result"], "2·x·ln(a)");
    assert_eq!(assume["domain"]["mode"], "assume");
    assert!(
        assume["required_conditions"]
            .as_array()
            .expect("assume required_conditions should be an array")
            .is_empty(),
        "assume output should not duplicate a > 0 as a separate required condition"
    );

    let assumptions = assume["assumptions_used"]
        .as_array()
        .expect("assumptions_used should be an array in assume mode");
    assert_eq!(assumptions.len(), 1);
    assert_eq!(assumptions[0]["display"], "a > 0");
    assert_eq!(assumptions[0]["expr_canonical"], "a");
    assert_eq!(assumptions[0]["rule"], "Log Even Power");

    let assume_messages = assume["wire"]["messages"]
        .as_array()
        .expect("assume wire messages should be an array");
    assert!(
        assume_messages.iter().any(|message| message["text"]
            .as_str()
            .is_some_and(|text| text.contains("Assume:"))),
        "assume mode should render the positivity condition as Assume: {assume_messages:?}"
    );
    assert!(
        !assume_messages.iter().any(|message| message["text"]
            .as_str()
            .is_some_and(|text| text.contains("Requires:"))),
        "assume integral output should not also render Requires: {assume_messages:?}"
    );

    let residual_expr = "diff(integrate(ln(a^2), x), x) - ln(a^2)";
    for mode in ["generic", "assume"] {
        let residual = eval_json_with_args(residual_expr, &["--domain", mode]);
        assert_eq!(
            residual["ok"], true,
            "residual eval should succeed in {mode}"
        );
        assert_eq!(
            residual["result"], "0",
            "antiderivative should verify by differentiation in {mode}"
        );
    }
}

#[test]
fn test_eval_integral_radical_abs_boundary_is_conservative_until_assumed() {
    let generic = eval_json_with_args("integrate(sqrt(x^2), x)", &["--domain", "generic"]);

    assert_eq!(generic["ok"], true);
    assert_eq!(generic["result"], "integrate(|x|, x)");
    assert_eq!(generic["domain"]["mode"], "generic");
    assert!(generic["assumptions_used"].is_null());
    assert!(
        generic["required_conditions"]
            .as_array()
            .expect("generic required_conditions should be an array")
            .is_empty(),
        "generic abs integral should not invent domain requirements"
    );

    let generic_residual = eval_json_with_args(
        "diff(integrate(sqrt(x^2), x), x) - sqrt(x^2)",
        &["--domain", "generic"],
    );
    assert_eq!(generic_residual["ok"], true);
    assert_eq!(
        generic_residual["result"], "diff(integrate(|x|, x), x) - |x|",
        "generic mode should leave the unsupported abs antiderivative as a residual"
    );

    let assume = eval_json_with_args("integrate(sqrt(x^2), x)", &["--domain", "assume"]);

    assert_eq!(assume["ok"], true);
    assert_eq!(assume["result"], "1/2·x^2");
    assert_eq!(assume["domain"]["mode"], "assume");
    assert!(
        assume["required_conditions"]
            .as_array()
            .expect("assume required_conditions should be an array")
            .is_empty(),
        "assume output should not duplicate x > 0 as a separate required condition"
    );

    let assumptions = assume["assumptions_used"]
        .as_array()
        .expect("assumptions_used should be an array in assume mode");
    assert_eq!(assumptions.len(), 1);
    assert_eq!(assumptions[0]["display"], "x > 0");
    assert_eq!(assumptions[0]["expr_canonical"], "x");
    assert_eq!(assumptions[0]["rule"], "Abs Under Positivity");

    let assume_messages = assume["wire"]["messages"]
        .as_array()
        .expect("assume wire messages should be an array");
    assert!(
        assume_messages.iter().any(|message| message["text"]
            .as_str()
            .is_some_and(|text| text.contains("Assume:"))),
        "assume mode should render the positivity condition as Assume: {assume_messages:?}"
    );
    assert!(
        !assume_messages.iter().any(|message| message["text"]
            .as_str()
            .is_some_and(|text| text.contains("Requires:"))),
        "assume radical integral output should not also render Requires: {assume_messages:?}"
    );

    let assume_residual = eval_json_with_args(
        "diff(integrate(sqrt(x^2), x), x) - sqrt(x^2)",
        &["--domain", "assume"],
    );
    assert_eq!(assume_residual["ok"], true);
    assert_eq!(
        assume_residual["result"], "0",
        "assumed radical antiderivative should verify by differentiation"
    );
    let residual_assumptions = assume_residual["assumptions_used"]
        .as_array()
        .expect("assumptions_used should be an array for assumed residual");
    assert!(
        residual_assumptions.iter().any(|assumption| {
            assumption["display"] == "x > 0" && assumption["rule"] == "Abs Under Positivity"
        }),
        "assumed residual should surface the positivity assumption: {residual_assumptions:?}"
    );
}

#[test]
fn test_eval_integral_independent_radical_constant_verifies_per_domain_mode() {
    let generic = eval_json_with_args("integrate(sqrt(a^2), x)", &["--domain", "generic"]);

    assert_eq!(generic["ok"], true);
    assert_eq!(generic["result"], "x·|a|");
    assert_eq!(generic["domain"]["mode"], "generic");
    assert!(generic["assumptions_used"].is_null());
    assert!(
        generic["required_conditions"]
            .as_array()
            .expect("generic required_conditions should be an array")
            .is_empty(),
        "generic independent radical constant should not invent domain requirements"
    );

    let generic_displayed_residual =
        eval_json_with_args("diff(x*abs(a), x) - sqrt(a^2)", &["--domain", "generic"]);
    assert_eq!(generic_displayed_residual["ok"], true);
    assert_eq!(
        generic_displayed_residual["result"], "0",
        "generic displayed antiderivative x*|a| should verify without assumptions"
    );

    let generic_assumed_form_residual =
        eval_json_with_args("diff(a*x, x) - sqrt(a^2)", &["--domain", "generic"]);
    assert_eq!(generic_assumed_form_residual["ok"], true);
    assert_eq!(
        generic_assumed_form_residual["result"], "a - |a|",
        "generic mode should not verify a*x without the positivity assumption"
    );

    let assume = eval_json_with_args("integrate(sqrt(a^2), x)", &["--domain", "assume"]);

    assert_eq!(assume["ok"], true);
    assert_eq!(assume["result"], "a·x");
    assert_eq!(assume["domain"]["mode"], "assume");
    assert!(
        assume["required_conditions"]
            .as_array()
            .expect("assume required_conditions should be an array")
            .is_empty(),
        "assume output should not duplicate a > 0 as a separate required condition"
    );

    let assumptions = assume["assumptions_used"]
        .as_array()
        .expect("assumptions_used should be an array in assume mode");
    assert_eq!(assumptions.len(), 1);
    assert_eq!(assumptions[0]["display"], "a > 0");
    assert_eq!(assumptions[0]["expr_canonical"], "a");
    assert_eq!(assumptions[0]["rule"], "Abs Under Positivity");

    let assume_displayed_residual =
        eval_json_with_args("diff(a*x, x) - sqrt(a^2)", &["--domain", "assume"]);
    assert_eq!(assume_displayed_residual["ok"], true);
    assert_eq!(
        assume_displayed_residual["result"], "0",
        "assume displayed antiderivative a*x should verify by differentiation"
    );
    let residual_assumptions = assume_displayed_residual["assumptions_used"]
        .as_array()
        .expect("assumptions_used should be an array for assumed displayed residual");
    assert!(
        residual_assumptions.iter().any(|assumption| {
            assumption["display"] == "a > 0" && assumption["rule"] == "Abs Under Positivity"
        }),
        "assumed displayed residual should surface the positivity assumption: {residual_assumptions:?}"
    );
}

#[test]
fn test_eval_integral_compound_independent_radical_constant_verifies_per_domain_mode() {
    let generic = eval_json_with_args("integrate(sqrt((a+1)^2), x)", &["--domain", "generic"]);

    assert_eq!(generic["ok"], true);
    assert_eq!(generic["result"], "x·|a + 1|");
    assert_eq!(generic["domain"]["mode"], "generic");
    assert!(generic["assumptions_used"].is_null());
    assert!(
        generic["required_conditions"]
            .as_array()
            .expect("generic required_conditions should be an array")
            .is_empty(),
        "generic compound radical constant should not invent domain requirements"
    );

    let generic_displayed_residual = eval_json_with_args(
        "diff(x*abs(a+1), x) - sqrt((a+1)^2)",
        &["--domain", "generic"],
    );
    assert_eq!(generic_displayed_residual["ok"], true);
    assert_eq!(
        generic_displayed_residual["result"], "0",
        "generic displayed antiderivative x*|a + 1| should verify without assumptions"
    );

    let generic_assumed_form_residual =
        eval_json_with_args("diff(x*(a+1), x) - sqrt((a+1)^2)", &["--domain", "generic"]);
    assert_eq!(generic_assumed_form_residual["ok"], true);
    assert_eq!(
        generic_assumed_form_residual["result"], "a + 1 - |a + 1|",
        "generic mode should not verify x*(a+1) without the compound positivity assumption"
    );

    let assume = eval_json_with_args("integrate(sqrt((a+1)^2), x)", &["--domain", "assume"]);

    assert_eq!(assume["ok"], true);
    assert_eq!(assume["result"], "x·(a + 1)");
    assert_eq!(assume["domain"]["mode"], "assume");
    assert!(
        assume["required_conditions"]
            .as_array()
            .expect("assume required_conditions should be an array")
            .is_empty(),
        "assume output should not duplicate a + 1 > 0 as a separate required condition"
    );

    let assumptions = assume["assumptions_used"]
        .as_array()
        .expect("assumptions_used should be an array in assume mode");
    assert_eq!(assumptions.len(), 1);
    assert_eq!(assumptions[0]["display"], "a + 1 > 0");
    assert_eq!(assumptions[0]["expr_canonical"], "a + 1");
    assert_eq!(assumptions[0]["rule"], "Abs Under Positivity");

    let assume_messages = assume["wire"]["messages"]
        .as_array()
        .expect("assume wire messages should be an array");
    assert!(
        assume_messages.iter().any(|message| message["text"]
            .as_str()
            .is_some_and(|text| text.contains("Assume:"))),
        "assume mode should render the compound positivity condition as Assume: {assume_messages:?}"
    );
    assert!(
        !assume_messages.iter().any(|message| message["text"]
            .as_str()
            .is_some_and(|text| text.contains("Requires:"))),
        "assume compound radical integral output should not also render Requires: {assume_messages:?}"
    );

    let assume_displayed_residual =
        eval_json_with_args("diff(x*(a+1), x) - sqrt((a+1)^2)", &["--domain", "assume"]);
    assert_eq!(assume_displayed_residual["ok"], true);
    assert_eq!(
        assume_displayed_residual["result"], "0",
        "assume displayed antiderivative x*(a+1) should verify by differentiation"
    );
    let residual_assumptions = assume_displayed_residual["assumptions_used"]
        .as_array()
        .expect("assumptions_used should be an array for assumed displayed residual");
    assert!(
        residual_assumptions.iter().any(|assumption| {
            assumption["display"] == "a + 1 > 0"
                && assumption["rule"] == "Abs Under Positivity"
        }),
        "assumed displayed residual should surface the compound positivity assumption: {residual_assumptions:?}"
    );
}

#[test]
fn test_eval_integral_reciprocal_compound_radical_constant_requires_abs_until_assumed() {
    let generic = eval_json_with_args("integrate(1/sqrt((a+1)^2), x)", &["--domain", "generic"]);

    assert_eq!(generic["ok"], true);
    assert_eq!(generic["result"], "x / |a + 1|");
    assert_eq!(generic["domain"]["mode"], "generic");
    assert!(generic["assumptions_used"].is_null());

    let generic_required = generic["required_conditions"]
        .as_array()
        .expect("generic required_conditions should be an array");
    assert_eq!(
        generic_required.len(),
        1,
        "expected denominator definability guard: {generic_required:?}"
    );
    assert_eq!(generic_required[0]["kind"], "NonZero");
    assert_eq!(generic_required[0]["expr_canonical"], "a + 1");

    let generic_displayed_residual = eval_json_with_args(
        "diff(x/abs(a+1), x) - 1/sqrt((a+1)^2)",
        &["--domain", "generic"],
    );
    assert_eq!(generic_displayed_residual["ok"], true);
    assert_eq!(
        generic_displayed_residual["result"], "0",
        "generic displayed antiderivative x/|a+1| should verify with nonzero guard"
    );

    let generic_assumed_form_residual = eval_json_with_args(
        "diff(x/(a+1), x) - 1/sqrt((a+1)^2)",
        &["--domain", "generic"],
    );
    assert_eq!(generic_assumed_form_residual["ok"], true);
    assert_eq!(
        generic_assumed_form_residual["result"], "(|a + 1| - a - 1) / (|a + 1|·(a + 1))",
        "generic mode should not verify x/(a+1) without the compound positivity assumption"
    );

    let assume = eval_json_with_args("integrate(1/sqrt((a+1)^2), x)", &["--domain", "assume"]);

    assert_eq!(assume["ok"], true);
    assert_eq!(assume["result"], "x / (a + 1)");
    assert_eq!(assume["domain"]["mode"], "assume");

    let assumptions = assume["assumptions_used"]
        .as_array()
        .expect("assumptions_used should be an array in assume mode");
    assert_eq!(assumptions.len(), 1);
    assert_eq!(assumptions[0]["display"], "a + 1 > 0");
    assert_eq!(assumptions[0]["expr_canonical"], "a + 1");
    assert_eq!(assumptions[0]["rule"], "Abs Under Positivity");
    assert!(
        assume["warnings"]
            .as_array()
            .expect("assume warnings should be an array")
            .is_empty(),
        "accepted assume-mode condition should not also be emitted as a warning"
    );
    assert_eq!(
        assume["required_conditions"]
            .as_array()
            .expect("assume required_conditions should be an array")
            .len(),
        0,
        "positive assumption should cover the redundant nonzero requirement"
    );
    assert_eq!(
        assume["required_display"]
            .as_array()
            .expect("assume required_display should be an array")
            .len(),
        0,
        "assume output should not render a covered Requires block"
    );
    let assume_messages = assume["wire"]["messages"]
        .as_array()
        .expect("assume wire messages should be an array");
    assert!(
        !assume_messages
            .iter()
            .any(|message| message["kind"] == "warn"),
        "accepted assume-mode condition should render only in Assume: {assume_messages:?}"
    );

    let assume_displayed_residual = eval_json_with_args(
        "diff(x/(a+1), x) - 1/sqrt((a+1)^2)",
        &["--domain", "assume"],
    );
    assert_eq!(assume_displayed_residual["ok"], true);
    assert_eq!(
        assume_displayed_residual["result"], "0",
        "assume displayed antiderivative x/(a+1) should verify by differentiation"
    );
    let residual_assumptions = assume_displayed_residual["assumptions_used"]
        .as_array()
        .expect("assumptions_used should be an array for assumed displayed residual");
    assert!(
        residual_assumptions.iter().any(|assumption| {
            assumption["display"] == "a + 1 > 0"
                && assumption["rule"] == "Abs Under Positivity"
        }),
        "assumed displayed residual should surface the compound positivity assumption: {residual_assumptions:?}"
    );
}

#[test]
fn test_eval_assume_mode_dedupes_shared_positive_assumption_across_rules() {
    let generic = eval_json_with_args("sqrt(abs(x))^2 + ln(x^2)", &["--domain", "generic"]);

    assert_eq!(generic["ok"], true);
    assert_eq!(generic["result"], "|x| + 2·ln(|x|)");
    let generic_required = generic["required_conditions"]
        .as_array()
        .expect("generic required_conditions should be an array");
    assert_eq!(
        generic_required.len(),
        1,
        "generic mode should preserve the log-domain nonzero guard: {generic_required:?}"
    );
    assert_eq!(generic_required[0]["kind"], "NonZero");
    assert_eq!(generic_required[0]["expr_canonical"], "x");

    let assume = eval_json_with_args("sqrt(abs(x))^2 + ln(x^2)", &["--domain", "assume"]);

    assert_eq!(assume["ok"], true);
    assert_eq!(assume["result"], "2·ln(x) + x");
    let assumptions = assume["assumptions_used"]
        .as_array()
        .expect("assumptions_used should be an array in assume mode");
    assert_eq!(
        assumptions.len(),
        1,
        "shared positive assumption should render once: {assumptions:?}"
    );
    assert_eq!(assumptions[0]["kind"], "positive");
    assert_eq!(assumptions[0]["display"], "x > 0");
    assert_eq!(assumptions[0]["expr_canonical"], "x");
    assert!(
        assume["required_conditions"]
            .as_array()
            .expect("assume required_conditions should be an array")
            .is_empty(),
        "positive assumption should cover the generic nonzero requirement"
    );

    let assume_messages = assume["wire"]["messages"]
        .as_array()
        .expect("assume wire messages should be an array");
    let assume_bullets = assume_messages
        .iter()
        .filter(|message| message["text"].as_str() == Some("  • x > 0"))
        .count();
    assert_eq!(
        assume_bullets, 1,
        "wire Assume block should not duplicate the same condition: {assume_messages:?}"
    );
}

#[test]
fn test_log_even_power_uses_inherited_negative_domain_for_abs_cleanup() {
    let generic = eval_json("ln(x^2) - 2*ln(-x)");

    assert_eq!(generic["ok"], true);
    assert_eq!(generic["result"], "0");
    let generic_required = generic["required_conditions"]
        .as_array()
        .expect("generic required_conditions should be an array");
    assert!(
        generic_required.iter().any(|condition| {
            condition["kind"] == "Positive" && condition["expr_display"] == "-x"
        }),
        "generic mode should preserve the inherited log-domain condition: {generic_required:?}"
    );
    assert!(
        generic
            .get("assumptions_used")
            .and_then(|value| value.as_array())
            .is_none_or(|assumptions| assumptions.is_empty()),
        "generic mode must not introduce assumptions"
    );

    let assume = eval_json_with_args("ln(x^2) - 2*ln(-x)", &["--domain", "assume"]);

    assert_eq!(assume["ok"], true);
    assert_eq!(assume["result"], "0");
    let empty_assume_assumptions = Vec::new();
    let assume_assumptions = assume
        .get("assumptions_used")
        .and_then(|value| value.as_array())
        .unwrap_or(&empty_assume_assumptions);
    assert!(
        !assume_assumptions
            .iter()
            .any(|assumption| assumption["display"] == "x > 0"),
        "assume mode should consume inherited -x > 0, not assume x > 0: {assume_assumptions:?}"
    );
    let assume_required = assume["required_conditions"]
        .as_array()
        .expect("assume required_conditions should be an array");
    assert!(
        assume_required.iter().any(|condition| {
            condition["kind"] == "Positive" && condition["expr_display"] == "-x"
        }),
        "assume mode should preserve the inherited log-domain condition: {assume_required:?}"
    );

    let log_abs_residual = eval_json("ln(abs(x)) - ln(-x)");
    assert_eq!(log_abs_residual["ok"], true);
    assert_eq!(
        log_abs_residual["result"], "0",
        "ln(|x|) should use inherited -x > 0 instead of staying one step short"
    );

    let scaled_negative = eval_json("ln(abs(x)) - ln(-x/2) - ln(2)");
    assert_eq!(scaled_negative["ok"], true);
    assert_eq!(
        scaled_negative["result"], "0",
        "scaled negative log domain should imply x < 0 for abs cleanup"
    );
    let scaled_required = scaled_negative["required_conditions"]
        .as_array()
        .expect("scaled required_conditions should be an array");
    assert!(
        scaled_required.iter().any(|condition| {
            condition["kind"] == "Positive" && condition["expr_display"] == "-x"
        }),
        "scaled negative domain should normalize to -x > 0: {scaled_required:?}"
    );

    let affine_negative = eval_json("ln(abs(2*x+1)) - ln(-(2*x+1))");
    assert_eq!(affine_negative["ok"], true);
    assert_eq!(
        affine_negative["result"], "0",
        "affine negative log domain should imply the affine subject is negative"
    );
    let affine_required = affine_negative["required_conditions"]
        .as_array()
        .expect("affine required_conditions should be an array");
    assert!(
        affine_required.iter().any(|condition| {
            condition["kind"] == "Positive" && condition["expr_display"] == "-2·x - 1"
        }),
        "affine negative domain should preserve the affine required condition: {affine_required:?}"
    );
}

#[test]
fn test_cancelled_log_domain_feeds_sqrt_square_abs_cleanup() {
    let positive = eval_json("sqrt(x^2)-x+ln(x)-ln(x)");

    assert_eq!(positive["ok"], true);
    assert_eq!(
        positive["result"], "0",
        "cancelled ln(x) domain should let sqrt(x^2) collapse through |x| -> x"
    );
    let positive_required = positive["required_conditions"]
        .as_array()
        .expect("positive required_conditions should be an array");
    assert!(
        positive_required.iter().any(|condition| {
            condition["kind"] == "Positive" && condition["expr_display"] == "x"
        }),
        "positive log domain should be retained as x > 0: {positive_required:?}"
    );

    let negative = eval_json("sqrt(x^2)+x+ln(-x)-ln(-x)");

    assert_eq!(negative["ok"], true);
    assert_eq!(
        negative["result"], "0",
        "cancelled ln(-x) domain should let sqrt(x^2) collapse through |x| -> -x"
    );
    let negative_required = negative["required_conditions"]
        .as_array()
        .expect("negative required_conditions should be an array");
    assert!(
        negative_required.iter().any(|condition| {
            condition["kind"] == "Positive" && condition["expr_display"] == "-x"
        }),
        "negative log domain should be retained as -x > 0: {negative_required:?}"
    );
}

#[test]
fn test_cancelled_sqrt_domain_feeds_abs_cleanup_but_nonzero_does_not() {
    let abs_cleanup = eval_json("abs(x)-x+sqrt(x)-sqrt(x)");

    assert_eq!(abs_cleanup["ok"], true);
    assert_eq!(
        abs_cleanup["result"], "0",
        "cancelled sqrt(x) domain should let |x| collapse to x"
    );
    let abs_required = abs_cleanup["required_conditions"]
        .as_array()
        .expect("abs cleanup required_conditions should be an array");
    assert!(
        abs_required.iter().any(|condition| {
            condition["kind"] == "NonNegative" && condition["expr_display"] == "x"
        }),
        "cancelled sqrt(x) should retain x >= 0: {abs_required:?}"
    );

    let sqrt_square_cleanup = eval_json("sqrt(x^2)-x+sqrt(x)-sqrt(x)");

    assert_eq!(sqrt_square_cleanup["ok"], true);
    assert_eq!(
        sqrt_square_cleanup["result"], "0",
        "cancelled sqrt(x) domain should let sqrt(x^2) collapse through |x| -> x"
    );
    let sqrt_square_required = sqrt_square_cleanup["required_conditions"]
        .as_array()
        .expect("sqrt square required_conditions should be an array");
    assert!(
        sqrt_square_required.iter().any(|condition| {
            condition["kind"] == "NonNegative" && condition["expr_display"] == "x"
        }),
        "cancelled sqrt(x) should retain x >= 0: {sqrt_square_required:?}"
    );

    let nonzero_control = eval_json("abs(x)-x+1/x-1/x");

    assert_eq!(nonzero_control["ok"], true);
    assert_eq!(
        nonzero_control["result"], "|x| - x",
        "cancelled reciprocal domain must not treat x != 0 as a sign condition"
    );
    let nonzero_required = nonzero_control["required_conditions"]
        .as_array()
        .expect("nonzero control required_conditions should be an array");
    assert!(
        nonzero_required.iter().any(|condition| {
            condition["kind"] == "NonZero" && condition["expr_display"] == "x"
        }),
        "cancelled reciprocal should retain only x != 0: {nonzero_required:?}"
    );
    assert!(
        !nonzero_required.iter().any(|condition| {
            condition["kind"] == "Positive" || condition["kind"] == "NonNegative"
        }),
        "nonzero control must not synthesize sign conditions: {nonzero_required:?}"
    );
}

#[test]
fn test_cancelled_non_ln_log_domains_feed_abs_cleanup() {
    for expr in [
        "abs(x)-x+log2(x)-log2(x)",
        "abs(x)-x+log10(x)-log10(x)",
        "abs(x)-x+log(2,x)-log(2,x)",
    ] {
        let json = eval_json(expr);

        assert_eq!(json["ok"], true, "expr: {expr}");
        assert_eq!(
            json["result"], "0",
            "cancelled non-ln log domain should let |x| collapse to x for {expr}"
        );

        let required = json["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array");
        assert!(
            required.iter().any(|condition| {
                condition["kind"] == "Positive" && condition["expr_display"] == "x"
            }),
            "cancelled non-ln log should retain x > 0 for {expr}: {required:?}"
        );
        assert_eq!(
            required.len(),
            1,
            "cancelled non-ln log should not add redundant domain guards for {expr}: {required:?}"
        );
    }
}

#[test]
fn test_cancelled_acosh_lower_bound_feeds_abs_cleanup_conservatively() {
    let direct = eval_json("abs(x)-x+acosh(x)-acosh(x)");

    assert_eq!(direct["ok"], true);
    assert_eq!(
        direct["result"], "0",
        "cancelled acosh(x) domain x >= 1 should let |x| collapse to x"
    );
    let direct_required = direct["required_conditions"]
        .as_array()
        .expect("direct required_conditions should be an array");
    assert!(
        direct_required.iter().any(|condition| {
            condition["kind"] == "LowerBound" && condition["expr_display"] == "x"
        }),
        "cancelled acosh(x) should retain the lower-bound domain: {direct_required:?}"
    );
    let direct_display = direct["required_display"]
        .as_array()
        .expect("direct required_display should be an array");
    assert!(
        direct_display
            .iter()
            .any(|condition| condition.as_str() == Some("x ≥ 1")),
        "cancelled acosh(x) should render x >= 1: {direct_display:?}"
    );

    let affine = eval_json("abs(x)-x+acosh(2*x+1)-acosh(2*x+1)");

    assert_eq!(affine["ok"], true);
    assert_eq!(
        affine["result"], "0",
        "cancelled acosh(2*x+1) domain should imply x >= 0 for abs cleanup"
    );
    let affine_required = affine["required_conditions"]
        .as_array()
        .expect("affine required_conditions should be an array");
    assert!(
        affine_required.iter().any(|condition| {
            condition["kind"] == "LowerBound" && condition["expr_display"] == "2·x + 1"
        }),
        "affine acosh should retain the original lower-bound domain: {affine_required:?}"
    );

    let negative_orientation = eval_json("abs(x)-x+acosh(1-x)-acosh(1-x)");

    assert_eq!(negative_orientation["ok"], true);
    assert_eq!(
        negative_orientation["result"], "-2·x",
        "acosh(1-x) domain implies x <= 0, so |x| should rewrite to -x, not x"
    );
    let negative_required = negative_orientation["required_conditions"]
        .as_array()
        .expect("negative orientation required_conditions should be an array");
    assert!(
        negative_required.iter().any(|condition| {
            condition["kind"] == "LowerBound" && condition["expr_display"] == "1 - x"
        }),
        "negative orientation acosh should retain its lower-bound domain: {negative_required:?}"
    );

    let negative_cancellation = eval_json("abs(x)+x+acosh(1-2*x)-acosh(1-2*x)");

    assert_eq!(negative_cancellation["ok"], true);
    assert_eq!(
        negative_cancellation["result"], "0",
        "acosh(1-2*x) domain implies x <= 0, so |x| + x should collapse"
    );
    let negative_cancellation_required = negative_cancellation["required_conditions"]
        .as_array()
        .expect("negative cancellation required_conditions should be an array");
    assert!(
        negative_cancellation_required.iter().any(|condition| {
            condition["kind"] == "LowerBound" && condition["expr_display"] == "1 - 2·x"
        }),
        "negative cancellation should retain its lower-bound domain: {negative_cancellation_required:?}"
    );

    let shifted_negative = eval_json("abs(x-1)+x-1+acosh(3-2*x)-acosh(3-2*x)");

    assert_eq!(shifted_negative["ok"], true);
    assert_eq!(
        shifted_negative["result"], "0",
        "acosh(3-2*x) domain implies x - 1 <= 0, so |x-1| + x - 1 should collapse"
    );
    let shifted_negative_required = shifted_negative["required_conditions"]
        .as_array()
        .expect("shifted negative required_conditions should be an array");
    assert!(
        shifted_negative_required.iter().any(|condition| {
            condition["kind"] == "LowerBound" && condition["expr_display"] == "3 - 2·x"
        }),
        "shifted negative should retain its lower-bound domain: {shifted_negative_required:?}"
    );

    let affine_negative_target = eval_json("abs(2*x+1)+2*x+1+acosh(-4*x-1)-acosh(-4*x-1)");

    assert_eq!(affine_negative_target["ok"], true);
    assert_eq!(
        affine_negative_target["result"], "0",
        "acosh(-4*x-1) domain implies 2*x + 1 <= 0, so |2*x+1| + 2*x + 1 should collapse"
    );
    let affine_negative_target_required = affine_negative_target["required_conditions"]
        .as_array()
        .expect("affine negative target required_conditions should be an array");
    assert!(
        affine_negative_target_required.iter().any(|condition| {
            condition["kind"] == "LowerBound" && condition["expr_display"] == "-4·x - 1"
        }),
        "affine negative target should retain its lower-bound domain: {affine_negative_target_required:?}"
    );

    let reciprocal_negative = eval_json("abs(1/(x-1))+1/(x-1)+acosh(3-2*x)-acosh(3-2*x)");

    assert_eq!(reciprocal_negative["ok"], true);
    assert_eq!(
        reciprocal_negative["result"], "0",
        "acosh(3-2*x) plus x-1 != 0 should imply 1/(x-1) < 0, so |1/(x-1)| + 1/(x-1) collapses"
    );
    let reciprocal_negative_required = reciprocal_negative["required_conditions"]
        .as_array()
        .expect("reciprocal negative required_conditions should be an array");
    assert!(
        reciprocal_negative_required.iter().any(|condition| {
            condition["kind"] == "LowerBound" && condition["expr_display"] == "3 - 2·x"
        }),
        "reciprocal negative should retain its lower-bound domain: {reciprocal_negative_required:?}"
    );
    assert!(
        reciprocal_negative_required
            .iter()
            .any(|condition| condition["kind"] == "NonZero" && condition["expr_display"] == "x - 1"),
        "reciprocal negative should retain its nonzero denominator guard: {reciprocal_negative_required:?}"
    );

    let reciprocal_even_power = eval_json("abs(1/(x-1)^2)-1/(x-1)^2+acosh(3-2*x)-acosh(3-2*x)");

    assert_eq!(reciprocal_even_power["ok"], true);
    assert_eq!(
        reciprocal_even_power["result"], "0",
        "x-1 != 0 should imply 1/(x-1)^2 > 0, so |1/(x-1)^2| collapses"
    );
    let reciprocal_even_power_required = reciprocal_even_power["required_conditions"]
        .as_array()
        .expect("reciprocal even power required_conditions should be an array");
    assert!(
        reciprocal_even_power_required
            .iter()
            .any(|condition| condition["kind"] == "NonZero" && condition["expr_display"] == "x - 1"),
        "reciprocal even power should retain its nonzero denominator guard: {reciprocal_even_power_required:?}"
    );

    let reciprocal_odd_power = eval_json("abs(1/(x-1)^3)+1/(x-1)^3+acosh(3-2*x)-acosh(3-2*x)");

    assert_eq!(reciprocal_odd_power["ok"], true);
    assert_eq!(
        reciprocal_odd_power["result"], "0",
        "acosh(3-2*x) plus x-1 != 0 should imply 1/(x-1)^3 < 0"
    );
    let reciprocal_odd_power_required = reciprocal_odd_power["required_conditions"]
        .as_array()
        .expect("reciprocal odd power required_conditions should be an array");
    assert!(
        reciprocal_odd_power_required.iter().any(|condition| {
            condition["kind"] == "LowerBound" && condition["expr_display"] == "3 - 2·x"
        }),
        "reciprocal odd power should retain its lower-bound domain: {reciprocal_odd_power_required:?}"
    );
    assert!(
        reciprocal_odd_power_required
            .iter()
            .any(|condition| condition["kind"] == "NonZero" && condition["expr_display"] == "x - 1"),
        "reciprocal odd power should retain its nonzero denominator guard: {reciprocal_odd_power_required:?}"
    );

    let product_positive =
        eval_json("abs(1/((x-1)*(x-2)))-1/((x-1)*(x-2))+acosh(3-2*x)-acosh(3-2*x)");

    assert_eq!(product_positive["ok"], true);
    assert_eq!(
        product_positive["result"], "0",
        "under x <= 1, both x-1 and x-2 are negative and nonzero, so their reciprocal product is positive"
    );
    let product_positive_required = product_positive["required_conditions"]
        .as_array()
        .expect("product positive required_conditions should be an array");
    assert!(
        product_positive_required.iter().any(|condition| {
            condition["kind"] == "LowerBound" && condition["expr_display"] == "3 - 2·x"
        }),
        "product positive should retain its lower-bound domain: {product_positive_required:?}"
    );
    assert!(
        product_positive_required.iter().any(|condition| {
            condition["kind"] == "NonZero" && condition["expr_display"] == "x - 1"
        }) && product_positive_required.iter().any(|condition| {
            condition["kind"] == "NonZero" && condition["expr_display"] == "x - 2"
        }),
        "product positive should retain both nonzero factor guards: {product_positive_required:?}"
    );

    let product_negative =
        eval_json("abs(1/((x-1)*(2-x)))+1/((x-1)*(2-x))+acosh(3-2*x)-acosh(3-2*x)");

    assert_eq!(product_negative["ok"], true);
    assert_eq!(
        product_negative["result"], "0",
        "under x <= 1, x-1 is negative and 2-x is positive, so the reciprocal product is negative"
    );
    let product_negative_required = product_negative["required_conditions"]
        .as_array()
        .expect("product negative required_conditions should be an array");
    assert!(
        product_negative_required.iter().any(|condition| {
            condition["kind"] == "LowerBound" && condition["expr_display"] == "3 - 2·x"
        }),
        "product negative should retain its lower-bound domain: {product_negative_required:?}"
    );
    assert!(
        product_negative_required.iter().any(|condition| {
            condition["kind"] == "NonZero" && condition["expr_display"] == "x - 1"
        }) && product_negative_required.iter().any(|condition| {
            condition["kind"] == "NonZero" && condition["expr_display"] == "x - 2"
        }),
        "product negative should retain both nonzero factor guards: {product_negative_required:?}"
    );

    let mixed_power_product_negative =
        eval_json("abs(1/((x-1)^2*(x-2)))+1/((x-1)^2*(x-2))+acosh(3-2*x)-acosh(3-2*x)");

    assert_eq!(mixed_power_product_negative["ok"], true);
    assert_eq!(
        mixed_power_product_negative["result"], "0",
        "under x <= 1, (x-1)^2 is positive and x-2 is negative, so the reciprocal product is negative"
    );
    let mixed_power_product_negative_required = mixed_power_product_negative["required_conditions"]
        .as_array()
        .expect("mixed power product negative required_conditions should be an array");
    assert!(
        mixed_power_product_negative_required.iter().any(|condition| {
            condition["kind"] == "LowerBound" && condition["expr_display"] == "3 - 2·x"
        }),
        "mixed power product negative should retain its lower-bound domain: {mixed_power_product_negative_required:?}"
    );
    assert!(
        mixed_power_product_negative_required.iter().any(|condition| {
            condition["kind"] == "NonZero" && condition["expr_display"] == "x - 1"
        }) && mixed_power_product_negative_required.iter().any(|condition| {
            condition["kind"] == "NonZero" && condition["expr_display"] == "x - 2"
        }),
        "mixed power product negative should retain both nonzero factor guards: {mixed_power_product_negative_required:?}"
    );

    let mixed_power_product_positive =
        eval_json("abs(1/((x-1)^2*(2-x)))-1/((x-1)^2*(2-x))+acosh(3-2*x)-acosh(3-2*x)");

    assert_eq!(mixed_power_product_positive["ok"], true);
    assert_eq!(
        mixed_power_product_positive["result"], "0",
        "under x <= 1, (x-1)^2 and 2-x are positive, so the reciprocal product is positive"
    );
    let mixed_power_product_positive_required = mixed_power_product_positive["required_conditions"]
        .as_array()
        .expect("mixed power product positive required_conditions should be an array");
    assert!(
        mixed_power_product_positive_required.iter().any(|condition| {
            condition["kind"] == "LowerBound" && condition["expr_display"] == "3 - 2·x"
        }),
        "mixed power product positive should retain its lower-bound domain: {mixed_power_product_positive_required:?}"
    );
    assert!(
        mixed_power_product_positive_required.iter().any(|condition| {
            condition["kind"] == "NonZero" && condition["expr_display"] == "x - 1"
        }) && mixed_power_product_positive_required.iter().any(|condition| {
            condition["kind"] == "NonZero" && condition["expr_display"] == "x - 2"
        }),
        "mixed power product positive should retain both nonzero factor guards: {mixed_power_product_positive_required:?}"
    );
}

#[test]
fn test_wire_output_message_present() {
    let wire = eval_wire("3*4");

    let messages = wire.get("messages").expect("messages should exist");
    let msgs = messages.as_array().unwrap();

    // Should have at least one output message
    let output_msgs: Vec<_> = msgs
        .iter()
        .filter(|m| m.get("kind") == Some(&Value::String("output".into())))
        .collect();

    assert!(
        !output_msgs.is_empty(),
        "should have at least one output message"
    );

    // Output message should contain result
    let output_text = output_msgs[0].get("text").unwrap().as_str().unwrap();
    assert!(
        output_text.contains("Result"),
        "output should contain 'Result'"
    );
    assert!(
        output_text.contains("12"),
        "output should contain the result value"
    );
}

#[test]
fn test_wire_steps_summary_when_enabled() {
    let output = Command::new(cargo::cargo_bin!("cas_cli"))
        .arg("eval")
        .arg("x^2 + 2*x + 1")
        .arg("--steps")
        .arg("on")
        .arg("--format")
        .arg("json")
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: Value = serde_json::from_str(&stdout).expect("Failed to parse JSON");
    let wire = json.get("wire").expect("wire should exist");

    let messages = wire.get("messages").expect("messages should exist");
    let msgs = messages.as_array().unwrap();

    // If there are steps, should have a steps summary message
    let steps_count = json
        .get("steps_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    if steps_count > 0 {
        let step_msgs: Vec<_> = msgs
            .iter()
            .filter(|m| m.get("kind") == Some(&Value::String("steps".into())))
            .collect();

        assert!(
            !step_msgs.is_empty(),
            "should have steps message when steps_count > 0"
        );
    }
}

#[test]
fn test_wire_message_order() {
    // Test that messages appear in expected order: warn, info, output, steps
    let wire = eval_wire("1/x"); // This might produce requires

    let messages = wire.get("messages").expect("messages should exist");
    let msgs = messages.as_array().unwrap();

    // Find position of output message
    let output_pos = msgs
        .iter()
        .position(|m| m.get("kind") == Some(&Value::String("output".into())));

    assert!(output_pos.is_some(), "should have output message");

    // Warn and info should come before output (if present)
    for (i, msg) in msgs.iter().enumerate() {
        let kind = msg.get("kind").and_then(|v| v.as_str()).unwrap_or("");
        if kind == "warn" || kind == "info" {
            assert!(
                i < output_pos.unwrap(),
                "warn/info messages should come before output"
            );
        }
    }
}

#[test]
fn test_wire_schema_version_stable() {
    // Verify schema_version is exactly 1 (not changing unexpectedly)
    for expr in &["1+1", "x^2", "sin(pi/2)"] {
        let wire = eval_wire(expr);
        assert_eq!(
            wire.get("schema_version"),
            Some(&Value::Number(1.into())),
            "schema_version should always be 1 for expr: {}",
            expr
        );
    }
}
