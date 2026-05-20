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

fn eval_json_with_stderr(expr: &str) -> (Value, String) {
    eval_json_with_args_and_stderr(expr, &[])
}

fn eval_json_with_args_and_stderr(expr: &str, args: &[&str]) -> (Value, String) {
    let mut command = Command::new(cargo::cargo_bin!("cas_cli"));
    command.arg("eval").arg(expr).arg("--format").arg("json");
    command.args(args);

    let output = command.output().expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    (
        serde_json::from_str(&stdout).expect("Failed to parse JSON"),
        stderr,
    )
}

#[test]
fn test_eval_json_sqrt_shifted_ln_diff_step_uses_compact_reciprocal_root_presentation() {
    let json = eval_json_with_args("diff(sqrt(ln(x)+1), x)", &["--steps", "on"]);

    assert_eq!(json["ok"], true);
    assert_eq!(json["result"], "1 / (2·x·sqrt(ln(x) + 1))");

    let steps = json["steps"].as_array().expect("steps should be an array");
    let diff_step = steps
        .iter()
        .find(|step| step["rule"] == "Calcular la derivada")
        .expect("expected symbolic differentiation step");
    let after = diff_step["after"].as_str().expect("after string");
    assert!(
        after.contains("1/(2 · x · sqrt(ln(x) + 1))") && !after.contains("1/2 - 1"),
        "diff step should render the post-calculus reciprocal-root form, got: {after}"
    );
    let after_latex = diff_step["after_latex"]
        .as_str()
        .expect("after_latex string");
    assert!(
        after_latex.contains("\\frac{1}{2\\cdot x\\cdot \\sqrt")
            && !after_latex.contains("\\frac{1}{2} - 1"),
        "diff step LaTeX should render the compact reciprocal-root form, got: {after_latex}"
    );
}

#[test]
fn test_eval_json_scaled_ln_diff_step_cancels_fraction_product_presentation_noise() {
    let cases = [
        (
            "diff(sqrt(2*ln(x)+3), x)",
            "1 / (x·sqrt(2·ln(x) + 3))",
            ["2/2", "1/2 - 1"],
            ["1/(x · sqrt(2 · ln(x) + 3))", "sqrt(2 · ln(x) + 3)"],
        ),
        (
            "diff(sqrt(ln(2*x+1)+1), x)",
            "1 / ((2·x + 1)·sqrt(ln(2·x + 1) + 1))",
            [" · 2/", "1/2 - 1"],
            ["2 · x + 1", "sqrt(ln(2 · x + 1) + 1)"],
        ),
    ];

    for (expr, expected_result, rejected_after_needles, expected_after_needles) in cases {
        let json = eval_json_with_args(expr, &["--steps", "on"]);

        assert_eq!(json["ok"], true, "expr: {expr}");
        assert_eq!(json["result"], expected_result, "expr: {expr}");

        let steps = json["steps"].as_array().expect("steps should be an array");
        let diff_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la derivada")
            .expect("expected symbolic differentiation step");
        let after = diff_step["after"].as_str().expect("after string");
        assert!(
            expected_after_needles
                .iter()
                .all(|needle| after.contains(needle)),
            "diff step should render the compact fraction/root form for {expr}, got: {after}"
        );
        assert!(
            rejected_after_needles
                .iter()
                .all(|needle| !after.contains(needle)),
            "diff step should not leak fraction-product noise for {expr}, got: {after}"
        );
    }
}

#[test]
fn test_eval_json_tan_exp_sqrt_diff_preserves_compact_presentation_without_timeout() {
    let json = eval_json_with_args("diff(sqrt(tan(x)+exp(x)+x), x)", &["--steps", "on"]);

    assert_eq!(json["ok"], true);
    assert_eq!(
        json["result"],
        "(cos(x)^2 + e^x·cos(x)^2 + 1) / (2·cos(x)^2·sqrt(tan(x) + e^x + x))"
    );

    let steps = json["steps"].as_array().expect("steps should be an array");
    assert_eq!(steps.len(), 1, "expected direct derivative step: {steps:?}");
    let after = steps[0]["after"].as_str().expect("after string");
    assert!(
        after.contains("sqrt(tan(x) + e^x + x)")
            && after.contains("cos(x)^2")
            && !after.contains("sin(x) / cos(x)"),
        "expected compact tan/exp sqrt derivative presentation, got: {after}"
    );

    let required = json["required_conditions"]
        .as_array()
        .expect("required_conditions should be an array");
    assert!(
        required.iter().any(|condition| {
            condition["kind"] == "NonZero" && condition["expr_canonical"] == "cos(x)"
        }) && required.iter().any(|condition| {
            condition["kind"] == "Positive" && condition["expr_canonical"] == "tan(x) + e^x + x"
        }),
        "expected cos nonzero and radicand positive guards: {required:?}"
    );
}

#[test]
fn test_eval_json_ln_sqrt_diff_omits_redundant_rationalize_expand_round_trip() {
    let json = eval_json_with_args("diff(ln(sqrt(x)+1), x)", &["--steps", "on"]);

    assert_eq!(json["ok"], true);
    assert_eq!(json["result"], "1 / (2·sqrt(x)·(sqrt(x) + 1))");

    let steps = json["steps"].as_array().expect("steps should be an array");
    assert_eq!(steps.len(), 1, "expected only the direct derivative step");
    assert_eq!(steps[0]["rule"], "Calcular la derivada");
    assert!(
        steps.iter().all(|step| {
            step["rule"] != "Racionalizar el denominador"
                && step["rule"] != "Expandir la expresión"
                && step["rule"] != "Presentar resultado de cálculo en forma compacta"
        }),
        "redundant rationalize/expand/presentation round trip should be hidden: {steps:?}"
    );
}

#[test]
fn test_eval_json_shifted_ln_sqrt_diff_omits_redundant_rationalize_cancel_round_trip() {
    let json = eval_json_with_args("diff(ln(sqrt(x+1)+1), x)", &["--steps", "on"]);

    assert_eq!(json["ok"], true);
    assert_eq!(json["result"], "1 / (2·sqrt(x + 1)·(sqrt(x + 1) + 1))");

    let steps = json["steps"].as_array().expect("steps should be an array");
    assert_eq!(steps.len(), 1, "expected only the direct derivative step");
    assert_eq!(steps[0]["rule"], "Calcular la derivada");
    assert!(
        steps.iter().all(|step| {
            step["rule"] != "Racionalizar el denominador"
                && step["rule"] != "Cancelar términos opuestos"
                && step["rule"] != "Presentar resultado de cálculo en forma compacta"
        }),
        "redundant rationalize/cancel/presentation round trip should be hidden: {steps:?}"
    );
}

#[test]
fn test_eval_json_quadratic_ln_sqrt_diff_keeps_presentation_without_rationalize_noise() {
    let json = eval_json_with_args("diff(ln(sqrt(x^2+1)+1), x)", &["--steps", "on"]);

    assert_eq!(json["ok"], true);
    assert_eq!(json["result"], "x / (sqrt(x^2 + 1)·(sqrt(x^2 + 1) + 1))");

    let steps = json["steps"].as_array().expect("steps should be an array");
    assert_eq!(
        steps.len(),
        2,
        "expected derivative plus useful presentation"
    );
    assert_eq!(steps[0]["rule"], "Calcular la derivada");
    assert_eq!(
        steps[1]["rule"],
        "Presentar resultado de cálculo en forma compacta"
    );
    assert!(
        steps
            .iter()
            .all(|step| step["rule"] != "Racionalizar el denominador"),
        "rationalization noise should be hidden before the useful presentation: {steps:?}"
    );
}

#[test]
fn test_eval_json_shifted_trig_ln_sqrt_diff_keeps_direct_presentation() {
    for (expr, result, after_fragment) in [
        (
            "diff(ln(1+sqrt(sin(x)+2)), x)",
            "cos(x) / (2·sqrt(sin(x) + 2)·(sqrt(sin(x) + 2) + 1))",
            "cos(x)/(2 · sqrt(sin(x) + 2) · (sqrt(sin(x) + 2) + 1))",
        ),
        (
            "diff(ln(1+sqrt(cos(x)+2)), x)",
            "-sin(x) / (2·sqrt(cos(x) + 2)·(sqrt(cos(x) + 2) + 1))",
            "-sin(x)/(2 · sqrt(cos(x) + 2) · (sqrt(cos(x) + 2) + 1))",
        ),
        (
            "diff(ln(1+sqrt(sin(x)+cos(x)+3)), x)",
            "(cos(x) - sin(x)) / (2·sqrt(sin(x) + cos(x) + 3)·(sqrt(sin(x) + cos(x) + 3) + 1))",
            "(cos(x) - sin(x))/(2 · sqrt(sin(x) + cos(x) + 3) · (sqrt(sin(x) + cos(x) + 3) + 1))",
        ),
        (
            "diff(ln(1+sqrt(2*sin(x)+cos(x)+4)), x)",
            "(2·cos(x) - sin(x)) / (2·sqrt(cos(x) + 2·sin(x) + 4)·(sqrt(cos(x) + 2·sin(x) + 4) + 1))",
            "(2 · cos(x) - sin(x))/(2 · sqrt(cos(x) + 2 · sin(x) + 4) · (sqrt(cos(x) + 2 · sin(x) + 4) + 1))",
        ),
    ] {
        let json = eval_json_with_args(expr, &["--steps", "on"]);

        assert_eq!(json["ok"], true);
        assert_eq!(json["result"], result);
        assert_eq!(json["required_display"], Value::Array(vec![]));

        let steps = json["steps"].as_array().expect("steps should be an array");
        assert_eq!(steps.len(), 1, "expected only the direct derivative step");
        assert_eq!(steps[0]["rule"], "Calcular la derivada");
        let after = steps[0]["after"].as_str().expect("after string");
        assert!(
            after.contains(after_fragment)
                && !after.contains("sqrt(sin(x) + 2) - 1")
                && !after.contains("sqrt(cos(x) + 2) - 1")
                && !after.contains("2 · sin(x) + 2")
                && !after.contains("2 · cos(x) + 2"),
            "shifted trig ln/sqrt diff should keep the direct educational form, got: {after}"
        );
    }
}

#[test]
fn test_eval_json_shifted_exp_ln_sqrt_diff_does_not_reintroduce_ln_e() {
    let json = eval_json_with_args("diff(ln(1+sqrt(exp(x)+1)), x)", &["--steps", "on"]);

    assert_eq!(json["ok"], true);
    assert_eq!(
        json["result"],
        "e^x / (2·sqrt(e^x + 1)·(sqrt(e^x + 1) + 1))"
    );
    assert_eq!(json["required_display"], Value::Array(vec![]));

    let steps = json["steps"].as_array().expect("steps should be an array");
    assert_eq!(steps.len(), 1, "expected only the direct derivative step");
    assert_eq!(steps[0]["rule"], "Calcular la derivada");
    let after = steps[0]["after"].as_str().expect("after string");
    assert!(
        after.contains("e^x/(2 · sqrt(e^x + 1) · (sqrt(e^x + 1) + 1))")
            && !after.contains("ln(e)")
            && !after.contains("sqrt(e^x + 1) - 1"),
        "shifted exp ln/sqrt diff should not reintroduce ln(e), got: {after}"
    );
}

#[test]
fn test_eval_json_scaled_trig_root_diff_step_cancels_fraction_product_presentation_noise() {
    let expr = "diff(sqrt(sin(2*x)+3), x)";
    let json = eval_json_with_args(expr, &["--steps", "on"]);

    assert_eq!(json["ok"], true, "expr: {expr}");
    assert_eq!(json["result"], "cos(2·x) / sqrt(sin(2·x) + 3)");

    let steps = json["steps"].as_array().expect("steps should be an array");
    let diff_step = steps
        .iter()
        .find(|step| step["rule"] == "Calcular la derivada")
        .expect("expected symbolic differentiation step");
    let after = diff_step["after"].as_str().expect("after string");
    assert!(
        after.contains("cos(2 · x)/sqrt(sin(2 · x) + 3)")
            && !after.contains("(1/2)/sqrt")
            && !after.contains("2 · cos(2 · x) · (1/2)"),
        "diff step should render the compact trig/root form for {expr}, got: {after}"
    );
    let after_latex = diff_step["after_latex"]
        .as_str()
        .expect("after_latex string");
    assert!(
        after_latex.contains("\\frac{\\cos(2\\cdot x)}{\\sqrt")
            && !after_latex.contains("\\frac{\\frac{1}{2}}{\\sqrt"),
        "diff step LaTeX should render the compact trig/root form for {expr}, got: {after_latex}"
    );
}

#[test]
fn test_eval_json_multi_function_trig_root_diff_skips_rationalization_noise() {
    let expr = "diff(sqrt(sin(2*x)+cos(x)+4), x)";
    let json = eval_json_with_args(expr, &["--steps", "on"]);

    assert_eq!(json["ok"], true, "expr: {expr}");
    assert_eq!(
        json["result"],
        "(cos(2·x) - 1/2·sin(x)) / sqrt(sin(2·x) + cos(x) + 4)"
    );

    let steps = json["steps"].as_array().expect("steps should be an array");
    assert!(
        steps
            .iter()
            .any(|step| step["rule"] == "Calcular la derivada"),
        "expected symbolic differentiation step for {expr}"
    );
    assert!(
        steps.iter().all(|step| {
            step["rule"] != "Rationalize Product Denominator"
                && step["rule"] != "Rationalize Linear Sqrt Denominator"
                && step["rule"] != "Expand to Cancel Fraction"
        }),
        "multi-function trig/root diff should keep the compact root denominator without rationalization noise: {steps:?}"
    );
}

#[test]
fn test_eval_json_trig_power_root_diff_uses_direct_presentation_without_cycle_noise() {
    let expr = "diff(sqrt(sin(x)^2+cos(x)+4), x)";
    let (json, stderr) = eval_json_with_args_and_stderr(expr, &["--steps", "on"]);

    assert_eq!(json["ok"], true, "expr: {expr}");
    assert!(
        !stderr.contains("depth_overflow"),
        "trig power root diff should not hit depth overflow: {stderr}"
    );
    assert_eq!(
        json["result"],
        "(2·sin(x)·cos(x) - sin(x)) / (2·sqrt(cos(x) + sin(x)^2 + 4))"
    );

    let steps = json["steps"].as_array().expect("steps should be an array");
    assert_eq!(
        steps.len(),
        1,
        "trig power root diff should stay on the direct derivative presentation route: {steps:?}"
    );
    assert!(
        steps.iter().all(|step| !step["rule"]
            .as_str()
            .is_some_and(|rule| rule.contains("ángulo doble"))),
        "trig power root diff should not cycle through double-angle expand/contract noise: {steps:?}"
    );
}

#[test]
fn test_eval_json_trig_root_diff_compacts_numeric_presentation_factors() {
    let scaled_expr = "diff(sqrt(2*sin(x)^2+cos(x)+4), x)";
    let scaled_json = eval_json_with_args(scaled_expr, &["--steps", "on"]);

    assert_eq!(scaled_json["ok"], true, "expr: {scaled_expr}");
    assert_eq!(
        scaled_json["result"],
        "(4·sin(x)·cos(x) - sin(x)) / (2·sqrt(cos(x) + 2·sin(x)^2 + 4))"
    );
    let scaled_steps = scaled_json["steps"]
        .as_array()
        .expect("steps should be an array");
    assert_eq!(
        scaled_steps.len(),
        1,
        "scaled trig root diff should stay on the direct presentation route: {scaled_steps:?}"
    );
    let scaled_after = scaled_steps[0]["after"].as_str().expect("after string");
    assert!(
        !scaled_after.contains("2 · 2"),
        "scaled trig root diff should combine numeric factors in the presentation result: {scaled_after}"
    );

    let half_expr = "diff(sqrt(sin(2*x)+cos(x)^2+4), x)";
    let half_json = eval_json_with_args(half_expr, &["--steps", "on"]);

    assert_eq!(half_json["ok"], true, "expr: {half_expr}");
    assert_eq!(
        half_json["result"],
        "(cos(2·x) - sin(x)·cos(x)) / sqrt(sin(2·x) + cos(x)^2 + 4)"
    );
    let half_steps = half_json["steps"]
        .as_array()
        .expect("steps should be an array");
    assert_eq!(
        half_steps.len(),
        1,
        "half-distributed trig root diff should stay on the direct presentation route: {half_steps:?}"
    );
    let half_after = half_steps[0]["after"].as_str().expect("after string");
    assert!(
        !half_after.contains("2/2") && !half_after.contains("1 ·"),
        "half-distributed trig root diff should not expose cancellable numeric factors: {half_after}"
    );
}

#[test]
fn test_eval_json_exp_root_diff_step_collapses_ln_e_presentation_noise() {
    let expr = "diff(sqrt(exp(x)+1), x)";
    let json = eval_json_with_args(expr, &["--steps", "on"]);

    assert_eq!(json["ok"], true, "expr: {expr}");
    assert_eq!(json["result"], "e^x / (2·sqrt(e^x + 1))");

    let steps = json["steps"].as_array().expect("steps should be an array");
    let diff_step = steps
        .iter()
        .find(|step| step["rule"] == "Calcular la derivada")
        .expect("expected symbolic differentiation step");
    let after = diff_step["after"].as_str().expect("after string");
    assert!(
        after.contains("e^x/(2 · sqrt(e^x + 1))") && !after.contains("ln(e)"),
        "diff step should collapse ln(e) presentation noise for {expr}, got: {after}"
    );
    let after_latex = diff_step["after_latex"]
        .as_str()
        .expect("after_latex string");
    assert!(
        after_latex.contains("\\frac{")
            && after_latex.contains("{e}^{x}")
            && after_latex.contains("2\\cdot \\sqrt")
            && !after_latex.contains("\\ln(e)"),
        "diff step LaTeX should collapse ln(e) presentation noise for {expr}, got: {after_latex}"
    );
}

#[test]
fn test_eval_json_exp_quadratic_diff_step_parenthesizes_nested_power_exponent() {
    let cases = [
        (
            "diff(exp(x^2), x)",
            "2·x·e^(x^2)",
            &["2 · x", "e^(x^2)"][..],
        ),
        (
            "diff(sin(x^2), x)",
            "2·x·cos(x^2)",
            &["2 · x", "cos(x^2)"][..],
        ),
        (
            "diff(sqrt(exp(x^2)+1), x)",
            "x·e^(x^2) / sqrt(e^(x^2) + 1)",
            &["e^(x^2)", "sqrt(e^(x^2) + 1)", "x"][..],
        ),
    ];

    for (expr, expected_result, expected_after_needles) in cases {
        let json = eval_json_with_args(expr, &["--steps", "on"]);

        assert_eq!(json["ok"], true, "expr: {expr}");
        assert_eq!(json["result"], expected_result, "expr: {expr}");

        let steps = json["steps"].as_array().expect("steps should be an array");
        let diff_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la derivada")
            .expect("expected symbolic differentiation step");
        let before = diff_step["before"].as_str().expect("before string");
        let after = diff_step["after"].as_str().expect("after string");
        assert!(
            !before.contains("e^x^2")
                && !after.contains("e^x^2")
                && !after.contains("x^(2 - 1)"),
            "diff step should not render ambiguous or unsimplified exponent text for {expr}, got before={before}, after={after}"
        );
        assert!(
            expected_after_needles
                .iter()
                .all(|needle| after.contains(needle)),
            "diff step should preserve grouped nested exponent text for {expr}, got: {after}"
        );
    }
}

#[test]
fn test_eval_json_arcsin_sqrt_diff_step_compacts_nested_reciprocal_division() {
    let expr = "diff(arcsin(sqrt(x)), x)";
    let json = eval_json_with_args(expr, &["--steps", "on"]);

    assert_eq!(json["ok"], true, "expr: {expr}");
    assert_eq!(json["result"], "1 / (2·sqrt(x)·sqrt(1 - x))");

    let steps = json["steps"].as_array().expect("steps should be an array");
    let diff_step = steps
        .iter()
        .find(|step| step["rule"] == "Calcular la derivada")
        .expect("expected symbolic differentiation step");
    let after = diff_step["after"].as_str().expect("after string");
    assert!(
        after.contains("1/(2 · sqrt(x) · sqrt(1 - sqrt(x)^2))")
            && !after.contains("(1/(2 · sqrt(x)))/sqrt(1 - sqrt(x)^2)"),
        "diff step should compact nested reciprocal division for {expr}, got: {after}"
    );
    let after_latex = diff_step["after_latex"]
        .as_str()
        .expect("after_latex string");
    assert!(
        after_latex.contains("\\frac{1}{2\\cdot \\sqrt{x}\\cdot \\sqrt")
            && !after_latex.contains("\\frac{\\frac{1}{2\\cdot \\sqrt{x}}}"),
        "diff step LaTeX should compact nested reciprocal division for {expr}, got: {after_latex}"
    );
    let root_step = steps
        .iter()
        .find(|step| step["rule"] == "Deshacer raíz y potencia")
        .expect("expected root-power cleanup step");
    let root_before = root_step["before"].as_str().expect("root before string");
    let root_after = root_step["after"].as_str().expect("root after string");
    assert_eq!(root_before, after);
    assert!(
        root_after.contains("1/(2 · sqrt(x) · sqrt(1 - x))")
            && !root_after.contains("(1/(2 · sqrt(x)))/sqrt(1 - x)"),
        "root-power step should keep the compact reciprocal-division presentation, got: {root_after}"
    );
}

#[test]
fn test_eval_json_arccos_sqrt_diff_step_compacts_signed_nested_reciprocal_division() {
    let expr = "diff(arccos(sqrt(x)), x)";
    let json = eval_json_with_args(expr, &["--steps", "on"]);

    assert_eq!(json["ok"], true, "expr: {expr}");
    assert_eq!(json["result"], "-1 / (2·sqrt(x)·sqrt(1 - x))");

    let steps = json["steps"].as_array().expect("steps should be an array");
    let diff_step = steps
        .iter()
        .find(|step| step["rule"] == "Calcular la derivada")
        .expect("expected symbolic differentiation step");
    let after = diff_step["after"].as_str().expect("after string");
    assert!(
        after.contains("-1/(2 · sqrt(x) · sqrt(1 - sqrt(x)^2))")
            && !after.contains("-(1/(2 · sqrt(x)))/sqrt(1 - sqrt(x)^2)"),
        "diff step should compact signed nested reciprocal division for {expr}, got: {after}"
    );
    let after_latex = diff_step["after_latex"]
        .as_str()
        .expect("after_latex string");
    assert!(
        after_latex.contains("-\\frac{1}{2\\cdot \\sqrt{x}\\cdot \\sqrt")
            && !after_latex.contains("\\frac{-\\frac{1}{2\\cdot \\sqrt{x}}}"),
        "diff step LaTeX should compact signed nested reciprocal division for {expr}, got: {after_latex}"
    );
    let root_step = steps
        .iter()
        .find(|step| step["rule"] == "Deshacer raíz y potencia")
        .expect("expected root-power cleanup step");
    let root_before = root_step["before"].as_str().expect("root before string");
    let root_after = root_step["after"].as_str().expect("root after string");
    assert_eq!(root_before, after);
    assert!(
        root_after.contains("-1/(2 · sqrt(x) · sqrt(1 - x))")
            && !root_after.contains("(-1/(2 · sqrt(x)))/sqrt(1 - x)"),
        "root-power step should keep signed compact reciprocal-division presentation, got: {root_after}"
    );
}

#[test]
fn test_eval_json_affine_inverse_trig_sqrt_diff_cancel_step_keeps_compact_reciprocal_division() {
    let cases = [
        (
            "diff(arcsin(sqrt(x+1)), x)",
            "1 / (2·sqrt(x + 1)·sqrt(-x))",
            "1/(2 · sqrt(x + 1) · sqrt(1 - (x + 1)))",
            "1/(2 · sqrt(-x) · sqrt(x + 1))",
            "(1/(2 · sqrt(x + 1)))/sqrt",
        ),
        (
            "diff(arccos(sqrt(x+1)), x)",
            "-1 / (2·sqrt(x + 1)·sqrt(-x))",
            "-1/(2 · sqrt(x + 1) · sqrt(1 - (x + 1)))",
            "-1/(2 · sqrt(-x) · sqrt(x + 1))",
            "(-1/(2 · sqrt(x + 1)))/sqrt",
        ),
    ];

    for (expr, expected_result, expected_before, expected_after, rejected_prefix) in cases {
        let json = eval_json_with_args(expr, &["--steps", "on"]);

        assert_eq!(json["ok"], true, "expr: {expr}");
        assert_eq!(json["result"], expected_result, "expr: {expr}");

        let steps = json["steps"].as_array().expect("steps should be an array");
        let cancel_step = steps
            .iter()
            .find(|step| step["rule"] == "Cancelar términos opuestos")
            .expect("expected exact additive cancellation step");
        let before = cancel_step["before"].as_str().expect("before string");
        let after = cancel_step["after"].as_str().expect("after string");
        assert!(
            before.contains(expected_before)
                && after.contains(expected_after)
                && !before.contains(rejected_prefix)
                && !after.contains(rejected_prefix),
            "cancel step should keep compact nested reciprocal division for {expr}, got before={before}, after={after}"
        );
    }
}

#[test]
fn test_eval_json_repeated_by_parts_integral_step_matches_presented_antiderivative() {
    let cases = [
        (
            "integrate(x^2*sin(x), x)",
            "2·x·sin(x) + (2 - x^2)·cos(x)",
            ["2·x·sin(x)", "(2 - x^2)·cos(x)"],
            ["cos(x)·", "sin(x)·"],
        ),
        (
            "integrate(x^2*cos(x), x)",
            "2·x·cos(x) + (x^2 - 2)·sin(x)",
            ["2·x·cos(x)", "(x^2 - 2)·sin(x)"],
            ["sin(x)·", "cos(x)·"],
        ),
        (
            "integrate(x*exp(2*x), x)",
            "(1/2·x - 1/4)·e^(2·x)",
            ["1/2·x - 1/4", "e^(2·x)"],
            ["e^(2 · x) ·", "(e^(2"],
        ),
        (
            "integrate(x^2*exp(2*x), x)",
            "1/4·e^(2·x)·(2·x^2 + 1 - 2·x)",
            ["1/4", "e^(2 · x) · (2 · x^2 + 1 - 2 · x)"],
            ["(e^(2", "e^(2 · x) ·"],
        ),
        (
            "integrate(x^2*sinh(x), x)",
            "(x^2 + 2)·cosh(x) - 2·x·sinh(x)",
            ["(x^2 + 2)·cosh(x)", "2·x·sinh(x)"],
            ["cosh(x)·", "sinh(x)·"],
        ),
        (
            "integrate(x^2*cosh(x), x)",
            "(x^2 + 2)·sinh(x) - 2·x·cosh(x)",
            ["(x^2 + 2)·sinh(x)", "2·x·cosh(x)"],
            ["sinh(x)·", "cosh(x)·"],
        ),
    ];

    for (expr, expected_result, expected_after_needles, rejected_prefixes) in cases {
        let json = eval_json_with_args(expr, &["--steps", "on"]);

        assert_eq!(json["ok"], true, "expr: {expr}");
        assert_eq!(json["result"], expected_result, "expr: {expr}");
        assert_eq!(
            json["required_conditions"],
            Value::Array(vec![]),
            "expr: {expr}"
        );

        let steps = json["steps"].as_array().expect("steps should be an array");
        let integrate_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected symbolic integration step");
        let after = integrate_step["after"].as_str().expect("after string");
        assert!(
            expected_after_needles
                .iter()
                .all(|needle| after.contains(needle)),
            "integration step should show the presented by-parts primitive for {expr}, got: {after}"
        );
        assert!(
            rejected_prefixes
                .iter()
                .all(|prefix| !after.starts_with(prefix)),
            "integration step should not keep the raw by-parts term order for {expr}, got: {after}"
        );
    }
}

#[test]
fn test_eval_json_integral_sqrt_product_keeps_reciprocal_sqrt_before_table_step() {
    let expr = "integrate(1/(sqrt(x+1)*sqrt(-x)), x)";
    let json = eval_json_with_args(expr, &["--steps", "on"]);

    assert_eq!(json["ok"], true, "expr: {expr}");
    assert_eq!(json["result"], "arcsin(2·x + 1)");

    let steps = json["steps"].as_array().expect("steps should be an array");
    let merge_step = steps
        .iter()
        .find(|step| step["rule"] == "Merge Sqrt Product")
        .expect("expected sqrt product merge step");
    let merge_after = merge_step["after"].as_str().expect("merge after string");
    assert!(
        merge_after.contains("1/sqrt(-x · (x + 1))"),
        "merge step should expose reciprocal-sqrt integrand for {expr}, got: {merge_after}"
    );

    let integrate_step = steps
        .iter()
        .find(|step| step["rule"] == "Calcular la integral")
        .expect("expected symbolic integration step");
    let before = integrate_step["before"].as_str().expect("before string");
    assert!(
        before.contains("1/sqrt(-(x · (x + 1)))")
            && !before.contains("^(-1/2)")
            && !before.contains(")^(-1/2)"),
        "integration table step should keep reciprocal-sqrt presentation for {expr}, got: {before}"
    );
    let before_latex = integrate_step["before_latex"]
        .as_str()
        .expect("before_latex string");
    assert!(
        before_latex.contains("\\frac{1}{\\sqrt")
            && !before_latex.contains("-\\frac{1}{2}"),
        "integration table step LaTeX should keep reciprocal-sqrt presentation for {expr}, got: {before_latex}"
    );
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
fn test_eval_calculus_undefined_suppresses_impossible_zero_nonzero_requirement() {
    for expr in [
        "diff(1/(sqrt(x^2+1)-sqrt(x^2+1)), x)",
        "integrate(1/(sqrt(x^2+1)-sqrt(x^2+1)), x)",
    ] {
        let json = eval_json(expr);

        assert_eq!(json["ok"], true, "eval should succeed for {expr}");
        assert_eq!(json["result"], "undefined", "expected undefined for {expr}");
        assert_eq!(
            json["required_conditions"]
                .as_array()
                .expect("required_conditions should be an array")
                .len(),
            0,
            "undefined calculus result should not expose NonZero(0): {json:?}"
        );
        assert_eq!(
            json["required_display"]
                .as_array()
                .expect("required_display should be an array")
                .len(),
            0,
            "undefined calculus result should not render 0 ≠ 0: {json:?}"
        );

        let messages = json["wire"]["messages"]
            .as_array()
            .expect("wire messages should be an array");
        assert!(
            !messages.iter().any(|message| message["text"]
                .as_str()
                .is_some_and(|text| text.contains("Requires:") || text.contains("0 ≠ 0"))),
            "undefined calculus result should not render impossible Requires: {messages:?}"
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
            "diff(sqrt(sin(x)+1), x)",
            "cos(x) / (2·sqrt(sin(x) + 1))",
            "((sin(x) + 1))^(1/2 - 1)",
            ["cos(x)", "sqrt(sin(x) + 1)"],
            Some("sin(x) + 1"),
        ),
        (
            "diff(sqrt(ln(x)), x)",
            "1 / (2·x·sqrt(ln(x)))",
            "ln(x)^(-1/2)",
            ["x", "sqrt(ln(x))"],
            Some("ln(x)"),
        ),
        (
            "diff(sqrt(ln(x)+1), x)",
            "1 / (2·x·sqrt(ln(x) + 1))",
            "((ln(x) + 1))^(1/2 - 1)",
            ["x", "sqrt(ln(x) + 1)"],
            Some("ln(x) + 1"),
        ),
        (
            "diff(sqrt(2*ln(x)+3), x)",
            "1 / (x·sqrt(2·ln(x) + 3))",
            "((2 · ln(x) + 3))^(1/2 - 1)",
            ["x", "sqrt(2 · ln(x) + 3)"],
            Some("2·ln(x) + 3"),
        ),
        (
            "diff(sqrt(ln(2*x+1)+1), x)",
            "1 / ((2·x + 1)·sqrt(ln(2·x + 1) + 1))",
            "((ln(2 · x + 1) + 1))^(1/2 - 1)",
            ["2 · x + 1", "sqrt(ln(2 · x + 1) + 1)"],
            Some("ln(2·x + 1) + 1"),
        ),
        (
            "diff(sqrt(ln(x^2+1)+1), x)",
            "x / ((x^2 + 1)·sqrt(ln(x^2 + 1) + 1))",
            "((ln(x^2 + 1) + 1))^(1/2 - 1)",
            ["sqrt(ln(x^2 + 1) + 1)", "x^2 + 1"],
            Some("ln(x^2 + 1) + 1"),
        ),
        (
            "diff(sqrt(log10(x)), x)",
            "1 / (2·x·ln(10)·sqrt(log10(x)))",
            "log10(x)^(-1/2)",
            ["ln(10)", "sqrt(log_10(x))"],
            Some("log10(x)"),
        ),
        (
            "diff(sqrt(log10(x)+1), x)",
            "1 / (2·x·ln(10)·sqrt(log10(x) + 1))",
            "((log_10(x) + 1))^(1/2 - 1)",
            ["ln(10)", "sqrt(log_10(x) + 1)"],
            Some("log10(x) + 1"),
        ),
        (
            "diff(sqrt(log10(x^2+1)+1), x)",
            "x / ((x^2 + 1)·ln(10)·sqrt(log10(x^2 + 1) + 1))",
            "((log_10(x^2 + 1) + 1))^(-1/2)",
            ["ln(10)", "sqrt(log_10(x^2 + 1) + 1)"],
            Some("log10(x^2 + 1) + 1"),
        ),
        (
            "diff(sqrt(log2(x)+1), x)",
            "1 / (2·x·ln(2)·sqrt(log2(x) + 1))",
            "((log2(x) + 1))^(1/2 - 1)",
            ["ln(2)", "sqrt(log2(x) + 1)"],
            Some("log2(x) + 1"),
        ),
        (
            "diff(sqrt(2*log10(3*x+1)+5), x)",
            "3 / ((3·x + 1)·ln(10)·sqrt(2·log10(3·x + 1) + 5))",
            "((2 · log_10(3 · x + 1) + 5))^(-1/2)",
            ["ln(10)", "sqrt(2 · log_10(3 · x + 1) + 5)"],
            Some("2·log10(3·x + 1) + 5"),
        ),
        (
            "diff(sqrt(exp(x)+1), x)",
            "e^x / (2·sqrt(e^x + 1))",
            "((e^x + 1))^(-1/2)",
            ["e^x", "sqrt(e^x + 1)"],
            None,
        ),
        (
            "diff(sqrt(exp(2*x+1)+1), x)",
            "e^(2·x + 1) / sqrt(e^(2·x + 1) + 1)",
            "((e^(2 · x + 1) + 1))^(-1/2)",
            ["e^(2 · x + 1)", "sqrt(e^(2 · x + 1) + 1)"],
            None,
        ),
        (
            "diff(sqrt(2*exp(x)+1), x)",
            "e^x / sqrt(2·e^x + 1)",
            "((2 · e^x + 1))^(-1/2)",
            ["e^x", "sqrt(2 · e^x + 1)"],
            None,
        ),
        (
            "diff(sqrt(3*exp(2*x+1)+5), x)",
            "3·e^(2·x + 1) / sqrt(3·e^(2·x + 1) + 5)",
            "((3 · e^(2 · x + 1) + 5))^(-1/2)",
            ["3 · e^(2 · x + 1)", "sqrt(3 · e^(2 · x + 1) + 5)"],
            None,
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
                step["before"] != step["after"]
                    && step["after"].as_str().is_some_and(|after| {
                        expected_after.iter().all(|needle| after.contains(needle))
                    })
            }) || steps.iter().any(|step| {
                step["rule"] == "Calcular la derivada"
                    && step["after"].as_str().is_some_and(|after| {
                        expected_after.iter().all(|needle| after.contains(needle))
                            || after.contains(expected_before)
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
            "diff(sqrt(ln(x)+1), x) - 1/(2*x*sqrt(ln(x)+1))",
            ["ln(x) + 1", "x"],
        ),
        (
            "diff(sqrt(2*ln(x)+3), x) - 1/(x*sqrt(2*ln(x)+3))",
            ["2·ln(x) + 3", "x"],
        ),
        (
            "diff(sqrt(ln(2*x+1)+1), x) - 1/((2*x+1)*sqrt(ln(2*x+1)+1))",
            ["ln(2·x + 1) + 1", "2·x + 1"],
        ),
        (
            "diff(sqrt(ln(x^2+1)+1), x) - x/((x^2+1)*sqrt(ln(x^2+1)+1))",
            ["ln(x^2 + 1) + 1", ""],
        ),
        (
            "diff(sqrt(log10(x)), x) - 1/(2*x*ln(10)*sqrt(log10(x)))",
            ["log10(x)", "x"],
        ),
        (
            "diff(sqrt(log10(x)+1), x) - 1/(2*x*ln(10)*sqrt(log10(x)+1))",
            ["log10(x) + 1", "x"],
        ),
        (
            "diff(sqrt(log10(x^2+1)+1), x) - x/((x^2+1)*ln(10)*sqrt(log10(x^2+1)+1))",
            ["log10(x^2 + 1) + 1", ""],
        ),
        (
            "diff(sqrt(log2(x)+1), x) - 1/(2*x*ln(2)*sqrt(log2(x)+1))",
            ["log2(x) + 1", "x"],
        ),
        (
            "diff(sqrt(log2(x^2+1)+1), x) - x/((x^2+1)*ln(2)*sqrt(log2(x^2+1)+1))",
            ["log2(x^2 + 1) + 1", ""],
        ),
        (
            "diff(sqrt(2*log10(3*x+1)+5), x) - 3/(ln(10)*(3*x+1)*sqrt(2*log10(3*x+1)+5))",
            ["2·log10(3·x + 1) + 5", "3·x + 1"],
        ),
        (
            "diff(sqrt(exp(x)+1), x) - exp(x)/(2*sqrt(exp(x)+1))",
            ["", ""],
        ),
        (
            "diff(sqrt(exp(2*x+1)+1), x) - exp(2*x+1)/sqrt(exp(2*x+1)+1)",
            ["", ""],
        ),
        (
            "diff(sqrt(2*exp(x)+1), x) - exp(x)/sqrt(2*exp(x)+1)",
            ["", ""],
        ),
        (
            "diff(sqrt(3*exp(2*x+1)+5), x) - 3*exp(2*x+1)/sqrt(3*exp(2*x+1)+5)",
            ["", ""],
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
fn test_eval_json_diff_sqrt_bounded_trig_positive_shift_uses_compact_trig_presentation() {
    let expr = "diff(sqrt(sin(2*x)+cos(x)+4), x)";
    let (json, stderr) = eval_json_with_args_and_stderr(expr, &["--steps", "on"]);

    assert_eq!(json["ok"], true, "expr: {expr}");
    assert!(
        !stderr.contains("depth_overflow"),
        "bounded trig sqrt diff should preserve the raw target and avoid depth overflow: {stderr}"
    );
    assert_eq!(
        json["result"], "(cos(2·x) - 1/2·sin(x)) / sqrt(sin(2·x) + cos(x) + 4)",
        "expr: {expr}"
    );

    let steps = json["steps"].as_array().expect("steps should be an array");
    assert!(
        steps
            .iter()
            .any(|step| step["rule"] == "Calcular la derivada"),
        "direct diff route should include a differentiation step for {expr}: {steps:?}"
    );
    assert!(
        steps.iter().all(|step| !step["rule"]
            .as_str()
            .is_some_and(|rule| rule.contains("ángulo doble"))),
        "raw diff target should avoid the double-angle expansion detour for {expr}: {steps:?}"
    );

    let required = json["required_conditions"]
        .as_array()
        .expect("required_conditions should be an array");
    assert!(
        required.is_empty(),
        "positive shifted bounded trig radicand should not emit a redundant guard for {expr}: {required:?}"
    );
}

#[test]
fn test_eval_json_diff_sqrt_trig_polynomial_sum_uses_direct_presentation_without_depth_overflow() {
    let expr = "diff(sqrt(sin(2*x)+cos(x)+x), x)";
    let (json, stderr) = eval_json_with_args_and_stderr(expr, &["--steps", "on"]);

    assert_eq!(json["ok"], true, "expr: {expr}");
    assert!(
        !stderr.contains("depth_overflow"),
        "mixed trig/polynomial sqrt diff should avoid the generic depth-overflow route: {stderr}"
    );
    assert_eq!(
        json["result"], "(cos(2·x) + 1/2 - 1/2·sin(x)) / sqrt(sin(2·x) + cos(x) + x)",
        "expr: {expr}"
    );

    let steps = json["steps"].as_array().expect("steps should be an array");
    assert_eq!(
        steps.len(),
        1,
        "expected one direct diff step for {expr}: {steps:?}"
    );
    assert_eq!(steps[0]["rule"], "Calcular la derivada");
    assert!(
        steps.iter().all(|step| !step["rule"]
            .as_str()
            .is_some_and(|rule| rule.contains("ángulo doble"))),
        "mixed trig/polynomial sqrt diff should keep the raw trig radicand: {steps:?}"
    );

    let required = json["required_conditions"]
        .as_array()
        .expect("required_conditions should be an array");
    assert!(
        required.iter().any(|condition| {
            condition["kind"] == "Positive"
                && condition["expr_canonical"] == "sin(2·x) + cos(x) + x"
        }),
        "unbounded mixed trig/polynomial radicand should expose a positive-domain guard for {expr}: {required:?}"
    );
}

#[test]
fn test_eval_json_diff_sqrt_tan_polynomial_sum_uses_direct_presentation_without_depth_overflow() {
    let expr = "diff(sqrt(tan(x)+x+2), x)";
    let (json, stderr) = eval_json_with_args_and_stderr(expr, &["--steps", "on"]);

    assert_eq!(json["ok"], true, "expr: {expr}");
    assert!(
        !stderr.contains("depth_overflow"),
        "mixed tan/polynomial sqrt diff should avoid the generic depth-overflow route: {stderr}"
    );
    assert_eq!(
        json["result"], "(cos(x)^2 + 1) / (2·cos(x)^2·sqrt(tan(x) + x + 2))",
        "expr: {expr}"
    );

    let steps = json["steps"].as_array().expect("steps should be an array");
    assert_eq!(
        steps.len(),
        1,
        "tan/polynomial sqrt diff should stay on the direct presentation route: {steps:?}"
    );
    assert_eq!(steps[0]["rule"], "Calcular la derivada");

    let required = json["required_conditions"]
        .as_array()
        .expect("required_conditions should be an array");
    assert!(
        required.iter().any(|condition| {
            condition["kind"] == "Positive" && condition["expr_canonical"] == "tan(x) + x + 2"
        }),
        "tan/polynomial radicand should expose a positive-domain guard for {expr}: {required:?}"
    );
    assert!(
        required.iter().any(|condition| {
            condition["kind"] == "NonZero" && condition["expr_canonical"] == "cos(x)"
        }),
        "tan/polynomial radicand should expose the tangent pole guard for {expr}: {required:?}"
    );
}

#[test]
fn test_eval_json_diff_sqrt_tan_sin_polynomial_sum_compacts_common_denominator_numerator() {
    let expr = "diff(sqrt(tan(x)+sin(x)+x), x)";
    let (json, stderr) = eval_json_with_args_and_stderr(expr, &["--steps", "on"]);

    assert_eq!(json["ok"], true, "expr: {expr}");
    assert!(
        !stderr.contains("depth_overflow"),
        "mixed tan/sin/polynomial sqrt diff should stay out of the generic cleanup loop: {stderr}"
    );
    assert_eq!(
        json["result"], "(cos(x)^2 + cos(x)^3 + 1) / (2·cos(x)^2·sqrt(sin(x) + tan(x) + x))",
        "expr: {expr}"
    );
    assert!(
        !json["result"]
            .as_str()
            .is_some_and(|result| result.contains("cos(x)·cos(x)^2")),
        "post-calculus presentation should compact repeated cos factors: {json:?}"
    );

    let steps = json["steps"].as_array().expect("steps should be an array");
    assert_eq!(
        steps.len(),
        1,
        "tan/sin/polynomial sqrt diff should remain a direct presentation step: {steps:?}"
    );

    let required = json["required_conditions"]
        .as_array()
        .expect("required_conditions should be an array");
    assert!(
        required.iter().any(|condition| {
            condition["kind"] == "Positive"
                && condition["expr_canonical"] == "sin(x) + tan(x) + x"
        }),
        "tan/sin/polynomial radicand should expose a positive-domain guard for {expr}: {required:?}"
    );
    assert!(
        required.iter().any(|condition| {
            condition["kind"] == "NonZero" && condition["expr_canonical"] == "cos(x)"
        }),
        "tan/sin/polynomial radicand should expose the tangent pole guard for {expr}: {required:?}"
    );
}

#[test]
fn test_eval_json_diff_sqrt_tan_cos_square_polynomial_sum_compacts_numerator_powers() {
    let expr = "diff(sqrt(tan(x)+cos(x)^2+x), x)";
    let (json, stderr) = eval_json_with_args_and_stderr(expr, &["--steps", "on"]);

    assert_eq!(json["ok"], true, "expr: {expr}");
    assert!(
        !stderr.contains("depth_overflow"),
        "mixed tan/cos-power/polynomial sqrt diff should stay out of cleanup loops: {stderr}"
    );
    assert_eq!(
        json["result"],
        "(cos(x)^2 + 1 - 2·sin(x)·cos(x)^3) / (2·cos(x)^2·sqrt(tan(x) + cos(x)^2 + x))",
        "expr: {expr}"
    );
    let result = json["result"].as_str().expect("result string");
    assert!(
        !result.contains("2 - 1") && !result.contains("cos(x)·cos(x)^2"),
        "post-calculus presentation should compact local cos powers: {result}"
    );

    let steps = json["steps"].as_array().expect("steps should be an array");
    assert_eq!(
        steps.len(),
        1,
        "tan/cos-power/polynomial sqrt diff should remain a direct presentation step: {steps:?}"
    );

    let required = json["required_conditions"]
        .as_array()
        .expect("required_conditions should be an array");
    assert!(
        required.iter().any(|condition| {
            condition["kind"] == "Positive"
                && condition["expr_canonical"] == "tan(x) + cos(x)^2 + x"
        }),
        "tan/cos-power/polynomial radicand should expose a positive-domain guard for {expr}: {required:?}"
    );
    assert!(
        required.iter().any(|condition| {
            condition["kind"] == "NonZero" && condition["expr_canonical"] == "cos(x)"
        }),
        "tan/cos-power/polynomial radicand should expose the tangent pole guard for {expr}: {required:?}"
    );
}

#[test]
fn test_eval_json_diff_sqrt_tan_ln_polynomial_sum_avoids_cycle_route() {
    let expr = "diff(sqrt(tan(x)+ln(x)+x), x)";
    let (json, stderr) = eval_json_with_args_and_stderr(expr, &["--steps", "on"]);

    assert_eq!(json["ok"], true, "expr: {expr}");
    assert!(
        !stderr.contains("cycle detected") && !stderr.contains("depth_overflow"),
        "tan/ln/polynomial sqrt diff should stay out of cleanup loops: {stderr}"
    );
    assert!(
        json.get("blocked_hints").is_none(),
        "direct tan/ln/polynomial sqrt diff should not emit blocked hints: {json:?}"
    );
    assert_eq!(
        json["result"], "(cos(x)^2 + x·cos(x)^2 + x) / (2·x·cos(x)^2·sqrt(tan(x) + ln(x) + x))",
        "expr: {expr}"
    );

    let steps = json["steps"].as_array().expect("steps should be an array");
    assert_eq!(
        steps.len(),
        1,
        "tan/ln/polynomial sqrt diff should use direct derivative presentation: {steps:?}"
    );
    assert_eq!(steps[0]["rule"], "Calcular la derivada");

    let required = json["required_conditions"]
        .as_array()
        .expect("required_conditions should be an array");
    assert!(
        required.iter().any(|condition| {
            condition["kind"] == "Positive" && condition["expr_canonical"] == "x"
        }),
        "ln(x) should expose x > 0 for {expr}: {required:?}"
    );
    assert!(
        required.iter().any(|condition| {
            condition["kind"] == "Positive"
                && condition["expr_canonical"] == "tan(x) + ln(x) + x"
        }),
        "tan/ln/polynomial radicand should expose the compact positive-domain guard for {expr}: {required:?}"
    );
    assert!(
        required.iter().any(|condition| {
            condition["kind"] == "NonZero" && condition["expr_canonical"] == "cos(x)"
        }),
        "tan/ln/polynomial radicand should expose the tangent pole guard for {expr}: {required:?}"
    );
}

#[test]
fn test_eval_json_diff_sqrt_tan_scaled_ln_polynomial_sum_uses_direct_route() {
    let expr = "diff(sqrt(tan(x)+2*ln(x)+x), x)";
    let (json, stderr) = eval_json_with_args_and_stderr(expr, &["--steps", "on"]);

    assert_eq!(json["ok"], true, "expr: {expr}");
    assert!(
        !stderr.contains("cycle detected") && !stderr.contains("depth_overflow"),
        "scaled tan/ln/polynomial sqrt diff should stay out of cleanup loops: {stderr}"
    );
    assert!(
        json.get("blocked_hints").is_none(),
        "scaled direct tan/ln/polynomial sqrt diff should not emit blocked hints: {json:?}"
    );
    assert_eq!(
        json["result"], "(2·cos(x)^2 + x·cos(x)^2 + x) / (2·x·cos(x)^2·sqrt(tan(x) + 2·ln(x) + x))",
        "expr: {expr}"
    );

    let steps = json["steps"].as_array().expect("steps should be an array");
    assert_eq!(
        steps.len(),
        1,
        "scaled tan/ln/polynomial sqrt diff should use direct derivative presentation: {steps:?}"
    );
    assert_eq!(steps[0]["rule"], "Calcular la derivada");

    let required = json["required_conditions"]
        .as_array()
        .expect("required_conditions should be an array");
    assert!(
        required.iter().any(|condition| {
            condition["kind"] == "Positive" && condition["expr_canonical"] == "x"
        }),
        "scaled ln(x) should expose x > 0 for {expr}: {required:?}"
    );
    assert!(
        required.iter().any(|condition| {
            condition["kind"] == "Positive"
                && condition["expr_canonical"] == "tan(x) + 2·ln(x) + x"
        }),
        "scaled tan/ln/polynomial radicand should expose the compact positive-domain guard for {expr}: {required:?}"
    );
    assert!(
        required.iter().any(|condition| {
            condition["kind"] == "NonZero" && condition["expr_canonical"] == "cos(x)"
        }),
        "scaled tan/ln/polynomial radicand should expose the tangent pole guard for {expr}: {required:?}"
    );
}

#[test]
fn test_eval_json_diff_sqrt_tan_negative_ln_polynomial_sum_uses_direct_route() {
    let expr = "diff(sqrt(tan(x)-ln(x)+x), x)";
    let (json, stderr) = eval_json_with_args_and_stderr(expr, &["--steps", "on"]);

    assert_eq!(json["ok"], true, "expr: {expr}");
    assert!(
        !stderr.contains("cycle detected") && !stderr.contains("depth_overflow"),
        "negative-log tan/polynomial sqrt diff should stay out of cleanup loops: {stderr}"
    );
    assert!(
        json.get("blocked_hints").is_none(),
        "negative-log direct tan/polynomial sqrt diff should not emit blocked hints: {json:?}"
    );
    assert_eq!(
        json["result"], "(x·cos(x)^2 + x - cos(x)^2) / (2·x·cos(x)^2·sqrt(tan(x) - ln(x) + x))",
        "expr: {expr}"
    );

    let steps = json["steps"].as_array().expect("steps should be an array");
    assert_eq!(
        steps.len(),
        1,
        "negative-log tan/polynomial sqrt diff should use direct derivative presentation: {steps:?}"
    );
    assert_eq!(steps[0]["rule"], "Calcular la derivada");

    let required = json["required_conditions"]
        .as_array()
        .expect("required_conditions should be an array");
    assert!(
        required.iter().any(|condition| {
            condition["kind"] == "Positive" && condition["expr_canonical"] == "x"
        }),
        "negative ln(x) should still expose x > 0 for {expr}: {required:?}"
    );
    assert!(
        required.iter().any(|condition| {
            condition["kind"] == "Positive"
                && condition["expr_canonical"] == "tan(x) - ln(x) + x"
        }),
        "negative-log tan/polynomial radicand should expose the compact positive-domain guard for {expr}: {required:?}"
    );
    assert!(
        required.iter().any(|condition| {
            condition["kind"] == "NonZero" && condition["expr_canonical"] == "cos(x)"
        }),
        "negative-log tan/polynomial radicand should expose the tangent pole guard for {expr}: {required:?}"
    );
}

#[test]
fn test_eval_json_diff_sqrt_tan_sqrt_variable_polynomial_sum_uses_direct_route() {
    let expr = "diff(sqrt(tan(x)+sqrt(x)+x), x)";
    let (json, stderr) = eval_json_with_args_and_stderr(expr, &["--steps", "on"]);

    assert_eq!(json["ok"], true, "expr: {expr}");
    assert!(
        !stderr.contains("cycle detected") && !stderr.contains("depth_overflow"),
        "tan/sqrt-variable/polynomial sqrt diff should stay out of cleanup loops: {stderr}"
    );
    assert!(
        json.get("blocked_hints").is_none(),
        "direct tan/sqrt-variable/polynomial sqrt diff should not emit blocked hints: {json:?}"
    );
    assert_eq!(
        json["result"],
        "(2·sqrt(x) + 2·sqrt(x)·sec(x)^2 + 1) / (4·sqrt(x)·sqrt(tan(x) + sqrt(x) + x))",
        "expr: {expr}"
    );

    let steps = json["steps"].as_array().expect("steps should be an array");
    assert_eq!(
        steps.len(),
        1,
        "tan/sqrt-variable/polynomial sqrt diff should use direct derivative presentation: {steps:?}"
    );
    assert_eq!(steps[0]["rule"], "Calcular la derivada");

    let required = json["required_conditions"]
        .as_array()
        .expect("required_conditions should be an array");
    assert!(
        required.iter().any(|condition| {
            condition["kind"] == "Positive" && condition["expr_canonical"] == "x"
        }),
        "sqrt(x) derivative should expose x > 0 for {expr}: {required:?}"
    );
    assert!(
        required.iter().any(|condition| {
            condition["kind"] == "NonZero" && condition["expr_canonical"] == "cos(x)"
        }),
        "tan/sqrt-variable/polynomial radicand should expose the tangent pole guard for {expr}: {required:?}"
    );
    assert!(
        required.iter().any(|condition| {
            condition["kind"] == "Positive"
                && condition["expr_canonical"] == "tan(x) + sqrt(x) + x"
        }),
        "tan/sqrt-variable/polynomial radicand should expose the compact positive-domain guard for {expr}: {required:?}"
    );
}

#[test]
fn test_eval_json_diff_sqrt_tan_reciprocal_sqrt_polynomial_sum_terminates_directly() {
    for (expr, expected, radicand) in [
        (
            "diff(sqrt(tan(x)+1/sqrt(x)+x), x)",
            "(2·x·sqrt(x) + 2·x·sqrt(x)·sec(x)^2 - 1) / (4·x·sqrt(x)·sqrt(tan(x) + 1 / sqrt(x) + x))",
            "tan(x) + 1 / sqrt(x) + x",
        ),
        (
            "diff(sqrt(tan(x)-1/sqrt(x)+x), x)",
            "(2·x·sqrt(x) + 2·x·sqrt(x)·sec(x)^2 + 1) / (4·x·sqrt(x)·sqrt(tan(x) - 1 / sqrt(x) + x))",
            "tan(x) - 1 / sqrt(x) + x",
        ),
    ] {
        let (json, stderr) = eval_json_with_args_and_stderr(expr, &["--steps", "on"]);

        assert_eq!(json["ok"], true, "expr: {expr}");
        assert!(
            !stderr.contains("cycle detected") && !stderr.contains("depth_overflow"),
            "tan/reciprocal-sqrt/polynomial sqrt diff should stay out of cleanup loops: {stderr}"
        );
        assert!(
            json.get("blocked_hints").is_none(),
            "direct tan/reciprocal-sqrt/polynomial sqrt diff should not emit blocked hints: {json:?}"
        );
        assert_eq!(json["result"], expected, "expr: {expr}");

        let steps = json["steps"].as_array().expect("steps should be an array");
        assert_eq!(
            steps.len(),
            1,
            "tan/reciprocal-sqrt/polynomial sqrt diff should use direct derivative presentation: {steps:?}"
        );
        assert_eq!(steps[0]["rule"], "Calcular la derivada");

        let required = json["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array");
        assert!(
            required.iter().any(|condition| {
                condition["kind"] == "Positive" && condition["expr_canonical"] == "x"
            }),
            "reciprocal sqrt derivative should expose x > 0 for {expr}: {required:?}"
        );
        assert!(
            required.iter().any(|condition| {
                condition["kind"] == "NonZero" && condition["expr_canonical"] == "cos(x)"
            }),
            "tan/reciprocal-sqrt/polynomial radicand should expose the tangent pole guard for {expr}: {required:?}"
        );
        assert!(
            required.iter().any(|condition| {
                condition["kind"] == "Positive" && condition["expr_canonical"] == radicand
            }),
            "tan/reciprocal-sqrt/polynomial radicand should expose the compact positive-domain guard for {expr}: {required:?}"
        );
    }
}

#[test]
fn test_eval_json_diff_sqrt_trig_quadratic_sum_keeps_direct_presentation() {
    let expr = "diff(sqrt(sin(2*x)+cos(x)+x^2), x)";
    let (json, stderr) = eval_json_with_args_and_stderr(expr, &["--steps", "on"]);

    assert_eq!(json["ok"], true, "expr: {expr}");
    assert!(
        !stderr.contains("depth_overflow"),
        "mixed trig/quadratic sqrt diff should avoid the generic depth-overflow route: {stderr}"
    );
    assert_eq!(
        json["result"], "(cos(2·x) + x - 1/2·sin(x)) / sqrt(sin(2·x) + cos(x) + x^2)",
        "expr: {expr}"
    );

    let steps = json["steps"].as_array().expect("steps should be an array");
    assert_eq!(
        steps.len(),
        1,
        "trig/quadratic sqrt diff should stay on the direct presentation route: {steps:?}"
    );
    assert_eq!(steps[0]["rule"], "Calcular la derivada");
    assert!(
        steps.iter().all(|step| !step["rule"]
            .as_str()
            .is_some_and(|rule| rule.contains("ángulo doble"))),
        "trig/quadratic sqrt diff should keep the raw trig radicand: {steps:?}"
    );

    let required = json["required_conditions"]
        .as_array()
        .expect("required_conditions should be an array");
    assert!(
        required.iter().any(|condition| {
            condition["kind"] == "Positive"
                && condition["expr_canonical"] == "sin(2·x) + cos(x) + x^2"
        }),
        "trig/quadratic radicand should expose a positive-domain guard for {expr}: {required:?}"
    );
}

#[test]
fn test_eval_json_diff_sqrt_trig_cubic_sum_keeps_direct_presentation() {
    let expr = "diff(sqrt(sin(2*x)+cos(x)+x^3), x)";
    let (json, stderr) = eval_json_with_args_and_stderr(expr, &["--steps", "on"]);

    assert_eq!(json["ok"], true, "expr: {expr}");
    assert!(
        !stderr.contains("depth_overflow"),
        "mixed trig/cubic sqrt diff should avoid the generic depth-overflow route: {stderr}"
    );
    assert_eq!(
        json["result"], "(cos(2·x) + 3/2·x^2 - 1/2·sin(x)) / sqrt(sin(2·x) + cos(x) + x^3)",
        "expr: {expr}"
    );

    let steps = json["steps"].as_array().expect("steps should be an array");
    assert_eq!(
        steps.len(),
        1,
        "trig/cubic sqrt diff should stay on the direct presentation route: {steps:?}"
    );
    assert_eq!(steps[0]["rule"], "Calcular la derivada");
    assert!(
        steps.iter().all(|step| !step["rule"]
            .as_str()
            .is_some_and(|rule| rule.contains("ángulo doble"))),
        "trig/cubic sqrt diff should keep the raw trig radicand: {steps:?}"
    );

    let required = json["required_conditions"]
        .as_array()
        .expect("required_conditions should be an array");
    assert!(
        required.iter().any(|condition| {
            condition["kind"] == "Positive"
                && condition["expr_canonical"] == "sin(2·x) + cos(x) + x^3"
        }),
        "trig/cubic radicand should expose a positive-domain guard for {expr}: {required:?}"
    );
}

#[test]
fn test_eval_json_diff_sqrt_trig_log_sum_keeps_direct_presentation() {
    let expr = "diff(sqrt(sin(2*x)+cos(x)+ln(x)), x)";
    let (json, stderr) = eval_json_with_args_and_stderr(expr, &["--steps", "on"]);

    assert_eq!(json["ok"], true, "expr: {expr}");
    assert!(
        !stderr.contains("depth_overflow"),
        "mixed trig/log sqrt diff should avoid the generic depth-overflow route: {stderr}"
    );
    assert_eq!(
        json["result"], "(2·x·cos(2·x) + 1 - x·sin(x)) / (2·x·sqrt(sin(2·x) + cos(x) + ln(x)))",
        "expr: {expr}"
    );

    let steps = json["steps"].as_array().expect("steps should be an array");
    assert_eq!(
        steps.len(),
        1,
        "trig/log sqrt diff should stay on the direct presentation route: {steps:?}"
    );
    assert_eq!(steps[0]["rule"], "Calcular la derivada");
    assert!(
        steps.iter().all(|step| !step["rule"]
            .as_str()
            .is_some_and(|rule| rule.contains("ángulo doble"))),
        "trig/log sqrt diff should keep the raw trig radicand: {steps:?}"
    );

    let required = json["required_conditions"]
        .as_array()
        .expect("required_conditions should be an array");
    assert!(
        required.iter().any(|condition| {
            condition["kind"] == "Positive"
                && condition["expr_canonical"] == "sin(2·x) + cos(x) + ln(x)"
        }),
        "trig/log radicand should expose a positive-domain guard for {expr}: {required:?}"
    );
    assert!(
        required
            .iter()
            .any(|condition| condition["kind"] == "Positive" && condition["expr_canonical"] == "x"),
        "ln term should preserve its positive-domain guard for {expr}: {required:?}"
    );
}

#[test]
fn test_eval_json_diff_sqrt_trig_exp_sum_keeps_direct_presentation() {
    let expr = "diff(sqrt(sin(2*x)+cos(x)+exp(x)), x)";
    let (json, stderr) = eval_json_with_args_and_stderr(expr, &["--steps", "on"]);

    assert_eq!(json["ok"], true, "expr: {expr}");
    assert!(
        !stderr.contains("depth_overflow"),
        "mixed trig/exp sqrt diff should avoid the generic depth-overflow route: {stderr}"
    );
    assert_eq!(
        json["result"], "(cos(2·x) + 1/2·e^x - 1/2·sin(x)) / sqrt(sin(2·x) + cos(x) + e^x)",
        "expr: {expr}"
    );

    let steps = json["steps"].as_array().expect("steps should be an array");
    assert_eq!(
        steps.len(),
        1,
        "trig/exp sqrt diff should stay on the direct presentation route: {steps:?}"
    );
    assert_eq!(steps[0]["rule"], "Calcular la derivada");
    assert!(
        steps.iter().all(|step| !step["rule"]
            .as_str()
            .is_some_and(|rule| rule.contains("ángulo doble"))),
        "trig/exp sqrt diff should keep the raw trig radicand: {steps:?}"
    );

    let required = json["required_conditions"]
        .as_array()
        .expect("required_conditions should be an array");
    assert!(
        required.iter().any(|condition| {
            condition["kind"] == "Positive"
                && condition["expr_canonical"] == "sin(2·x) + cos(x) + e^x"
        }),
        "trig/exp radicand should expose a positive-domain guard for {expr}: {required:?}"
    );
}

#[test]
fn test_eval_json_diff_sqrt_trig_sqrt_var_sum_keeps_direct_presentation() {
    let expr = "diff(sqrt(sin(2*x)+cos(x)+sqrt(x)), x)";
    let (json, stderr) = eval_json_with_args_and_stderr(expr, &["--steps", "on"]);

    assert_eq!(json["ok"], true, "expr: {expr}");
    assert!(
        !stderr.contains("depth_overflow"),
        "mixed trig/sqrt-variable root diff should avoid the generic depth-overflow route: {stderr}"
    );
    assert_eq!(
        json["result"],
        "(cos(2·x) + 1 / (4·sqrt(x)) - 1/2·sin(x)) / sqrt(sin(2·x) + cos(x) + sqrt(x))",
        "expr: {expr}"
    );

    let steps = json["steps"].as_array().expect("steps should be an array");
    assert_eq!(
        steps.len(),
        1,
        "trig/sqrt-variable root diff should stay on the direct presentation route: {steps:?}"
    );
    assert_eq!(steps[0]["rule"], "Calcular la derivada");
    assert!(
        steps.iter().all(|step| !step["rule"]
            .as_str()
            .is_some_and(|rule| rule.contains("ángulo doble"))),
        "trig/sqrt-variable root diff should keep the raw trig radicand: {steps:?}"
    );

    let required = json["required_conditions"]
        .as_array()
        .expect("required_conditions should be an array");
    assert!(
        required.iter().any(|condition| {
            condition["kind"] == "Positive"
                && condition["expr_canonical"] == "sin(2·x) + cos(x) + sqrt(x)"
        }),
        "trig/sqrt-variable radicand should expose a positive-domain guard for {expr}: {required:?}"
    );
    assert!(
        required
            .iter()
            .any(|condition| condition["kind"] == "Positive" && condition["expr_canonical"] == "x"),
        "sqrt-variable term should preserve the derivative-domain guard for {expr}: {required:?}"
    );
}

#[test]
fn test_eval_json_diff_sqrt_trig_negative_sqrt_var_sum_keeps_direct_presentation() {
    let expr = "diff(sqrt(sin(2*x)+cos(x)-sqrt(x)), x)";
    let (json, stderr) = eval_json_with_args_and_stderr(expr, &["--steps", "on"]);

    assert_eq!(json["ok"], true, "expr: {expr}");
    assert!(
        !stderr.contains("depth_overflow"),
        "mixed trig/negative-sqrt-variable root diff should avoid the generic depth-overflow route: {stderr}"
    );
    assert_eq!(
        json["result"],
        "(cos(2·x) - 1/2·sin(x) - 1 / (4·sqrt(x))) / sqrt(sin(2·x) + cos(x) - sqrt(x))",
        "expr: {expr}"
    );
    assert!(
        json["result_latex"]
            .as_str()
            .is_some_and(|latex| latex.contains("\\frac{1}{4\\cdot \\sqrt{x}}")),
        "negative reciprocal sqrt term should stay compact in LaTeX too: {}",
        json["result_latex"]
    );

    let steps = json["steps"].as_array().expect("steps should be an array");
    assert_eq!(
        steps.len(),
        1,
        "trig/negative-sqrt-variable root diff should stay on the direct presentation route: {steps:?}"
    );
    assert_eq!(steps[0]["rule"], "Calcular la derivada");

    let required = json["required_conditions"]
        .as_array()
        .expect("required_conditions should be an array");
    assert!(
        required.iter().any(|condition| {
            condition["kind"] == "Positive"
                && condition["expr_canonical"] == "sin(2·x) + cos(x) - sqrt(x)"
        }),
        "trig/negative-sqrt-variable radicand should expose a positive-domain guard for {expr}: {required:?}"
    );
    assert!(
        required
            .iter()
            .any(|condition| condition["kind"] == "Positive" && condition["expr_canonical"] == "x"),
        "negative sqrt-variable term should preserve the derivative-domain guard for {expr}: {required:?}"
    );
}

#[test]
fn test_eval_json_diff_sqrt_trig_scaled_sqrt_var_sum_keeps_direct_presentation() {
    for (expr, expected_result, radicand) in [
        (
            "diff(sqrt(sin(2*x)+cos(x)+2*sqrt(x)), x)",
            "(cos(2·x) + 1 / (2·sqrt(x)) - 1/2·sin(x)) / sqrt(sin(2·x) + cos(x) + 2·sqrt(x))",
            "sin(2·x) + cos(x) + 2·sqrt(x)",
        ),
        (
            "diff(sqrt(sin(2*x)+cos(x)-2*sqrt(x)), x)",
            "(cos(2·x) - 1/2·sin(x) - 1 / (2·sqrt(x))) / sqrt(sin(2·x) + cos(x) - 2·sqrt(x))",
            "sin(2·x) + cos(x) - 2·sqrt(x)",
        ),
    ] {
        let (json, stderr) = eval_json_with_args_and_stderr(expr, &["--steps", "on"]);

        assert_eq!(json["ok"], true, "expr: {expr}");
        assert!(
            !stderr.contains("depth_overflow"),
            "scaled mixed trig/sqrt-variable root diff should avoid depth overflow for {expr}: {stderr}"
        );
        assert_eq!(json["result"], expected_result, "expr: {expr}");

        let steps = json["steps"].as_array().expect("steps should be an array");
        assert_eq!(
            steps.len(),
            1,
            "scaled trig/sqrt-variable root diff should stay on the direct route for {expr}: {steps:?}"
        );
        assert_eq!(steps[0]["rule"], "Calcular la derivada");
        assert!(
            steps.iter().all(|step| !step["rule"]
                .as_str()
                .is_some_and(|rule| rule.contains("ángulo doble"))),
            "scaled trig/sqrt-variable root diff should keep the raw trig radicand for {expr}: {steps:?}"
        );

        let required = json["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array");
        assert!(
            required.iter().any(|condition| {
                condition["kind"] == "Positive" && condition["expr_canonical"] == radicand
            }),
            "scaled trig/sqrt-variable radicand should expose a positive-domain guard for {expr}: {required:?}"
        );
        assert!(
            required.iter().any(
                |condition| condition["kind"] == "Positive" && condition["expr_canonical"] == "x"
            ),
            "scaled sqrt-variable term should preserve the derivative-domain guard for {expr}: {required:?}"
        );
    }
}

#[test]
fn test_eval_json_diff_sqrt_trig_scaled_elementary_sum_keeps_direct_presentation() {
    for (expr, expected_result, radicand, extra_condition) in [
        (
            "diff(sqrt(sin(2*x)+cos(x)+2*ln(x)), x)",
            "(2·x·cos(2·x) + 2 - x·sin(x)) / (2·x·sqrt(sin(2·x) + cos(x) + 2·ln(x)))",
            "sin(2·x) + cos(x) + 2·ln(x)",
            Some("x"),
        ),
        (
            "diff(sqrt(sin(2*x)+cos(x)-3*ln(x)), x)",
            "(2·x·cos(2·x) - x·sin(x) - 3) / (2·x·sqrt(sin(2·x) + cos(x) - 3·ln(x)))",
            "sin(2·x) + cos(x) - 3·ln(x)",
            Some("x"),
        ),
        (
            "diff(sqrt(sin(2*x)+cos(x)+2*exp(x)), x)",
            "(cos(2·x) + e^x - 1/2·sin(x)) / sqrt(sin(2·x) + cos(x) + 2·e^x)",
            "sin(2·x) + cos(x) + 2·e^x",
            None,
        ),
        (
            "diff(sqrt(sin(2*x)+cos(x)-3*exp(x)), x)",
            "(cos(2·x) - 3/2·e^x - 1/2·sin(x)) / sqrt(sin(2·x) + cos(x) - 3·e^x)",
            "sin(2·x) + cos(x) - 3·e^x",
            None,
        ),
    ] {
        let (json, stderr) = eval_json_with_args_and_stderr(expr, &["--steps", "on"]);

        assert_eq!(json["ok"], true, "expr: {expr}");
        assert!(
            !stderr.contains("depth_overflow"),
            "scaled elementary trig-root diff should avoid depth overflow for {expr}: {stderr}"
        );
        assert_eq!(json["result"], expected_result, "expr: {expr}");

        let steps = json["steps"].as_array().expect("steps should be an array");
        assert_eq!(
            steps.len(),
            1,
            "scaled elementary trig-root diff should stay on the direct route for {expr}: {steps:?}"
        );
        assert_eq!(steps[0]["rule"], "Calcular la derivada");
        assert!(
            steps.iter().all(|step| !step["rule"]
                .as_str()
                .is_some_and(|rule| rule.contains("ángulo doble"))),
            "scaled elementary trig-root diff should keep the raw trig radicand for {expr}: {steps:?}"
        );

        let required = json["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array");
        assert!(
            required.iter().any(|condition| {
                condition["kind"] == "Positive" && condition["expr_canonical"] == radicand
            }),
            "scaled elementary radicand should expose a positive-domain guard for {expr}: {required:?}"
        );
        if let Some(extra_condition) = extra_condition {
            assert!(
                required.iter().any(|condition| {
                    condition["kind"] == "Positive"
                        && condition["expr_canonical"] == extra_condition
                }),
                "scaled logarithmic term should preserve its derivative-domain guard for {expr}: {required:?}"
            );
        }
    }
}

#[test]
fn test_eval_json_diff_sqrt_polynomial_quotient_power_presentation_cancels_common_factor() {
    let expr = "diff(sqrt(x)/(x+1)^2, x)";
    let json = eval_json(expr);

    assert_eq!(json["ok"], true, "expr: {expr}");
    assert_eq!(json["result"], "(1 - 3·x) / (2·sqrt(x)·(x + 1)^3)");

    let residual = eval_json("diff(sqrt(x)/(x+1)^2, x) - (1-3*x)/(2*sqrt(x)*(x+1)^3)");
    assert_eq!(residual["ok"], true, "residual for {expr}");
    assert_eq!(residual["result"], "0", "residual for {expr}");

    let required = json["required_conditions"]
        .as_array()
        .expect("required_conditions should be an array");
    assert!(
        required.iter().any(|condition| {
            condition["kind"] == "Positive" && condition["expr_canonical"] == "x"
        }),
        "presentation should preserve the sqrt-domain guard for {expr}: {required:?}"
    );
}

#[test]
fn test_eval_json_diff_sqrt_linear_quotient_post_calculus_step_avoids_double_power_parens() {
    let expr = "diff(sqrt(x)/(x+1), x)";
    let json = eval_json_with_args(expr, &["--steps", "on"]);

    assert_eq!(json["ok"], true, "expr: {expr}");
    assert_eq!(json["result"], "(1 - x) / (2·sqrt(x)·(x + 1)^2)");

    let steps = json["steps"].as_array().expect("steps should be an array");
    let presentation_step = steps
        .iter()
        .find(|step| step["rule"] == "Presentar resultado de cálculo en forma compacta")
        .expect("expected post-calculus presentation step");
    let after = presentation_step["after"].as_str().expect("after string");
    assert!(
        after.contains("(x + 1)^2") && !after.contains("((x + 1))^2"),
        "post-calculus presentation step should not double-parenthesize power bases for {expr}, got: {after}"
    );
}

#[test]
fn test_eval_json_diff_sqrt_expanded_affine_square_quotient_presentation_cancels_common_factor() {
    let expr = "diff(sqrt(x)/(x^2+2*x+1), x)";
    let json = eval_json(expr);

    assert_eq!(json["ok"], true, "expr: {expr}");
    assert_eq!(json["result"], "(1 - 3·x) / (2·sqrt(x)·(x + 1)^3)");

    let residual = eval_json("diff(sqrt(x)/(x^2+2*x+1), x) - (1-3*x)/(2*sqrt(x)*(x+1)^3)");
    assert_eq!(residual["ok"], true, "residual for {expr}");
    assert_eq!(residual["result"], "0", "residual for {expr}");

    let required = json["required_conditions"]
        .as_array()
        .expect("required_conditions should be an array");
    assert!(
        required.iter().any(|condition| {
            condition["kind"] == "Positive" && condition["expr_canonical"] == "x"
        }),
        "presentation should preserve the sqrt-domain guard for {expr}: {required:?}"
    );
}

#[test]
fn test_eval_json_diff_sqrt_power_of_expanded_affine_square_quotient_presentation_cancels_common_factor(
) {
    let expr = "diff(sqrt(x)/(x^2+2*x+1)^2, x)";
    let json = eval_json(expr);

    assert_eq!(json["ok"], true, "expr: {expr}");
    assert_eq!(json["result"], "(1 - 7·x) / (2·sqrt(x)·(x + 1)^5)");

    let residual = eval_json("diff(sqrt(x)/(x^2+2*x+1)^2, x) - (1-7*x)/(2*sqrt(x)*(x+1)^5)");
    assert_eq!(residual["ok"], true, "residual for {expr}");
    assert_eq!(residual["result"], "0", "residual for {expr}");

    let required = json["required_conditions"]
        .as_array()
        .expect("required_conditions should be an array");
    assert!(
        required.iter().any(|condition| {
            condition["kind"] == "Positive" && condition["expr_canonical"] == "x"
        }),
        "presentation should preserve the sqrt-domain guard for {expr}: {required:?}"
    );

    let steps_on = eval_json_with_args(expr, &["--steps", "on"]);
    assert_eq!(steps_on["ok"], true, "steps-on expr: {expr}");
    assert_eq!(steps_on["result"], json["result"]);
    assert!(
        steps_on["steps_count"]
            .as_u64()
            .is_some_and(|count| (2..=4).contains(&count)),
        "steps-on mode should use the bounded derivative trace for {expr}: {steps_on:?}"
    );
    let steps = steps_on["steps"]
        .as_array()
        .expect("steps should be an array");
    assert!(
        steps
            .iter()
            .any(|step| step["rule"] == "Calcular la derivada"),
        "steps-on mode should still expose the derivative step for {expr}: {steps:?}"
    );
    assert!(
        steps
            .iter()
            .any(|step| step["rule"] == "Presentar resultado de cálculo en forma compacta"),
        "steps-on mode should expose bounded post-calculus presentation for {expr}: {steps:?}"
    );
}

#[test]
fn test_eval_json_diff_sqrt_scaled_expanded_affine_square_quotient_presentation_cancels_common_factor(
) {
    let expr = "diff(sqrt(x)/(4*x^2+4*x+1), x)";
    let json = eval_json(expr);

    assert_eq!(json["ok"], true, "expr: {expr}");
    assert_eq!(json["result"], "(1 - 6·x) / (2·sqrt(x)·(2·x + 1)^3)");

    let residual = eval_json("diff(sqrt(x)/(4*x^2+4*x+1), x) - (1-6*x)/(2*sqrt(x)*(2*x+1)^3)");
    assert_eq!(residual["ok"], true, "residual for {expr}");
    assert_eq!(residual["result"], "0", "residual for {expr}");

    let required = json["required_conditions"]
        .as_array()
        .expect("required_conditions should be an array");
    assert!(
        required.iter().any(|condition| {
            condition["kind"] == "Positive" && condition["expr_canonical"] == "x"
        }),
        "presentation should preserve the sqrt-domain guard for {expr}: {required:?}"
    );
}

#[test]
fn test_eval_json_diff_sqrt_of_polynomial_quotient_presents_reciprocal_sqrt_denominator() {
    let expr = "diff(sqrt((x+1)/(x+2)), x)";
    let json = eval_json(expr);

    assert_eq!(json["ok"], true, "expr: {expr}");
    assert_eq!(json["result"], "1 / (2·sqrt((x + 1) / (x + 2))·(x + 2)^2)");
    assert!(
        !json["result"]
            .as_str()
            .expect("result should be a string")
            .contains("^(-1/2)"),
        "post-calculus presentation should avoid reciprocal half-power notation: {json:?}"
    );

    let residual = eval_json("diff(sqrt((x+1)/(x+2)), x) - 1/(2*sqrt((x+1)/(x+2))*(x+2)^2)");
    assert_eq!(residual["ok"], true, "residual for {expr}");
    assert_eq!(residual["result"], "0", "residual for {expr}");

    let required = json["required_conditions"]
        .as_array()
        .expect("required_conditions should be an array");
    assert!(
        required.iter().any(|condition| {
            condition["kind"] == "Positive" && condition["expr_canonical"] == "x^2 + 3·x + 2"
        }),
        "presentation should preserve the quotient sqrt-domain guard for {expr}: {required:?}"
    );
    assert_eq!(
        json["required_display"]
            .as_array()
            .expect("required_display should be an array"),
        &[serde_json::json!("x < -2 or x > -1")]
    );

    let steps_on = eval_json_with_args(expr, &["--steps", "on"]);
    assert_eq!(steps_on["ok"], true, "steps-on expr: {expr}");
    assert_eq!(steps_on["result"], json["result"]);
    let steps = steps_on["steps"]
        .as_array()
        .expect("steps should be an array");
    assert!(
        steps
            .iter()
            .any(|step| step["rule"] == "Presentar resultado de cálculo en forma compacta"),
        "steps-on mode should expose the post-calculus presentation recovery for {expr}: {steps:?}"
    );
}

#[test]
fn test_eval_json_diff_sqrt_of_power_denominator_quotient_cancels_partial_base_factor() {
    let expr = "diff(sqrt((x+1)/(x+2)^2), x)";
    let json = eval_json(expr);

    assert_eq!(json["ok"], true, "expr: {expr}");
    assert_eq!(
        json["result"],
        "-x / (2·sqrt((x + 1) / (x + 2)^2)·(x + 2)^3)"
    );
    assert!(
        !json["result"]
            .as_str()
            .expect("result should be a string")
            .contains("^2^2"),
        "post-calculus presentation should not render nested power text: {json:?}"
    );

    let residual = eval_json("diff(sqrt((x+1)/(x+2)^2), x) + x/(2*sqrt((x+1)/(x+2)^2)*(x+2)^3)");
    assert_eq!(residual["ok"], true, "residual for {expr}");
    assert_eq!(residual["result"], "0", "residual for {expr}");

    let required = json["required_conditions"]
        .as_array()
        .expect("required_conditions should be an array");
    assert!(
        required.iter().any(|condition| {
            condition["kind"] == "Positive" && condition["expr_canonical"] == "x + 1"
        }),
        "presentation should preserve the quotient sqrt-domain guard for {expr}: {required:?}"
    );
    assert_eq!(
        json["required_display"]
            .as_array()
            .expect("required_display should be an array"),
        &[serde_json::json!("x > -1")]
    );

    let steps_on = eval_json_with_args(expr, &["--steps", "on"]);
    assert_eq!(steps_on["ok"], true, "steps-on expr: {expr}");
    assert_eq!(steps_on["result"], json["result"]);
    let steps = steps_on["steps"]
        .as_array()
        .expect("steps should be an array");
    assert!(
        steps
            .iter()
            .any(|step| step["rule"] == "Presentar resultado de cálculo en forma compacta"),
        "steps-on mode should expose post-calculus recovery after intermediate simplification for {expr}: {steps:?}"
    );
}

#[test]
fn test_eval_json_diff_sqrt_polynomial_over_same_polynomial_cancels_denominator_factor() {
    let positive = eval_json("diff(sqrt(x^2+1)/(x^2+1), x)");
    assert_eq!(positive["ok"], true);
    assert_eq!(positive["result"], "-x / (sqrt(x^2 + 1)·(x^2 + 1))");
    assert!(
        positive["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .is_empty(),
        "globally positive radicand should not add conditions: {positive:?}"
    );
    let positive_residual = eval_json("diff(sqrt(x^2+1)/(x^2+1), x) + x/(sqrt(x^2+1)*(x^2+1))");
    assert_eq!(positive_residual["ok"], true);
    assert_eq!(positive_residual["result"], "0");

    let conditional = eval_json("diff(sqrt(x^2-1)/(x^2-1), x)");
    assert_eq!(conditional["ok"], true);
    assert_eq!(conditional["result"], "-x / (sqrt(x^2 - 1)·(x^2 - 1))");
    assert!(
        conditional["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .iter()
            .any(|condition| {
                condition["kind"] == "Positive" && condition["expr_canonical"] == "x^2 - 1"
            }),
        "conditional radicand should preserve its domain guard: {conditional:?}"
    );
    let conditional_residual = eval_json("diff(sqrt(x^2-1)/(x^2-1), x) + x/(sqrt(x^2-1)*(x^2-1))");
    assert_eq!(conditional_residual["ok"], true);
    assert_eq!(conditional_residual["result"], "0");
}

#[test]
fn test_eval_json_diff_sqrt_scaled_expanded_affine_square_keeps_steps_on_didactic_trace() {
    let expr = "diff(sqrt(x)/(4*x^2+4*x+1), x)";
    let json = eval_json_with_args(expr, &["--steps", "on"]);

    assert_eq!(json["ok"], true, "expr: {expr}");
    assert_eq!(json["result"], "(1 - 6·x) / (2·sqrt(x)·(2·x + 1)^3)");
    assert!(
        json["steps_count"].as_u64().is_some_and(|count| count > 1),
        "steps-on mode should keep the explanatory derivative trace for {expr}: {json:?}"
    );

    let steps = json["steps"].as_array().expect("steps should be an array");
    assert!(
        steps
            .iter()
            .any(|step| step["rule"] == "Presentar resultado de cálculo en forma compacta"),
        "steps-on mode should still show post-calculus presentation for {expr}: {steps:?}"
    );
}

#[test]
fn test_eval_json_diff_reciprocal_sqrt_polynomial_power_presentation_cancels_common_factor() {
    let expr = "diff(1/(sqrt(x)*(x+1)^2), x)";
    let json = eval_json(expr);

    assert_eq!(json["ok"], true, "expr: {expr}");
    assert_eq!(json["result"], "-(5·x + 1) / (2·x·sqrt(x)·(x + 1)^3)");

    let residual = eval_json("diff(1/(sqrt(x)*(x+1)^2), x) + (5*x+1)/(2*x*sqrt(x)*(x+1)^3)");
    assert_eq!(residual["ok"], true, "residual for {expr}");
    assert_eq!(residual["result"], "0", "residual for {expr}");

    let required = json["required_conditions"]
        .as_array()
        .expect("required_conditions should be an array");
    assert!(
        required.iter().any(|condition| {
            condition["kind"] == "Positive" && condition["expr_canonical"] == "x"
        }),
        "presentation should preserve the sqrt-domain guard for {expr}: {required:?}"
    );
}

#[test]
fn test_eval_json_diff_arctan_sqrt_plus_rational_residual_collapses_without_depth_overflow() {
    for expr in [
        "diff(arctan(sqrt(x)) + sqrt(x)/(x+1), x) - 1/(sqrt(x)*(x+1)^2)",
        "diff(2*arctan(sqrt(x)) + 2*sqrt(x)/(x+1), x) - 2/(sqrt(x)*(x+1)^2)",
    ] {
        let (json, stderr) = eval_json_with_stderr(expr);

        assert_eq!(json["ok"], true, "expr: {expr}");
        assert_eq!(json["result"], "0", "expr: {expr}");
        assert!(
            !stderr.contains("depth_overflow"),
            "arctan-sqrt residual should not emit depth_overflow for {expr}\nstderr:\n{stderr}"
        );

        let required = json["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array");
        assert!(
            required.iter().any(|condition| {
                condition["kind"] == "Positive" && condition["expr_canonical"] == "x"
            }),
            "residual shortcut should preserve the sqrt-domain guard for {expr}: {required:?}"
        );
    }
}

#[test]
fn test_eval_json_diff_reciprocal_sqrt_polynomial_power_keeps_steps_on_trace() {
    let expr = "diff(1/(sqrt(x)*(x+1)^2), x)";
    let json = eval_json_with_args(expr, &["--steps", "on"]);

    assert_eq!(json["ok"], true, "expr: {expr}");
    assert_eq!(json["result"], "-(5·x + 1) / (2·x·sqrt(x)·(x + 1)^3)");
    assert!(
        json["steps_count"].as_u64().is_some_and(|count| count >= 1),
        "steps-on mode should keep an explanatory derivative trace for {expr}: {json:?}"
    );

    let required = json["required_conditions"]
        .as_array()
        .expect("required_conditions should be an array");
    assert!(
        required.iter().any(|condition| {
            condition["kind"] == "Positive" && condition["expr_canonical"] == "x"
        }),
        "steps-on mode should preserve the sqrt-domain guard for {expr}: {required:?}"
    );
}

#[test]
fn test_eval_json_diff_polynomial_over_positive_quadratic_sqrt_presents_compact_fraction() {
    let expr = "diff(x/sqrt(x^2+1), x)";
    let json = eval_json(expr);

    assert_eq!(json["ok"], true, "expr: {expr}");
    assert_eq!(json["result"], "1 / ((x^2 + 1)·sqrt(x^2 + 1))");
    assert!(
        json["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .is_empty(),
        "strictly positive quadratic radicand should not add domain guards for {expr}: {json:?}"
    );

    let residual = eval_json("diff(x/sqrt(x^2+1), x) - 1/(sqrt(x^2+1)*(x^2+1))");
    assert_eq!(residual["ok"], true, "residual for {expr}");
    assert_eq!(residual["result"], "0", "residual for {expr}");

    let steps_on = eval_json_with_args(expr, &["--steps", "on"]);
    assert_eq!(steps_on["ok"], true, "steps-on expr: {expr}");
    assert_eq!(steps_on["result"], "1 / ((x^2 + 1)·sqrt(x^2 + 1))");
    assert!(
        steps_on["steps_count"]
            .as_u64()
            .is_some_and(|count| count >= 2),
        "steps-on mode should expose derivative plus compact presentation for {expr}: {steps_on:?}"
    );
    let steps = steps_on["steps"]
        .as_array()
        .expect("steps should be an array");
    assert!(
        steps
            .iter()
            .any(|step| step["rule"] == "Presentar resultado de cálculo en forma compacta"),
        "steps-on mode should show post-calculus presentation for {expr}: {steps:?}"
    );
}

#[test]
fn test_eval_json_diff_polynomial_over_positive_even_sqrt_presents_compact_fraction() {
    let expr = "diff(x/sqrt(x^4+1), x)";
    let json = eval_json(expr);

    assert_eq!(json["ok"], true, "expr: {expr}");
    assert_eq!(json["result"], "(1 - x^4) / ((x^4 + 1)·sqrt(x^4 + 1))");
    assert!(
        json["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .is_empty(),
        "strictly positive even radicand should not add domain guards for {expr}: {json:?}"
    );

    let residual = eval_json("diff(x/sqrt(x^4+1), x) - (1-x^4)/(sqrt(x^4+1)*(x^4+1))");
    assert_eq!(residual["ok"], true, "residual for {expr}");
    assert_eq!(residual["result"], "0", "residual for {expr}");

    let steps_on = eval_json_with_args(expr, &["--steps", "on"]);
    assert_eq!(steps_on["ok"], true, "steps-on expr: {expr}");
    assert_eq!(steps_on["result"], "(1 - x^4) / ((x^4 + 1)·sqrt(x^4 + 1))");
    let steps = steps_on["steps"]
        .as_array()
        .expect("steps should be an array");
    assert!(
        steps
            .iter()
            .any(|step| step["rule"] == "Presentar resultado de cálculo en forma compacta"),
        "steps-on mode should show post-calculus presentation for {expr}: {steps:?}"
    );

    let non_positive = eval_json("diff(x/sqrt(x^4-x^2), x)");
    assert_eq!(non_positive["ok"], true, "non-positive radicand case");
    assert_eq!(
        non_positive["result"],
        "-x^4 / ((x^4 - x^2)·sqrt(x^4 - x^2))"
    );
    assert!(
        non_positive["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .iter()
            .any(|condition| {
                condition["kind"] == "Positive" && condition["expr_canonical"] == "x^4 - x^2"
            }),
        "non-positive radicand should preserve its domain guard: {non_positive:?}"
    );

    let conditional = eval_json("diff(x/sqrt(x^2-1), x)");
    assert_eq!(conditional["ok"], true, "conditional radicand case");
    assert_eq!(conditional["result"], "-1 / ((x^2 - 1)·sqrt(x^2 - 1))");
    assert!(
        conditional["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .iter()
            .any(|condition| {
                condition["kind"] == "Positive" && condition["expr_canonical"] == "x^2 - 1"
            }),
        "conditional radicand should preserve its domain guard: {conditional:?}"
    );

    let conditional_residual = eval_json("diff(x/sqrt(x^2-1), x) + 1/(sqrt(x^2-1)*(x^2-1))");
    assert_eq!(conditional_residual["ok"], true, "conditional residual");
    assert_eq!(conditional_residual["result"], "0", "conditional residual");

    let conditional_steps_on = eval_json_with_args("diff(x/sqrt(x^2-1), x)", &["--steps", "on"]);
    assert_eq!(conditional_steps_on["ok"], true, "conditional steps-on");
    assert_eq!(
        conditional_steps_on["result"],
        "-1 / ((x^2 - 1)·sqrt(x^2 - 1))"
    );
    assert!(
        conditional_steps_on["steps_count"]
            .as_u64()
            .is_some_and(|count| count >= 2),
        "steps-on mode should keep derivative plus presentation trace: {conditional_steps_on:?}"
    );
    assert!(
        conditional_steps_on["steps"]
            .as_array()
            .expect("steps should be an array")
            .iter()
            .any(|step| step["rule"] == "Presentar resultado de cálculo en forma compacta"),
        "steps-on mode should still show post-calculus presentation: {conditional_steps_on:?}"
    );
}

#[test]
fn test_eval_json_diff_polynomial_over_its_sqrt_cancels_radicand_factor() {
    let positive = eval_json("diff((x^2+1)/sqrt(x^2+1), x)");
    assert_eq!(positive["ok"], true);
    assert_eq!(positive["result"], "x / sqrt(x^2 + 1)");
    assert!(
        positive["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .is_empty(),
        "globally positive radicand should not add conditions: {positive:?}"
    );
    let positive_residual = eval_json("diff((x^2+1)/sqrt(x^2+1), x) - x/sqrt(x^2+1)");
    assert_eq!(positive_residual["ok"], true);
    assert_eq!(positive_residual["result"], "0");

    let conditional = eval_json("diff((x^2-1)/sqrt(x^2-1), x)");
    assert_eq!(conditional["ok"], true);
    assert_eq!(conditional["result"], "x / sqrt(x^2 - 1)");
    assert!(
        conditional["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .iter()
            .any(|condition| {
                condition["kind"] == "Positive" && condition["expr_canonical"] == "x^2 - 1"
            }),
        "conditional radicand should preserve its domain guard: {conditional:?}"
    );
    let conditional_residual = eval_json("diff((x^2-1)/sqrt(x^2-1), x) - x/sqrt(x^2-1)");
    assert_eq!(conditional_residual["ok"], true);
    assert_eq!(conditional_residual["result"], "0");
}

#[test]
fn test_eval_json_diff_polynomial_square_over_its_sqrt_lifts_radicand_to_sqrt() {
    let linear = eval_json("diff((x+1)^2/sqrt(x+1), x)");
    assert_eq!(linear["ok"], true);
    assert_eq!(linear["result"], "3·sqrt(x + 1) / (2)");
    assert!(
        linear["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .iter()
            .any(|condition| {
                condition["kind"] == "Positive" && condition["expr_canonical"] == "x + 1"
            }),
        "linear radicand should preserve its domain guard: {linear:?}"
    );
    let linear_residual = eval_json("diff((x+1)^2/sqrt(x+1), x) - 3*sqrt(x+1)/2");
    assert_eq!(linear_residual["ok"], true);
    assert_eq!(linear_residual["result"], "0");

    let scaled = eval_json("diff((2*x+3)^2/sqrt(2*x+3), x)");
    assert_eq!(scaled["ok"], true);
    assert_eq!(scaled["result"], "3·sqrt(2·x + 3)");
    assert!(
        scaled["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .iter()
            .any(|condition| {
                condition["kind"] == "Positive" && condition["expr_canonical"] == "2·x + 3"
            }),
        "scaled radicand should preserve its domain guard: {scaled:?}"
    );
    let scaled_residual = eval_json("diff((2*x+3)^2/sqrt(2*x+3), x) - 3*sqrt(2*x+3)");
    assert_eq!(scaled_residual["ok"], true);
    assert_eq!(scaled_residual["result"], "0");
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
