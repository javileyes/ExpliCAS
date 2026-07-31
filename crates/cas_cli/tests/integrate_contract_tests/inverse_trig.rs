use super::*;

#[test]
fn integrate_contract_reciprocal_shifted_arcsin_residual_keeps_radical_domain_requires() {
    let input = "1/((diff(integrate(1/sqrt(1-x^2), x), x) - 1/sqrt(1-x^2)) + x + 2) - 1/(x+2)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert!(
        stderr.is_empty(),
        "unexpected stderr for reciprocal shifted arcsin residual: {stderr}"
    );
    assert_eq!(wire["result"], "0");
    assert_eq!(
        wire["required_display"],
        serde_json::json!(["-1 < x < 1", "x ≠ -2"])
    );
}
#[test]
fn integrate_contract_reciprocal_shifted_affine_arcsin_residual_keeps_radical_domain_requires() {
    let input =
        "1/((diff(integrate(1/sqrt(4-(x+1)^2), x), x) - 1/sqrt(4-(x+1)^2)) + x + 2) - 1/(x+2)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert!(
        stderr.is_empty(),
        "unexpected stderr for reciprocal shifted affine arcsin residual: {stderr}"
    );
    assert_eq!(wire["result"], "0");
    assert_eq!(
        wire["required_display"],
        serde_json::json!(["-3 < x < 1", "x ≠ -2"])
    );
}
#[test]
fn integrate_contract_reciprocal_shifted_arctan_sqrt_residual_keeps_positive_domain_requires() {
    let input =
        "1/((diff(integrate(1/(sqrt(x)*(x+1)), x), x) - 1/(sqrt(x)*(x+1))) + x + 2) - 1/(x+2)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert!(
        stderr.is_empty(),
        "unexpected stderr for reciprocal shifted arctan sqrt residual: {stderr}"
    );
    assert_eq!(wire["result"], "0");
    assert_eq!(wire["required_display"], serde_json::json!(["x > 0"]));
}
#[test]
fn integrate_contract_reciprocal_shifted_asinh_residual_compacts_without_timeout() {
    let input =
        "1/((diff(integrate(1/sqrt(4+(x+1)^2), x), x) - 1/sqrt(4+(x+1)^2)) + x + 2) - 1/(x+2)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert!(
        stderr.is_empty(),
        "unexpected stderr for reciprocal shifted asinh residual: {stderr}"
    );
    assert_eq!(wire["result"], "0");
    assert_eq!(wire["required_display"], serde_json::json!(["x ≠ -2"]));
}
#[test]
fn integrate_contract_shifted_quotient_asinh_residual_compacts_without_timeout() {
    let input =
        "1 - (x+2)/((diff(integrate(1/sqrt(4+(x+1)^2), x), x) - 1/sqrt(4+(x+1)^2)) + x + 2)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert!(
        stderr.is_empty(),
        "unexpected stderr for shifted quotient asinh residual: {stderr}"
    );
    assert_eq!(wire["result"], "0");
    assert_eq!(wire["required_display"], serde_json::json!(["x ≠ -2"]));
}
#[test]
fn integrate_contract_acosh_root_product_uses_compact_unit_offset_affine_arg() {
    for (input, expected_result) in [
        ("integrate(1/(sqrt(x)*sqrt(x+3)), x)", "acosh(2/3 * x + 1)"),
        ("integrate(1/(sqrt(x)*sqrt(x+5)), x)", "acosh(2/5 * x + 1)"),
        (
            "integrate(1/(sqrt(x)*sqrt(2*x+4)), x)",
            "acosh(x + 1) / sqrt(2)",
        ),
    ] {
        let (result, required) = evaluated_integral_with_required_conditions(input);

        assert_eq!(result, expected_result, "input: {input}");
        assert!(
            !result.contains("*(x +") && !result.contains("* (x +"),
            "post-calculus presentation should not leave a scaled shifted group in acosh: {result}"
        );
        assert_eq!(
            required,
            vec!["x > 0".to_string()],
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert_antiderivative_verifies(input);
        assert_rendered_antiderivative_verifies(input, &result);

        let residual = format!(
            "diff({result}, x) - {}",
            input
                .strip_prefix("integrate(")
                .and_then(|rest| rest.strip_suffix(", x)"))
                .expect("test input should be explicit integrate(expr, x)")
        );
        let (residual_result, residual_required) =
            evaluated_expr_with_required_conditions(&residual);
        assert_eq!(residual_result, "0", "residual: {residual}");
        assert_eq!(
            residual_required,
            vec!["x > 0".to_string()],
            "unexpected residual required_conditions for {input}: {residual_required:?}"
        );
    }
}
#[test]
fn integrate_contract_linear_numerator_over_positive_quadratic_decomposes_to_log_plus_arctan() {
    let input = "integrate((x+1)/(x^2+1), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "arctan(x) + 1/2 * ln(x^2 + 1)");
    assert!(
        required.is_empty(),
        "positive quadratic denominator should not add synthetic required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);
    assert!(
        stderr.is_empty(),
        "positive quadratic linear-numerator trace should stay quiet\nstderr:\n{stderr}"
    );
    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    let integration_step = steps
        .iter()
        .find(|step| step["rule"] == "Calcular la integral")
        .expect("expected public symbolic integration step");
    let substeps = integration_step["substeps"]
        .as_array()
        .expect("integration step should expose didactic substeps");
    let decomposition_substep = substeps
        .iter()
        .find(|substep| substep["title"] == "Descomponer en fracciones parciales")
        .expect("expected positive quadratic numerator decomposition substep");
    let decomposition_latex = decomposition_substep["after_latex"]
        .as_str()
        .expect("decomposition substep should expose concrete after_latex");
    assert!(
        decomposition_latex.contains("\\frac{x}{{x}^{2} + 1}")
            && decomposition_latex.contains("\\frac{1}{{x}^{2} + 1}"),
        "decomposition should expose derivative and arctan parts, got {decomposition_latex}"
    );
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Integrar los términos simples"),
        "expected simple-term integration substep, got {substeps:?}"
    );

    let input = "integrate(x/(x^2+2*x+2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "1/2 * ln(x^2 + 2 * x + 2) - arctan(x + 1)");
    assert!(
        required.is_empty(),
        "positive shifted quadratic denominator should not add synthetic required conditions: {required:?}"
    );

    let residual = "diff(integrate(x/(x^2+2*x+2), x), x) - x/(x^2+2*x+2)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "nested verification should not invent denominator conditions for a positive quadratic"
    );

    let input = "integrate(1/(x^2+x+1), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "2 * arctan((2 * x + 1) / sqrt(3)) / sqrt(3)");
    assert!(
        !result.contains("sqrt(3/4)"),
        "positive quadratic presentation should reduce rational surd width: {result}"
    );
    assert!(
        required.is_empty(),
        "positive shifted quadratic denominator should not add synthetic required conditions: {required:?}"
    );

    let residual = "diff(integrate(1/(x^2+x+1), x), x) - 1/(x^2+x+1)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "nested verification for reduced surd width should not invent denominator conditions"
    );

    let input = "integrate(1/(1/2*x^2+1/2*x+1/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "4 * arctan((2 * x + 1) / sqrt(3)) / sqrt(3)");
    assert!(
        !result.contains("sqrt(3/4)"),
        "scaled positive quadratic presentation should reduce rational surd width: {result}"
    );
    assert!(
        required.is_empty(),
        "scaled positive shifted quadratic denominator should not add synthetic required conditions: {required:?}"
    );

    let residual = "diff(integrate(1/(1/2*x^2+1/2*x+1/2), x), x) - 1/(1/2*x^2+1/2*x+1/2)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "nested verification for scaled reduced surd width should not invent denominator conditions"
    );

    let input = "integrate((x+3)/(2*x^2+4*x+4), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "arctan(x + 1) + 1/4 * ln(x^2 + 2 * x + 2)");
    assert!(
        required.is_empty(),
        "positive scaled quadratic denominator should not add synthetic required conditions: {required:?}"
    );

    let residual = "diff(integrate((x+3)/(2*x^2+4*x+4), x), x) - (x+3)/(2*x^2+4*x+4)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "scaled nested verification should not invent denominator conditions"
    );

    let input = "integrate((x+1)/(1/2*x^2+1/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "ln(x^2 + 1) + 2 * arctan(x)");
    assert!(
        required.is_empty(),
        "fractionally scaled positive quadratic denominator should not add synthetic required conditions: {required:?}"
    );

    let residual = "diff(integrate((x+1)/(1/2*x^2+1/2), x), x) - (x+1)/(1/2*x^2+1/2)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "fractionally scaled nested verification should not invent denominator conditions"
    );

    let input = "integrate((x+1)/(-2*x^2-2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "-1/2 * arctan(x) - 1/4 * ln(x^2 + 1)");
    assert!(
        required.is_empty(),
        "negative scaled positive quadratic denominator should not add synthetic required conditions: {required:?}"
    );

    let residual = "diff(integrate((x+1)/(-2*x^2-2), x), x) - (x+1)/(-2*x^2-2)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "negative scaled nested verification should not invent denominator conditions"
    );

    let input = "integrate((x+1)/(-1/2*x^2-1/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "-ln(x^2 + 1) - 2 * arctan(x)");
    assert!(
        required.is_empty(),
        "negative fractionally scaled positive quadratic denominator should not add synthetic required conditions: {required:?}"
    );

    let residual = "diff(integrate((x+1)/(-1/2*x^2-1/2), x), x) - (x+1)/(-1/2*x^2-1/2)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "negative fractionally scaled nested verification should not invent denominator conditions"
    );

    let input = "integrate(x/(-x^2-2*x-2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "arctan(x + 1) - 1/2 * ln(x^2 + 2 * x + 2)");
    assert!(
        required.is_empty(),
        "negative shifted positive quadratic denominator should not add synthetic required conditions: {required:?}"
    );

    let residual = "diff(integrate(x/(-x^2-2*x-2), x), x) - x/(-x^2-2*x-2)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "negative shifted nested verification should not invent denominator conditions"
    );

    let input = "integrate((x^3+2*x^2+3*x+4)/(x^2+2*x+2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "1/2 * ln(x^2 + 2 * x + 2) + 3 * arctan(x + 1) + x^2 / 2"
    );
    assert!(
        required.is_empty(),
        "shifted positive quadratic denominator should not add synthetic required conditions: {required:?}"
    );

    let residual =
        "diff(integrate((x^3+2*x^2+3*x+4)/(x^2+2*x+2), x), x) - (x^3+2*x^2+3*x+4)/(x^2+2*x+2)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "shifted nested verification should not invent denominator conditions"
    );

    let negative_residual =
        "diff(integrate((x^3+2*x^2+3*x+4)/(-x^2-2*x-2), x), x) - (x^3+2*x^2+3*x+4)/(-x^2-2*x-2)";
    let (wire, stderr) = cli_eval_json_with_stderr(negative_residual);
    assert_eq!(wire["result"].as_str(), Some("0"));
    assert!(
        !stderr.contains("depth_overflow"),
        "negative shifted improper quadratic verification should avoid depth_overflow\nstderr:\n{stderr}"
    );
}
#[test]
fn integrate_contract_positive_quadratic_square_decomposes_to_arctan_plus_rational() {
    let input = "integrate(1/(x^2+1)^2, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "1/2 * arctan(x) + x / (2 * (x^2 + 1))");
    assert!(
        required.is_empty(),
        "positive quadratic square should not add synthetic required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);
    assert!(
        stderr.is_empty(),
        "positive quadratic square trace should stay quiet\nstderr:\n{stderr}"
    );
    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    let integration_step = steps
        .iter()
        .find(|step| step["rule"] == "Calcular la integral")
        .expect("expected public symbolic integration step");
    let substeps = integration_step["substeps"]
        .as_array()
        .expect("integration step should expose didactic substeps");
    let reduction_substep = substeps
        .iter()
        .find(|substep| substep["title"] == "Reducir el cuadrático positivo al cuadrado")
        .expect("expected positive quadratic square reduction substep");
    let reduction_latex = reduction_substep["after_latex"]
        .as_str()
        .expect("reduction substep should expose concrete after_latex");
    assert!(
        reduction_latex.contains("\\frac{1}{2\\cdot ({x}^{2} + 1)}")
            && reduction_latex.contains("{({x}^{2} + 1)}^{2}"),
        "reduction should expose the arctan integrand and rational derivative part, got {reduction_latex}"
    );
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Integrar la parte arctan y la parte racional"),
        "expected final integration substep, got {substeps:?}"
    );

    let residual = "diff(integrate(1/(x^2+1)^2, x), x) - 1/(x^2+1)^2";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "nested verification should not invent denominator conditions for a positive quadratic square"
    );

    let input = "integrate(x^2/(x^2+1)^2, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "1/2 * arctan(x) - x / (2 * (x^2 + 1))");
    assert!(
        required.is_empty(),
        "quadratic numerator over a positive quadratic square should not add synthetic required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);

    let residual = "diff(integrate(x^2/(x^2+1)^2, x), x) - x^2/(x^2+1)^2";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "nested verification should not invent denominator conditions for a quadratic numerator over a positive quadratic square"
    );

    let input = "integrate(1/((x+1)^2+1)^2, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "1/2 * arctan(x + 1) + (x + 1) / (2 * (x^2 + 2 * x + 2))"
    );
    assert!(
        required.is_empty(),
        "shifted positive quadratic square should not add synthetic required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);

    let input = "integrate(1/(x^2+2*x+5)^2, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "1/16 * arctan(1/2 * x + 1/2) + (x + 1) / (8 * (x^2 + 2 * x + 5))"
    );
    assert!(
        required.is_empty(),
        "wide shifted positive quadratic square should not add synthetic required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);

    let residual = "diff(integrate(1/(x^2+2*x+5)^2, x), x) - 1/(x^2+2*x+5)^2";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "nested verification should not invent denominator conditions for a wide shifted positive quadratic square"
    );

    let input = "integrate(1/(4*x^2+1)^2, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "1/4 * arctan(2 * x) + x / (2 * (4 * x^2 + 1))");
    assert!(
        required.is_empty(),
        "scaled positive quadratic square should not add synthetic required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);

    let direct_diff = "diff(1/4*arctan(2*x)+x/(2*(4*x^2+1)), x)";
    let (direct_diff_result, direct_diff_required) =
        evaluated_expr_with_required_conditions(direct_diff);
    assert_eq!(direct_diff_result, "1 / (4 * x^2 + 1)^2");
    assert!(
        direct_diff_required.is_empty(),
        "compact post-diff presentation should not add synthetic required conditions"
    );

    let residual = "diff(integrate(1/(4*x^2+1)^2, x), x) - 1/(4*x^2+1)^2";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "nested verification should not invent denominator conditions for a scaled positive quadratic square"
    );
}
#[test]
fn integrate_contract_arctan_kernel() {
    assert_eq!(simplified_integral("integrate(1/(x^2+1), x)"), "arctan(x)");
}
#[test]
fn integrate_contract_arctan_sqrt_kernel_inverts_diff_output() {
    let input = "integrate(1/(2*sqrt(x)*(x+1)), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "arctan(sqrt(x))");
    assert_eq!(
        required,
        vec!["x > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    let step_rules = evaluated_integral_step_rules(input);
    assert_eq!(
        step_rules,
        vec!["Symbolic Integration".to_string()],
        "arctan sqrt reciprocal kernel should integrate directly without pre-expanding the denominator: {step_rules:?}"
    );
    assert_antiderivative_verifies(input);
    let (nested_derivative, nested_required) =
        evaluated_expr_with_required_conditions("diff(integrate(1/(2*sqrt(x)*(x+1)), x), x)");
    assert_eq!(nested_derivative, "1 / (2 * sqrt(x) * (x + 1))");
    assert_eq!(
        nested_required,
        vec!["x > 0".to_string()],
        "nested arctan sqrt derivative should preserve the positive radicand condition"
    );
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(1/(2*sqrt(x)*(x+1)), x), x) - 1/(2*sqrt(x)*(x+1))",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["x > 0".to_string()],
        "nested arctan sqrt verification should preserve the positive radicand condition"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(1/(sqrt(x)*(x+1)), x)");
    assert_eq!(result, "2 * arctan(sqrt(x))");
    assert_eq!(
        required,
        vec!["x > 0".to_string()],
        "unexpected scaled required_conditions: {required:?}"
    );
    let step_rules = evaluated_integral_step_rules("integrate(1/(sqrt(x)*(x+1)), x)");
    assert_eq!(
        step_rules,
        vec!["Symbolic Integration".to_string()],
        "scaled arctan sqrt reciprocal kernel should integrate directly without pre-expanding the denominator: {step_rules:?}"
    );
    assert_antiderivative_verifies("integrate(1/(sqrt(x)*(x+1)), x)");

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(1/(sqrt(x)*(4*x+1)), x)");
    assert_eq!(result, "arctan(2 * sqrt(x))");
    assert_eq!(
        required,
        vec!["x > 0".to_string()],
        "unexpected scaled linear required_conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(1/(sqrt(x)*(4*x+1)), x)");
    let (nested_derivative, nested_required) =
        evaluated_expr_with_required_conditions("diff(integrate(1/(sqrt(x)*(4*x+1)), x), x)");
    assert_eq!(nested_derivative, "1 / (sqrt(x) * (4 * x + 1))");
    assert_eq!(
        nested_required,
        vec!["x > 0".to_string()],
        "scaled linear nested derivative should preserve the positive radicand condition"
    );
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(1/(sqrt(x)*(4*x+1)), x), x) - 1/(sqrt(x)*(4*x+1))",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["x > 0".to_string()],
        "scaled linear verification should preserve the positive radicand condition"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(1/(sqrt(x)*(x+4)), x)");
    assert_eq!(result, "arctan(sqrt(x) / 2)");
    assert_eq!(
        required,
        vec!["x > 0".to_string()],
        "unexpected offset linear required_conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(1/(sqrt(x)*(x+4)), x)");

    let input = "integrate(1/(sqrt(4*x+1)*(2*x+1)), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    assert_eq!(result, "arctan(sqrt(4 * x + 1))");
    assert_eq!(
        required,
        vec!["x > -1/4".to_string()],
        "unexpected affine required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    let (nested_derivative, nested_required) =
        evaluated_expr_with_required_conditions("diff(integrate(1/(sqrt(4*x+1)*(2*x+1)), x), x)");
    assert!(
        !nested_derivative.contains("integrate("),
        "nested derivative should not leave an integration residual: {nested_derivative}"
    );
    assert_eq!(
        nested_required,
        vec!["x > -1/4".to_string()],
        "affine nested derivative should preserve the positive radicand condition"
    );
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(1/(sqrt(4*x+1)*(2*x+1)), x), x) - 1/(sqrt(4*x+1)*(2*x+1))",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["x > -1/4".to_string()],
        "affine verification should preserve the positive radicand condition"
    );

    let input = "integrate(-1/(2*sqrt(5-3*x)*(2-x)), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    assert_eq!(result, "arctan(sqrt(5 - 3 * x))");
    assert_eq!(
        required,
        vec!["x < 5/3".to_string()],
        "unexpected negative-slope affine required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_antiderivative_equiv_verifies(input);
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(-1/(2*sqrt(5-3*x)*(2-x)), x), x) - (-1/(2*sqrt(5-3*x)*(2-x)))",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["x < 5/3".to_string()],
        "negative-slope affine residual verification should preserve the positive radicand condition"
    );
    let (nested_equiv, nested_required) = evaluated_equiv_with_required_conditions(
        "diff(integrate(-1/(2*sqrt(5-3*x)*(2-x)), x), x)",
        "-1/(2*sqrt(5-3*x)*(2-x))",
    );
    assert!(nested_equiv);
    assert_eq!(
        nested_required,
        vec!["x < 5/3".to_string()],
        "negative-slope affine derivative equivalence should preserve the positive radicand condition"
    );

    let (nested_equiv, nested_required) = evaluated_equiv_with_required_conditions(
        "diff(integrate(-1/(2*sqrt(1-x)*(2-x)), x), x)",
        "-1/(2*sqrt(1-x)*(2-x))",
    );
    assert!(
        nested_equiv,
        "public equivalence should accept the directly simplified zero residual"
    );
    assert_eq!(
        nested_required,
        vec!["x < 1".to_string()],
        "unit-slope negative affine derivative equivalence should preserve the positive radicand condition"
    );
}
#[test]
fn integrate_contract_arctan_sqrt_unit_shift_square_inverts_diff_output() {
    let input = "integrate(1/(sqrt(x)*(x+1)^2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "arctan(sqrt(x)) + sqrt(x) / (x + 1)");
    assert_eq!(
        required,
        vec!["x > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    let step_rules = evaluated_integral_step_rules(input);
    assert!(
        step_rules
            .iter()
            .any(|rule| rule == "Symbolic Integration"),
        "unit-shift square arctan sqrt reciprocal kernel should reach symbolic integration: {step_rules:?}"
    );
    assert_rendered_antiderivative_verifies(input, &result);

    let scaled_input = "integrate(1/(2*sqrt(x)*(x+1)^2), x)";
    let (scaled_result, scaled_required) =
        evaluated_integral_with_required_conditions(scaled_input);
    assert_eq!(scaled_result, "1/2 * (arctan(sqrt(x)) + sqrt(x) / (x + 1))");
    assert_eq!(
        scaled_required,
        vec!["x > 0".to_string()],
        "unexpected scaled required_conditions: {scaled_required:?}"
    );
    assert_eq!(
        integrate_call_antiderivative_residual_result(scaled_input),
        "0"
    );
    assert_rendered_antiderivative_verifies(scaled_input, &scaled_result);

    let shifted_input = "integrate(1/(sqrt(x)*(x+4)^2), x)";
    let (shifted_result, shifted_required) =
        evaluated_integral_with_required_conditions(shifted_input);
    assert_eq!(
        shifted_result,
        "1/8 * arctan(sqrt(x) / 2) + sqrt(x) / (4 * (x + 4))"
    );
    assert_eq!(
        shifted_required,
        vec!["x > 0".to_string()],
        "unexpected shifted required_conditions: {shifted_required:?}"
    );
    assert_eq!(
        integrate_call_antiderivative_residual_result(shifted_input),
        "0"
    );
    assert_eq!(
        assert_antiderivative_verifies(shifted_input),
        AntiderivativeVerificationRoute::PublicResidual
    );
    assert_rendered_antiderivative_verifies(shifted_input, &shifted_result);

    let rational_shift_input = "integrate(1/(sqrt(x)*(x+1/4)^2), x)";
    let (rational_shift_result, rational_shift_required) =
        evaluated_integral_with_required_conditions(rational_shift_input);
    assert_eq!(
        rational_shift_result,
        "8 * arctan(2 * sqrt(x)) + 4 * sqrt(x) / (x + 1/4)"
    );
    assert_eq!(
        rational_shift_required,
        vec!["x > 0".to_string()],
        "unexpected rational shift required_conditions: {rational_shift_required:?}"
    );
    assert_eq!(
        integrate_call_antiderivative_residual_result(rational_shift_input),
        "0"
    );
    assert_rendered_antiderivative_verifies(rational_shift_input, &rational_shift_result);
    let (rational_shift_displayed_derivative, rational_shift_displayed_required) =
        evaluated_expr_with_required_conditions("diff(8*arctan(2*sqrt(x)) + 4*sqrt(x)/(x+1/4), x)");
    assert_eq!(
        rational_shift_displayed_derivative,
        "1 / ((x + 1/4)^2 * sqrt(x))"
    );
    assert_eq!(
        rational_shift_displayed_required,
        vec!["x > 0".to_string()],
        "displayed rational-shift derivative should preserve the positive radicand condition"
    );

    let externally_scaled_rational_shift_input = "integrate(1/(3*sqrt(x)*(x+1/4)^2), x)";
    let (externally_scaled_rational_shift_result, externally_scaled_rational_shift_required) =
        evaluated_integral_with_required_conditions(externally_scaled_rational_shift_input);
    assert_eq!(
        externally_scaled_rational_shift_result,
        "8/3 * arctan(2 * sqrt(x)) + 4/3 * sqrt(x) / (x + 1/4)"
    );
    assert_eq!(
        externally_scaled_rational_shift_required,
        vec!["x > 0".to_string()],
        "unexpected externally scaled rational shift required_conditions: {externally_scaled_rational_shift_required:?}"
    );
    assert_eq!(
        integrate_call_antiderivative_residual_result(externally_scaled_rational_shift_input),
        "0"
    );
    assert_rendered_antiderivative_verifies(
        externally_scaled_rational_shift_input,
        &externally_scaled_rational_shift_result,
    );
}
#[test]
fn integrate_contract_scaled_asinh_residual_stays_quiet_on_stderr() {
    let input = concat!(
        "diff(integrate(1/((6-2*x)*sqrt(8-2*x)), x), x) ",
        "- 1/((6-2*x)*sqrt(8-2*x))"
    );
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert_eq!(wire["result"], "0");
    assert_eq!(
        wire["required_display"],
        serde_json::json!(["x < 3"]),
        "scaled asinh residual should preserve the antiderivative domain condition"
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "scaled asinh residual should not emit depth_overflow to stderr, got: {stderr}"
    );
}
#[test]
fn integrate_contract_scaled_affine_arctan_kernel_survives_quadratic_normalization() {
    let input = "integrate(2/(1+(2*x+1)^2), x)";
    assert_eq!(simplified_integral(input), "arctan(2 * x + 1)");
    assert_antiderivative_verifies(input);
}
#[test]
fn integrate_contract_arctan_positive_quadratic_with_surd_width() {
    let input = "integrate(1/(2*x^2+4*x+5), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "arctan((2 * x + 2) / sqrt(6)) / sqrt(6)");
    assert!(
        required.is_empty(),
        "positive quadratic arctan kernel should not add required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}
#[test]
fn integrate_contract_arctan_symbolic_scaled_positive_quadratic() {
    let cases = [
        ("integrate(1/(a^2*x^2+1), x)", "arctan(a * x) / a"),
        (
            "integrate(1/(a^2*(x+b)^2+1), x)",
            "arctan(a * b + a * x) / a",
        ),
        ("integrate(1/((a*x+b)^2+1), x)", "arctan(a * x + b) / a"),
        (
            "integrate(1/((a*x+b)^2+4), x)",
            "arctan((a * x + b) / 2) / (2 * a)",
        ),
        (
            "integrate(1/((a*x+b)^2+2), x)",
            "arctan(sqrt(2) * (a * x + b) / 2) / (sqrt(2) * a)",
        ),
        (
            "integrate(1/(a^2*(2*x+1)^2+1), x)",
            "arctan(2 * a * x + a) / (2 * a)",
        ),
        ("integrate(1/(a^2*(1-x)^2+1), x)", "-arctan(a - a * x) / a"),
    ];

    for (input, expected) in cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);

        assert_eq!(
            result, expected,
            "unexpected symbolic arctan integral for {input}"
        );
        assert_eq!(
            required,
            vec!["a ≠ 0".to_string()],
            "symbolic arctan scale should expose the nonzero parameter condition for {input}"
        );
        assert_antiderivative_verifies(input);
    }

    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(1/((a*x+b)^2+4), x), x) - 1/((a*x+b)^2+4)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["a ≠ 0".to_string()],
        "nested verification should preserve the nonzero parameter condition"
    );

    let (irrational_radius_residual, irrational_radius_required) =
        evaluated_expr_with_required_conditions(
            "diff(integrate(1/((a*x+b)^2+2), x), x) - 1/((a*x+b)^2+2)",
        );
    assert_eq!(irrational_radius_residual, "0");
    assert_eq!(
        irrational_radius_required,
        vec!["a ≠ 0".to_string()],
        "nested irrational-radius verification should preserve the nonzero parameter condition"
    );
}
#[test]
fn integrate_contract_atanh_quadratic_kernel_with_surd_width_preserves_positive_domain() {
    let input = "integrate(1/(3-x^2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "atanh(x / sqrt(3)) / sqrt(3)");
    assert_eq!(
        required,
        vec!["-sqrt(3) < x < sqrt(3)".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);

    let input = "integrate(1/(3/4-(x+1/2)^2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "2 * atanh((2 * x + 1) / sqrt(3)) / sqrt(3)");
    assert!(
        !result.contains("sqrt(3/4)"),
        "atanh quadratic presentation should reduce rational surd width: {result}"
    );
    assert_eq!(
        required,
        vec!["-1/2 - sqrt(3)/2 < x < -1/2 + sqrt(3)/2".to_string()],
        "shifted atanh quadratic should preserve its positive-domain condition"
    );

    let residual = "diff(integrate(1/(3/4-(x+1/2)^2), x), x) - 1/(3/4-(x+1/2)^2)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["-1/2 - sqrt(3)/2 < x < -1/2 + sqrt(3)/2".to_string()],
        "nested shifted atanh verification should preserve the positive-domain condition"
    );
}
#[test]
fn integrate_contract_scaled_atanh_quadratic_kernel_reduces_surd_width() {
    let input = "integrate(1/(12-4*x^2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "1/4 * atanh(x / sqrt(3)) / sqrt(3)");
    assert_eq!(
        required,
        vec!["-sqrt(3) < x < sqrt(3)".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}
#[test]
fn integrate_contract_polynomial_atanh_surd_width_uses_compact_positive_domain() {
    let input = "integrate(2*x/(3-x^4), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "atanh(x^2 / sqrt(3)) / sqrt(3)");
    assert_eq!(
        required,
        vec!["3 - x^4 > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    let (nested_residual, nested_required) =
        evaluated_expr_with_required_conditions("diff(integrate(2*x/(3-x^4), x), x) - 2*x/(3-x^4)");
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["3 - x^4 > 0".to_string()],
        "nested atanh substitution should preserve its positive open-interval condition"
    );
}
#[test]
fn integrate_contract_shifted_polynomial_atanh_surd_width_uses_compact_positive_domain() {
    let input = "integrate((2*x+2)/(3-(x+1)^4), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "atanh((x^2 + 2 * x + 1) / sqrt(3)) / sqrt(3)");
    assert_eq!(
        required,
        vec!["3 - (x + 1)^4 > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}
#[test]
fn integrate_contract_expanded_square_atanh_surd_width_uses_compact_positive_domain() {
    let input = "integrate((2*x+2)/(5-(x^2+2*x+1)^2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "atanh((x^2 + 2 * x + 1) / sqrt(5)) / sqrt(5)");
    assert_eq!(
        required,
        vec!["5 - (x + 1)^4 > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}
#[test]
fn integrate_contract_polynomial_derivative_arcsin_surd_width_preserves_positive_domain() {
    let input = "integrate(2*x/sqrt(3-x^4), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "arcsin(x^2 / sqrt(3))");
    assert_eq!(
        required,
        vec!["3 - x^4 > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(2*x/sqrt(3-x^4), x), x) - 2*x/sqrt(3-x^4)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["3 - x^4 > 0".to_string()],
        "nested surd-width arcsin substitution should preserve its positive radicand condition"
    );
}
#[test]
fn integrate_contract_expanded_shifted_polynomial_arcsin_surd_width_dedupes_positive_domain() {
    let input = "integrate((2*x+2)/sqrt(2 - x^4 - 4*x^3 - 6*x^2 - 4*x), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "arcsin((x + 1)^2 / sqrt(3))");
    assert_eq!(
        required,
        vec!["3 - (x + 1)^4 > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}
#[test]
fn integrate_contract_factored_shifted_polynomial_arcsin_surd_width_verifies_positive_domain() {
    let input = "integrate((2*x+2)/sqrt(3-(x^2+2*x+1)^2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "arcsin((x + 1)^2 / sqrt(3))");
    assert_eq!(
        required,
        vec!["3 - (x + 1)^4 > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}
#[test]
fn integrate_contract_shifted_sqrt_arcsin_kernel_verifies_public_residual() {
    let input = "integrate(1/(sqrt(x)*sqrt(sqrt(x)-x)), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);
    let mut expected_required = vec!["sqrt(x) - x > 0".to_string(), "x > 0".to_string()];
    required.sort();
    expected_required.sort();

    assert_eq!(result, "2 * arcsin(2 * sqrt(x) - 1)");
    assert_eq!(
        required, expected_required,
        "shifted sqrt arcsin kernel should preserve minimal denominator conditions"
    );
    assert_antiderivative_verifies(input);

    let (nested_residual, mut nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(1/(sqrt(x)*sqrt(sqrt(x)-x)), x), x) - 1/(sqrt(x)*sqrt(sqrt(x)-x))",
    );
    let mut expected_nested_required = vec!["sqrt(x) - x > 0".to_string(), "x > 0".to_string()];
    nested_required.sort();
    expected_nested_required.sort();
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required, expected_nested_required,
        "nested shifted sqrt arcsin residual should preserve denominator conditions"
    );
}
#[test]
fn integrate_contract_polynomial_derivative_asinh_residual_omits_cycle_blocked_hint_after_zero() {
    let input = "diff(integrate(2*x/sqrt(1+x^4), x), x) - 2*x/sqrt(1+x^4)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert_eq!(wire["result"], "0");
    assert!(
        wire.get("blocked_hints").is_none(),
        "successful residual should not surface non-actionable cycle blocked hints: {wire:?}"
    );
    assert!(
        !stderr.contains("cycle detected"),
        "successful residual should not print cycle detected hint\nstderr:\n{stderr}"
    );
}
#[test]
fn integrate_contract_polynomial_derivative_asinh_surd_width_remains_unconditional() {
    let input = "integrate(2*x/sqrt(3+x^4), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "asinh(x^2 / sqrt(3))");
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}
#[test]
fn integrate_contract_asinh_rational_surd_width_reduces_inner_offset() {
    let input = "integrate(x/sqrt(x^4+3/4), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "1/2 * asinh(2 * x^2 / sqrt(3))");
    assert!(
        !result.contains("sqrt(3/4)"),
        "asinh substitution presentation should reduce rational surd width: {result}"
    );
    assert!(
        required.is_empty(),
        "asinh positive radicand substitution should remain unconditional: {required:?}"
    );
    assert_antiderivative_equiv_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);

    let residual = "diff(integrate(x/sqrt(x^4+3/4), x), x) - x/sqrt(x^4+3/4)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "nested asinh verification should remain unconditional: {residual_required:?}"
    );

    let explicit_residual = "diff(1/2*asinh(2*x^2/sqrt(3)), x) - x/sqrt(x^4+3/4)";
    let (explicit_residual_result, explicit_residual_required) =
        evaluated_expr_with_required_conditions(explicit_residual);
    assert_eq!(explicit_residual_result, "0");
    assert!(
        explicit_residual_required.is_empty(),
        "explicit asinh verification should remain unconditional: {explicit_residual_required:?}"
    );

    let additive_residual = "diff(-1/2*asinh(2*x^2/sqrt(3)), x) + x/sqrt(x^4+3/4)";
    let (additive_residual_result, additive_residual_required) =
        evaluated_expr_with_required_conditions(additive_residual);
    assert_eq!(additive_residual_result, "0");
    assert!(
        additive_residual_required.is_empty(),
        "negative explicit asinh verification should remain unconditional: {additive_residual_required:?}"
    );
}
#[test]
fn integrate_contract_arcsin_rational_surd_width_reduces_inner_offset() {
    let input = "integrate((2*x+1)/sqrt(3/4-(x^2+x+1)^2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "arcsin((2 * x^2 + 2 * x + 2) / sqrt(3))");
    assert!(
        !result.contains("sqrt(3/4)"),
        "arcsin substitution presentation should reduce rational surd width: {result}"
    );
    assert!(
        required == ["3/4 - (x^2 + x + 1)^2 > 0"],
        "arcsin substitution presentation should preserve the real radicand condition: {required:?}"
    );
    assert_antiderivative_equiv_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);

    let residual = "diff(arcsin((2*x^2+2*x+2)/sqrt(3)), x) - (2*x+1)/sqrt(3/4-(x^2+x+1)^2)";
    let (residual_result, _residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
}
#[test]
fn integrate_contract_arcsin_scaled_rational_surd_width_verifies() {
    let input = "integrate((x+1/2)/sqrt(3/4-(x^2+x+1)^2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "1/2 * arcsin((2 * x^2 + 2 * x + 2) / sqrt(3))");
    assert!(
        required == ["3/4 - (x^2 + x + 1)^2 > 0"],
        "scaled arcsin substitution should preserve the real radicand condition: {required:?}"
    );
    assert_antiderivative_equiv_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);

    let residual = "diff(1/2*arcsin((2*x^2+2*x+2)/sqrt(3)), x) - (x+1/2)/sqrt(3/4-(x^2+x+1)^2)";
    let (residual_result, _residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");

    let negative_input = "integrate(-(x+1/2)/sqrt(3/4-(x^2+x+1)^2), x)";
    let (negative_result, negative_required) =
        evaluated_integral_with_required_conditions(negative_input);
    assert_eq!(
        negative_result,
        "-1/2 * arcsin((2 * x^2 + 2 * x + 2) / sqrt(3))"
    );
    assert!(
        negative_required == ["3/4 - (x^2 + x + 1)^2 > 0"],
        "negative scaled arcsin substitution should preserve the real radicand condition: {negative_required:?}"
    );
    assert_antiderivative_equiv_verifies(negative_input);
    assert_rendered_antiderivative_verifies(negative_input, &negative_result);

    let additive_residual =
        "diff(-1/2*arcsin((2*x^2+2*x+2)/sqrt(3)), x) + (x+1/2)/sqrt(3/4-(x^2+x+1)^2)";
    let (additive_residual_result, _additive_residual_required) =
        evaluated_expr_with_required_conditions(additive_residual);
    assert_eq!(additive_residual_result, "0");
}
#[test]
fn integrate_contract_asinh_kernel() {
    assert_eq!(
        simplified_integral("integrate((x^2+1)^(-1/2), x)"),
        "asinh(x)"
    );
}
