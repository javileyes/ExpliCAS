use super::*;

#[test]
fn integrate_contract_simple_linear_partial_fractions_normalize_negative_factor() {
    let input = "integrate((x+2)/((x-1)*(-x-1)), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert!(
        result.contains("ln(|x + 1|)")
            && result.contains("ln(|x - 1|)")
            && !result.contains("ln(|-x - 1|)"),
        "expected normalized logarithmic simple-pole terms for negative factor orientation, got {result}"
    );
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "unexpected negative-factor simple-pole required_conditions: {required:?}"
    );

    let residual = "diff(integrate((x+2)/((x-1)*(-x-1)), x), x) - (x+2)/((x-1)*(-x-1))";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "negative-factor simple-pole verification should preserve the source denominator domain"
    );
}
#[test]
fn integrate_contract_rational_partial_fraction_with_two_linear_factors_and_positive_quadratic() {
    let input = "integrate(1/(x^4-1), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "1/4 * ln(|x - 1|) - 1/2 * arctan(x) - 1/4 * ln(|x + 1|)"
    );
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "positive quadratic remainder should not add real-domain conditions: {required:?}"
    );

    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);
    assert!(
        stderr.is_empty(),
        "positive-quadratic partial-fraction trace should stay quiet\nstderr:\n{stderr}"
    );
    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    let integration_step = steps
        .iter()
        .find(|step| step["rule"] == "Calcular la integral")
        .expect("expected integration step");
    let substeps = integration_step["substeps"]
        .as_array()
        .expect("integration step should expose didactic substeps");
    let decomposition_substep = substeps
        .iter()
        .find(|substep| substep["title"] == "Descomponer en fracciones parciales")
        .expect("expected positive-quadratic partial-fraction decomposition substep");
    let decomposition_latex = decomposition_substep["after_latex"]
        .as_str()
        .expect("partial-fraction substep should expose concrete after_latex");
    assert!(
        decomposition_latex.contains("x - 1")
            && decomposition_latex.contains("x + 1")
            && decomposition_latex.contains("{x}^{2} + 1"),
        "partial-fraction substep should show linear factors and positive quadratic, got {decomposition_latex}"
    );
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Integrar los términos simples"),
        "expected simple-term integration substep, got {substeps:?}"
    );

    let residual = "diff(integrate(1/(x^4-1), x), x) - 1/(x^4-1)";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "nested verification should preserve only the source denominator domain"
    );

    let input = "integrate((x-1)/(x^4-1), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "1/2 * arctan(x) - 1/4 * ln(x^2 + 1) + 1/2 * ln(|x + 1|)"
    );
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "cancelled pole in the partial-fraction numerator must still retain the source denominator domain: {required:?}"
    );

    let residual = "diff(integrate((x-1)/(x^4-1), x), x) - (x-1)/(x^4-1)";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "cancelled-pole nested verification should preserve the source denominator domain"
    );

    let input = "integrate((x^2-1)/(x^4-1), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "arctan(x)");
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "fully cancelled real poles must still retain the source denominator domain: {required:?}"
    );

    let residual = "diff(integrate((x^2-1)/(x^4-1), x), x) - (x^2-1)/(x^4-1)";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "fully-cancelled-pole nested verification should preserve the source denominator domain"
    );

    let input = "integrate((x^2-1)/((x-1)*(x+1)*(x^2+1)), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "arctan(x)");
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "factored denominator cancellation must still retain only the real source denominator domain: {required:?}"
    );

    let residual =
        "diff(integrate((x^2-1)/((x-1)*(x+1)*(x^2+1)), x), x) - (x^2-1)/((x-1)*(x+1)*(x^2+1))";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "factored-denominator nested verification should preserve only the real source denominator domain"
    );

    let input = "integrate((x^2-1)/(-(x-1)*(x+1)*(x^2+1)), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "-arctan(x)");
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "negative factored denominator cancellation must retain only the real source denominator domain: {required:?}"
    );

    let residual =
        "diff(integrate((x^2-1)/(-(x-1)*(x+1)*(x^2+1)), x), x) - (x^2-1)/(-(x-1)*(x+1)*(x^2+1))";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "negative-factored-denominator verification should preserve only the real source denominator domain"
    );
}
#[test]
fn integrate_contract_rational_partial_fractions_over_repeated_linear_factors() {
    let input = "integrate((3*x+5)/(x^3-x^2-x+1), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert!(
        !result.starts_with("integrate("),
        "expected a proper antiderivative, got {result}"
    );
    assert_eq!(result, "1/2 * ln(|(x + 1) / (x - 1)|) - 4 / (x - 1)");
    assert!(
        result.contains("4 / (x - 1)"),
        "expected repeated-pole rational term, got {result}"
    );
    assert!(
        result.contains(" - 4 / (x - 1)") && !result.contains("+ -"),
        "expected a clean subtraction for the repeated-pole rational term, got {result}"
    );
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "unexpected required_conditions: {required:?}"
    );

    let residual = "diff(integrate((3*x+5)/(x^3-x^2-x+1), x), x) - (3*x+5)/(x^3-x^2-x+1)";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "nested verification should preserve the source denominator domain"
    );
    assert_rendered_antiderivative_verifies(input, &result);

    let reordered_residual =
        "diff((-1/2)*ln(abs(x-1)) - 4/(x-1) + (1/2)*ln(abs(x+1)), x) - (3*x+5)/(x^3-x^2-x+1)";
    let (reordered_result, mut reordered_required) =
        evaluated_expr_with_required_conditions(reordered_residual);
    reordered_required.sort();
    assert_eq!(reordered_result, "0");
    assert_eq!(
        reordered_required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "reordered rendered antiderivative verification should preserve the source denominator domain"
    );

    let input = "integrate((3*x+5)/((x-2)^2*(x+3)), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "4/25 * ln(|(x - 2) / (x + 3)|) - 11 / (5 * (x - 2))"
    );
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -3".to_string(), "x ≠ 2".to_string()],
        "unexpected shifted repeated-pole required_conditions: {required:?}"
    );

    let residual = "diff(integrate((3*x+5)/((x-2)^2*(x+3)), x), x) - (3*x+5)/((x-2)^2*(x+3))";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -3".to_string(), "x ≠ 2".to_string()],
        "shifted repeated-pole nested verification should preserve the source denominator domain"
    );

    let input = "integrate((3*x+5)/(x^3-x^2-8*x+12), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "4/25 * ln(|(x - 2) / (x + 3)|) - 11 / (5 * (x - 2))"
    );
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -3".to_string(), "x ≠ 2".to_string()],
        "unexpected expanded shifted repeated-pole required_conditions: {required:?}"
    );

    let residual = "diff(integrate((3*x+5)/(x^3-x^2-8*x+12), x), x) - (3*x+5)/(x^3-x^2-8*x+12)";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -3".to_string(), "x ≠ 2".to_string()],
        "expanded shifted repeated-pole nested verification should preserve the source denominator domain"
    );

    let input = "integrate((3*x+5)/((1-x)^2*(x+1)), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "1/2 * ln(|(x + 1) / (x - 1)|) - 4 / (x - 1)");
    assert!(
        result.contains(" - 4 / (x - 1)") && !result.contains("+ -"),
        "expected clean repeated-pole term for factored orientation, got {result}"
    );
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "unexpected factored-orientation required_conditions: {required:?}"
    );

    let residual = "diff(integrate((3*x+5)/((1-x)^2*(x+1)), x), x) - (3*x+5)/((1-x)^2*(x+1))";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "factored-orientation nested verification should preserve the source denominator domain"
    );

    let input = "integrate((3*x+5)/((x-1)^2*(-x-1)), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "1/2 * ln(|(x - 1) / (x + 1)|) + 4 / (x - 1)");
    assert!(
        result.contains("4 / (x - 1)") && !result.contains("+ -"),
        "expected clean repeated-pole term for negative factor orientation, got {result}"
    );
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "unexpected negative-factor required_conditions: {required:?}"
    );

    let residual = "diff(integrate((3*x+5)/((x-1)^2*(-x-1)), x), x) - (3*x+5)/((x-1)^2*(-x-1))";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "negative-factor nested verification should preserve the source denominator domain"
    );
}
#[test]
fn integrate_contract_scaled_repeated_linear_partial_fractions_normalize_log_arguments() {
    let input = "integrate((3*x+5)/(2*x^3-2*x^2-2*x+2), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "1/4 * ln(|(x + 1) / (x - 1)|) - 2 / (x - 1)");
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "unexpected required_conditions: {required:?}"
    );

    let residual =
        "diff(integrate((3*x+5)/(2*x^3-2*x^2-2*x+2), x), x) - (3*x+5)/(2*x^3-2*x^2-2*x+2)";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "nested verification should preserve the source denominator domain"
    );

    let input = "integrate((3*x+5)/(1/2*x^3-1/2*x^2-1/2*x+1/2), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "ln(|(x + 1) / (x - 1)|) - 8 / (x - 1)");
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "unexpected fractional-scale required_conditions: {required:?}"
    );

    let residual =
        "diff(integrate((3*x+5)/(1/2*x^3-1/2*x^2-1/2*x+1/2), x), x) - (3*x+5)/(1/2*x^3-1/2*x^2-1/2*x+1/2)";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "fractional nested verification should preserve the source denominator domain"
    );

    let input = "integrate(1/((2*x+2)^2*(x-1)), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert!(
        !result.starts_with("integrate("),
        "expected scaled repeated linear factors to integrate, got {result}"
    );
    assert_eq!(result, "1/16 * ln(|(x - 1) / (x + 1)|) + 1 / (8 * (x + 1))");
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "unexpected scaled repeated-factor required_conditions: {required:?}"
    );

    let residual = "diff(integrate(1/((2*x+2)^2*(x-1)), x), x) - 1/((2*x+2)^2*(x-1))";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "scaled repeated-factor nested verification should preserve the source denominator domain"
    );
}
#[test]
fn integrate_contract_degree_five_linear_partial_fractions_verify_by_diff() {
    let input = "integrate(1/((x-2)*(x-1)*x*(x+1)*(x+2)), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert!(
        !result.starts_with("integrate("),
        "expected a proper antiderivative for five linear factors, got {result}"
    );
    for log_arg in [
        "ln(|x - 2|)",
        "ln(|x - 1|)",
        "ln(|x|)",
        "ln(|x + 1|)",
        "ln(|x + 2|)",
    ] {
        assert!(
            result.contains(log_arg),
            "expected {log_arg} in five-factor partial-fraction antiderivative, got {result}"
        );
    }
    required.sort();
    assert_eq!(
        required,
        vec![
            "x ≠ -1".to_string(),
            "x ≠ -2".to_string(),
            "x ≠ 0".to_string(),
            "x ≠ 1".to_string(),
            "x ≠ 2".to_string(),
        ],
        "unexpected five-factor partial-fraction required_conditions: {required:?}"
    );

    let residual =
        "diff(integrate(1/((x-2)*(x-1)*x*(x+1)*(x+2)), x), x) - 1/((x-2)*(x-1)*x*(x+1)*(x+2))";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec![
            "x ≠ -1".to_string(),
            "x ≠ -2".to_string(),
            "x ≠ 0".to_string(),
            "x ≠ 1".to_string(),
            "x ≠ 2".to_string(),
        ],
        "five-factor nested verification should preserve the source denominator domain"
    );

    let public_equiv = "equiv(diff(integrate(1/((x-2)*(x-1)*x*(x+1)*(x+2)), x), x), 1/((x-2)*(x-1)*x*(x+1)*(x+2)))";
    let (equiv_wire, equiv_stderr) = cli_eval_json_with_stderr(public_equiv);
    assert!(
        equiv_stderr.is_empty(),
        "public five-factor antiderivative check should avoid depth overflow: {equiv_stderr}"
    );
    assert_eq!(equiv_wire["result"], "true");
    assert_eq!(
        equiv_wire["required_display"],
        serde_json::json!(["x ≠ 0", "x ≠ 1", "x ≠ -1", "x ≠ 2", "x ≠ -2"])
    );

    let direct_diff = "diff(integrate(1/((x-2)*(x-1)*x*(x+1)*(x+2)), x), x)";
    let (direct_wire, direct_stderr) = cli_eval_json_with_stderr(direct_diff);
    assert!(
        direct_stderr.is_empty(),
        "direct five-factor diff/integrate should avoid depth overflow: {direct_stderr}"
    );
    assert_eq!(
        direct_wire["result"],
        "1 / (x·(x + 1)·(x + 2)·(x - 1)·(x - 2))"
    );
    assert_eq!(
        direct_wire["required_display"],
        serde_json::json!(["x ≠ -1", "x ≠ -2", "x ≠ 0", "x ≠ 1", "x ≠ 2"])
    );
    assert_rendered_antiderivative_verifies(input, &result);
}
#[test]
fn integrate_contract_degree_six_linear_partial_fractions_verify_by_diff() {
    let input = "integrate(1/((x-3)*(x-2)*(x-1)*x*(x+1)*(x+2)), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert!(
        !result.starts_with("integrate("),
        "expected a proper antiderivative for six linear factors, got {result}"
    );
    for log_arg in [
        "ln(|x - 3|)",
        "ln(|x - 2|)",
        "ln(|x - 1|)",
        "ln(|x|)",
        "ln(|x + 1|)",
        "ln(|x + 2|)",
    ] {
        assert!(
            result.contains(log_arg),
            "expected {log_arg} in six-factor partial-fraction antiderivative, got {result}"
        );
    }
    required.sort();
    assert_eq!(
        required,
        vec![
            "x ≠ -1".to_string(),
            "x ≠ -2".to_string(),
            "x ≠ 0".to_string(),
            "x ≠ 1".to_string(),
            "x ≠ 2".to_string(),
            "x ≠ 3".to_string(),
        ],
        "unexpected six-factor partial-fraction required_conditions: {required:?}"
    );

    let residual =
        "diff(integrate(1/((x-3)*(x-2)*(x-1)*x*(x+1)*(x+2)), x), x) - 1/((x-3)*(x-2)*(x-1)*x*(x+1)*(x+2))";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec![
            "x ≠ -1".to_string(),
            "x ≠ -2".to_string(),
            "x ≠ 0".to_string(),
            "x ≠ 1".to_string(),
            "x ≠ 2".to_string(),
            "x ≠ 3".to_string(),
        ],
        "six-factor nested verification should preserve the source denominator domain"
    );
    assert_rendered_antiderivative_verifies(input, &result);
}
#[test]
fn integrate_contract_linear_times_positive_quadratic_partial_fraction_verify_by_diff() {
    let input = "integrate(1/((x+1)*(x^2+1)), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert!(
        !result.starts_with("integrate("),
        "expected a proper antiderivative for linear times positive quadratic, got {result}"
    );
    assert!(
        result.contains("ln(|x + 1|)")
            && result.contains("ln(x^2 + 1)")
            && result.contains("arctan(x)"),
        "expected log-linear plus positive-quadratic log/arctan terms, got {result}"
    );
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string()],
        "unexpected linear-positive-quadratic required_conditions: {required:?}"
    );

    let residual = "diff(integrate(1/((x+1)*(x^2+1)), x), x) - 1/((x+1)*(x^2+1))";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string()],
        "linear-positive-quadratic verification should keep only the linear-pole domain"
    );
    assert_rendered_antiderivative_verifies(input, &result);

    let input = "integrate(1/((x+2)*(x^2+2*x+5)), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert!(
        !result.starts_with("integrate("),
        "expected a proper antiderivative for shifted linear-positive-quadratic partial fractions, got {result}"
    );
    assert!(
        result.contains("ln(|x + 2|)")
            && result.contains("ln(x^2 + 2 * x + 5)")
            && result.contains("arctan(1/2 * x + 1/2)"),
        "expected shifted log-linear plus scaled positive-quadratic log/arctan terms, got {result}"
    );
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -2".to_string()],
        "shifted positive-quadratic denominator should not add synthetic required conditions: {required:?}"
    );

    let residual = "diff(integrate(1/((x+2)*(x^2+2*x+5)), x), x) - 1/((x+2)*(x^2+2*x+5))";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -2".to_string()],
        "shifted linear-positive-quadratic verification should keep only the linear-pole domain"
    );
    let (equivalent, mut equiv_required) = evaluated_equiv_with_required_conditions(
        "diff(integrate(1/((x+2)*(x^2+2*x+5)), x), x)",
        "1/((x+2)*(x^2+2*x+5))",
    );
    equiv_required.sort();
    assert!(
        equivalent,
        "public equivalence should verify the shifted linear-positive-quadratic antiderivative by differentiation"
    );
    assert_eq!(
        equiv_required,
        vec!["x ≠ -2".to_string()],
        "public equivalence should keep only the shifted linear-pole domain"
    );

    let direct_diff = "diff(integrate(1/((x+2)*(x^2+2*x+5)), x), x)";
    let (direct_wire, direct_stderr) = cli_eval_json_with_stderr(direct_diff);
    assert!(
        direct_stderr.is_empty(),
        "direct shifted linear-positive-quadratic diff/integrate should avoid depth overflow: {direct_stderr}"
    );
    let direct_result = direct_wire["result"].as_str().unwrap_or_default();
    assert!(
        direct_result.contains("x + 2") && direct_result.contains("x^2 + 2·x + 5"),
        "direct shifted linear-positive-quadratic diff/integrate should preserve a compact denominator, got {direct_result}"
    );
    assert_eq!(
        direct_wire["required_display"],
        serde_json::json!(["x ≠ -2"])
    );
    assert_rendered_antiderivative_verifies(input, &result);

    let input = "integrate((x+2)/(x^3+1), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert!(
        !result.starts_with("integrate("),
        "expected a proper antiderivative for expanded cubic linear-positive-quadratic partial fractions, got {result}"
    );
    assert!(
        result.contains("ln(|x + 1|)")
            && result.contains("ln(x^2 + 1 - x)")
            && result.contains("arctan((2 * x - 1) / sqrt(3))"),
        "expected expanded cubic to decompose into linear-log plus positive-quadratic log/arctan terms, got {result}"
    );
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string()],
        "expanded cubic denominator should only require the real linear pole domain: {required:?}"
    );

    let residual = "diff(integrate((x+2)/(x^3+1), x), x) - (x+2)/(x^3+1)";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string()],
        "expanded cubic verification should preserve only the real linear pole domain"
    );

    let input = "integrate((x^2+1)/(x^3+1), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert!(
        !result.starts_with("integrate("),
        "expected a proper antiderivative for expanded cubic quadratic-numerator partial fractions, got {result}"
    );
    assert!(
        result.contains("ln(|x + 1|)")
            && result.contains("ln(x^2 + 1 - x)")
            && result.contains("arctan((2 * x - 1) / sqrt(3))"),
        "expected expanded cubic quadratic numerator to decompose into linear-log plus positive-quadratic log/arctan terms, got {result}"
    );
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string()],
        "expanded cubic quadratic numerator should only require the real linear pole domain: {required:?}"
    );

    let residual = "diff(integrate((x^2+1)/(x^3+1), x), x) - (x^2+1)/(x^3+1)";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string()],
        "expanded cubic quadratic-numerator verification should preserve only the real linear pole domain"
    );
}
#[test]
fn integrate_contract_repeated_linear_partial_fraction_preserves_nonzero_domain() {
    let input = "integrate((3*x+5)/(x^3-x^2-x+1), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "1/2 * ln(|(x + 1) / (x - 1)|) - 4 / (x - 1)");
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "unexpected repeated-linear partial fraction required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);

    let residual = "diff(integrate((3*x+5)/(x^3-x^2-x+1), x), x) - (3*x+5)/(x^3-x^2-x+1)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "unexpected repeated-linear partial fraction residual conditions: {residual_required:?}"
    );
}
