use super::*;

/// Honest-residual contract for the universal closure's gates: symbolic
/// coefficients, non-squarefree denominators and non-squarefree resultants
/// must stay residual — declining, never emitting a wrong antiderivative.
/// (Every earlier algebraic-extension pin graduated: √5/∛2 via Cap. C/D,
/// the octics via R3, and the Galois-obstructed class via the E-iv RootSum.)
#[test]
fn integrate_contract_algebraic_extension_denominators_stay_residual() {
    for input in [
        "integrate(1/(x^3-a), x)", // symbolic coefficient: no numeric resultant
        "integrate(1/((x^3-x-1)^2), x)", // non-squarefree denominator
        "integrate(1/(x^8+x^4+1), x)", // non-squarefree Rothstein-Trager resultant
    ] {
        let (result, _required) = evaluated_integral_with_required_conditions(input);
        assert!(
            result.contains("integrate("),
            "should stay an honest residual until its sub-cycle lands: {input} -> {result}"
        );
    }
}
#[test]
fn integrate_contract_positive_quadratic_rational_double_nested_residual_stays_quiet() {
    let input =
        "(((((diff(integrate((3*x + 5)/(x^2+x+1),x),x)-((3*x + 5)/(x^2+x+1)))+1)/(x+2))/(x+3))/(x+4))";
    let (wire, stderr) = cli_eval_json_with_stderr(input);
    assert!(
        stderr.is_empty(),
        "positive-quadratic rational residual should avoid depth overflow: {stderr}"
    );
    assert_eq!(wire["result"], "1 / ((x + 2)·(x + 3)·(x + 4))");
    assert_eq!(
        wire["required_display"],
        serde_json::json!(["x ≠ -2", "x ≠ -3", "x ≠ -4"])
    );
}
#[test]
fn integrate_contract_polynomial_derivative_over_denominator_power_preserves_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x/(x^2-1)^2, x)");

    assert_eq!(result, "-1 / (x^2 - 1)");
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_shifted_polynomial_derivative_over_syntactic_denominator_square_preserves_nonzero_domain(
) {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((2*x+1)/(x^2+x-1)^2, x)");

    assert_eq!(result, "-1 / (x^2 + x - 1)");
    assert_eq!(
        required,
        vec!["x^2 + x - 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_scaled_syntactic_denominator_square_preserves_nonzero_domain() {
    let input = "integrate((2*x+1)/(3*(x^2+x-1)^2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    let step_rules = evaluated_integral_step_rules(input);

    assert_eq!(result, "-1 / (3 * (x^2 + x - 1))");
    assert_eq!(
        required,
        vec!["x^2 + x - 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_eq!(
        step_rules,
        vec!["Symbolic Integration".to_string()],
        "scaled denominator power integration should render as a direct compact reciprocal: {step_rules:?}"
    );
    assert_antiderivative_verifies(input);
}
#[test]
fn integrate_contract_negative_power_denominator_displays_base_nonzero_domain() {
    let input = "integrate((2*x+1)/(3*(x^2+x-1)^(-2)), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    let step_rules = evaluated_integral_step_rules(input);

    assert_eq!(result, "1/9 * (x^2 + x - 1)^3");
    assert_eq!(
        required,
        vec!["x^2 + x - 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert!(
        step_rules
            .iter()
            .any(|rule_name| rule_name == "Symbolic Integration"),
        "expected symbolic integration step, got {step_rules:?}"
    );
    assert!(
        !step_rules
            .iter()
            .any(|rule_name| rule_name == "Simplify Complex Fraction"
                || rule_name == "Sacar factor común"),
        "negative-power denominator integration should not expand then refactor before integrating: {step_rules:?}"
    );
    assert_rendered_antiderivative_verifies(input, &result);
}
#[test]
fn integrate_contract_reciprocal_power_denominator_quotient_integrates_directly_with_domain() {
    let input = "integrate((2*x+1)/(3/(x^2+x-1)^2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    let step_rules = evaluated_integral_step_rules(input);

    assert_eq!(result, "1/9 * (x^2 + x - 1)^3");
    assert_eq!(
        required,
        vec!["x^2 + x - 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert!(
        step_rules
            .iter()
            .any(|rule_name| rule_name == "Symbolic Integration"),
        "expected symbolic integration step, got {step_rules:?}"
    );
    assert!(
        !step_rules
            .iter()
            .any(|rule_name| rule_name == "Simplify Complex Fraction"
                || rule_name == "Sacar factor común"),
        "reciprocal quotient denominator integration should not expand then refactor before integrating: {step_rules:?}"
    );
    assert_rendered_antiderivative_verifies(input, &result);
}
#[test]
fn integrate_contract_reciprocal_negative_power_denominator_quotient_integrates_directly_with_domain(
) {
    let input = "integrate((2*x+1)/(3/((x^2+x-1)^(-2))), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    let step_rules = evaluated_integral_step_rules(input);

    assert_eq!(result, "-1 / (3 * (x^2 + x - 1))");
    assert_eq!(
        required,
        vec!["x^2 + x - 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_eq!(
        step_rules,
        vec!["Symbolic Integration".to_string()],
        "reciprocal negative-power denominator integration should not need a nested-fraction pre-step: {step_rules:?}"
    );
    assert_rendered_antiderivative_verifies(input, &result);
}
#[test]
fn integrate_contract_shifted_polynomial_derivative_over_syntactic_denominator_cube_preserves_nonzero_domain(
) {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((2*x+1)/(x^2+x-1)^3, x)");

    assert_eq!(result, "-1 / (2 * (x^2 + x - 1)^2)");
    assert_eq!(
        required,
        vec!["x^2 + x - 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_positive_quadratic_denominator_cube_preserves_compact_antiderivative() {
    let input = "integrate(2*x/(x^2+1)^3, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    let step_rules = evaluated_integral_step_rules(input);

    assert_eq!(result, "-1 / (2 * (x^2 + 1)^2)");
    assert!(
        required.is_empty(),
        "positive quadratic denominator should not add required conditions: {required:?}"
    );
    assert!(
        !result.contains("x^4"),
        "post-calculus presentation should keep the denominator factored: {result}"
    );
    assert!(
        !step_rules
            .iter()
            .any(|rule_name| rule_name == "Cancelar factores en una fracción"),
        "scaled denominator-power integration should not expand through cancellation: {step_rules:?}"
    );
    assert_rendered_antiderivative_verifies(input, &result);
}
#[test]
fn integrate_contract_negative_derivative_over_denominator_cube_keeps_compact_domain_signal() {
    let cases = [
        (
            "integrate(-2*x/(x^2+1)^3, x)",
            "1 / (2·(x^2 + 1)^2)",
            "\\frac{1}{2\\cdot {({x}^{2} + 1)}^{2}}",
            serde_json::json!([]),
            "diff(integrate(-2*x/(x^2+1)^3, x), x) + 2*x/(x^2+1)^3",
        ),
        (
            "integrate(-(2*x+1)/(x^2+x-1)^3, x)",
            "1 / (2·(x^2 + x - 1)^2)",
            "\\frac{1}{2\\cdot {({x}^{2} + x - 1)}^{2}}",
            serde_json::json!(["x^2 + x - 1 ≠ 0"]),
            "diff(integrate(-(2*x+1)/(x^2+x-1)^3, x), x) + (2*x+1)/(x^2+x-1)^3",
        ),
    ];

    for (input, expected_result, expected_latex, expected_required, residual) in cases {
        let (wire, stderr) = cli_eval_json_with_stderr(input);
        assert!(
            stderr.is_empty(),
            "unexpected stderr for negative rational denominator-power primitive: {stderr}"
        );
        assert_eq!(wire["result"], expected_result);
        assert_eq!(wire["result_latex"], expected_latex);
        assert_eq!(wire["required_display"], expected_required);

        let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual);
        assert!(
            residual_stderr.is_empty(),
            "unexpected stderr for negative rational denominator-power residual: {residual_stderr}"
        );
        assert_eq!(residual_wire["result"], "0");
        assert_eq!(residual_wire["required_display"], expected_required);
    }
}
#[test]
fn integrate_contract_scaled_polynomial_derivative_over_higher_denominator_power_preserves_compact_form_and_nonzero_domain(
) {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((3*x+3/2)/(x^2+x-1)^4, x)");

    assert_eq!(result, "-1 / (2 * (x^2 + x - 1)^3)");
    assert_eq!(
        required,
        vec!["x^2 + x - 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_scaled_syntactic_denominator_power_preserves_full_nonzero_domain() {
    let (result, mut required) =
        evaluated_integral_with_required_conditions("integrate((8*x+2)/(3*(2*x^2+x-1)^3), x)");

    assert_eq!(result, "-1 / (3 * (2 * x^2 + x - 1)^2)");
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string(), "x ≠ 1/2".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
