use super::*;

/// G1 sub-cycle Cap. A (2026-07-14): a rational denominator whose squarefree
/// factorization over ℚ leaves a quadratic with a POSITIVE discriminant (real
/// irrational roots, e.g. `x^2 - 2` inside `x^4 - 4 = (x^2-2)(x^2+2)`) now
/// integrates to a real-log ratio `ln|(x-√2)/(x+√2)|` instead of declining. The
/// irreducible (Δ<0) factor still renders as arctan, unchanged.
/// See docs/G1_RATIONAL_INTEGRATION_SCOPING.md.
///
/// The `x^4-4` case is also in REPRESENTATIVE_ANTIDERIVATIVE_VERIFICATION_CASES,
/// where its antiderivative is confirmed by differentiate-back. The odd-surd
/// variants (`x^4-9`, the `(x^2-3)` factor) are equally correct — verified
/// numerically — but the differentiate-back simplifier does not yet fold
/// `sqrt(3)*sqrt(3)` inside the rational cancellation, so they are asserted here
/// by render form + support only (the surd self-verification is a simplifier
/// residual, not an integration one — no wrong answer is emitted).
#[test]
fn integrate_contract_real_root_quadratic_factor_renders_real_log_ratio() {
    for input in [
        "integrate(1/(x^4-4), x)",
        "integrate(1/(x^4-9), x)",
        "integrate(1/((x^2-2)*(x^2+1)), x)",
        "integrate(1/((x^2-2)*(x^2-3)), x)",
        "integrate(x/(x^4-4), x)",
    ] {
        let (result, _required) = evaluated_integral_with_required_conditions(input);
        assert!(
            !result.contains("integrate("),
            "should no longer be residual: {input} -> {result}"
        );
        // Real-root factor contributes a log ratio (ln of an absolute value).
        assert!(
            result.contains("ln(|"),
            "expected a real-log ratio term for {input} -> {result}"
        );
    }

    // The mixed denominator keeps the arctan term for its irreducible factor.
    let (mixed, _) =
        evaluated_integral_with_required_conditions("integrate(1/((x^2-2)*(x^2+1)), x)");
    assert!(
        mixed.contains("arctan"),
        "irreducible factor should still render arctan: {mixed}"
    );
}
#[test]
fn integrate_contract_antiderivative_verification_uses_bounded_public_residual_for_log_reciprocal_derivative(
) {
    for input in [
        "integrate(1/(x*ln(x)), x)",
        "integrate(2*x/((x^2+1)*ln(x^2+1)^2), x)",
    ] {
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "{input} should verify through the bounded public residual route"
        );
    }
}
#[test]
fn integrate_contract_reciprocal_shifted_log_reciprocal_residual_keeps_domain_requires() {
    let input = "1/((diff(integrate(1/(x*ln(x)), x), x) - 1/(x*ln(x))) + x + 2) - 1/(x+2)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert!(
        stderr.is_empty(),
        "unexpected stderr for reciprocal shifted log reciprocal residual: {stderr}"
    );
    assert_eq!(wire["result"], "0");
    assert_eq!(
        wire["required_display"],
        serde_json::json!(["x > 0", "x ≠ 1"])
    );
}
#[test]
fn integrate_contract_linear_log_table_preserves_positive_domain() {
    let (result, required) = evaluated_integral_with_required_conditions("integrate(ln(x), x)");

    assert_eq!(result, "x * ln(x) - x");
    assert_eq!(
        required,
        vec!["x > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_rendered_antiderivative_verifies("integrate(ln(x), x)", &result);

    let (result, required) = evaluated_integral_with_required_conditions("integrate(ln(2*x+1), x)");

    assert_eq!(result, "1/2 * (2 * x + 1) * (ln(2 * x + 1) - 1)");
    assert_eq!(
        required,
        vec!["x > -1/2".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(ln(2*x+1), x)");
    assert_rendered_antiderivative_verifies("integrate(ln(2*x+1), x)", &result);
}
#[test]
fn integrate_contract_log_reciprocal_derivative_preserves_real_domain() {
    let cases = [
        (
            "integrate(1/(x*ln(x)), x)",
            "ln(|ln(x)|)",
            "diff(integrate(1/(x*ln(x)), x), x) - 1/(x*ln(x))",
            vec!["x > 0".to_string(), "x ≠ 1".to_string()],
        ),
        (
            "integrate(2/((2*x+1)*ln(2*x+1)), x)",
            "ln(|ln(2 * x + 1)|)",
            "diff(integrate(2/((2*x+1)*ln(2*x+1)), x), x) - 2/((2*x+1)*ln(2*x+1))",
            vec!["x > -1/2".to_string(), "x ≠ 0".to_string()],
        ),
        (
            "integrate(2*x/((x^2+1)*ln(x^2+1)), x)",
            "ln(|ln(x^2 + 1)|)",
            "diff(integrate(2*x/((x^2+1)*ln(x^2+1)), x), x) - 2*x/((x^2+1)*ln(x^2+1))",
            vec!["x ≠ 0".to_string()],
        ),
        (
            "integrate((2*x+1)/((x^2+x+1)*ln(x^2+x+1)), x)",
            "ln(|ln(x^2 + x + 1)|)",
            "diff(integrate((2*x+1)/((x^2+x+1)*ln(x^2+x+1)), x), x) - (2*x+1)/((x^2+x+1)*ln(x^2+x+1))",
            vec!["x ≠ -1".to_string(), "x ≠ 0".to_string()],
        ),
        (
            "integrate(2*x/((x^2-1)*ln(x^2-1)), x)",
            "ln(|ln(x^2 - 1)|)",
            "diff(integrate(2*x/((x^2-1)*ln(x^2-1)), x), x) - 2*x/((x^2-1)*ln(x^2-1))",
            vec!["x < -1 or x > 1".to_string(), "x^2 - 2 ≠ 0".to_string()],
        ),
        (
            "integrate(2*x/((x^2+1)*ln(x^2+1)^2), x)",
            "-1 / ln(x^2 + 1)",
            "diff(integrate(2*x/((x^2+1)*ln(x^2+1)^2), x), x) - 2*x/((x^2+1)*ln(x^2+1)^2)",
            vec!["x ≠ 0".to_string()],
        ),
        (
            "integrate(2*x/((x^2-1)*ln(x^2-1)^2), x)",
            "-1 / ln(x^2 - 1)",
            "diff(integrate(2*x/((x^2-1)*ln(x^2-1)^2), x), x) - 2*x/((x^2-1)*ln(x^2-1)^2)",
            vec!["x < -1 or x > 1".to_string(), "x^2 - 2 ≠ 0".to_string()],
        ),
        (
            "integrate((2*x+1)/((x^2+x-1)*ln(x^2+x-1)^2), x)",
            "-1 / ln(x^2 + x - 1)",
            "diff(integrate((2*x+1)/((x^2+x-1)*ln(x^2+x-1)^2), x), x) - (2*x+1)/((x^2+x-1)*ln(x^2+x-1)^2)",
            vec![
                "x < -1/2 - sqrt(5)/2 or x > -1/2 + sqrt(5)/2".to_string(),
                "x ≠ -2".to_string(),
                "x ≠ 1".to_string(),
            ],
        ),
        (
            "integrate((2*x+1)/((x^2+x-1)*ln(x^2+x-1)^3), x)",
            "-1 / (2 * ln(x^2 + x - 1)^2)",
            "diff(integrate((2*x+1)/((x^2+x-1)*ln(x^2+x-1)^3), x), x) - (2*x+1)/((x^2+x-1)*ln(x^2+x-1)^3)",
            vec![
                "x < -1/2 - sqrt(5)/2 or x > -1/2 + sqrt(5)/2".to_string(),
                "x ≠ -2".to_string(),
                "x ≠ 1".to_string(),
            ],
        ),
    ];

    for (input, expected, residual, expected_required) in cases {
        let (result, mut required) = evaluated_integral_with_required_conditions(input);
        required.sort();
        assert_eq!(result, expected, "input: {input}");
        assert_eq!(required, expected_required, "input: {input}");
        assert_rendered_antiderivative_verifies(input, &result);

        let (residual_result, mut residual_required) =
            evaluated_expr_with_required_conditions(residual);
        residual_required.sort();
        assert_eq!(residual_result, "0", "input: {input}");
        assert_eq!(residual_required, expected_required, "input: {input}");
    }
}
#[test]
fn integrate_contract_log_reciprocal_derivative_preserves_compact_prep_trace() {
    let input = "integrate(2*x/((x^2+1)*ln(x^2+1)), x)";
    let step_rules = evaluated_integral_step_rules(input);

    assert_eq!(
        step_rules,
        vec![
            "Pull Constant From Fraction".to_string(),
            "Symbolic Integration".to_string(),
        ],
        "log-reciprocal substitution should not expand and refactor its compact denominator: {step_rules:?}"
    );
}
#[test]
fn integrate_contract_cubic_times_shifted_positive_quadratic_log_flattens_remainder() {
    let input = "integrate((x^3+x^2+x+1)*ln(x^2+x+1), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "ln(x^2 + x + 1) * (1/4 * x^4 + 1/3 * x^3 + 1/2 * x^2 + x + 13/24) + 9/4 * arctan((2 * x + 1) / sqrt(3)) / sqrt(3) - 1/8 * x^4 - 5/36 * x^3 - 5/24 * x^2 - 5/3 * x"
    );
    assert!(
        !result.contains(" - ("),
        "cubic shifted positive-quadratic log by-parts presentation should flatten subtracting a remainder group, got {result}"
    );
    assert!(
        required.is_empty(),
        "shifted positive quadratic log argument should not add conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);

    let residual = "diff(integrate((x^3+x^2+x+1)*ln(x^2+x+1), x), x) - (x^3+x^2+x+1)*ln(x^2+x+1)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "cubic shifted positive-quadratic log by-parts residual should not add conditions: {residual_required:?}"
    );
}
#[test]
fn integrate_contract_quartic_times_shifted_positive_quadratic_log_flattens_remainder() {
    let input = "integrate((x^4+x^3+x^2+x+1)*ln(x^2+x+1), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "ln(x^2 + x + 1) * (1/5 * x^5 + 1/4 * x^4 + 1/3 * x^3 + 1/2 * x^2 + x + 77/120) + 33/20 * arctan((2 * x + 1) / sqrt(3)) / sqrt(3) - 2/25 * x^5 - 3/40 * x^4 - 13/180 * x^3 - 49/120 * x^2 - 22/15 * x"
    );
    assert!(
        !result.contains(" - ("),
        "quartic shifted positive-quadratic log by-parts presentation should flatten subtracting a remainder group, got {result}"
    );
    assert!(
        required.is_empty(),
        "quartic shifted positive quadratic log argument should not add conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);

    let residual =
        "diff(integrate((x^4+x^3+x^2+x+1)*ln(x^2+x+1), x), x) - (x^4+x^3+x^2+x+1)*ln(x^2+x+1)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "quartic shifted positive-quadratic log by-parts residual should not add conditions: {residual_required:?}"
    );
}
#[test]
fn integrate_contract_polynomial_log_product_preserves_source_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x*ln(x^2-1), x)");

    assert_eq!(result, "(ln(x^2 - 1) - 1) * (x^2 - 1)");
    assert_eq!(
        required,
        vec!["x < -1 or x > 1".to_string()],
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((2*x+1)*ln(x^2+x+1), x)");

    assert_eq!(result, "(ln(x^2 + x + 1) - 1) * (x^2 + x + 1)");
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_reciprocal_linear_uses_abs_log() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(1/(2*x + 1), x)");

    assert_eq!(result, "1/2 * ln(|2 * x + 1|)");
    assert_eq!(
        required,
        vec!["x ≠ -1/2".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_rendered_antiderivative_verifies("integrate(1/(2*x + 1), x)", &result);
}
#[test]
fn integrate_contract_direct_reciprocal_uses_abs_log_and_nonzero_domain() {
    let (result, required) = evaluated_integral_with_required_conditions("integrate(1/x, x)");

    assert_eq!(result, "ln(|x|)");
    assert_eq!(
        required,
        vec!["x ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_factorable_quadratic_log_ratio_simplifies_linear_factors() {
    let cases = [
        (
            "integrate(1/(x^2+x), x)",
            "ln(|x / (x + 1)|)",
            vec!["x ≠ 0".to_string(), "x ≠ -1".to_string()],
        ),
        (
            "integrate(1/(x^2-x), x)",
            "ln(|(x - 1) / x|)",
            vec!["x ≠ 1".to_string(), "x ≠ 0".to_string()],
        ),
        (
            "integrate(1/(x^2+3*x+2), x)",
            "ln(|(x + 1) / (x + 2)|)",
            vec!["x ≠ -1".to_string(), "x ≠ -2".to_string()],
        ),
        (
            "integrate(1/(4*x^2+4*x), x)",
            "1/4 * ln(|x / (x + 1)|)",
            vec!["x ≠ 0".to_string(), "x ≠ -1".to_string()],
        ),
        (
            "integrate(1/(4*x^2+12*x+8), x)",
            "1/4 * ln(|(x + 1) / (x + 2)|)",
            vec!["x ≠ -1".to_string(), "x ≠ -2".to_string()],
        ),
        (
            "integrate(1/(4*x^2-4), x)",
            "1/8 * ln(|(x - 1) / (x + 1)|)",
            vec!["x ≠ 1".to_string(), "x ≠ -1".to_string()],
        ),
        (
            "integrate(1/(2*x^2+3*x+1), x)",
            "ln(|(2 * x + 1) / (x + 1)|)",
            vec!["x ≠ -1/2".to_string(), "x ≠ -1".to_string()],
        ),
        (
            "integrate(1/(6*x^2+9*x+3), x)",
            "1/3 * ln(|(2 * x + 1) / (x + 1)|)",
            vec!["x ≠ -1/2".to_string(), "x ≠ -1".to_string()],
        ),
        (
            "integrate(1/(3*x^2+7*x+4), x)",
            "ln(|(x + 1) / (3 * x + 4)|)",
            vec!["x ≠ -1".to_string(), "x ≠ -4/3".to_string()],
        ),
    ];

    for (input, expected_result, expected_required) in cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected_result, "input: {input}");
        assert_eq!(
            required, expected_required,
            "input: {input}, unexpected required_conditions: {required:?}"
        );
        assert_rendered_antiderivative_verifies(input, &result);
    }
}
#[test]
fn integrate_contract_shifted_polynomial_derivative_over_expanded_denominator_square_preserves_nonzero_domain(
) {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((2*x+1)/(x^4+2*x^3-x^2-2*x+1), x)");

    assert_eq!(result, "-1 / (x^2 + x - 1)");
    assert_eq!(
        required,
        vec!["x^2 + x - 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );

    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate((2*x+1)/(x^4+2*x^3-x^2-2*x+1), x), x) - (2*x+1)/(x^4+2*x^3-x^2-2*x+1)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["x^2 + x - 1 ≠ 0".to_string()],
        "expanded denominator antiderivative verification should preserve the compact nonzero domain"
    );
}
#[test]
fn integrate_contract_shifted_polynomial_derivative_over_expanded_denominator_cube_recovers_compact_base_and_nonzero_domain(
) {
    let (result, required) = evaluated_integral_with_required_conditions(
        "integrate((2*x+1)/(x^6+3*x^5-5*x^3+3*x-1), x)",
    );

    assert_eq!(result, "-1 / (2 * (x^2 + x - 1)^2)");
    assert_eq!(
        required,
        vec!["x^2 + x - 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_shifted_polynomial_derivative_over_scaled_expanded_denominator_cube_recovers_compact_base_and_nonzero_domain(
) {
    let (result, required) = evaluated_integral_with_required_conditions(
        "integrate((2*x+1)/(4*x^6+12*x^5-20*x^3+12*x-4), x)",
    );

    assert_eq!(result, "-1 / (8 * (x^2 + x - 1)^2)");
    assert_eq!(
        required,
        vec!["x^2 + x - 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_shifted_polynomial_derivative_over_negatively_scaled_expanded_denominator_cube_preserves_sign_and_domain(
) {
    let (result, required) = evaluated_integral_with_required_conditions(
        "integrate((2*x+1)/(-4*x^6-12*x^5+20*x^3-12*x+4), x)",
    );

    assert_eq!(result, "1 / (8 * (x^2 + x - 1)^2)");
    assert_eq!(
        required,
        vec!["x^2 + x - 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );

    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate((2*x+1)/(-4*x^6-12*x^5+20*x^3-12*x+4), x), x) - (2*x+1)/(-4*x^6-12*x^5+20*x^3-12*x+4)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["x^2 + x - 1 ≠ 0".to_string()],
        "negative expanded denominator antiderivative verification should preserve the compact nonzero domain"
    );
}
#[test]
fn integrate_contract_scaled_polynomial_derivative_over_expanded_denominator_fourth_power_recovers_compact_base_and_nonzero_domain(
) {
    let (result, required) = evaluated_integral_with_required_conditions(
        "integrate((3*x+3/2)/(x^8+4*x^7+2*x^6-8*x^5-5*x^4+8*x^3+2*x^2-4*x+1), x)",
    );

    assert_eq!(result, "-1 / (2 * (x^2 + x - 1)^3)");
    assert_eq!(
        required,
        vec!["x^2 + x - 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_expanded_linear_denominator_fifth_power_recovers_compact_base_and_nonzero_domain(
) {
    let (result, required) = evaluated_integral_with_required_conditions(
        "integrate(1/(x^5+5*x^4+10*x^3+10*x^2+5*x+1), x)",
    );

    assert_eq!(result, "-1 / (4 * (x + 1)^4)");
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_near_expanded_denominator_power_remains_residual_without_compact_base_domain()
{
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((2*x+1)/(x^6+3*x^5-5*x^3+3*x), x)");

    assert!(
        result.starts_with("integrate("),
        "near denominator power should remain residual, got {result}"
    );
    assert!(
        !result.contains("-1 / (2 * (x^2 + x - 1)^2)"),
        "near denominator power must not reuse the exact-power antiderivative"
    );
    assert!(
        !required.contains(&"x^2 + x - 1 ≠ 0".to_string()),
        "near denominator power must not collapse domain to compact base: {required:?}"
    );
}
#[test]
fn integrate_contract_polynomial_log_derivative_uses_abs_log_and_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((2*x + 1)/(x^2 + x - 1), x)");

    assert_eq!(result, "ln(|x^2 + x - 1|)");
    assert_eq!(
        required,
        vec!["x^2 + x - 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_abs_log_preserves_compact_positive_leading_polynomial_base() {
    let input = "integrate((2*x-1)/(x^2-x-1), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "ln(|x^2 - x - 1|)");
    assert_eq!(
        required,
        vec!["x^2 - x - 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);
}
#[test]
fn integrate_contract_scaled_polynomial_log_derivative() {
    assert_eq!(
        simplified_integral("integrate((4*x + 2)/(x^2 + x + 1), x)"),
        "2 * ln(x^2 + x + 1)"
    );
}
#[test]
fn integrate_contract_negative_scaled_abs_log_derivative_power_preserves_sign_and_domain() {
    let input = "integrate(-2*((2*x+1)/(x^2+x-1)*ln(abs(x^2+x-1))^2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "-2/3 * ln(|x^2 + x - 1|)^3");
    assert_eq!(
        required,
        vec!["x^2 + x - 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);
}
