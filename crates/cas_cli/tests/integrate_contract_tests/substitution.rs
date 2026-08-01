use super::*;

#[test]
fn integrate_contract_antiderivative_verification_uses_bounded_public_residual_for_trig_log_substitution(
) {
    for input in [
        "integrate(tan(2*x+1), x)",
        "integrate(cot(2*x+1), x)",
        "integrate(sec(2*x+1), x)",
        "integrate(csc(2*x+1), x)",
    ] {
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "{input} should verify through the bounded public residual route"
        );
    }
}
#[test]
fn integrate_contract_antiderivative_verification_uses_bounded_public_residual_for_hyperbolic_quotient_substitution(
) {
    for input in [
        "integrate(sinh(2*x+1)/cosh(2*x+1), x)",
        "integrate(cosh(2*x+1)/sinh(2*x+1), x)",
    ] {
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "{input} should verify through the bounded public residual route"
        );
    }
}
#[test]
fn integrate_contract_odd_power_quadratic_inner_substitution_resolves_and_respects_boundary() {
    // x^(2k+1) f(x^2) for f in {exp,sin,cos}, odd power >= 3: substitute u = x^2
    // and delegate u^k f(c u) to the poly*{exp,trig} by-parts owner.
    for (input, expected) in [
        ("integrate(x^3*exp(x^2), x)", "1/2 * e^(x^2) * (x^2 - 1)"),
        (
            "integrate(x^5*exp(x^2), x)",
            "1/2 * e^(x^2) * (x^4 + 2 - 2 * x^2)",
        ),
        (
            "integrate(x^3*sin(x^2), x)",
            "1/2 * (sin(x^2) - cos(x^2) * x^2)",
        ),
        (
            "integrate(x^3*cos(x^2), x)",
            "1/2 * (cos(x^2) + sin(x^2) * x^2)",
        ),
        (
            "integrate(x^3*exp(2*x^2), x)",
            "1/8 * e^(2 * x^2) * (2 * x^2 - 1)",
        ),
    ] {
        let (result, required) = evaluated_expr_with_required_conditions(input);
        assert_eq!(result, expected, "result for {input}");
        assert!(required.is_empty(), "no domain condition for {input}");
        let integrand = &input["integrate(".len()..input.len() - ", x)".len()];
        let (rt, _) = evaluated_expr_with_required_conditions(&format!(
            "diff(integrate({integrand}, x), x) - ({integrand})"
        ));
        assert_eq!(rt, "0", "round-trip for {input}");
    }

    // Boundary: the k=0 case x*f(x^2) is owned by the derivative-substitution
    // rule; an even power, a non-x^2 inner, and a non-elementary f all decline.
    assert_eq!(
        evaluated_expr_with_required_conditions("integrate(x*exp(x^2), x)").0,
        "1/2 * e^(x^2)"
    );
    for residual in [
        "integrate(x^2*exp(x^2), x)", // even power, no elementary closed form
        "integrate(x^3*ln(x^2), x)",  // f = ln, not exp/sin/cos
    ] {
        let (result, _) = evaluated_expr_with_required_conditions(residual);
        assert!(
            result.starts_with("integrate("),
            "{residual} should stay an honest residual, got {result}"
        );
    }
    // x^3*exp(x) (linear inner) keeps the ordinary polynomial-by-parts result.
    assert_eq!(
        evaluated_expr_with_required_conditions("integrate(x^3*exp(x), x)").0,
        "e^x * (x^3 + 6 * x - 3 * x^2 - 6)"
    );
}
#[test]
fn integrate_contract_hyperbolic_reciprocal_fourth_substitution_verifies() {
    let cases = [
        (
            "integrate(1/cosh(x)^4, x)",
            "1/3 * (3 * tanh(x) - tanh(x)^3)",
            "diff(integrate(1/cosh(x)^4, x), x) - 1/cosh(x)^4",
            vec![],
        ),
        (
            "integrate(1/cosh(2*x+1)^4, x)",
            "1/6 * (3 * tanh(2 * x + 1) - tanh(2 * x + 1)^3)",
            "diff(integrate(1/cosh(2*x+1)^4, x), x) - 1/cosh(2*x+1)^4",
            vec![],
        ),
        (
            "integrate(2*x/cosh(x^2)^4, x)",
            "1/3 * (3 * tanh(x^2) - tanh(x^2)^3)",
            "diff(integrate(2*x/cosh(x^2)^4, x), x) - 2*x/cosh(x^2)^4",
            vec![],
        ),
        (
            "integrate(1/sinh(2*x+1)^4, x)",
            "1/2 / tanh(2 * x + 1) - 1/6 / tanh(2 * x + 1)^3",
            "diff(integrate(1/sinh(2*x+1)^4, x), x) - 1/sinh(2*x+1)^4",
            vec!["sinh(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(2*x/sinh(x^2)^4, x)",
            "1 / tanh(x^2) - 1/3 / tanh(x^2)^3",
            "diff(integrate(2*x/sinh(x^2)^4, x), x) - 2*x/sinh(x^2)^4",
            vec!["sinh(x^2) ≠ 0"],
        ),
        (
            "integrate(2*k*x/sinh(x^2+b)^4, x)",
            "k / tanh(x^2 + b) - k / (3 * tanh(x^2 + b)^3)",
            "diff(integrate(2*k*x/sinh(x^2+b)^4, x), x) - 2*k*x/sinh(x^2+b)^4",
            vec!["sinh(x^2 + b) ≠ 0"],
        ),
    ];

    for (input, expected, residual, expected_required) in cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected, "input: {input}");
        assert_eq!(
            required, expected_required,
            "unexpected required conditions for {input}: {required:?}"
        );
        let (residual_result, residual_required) =
            evaluated_integral_with_required_conditions(residual);
        assert_eq!(residual_result, "0", "residual: {residual}");
        assert_eq!(
            residual_required, expected_required,
            "unexpected residual required conditions for {input}: {residual_required:?}"
        );
    }
}
#[test]
fn integrate_contract_scaled_inverse_sqrt_polynomial_power_substitution() {
    let input = "integrate((4*x^3+6*x^2+6*x+2)/sqrt(2-3*(x^2+x+1)^4), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "arcsin((x^2 + x + 1)^2 / sqrt(2/3)) / sqrt(3)");
    assert_eq!(required, vec!["2 - 3 * (x^2 + x + 1)^4 > 0".to_string()]);
    assert_antiderivative_equiv_verifies(input);
    assert_inverse_trig_polynomial_substitution_keeps_compact_steps(input);

    let scaled = "integrate(2*(2*x^3+3*x^2+3*x+1)*sqrt(3)/sqrt(2-3*(x^2+x+1)^4), x)";
    let (result, required) = evaluated_integral_with_required_conditions(scaled);
    assert_eq!(result, "arcsin((x^2 + x + 1)^2 / sqrt(2/3))");
    assert_eq!(required, vec!["2 - 3 * (x^2 + x + 1)^4 > 0".to_string()]);
    assert_antiderivative_equiv_verifies(scaled);
    assert_inverse_trig_polynomial_substitution_keeps_compact_steps(scaled);
}
#[test]
fn integrate_contract_linear_sine_substitution() {
    assert_eq!(
        simplified_integral("integrate(sin(2*x), x)"),
        "-1/2 * cos(2 * x)"
    );
}
#[test]
fn integrate_contract_linear_exp_substitution() {
    assert_eq!(
        simplified_integral("integrate(exp(3*x + 1), x)"),
        "1/3 * e^(3 * x + 1)"
    );
}
#[test]
fn integrate_contract_polynomial_derivative_exp_substitution() {
    assert_eq!(simplified_integral("integrate(2*x*exp(x^2), x)"), "e^(x^2)");
}
#[test]
fn integrate_contract_symbolic_scale_tanh_substitution_exposes_concrete_trace() {
    let input = "integrate(2*k*x*tanh(x^2+b), x)";
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

    assert_eq!(wire["result"], "k·ln(cosh(x^2 + b))");
    assert!(
        wire["required_display"]
            .as_array()
            .expect("required_display should be an array")
            .is_empty(),
        "symbolic tanh substitution should not invent domain conditions: {:?}",
        wire["required_display"]
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "symbolic tanh substitution trace should not emit depth_overflow warning\nstderr:\n{stderr}"
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
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Usar la regla de tanh(u) -> ln(cosh(u))"),
        "expected tanh table substep for symbolic scale case, got {substeps:?}"
    );
    assert_u_du_substep_labels(substeps, input);
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Ajustar el factor constante"),
        "expected symbolic scale adjustment substep, got {substeps:?}"
    );
    assert!(
        substeps
            .iter()
            .all(|substep| substep["title"] != "Usar sustitución"),
        "symbolic tanh substitution should not fall back to generic substitution only: {substeps:?}"
    );
    assert!(
        steps
            .iter()
            .any(|step| step["rule"] == "Abs Under Positivity"),
        "expected cosh positivity cleanup step for symbolic tanh substitution, got {steps:?}"
    );
    assert_antiderivative_verifies(input);
}
#[test]
fn integrate_contract_polynomial_derivative_cos_substitution() {
    assert_eq!(
        simplified_integral("integrate(2*x*cos(x^2), x)"),
        "sin(x^2)"
    );
}
#[test]
fn integrate_contract_polynomial_derivative_sin_substitution() {
    assert_eq!(
        simplified_integral("integrate(2*x*sin(x^2), x)"),
        "-cos(x^2)"
    );
}
#[test]
fn integrate_contract_linear_hyperbolic_substitution() {
    assert_eq!(
        simplified_integral("integrate(sinh(2*x + 1), x)"),
        "1/2 * cosh(2 * x + 1)"
    );
    assert_eq!(
        simplified_integral("integrate(cosh(2*x + 1), x)"),
        "1/2 * sinh(2 * x + 1)"
    );
}
#[test]
fn integrate_contract_polynomial_derivative_hyperbolic_substitution() {
    assert_eq!(
        simplified_integral("integrate(2*x*sinh(x^2), x)"),
        "cosh(x^2)"
    );
    assert_eq!(
        simplified_integral("integrate(2*x*cosh(x^2), x)"),
        "sinh(x^2)"
    );
    assert_eq!(
        simplified_integral("integrate(2*x*tanh(x^2), x)"),
        "ln(cosh(x^2))"
    );
}
#[test]
fn integrate_contract_hyperbolic_arctan_derivative_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(cosh(x)/(1+sinh(x)^2), x)");

    assert_eq!(result, "arctan(sinh(x))");
    assert!(
        required.is_empty(),
        "positive one-plus-square denominator should not require source conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(cosh(x)/(1+sinh(x)^2), x)");

    let (result, required) = evaluated_integral_with_required_conditions(
        "integrate(2*cosh(2*x+1)/(1+sinh(2*x+1)^2), x)",
    );

    assert_eq!(result, "arctan(sinh(2 * x + 1))");
    assert!(
        required.is_empty(),
        "positive one-plus-square denominator should not require source conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(2*cosh(2*x+1)/(1+sinh(2*x+1)^2), x)");

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x*cosh(x^2)/(1+sinh(x^2)^2), x)");

    assert_eq!(result, "arctan(sinh(x^2))");
    assert!(
        required.is_empty(),
        "positive one-plus-square denominator should not require source conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(2*x*cosh(x^2)/(1+sinh(x^2)^2), x)");

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x*cosh(x^2)/(sinh(x^2)^2+1), x)");

    assert_eq!(result, "arctan(sinh(x^2))");
    assert!(
        required.is_empty(),
        "positive one-plus-square denominator should not require source conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(2*x*cosh(x^2)/(sinh(x^2)^2+1), x)");

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(sinh(x)/(1+cosh(x)^2), x)");

    assert_eq!(result, "arctan(cosh(x))");
    assert!(
        required.is_empty(),
        "positive one-plus-square denominator should not require source conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(sinh(x)/(1+cosh(x)^2), x)");

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(-2*x*sinh(x^2)/(1+cosh(x^2)^2), x)");

    assert_eq!(result, "-arctan(cosh(x^2))");
    assert!(
        required.is_empty(),
        "positive one-plus-square denominator should not require source conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(-2*x*sinh(x^2)/(1+cosh(x^2)^2), x)");
}
#[test]
fn integrate_contract_trig_arctan_derivative_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(cos(x)/(1+sin(x)^2), x)");

    assert_eq!(result, "arctan(sin(x))");
    assert!(
        required.is_empty(),
        "positive one-plus-square denominator should not require source conditions: {required:?}"
    );
    assert_rendered_antiderivative_verifies("integrate(cos(x)/(1+sin(x)^2), x)", &result);

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*cos(2*x+1)/(1+sin(2*x+1)^2), x)");

    assert_eq!(result, "arctan(sin(2 * x + 1))");
    assert!(
        required.is_empty(),
        "positive one-plus-square denominator should not require source conditions: {required:?}"
    );
    assert_rendered_antiderivative_verifies("integrate(2*cos(2*x+1)/(1+sin(2*x+1)^2), x)", &result);

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*cos(2*x+1)/(sin(2*x+1)^2+1), x)");

    assert_eq!(result, "arctan(sin(2 * x + 1))");
    assert!(
        required.is_empty(),
        "positive one-plus-square denominator should not require source conditions: {required:?}"
    );
    assert_rendered_antiderivative_verifies("integrate(2*cos(2*x+1)/(sin(2*x+1)^2+1), x)", &result);

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x*cos(x^2)/(1+sin(x^2)^2), x)");

    assert_eq!(result, "arctan(sin(x^2))");
    assert!(
        required.is_empty(),
        "positive one-plus-square denominator should not require source conditions: {required:?}"
    );
    assert_rendered_antiderivative_verifies("integrate(2*x*cos(x^2)/(1+sin(x^2)^2), x)", &result);

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x*cos(x^2)/(sin(x^2)^2+1), x)");

    assert_eq!(result, "arctan(sin(x^2))");
    assert!(
        required.is_empty(),
        "positive one-plus-square denominator should not require source conditions: {required:?}"
    );
    assert_rendered_antiderivative_verifies("integrate(2*x*cos(x^2)/(sin(x^2)^2+1), x)", &result);

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(-sin(x)/(1+cos(x)^2), x)");

    assert_eq!(result, "arctan(cos(x))");
    assert!(
        required.is_empty(),
        "positive one-plus-square denominator should not require source conditions: {required:?}"
    );
    assert_rendered_antiderivative_verifies("integrate(-sin(x)/(1+cos(x)^2), x)", &result);

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(-2*sin(2*x+1)/(cos(2*x+1)^2+1), x)");

    assert_eq!(result, "arctan(cos(2 * x + 1))");
    assert!(
        required.is_empty(),
        "positive one-plus-square denominator should not require source conditions: {required:?}"
    );
    assert_rendered_antiderivative_verifies(
        "integrate(-2*sin(2*x+1)/(cos(2*x+1)^2+1), x)",
        &result,
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(-2*x*sin(x^2)/(cos(x^2)^2+1), x)");

    assert_eq!(result, "arctan(cos(x^2))");
    assert!(
        required.is_empty(),
        "positive one-plus-square denominator should not require source conditions: {required:?}"
    );
    assert_rendered_antiderivative_verifies("integrate(-2*x*sin(x^2)/(cos(x^2)^2+1), x)", &result);
}
#[test]
fn integrate_contract_polynomial_hyperbolic_coth_ratio_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x*cosh(x^2)/sinh(x^2), x)");

    assert_eq!(result, "ln(|sinh(x^2)|)");
    assert_eq!(
        required,
        vec!["sinh(x^2) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_polynomial_hyperbolic_tanh_reciprocal_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x/tanh(x^2), x)");

    assert_eq!(result, "ln(|sinh(x^2)|)");
    assert_eq!(
        required,
        vec!["sinh(x^2) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_linear_hyperbolic_tanh_derivative_square_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(1/cosh(2*x + 1)^2, x)");

    assert_eq!(result, "1/2 * tanh(2 * x + 1)");
    assert_eq!(
        required,
        Vec::<String>::new(),
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_polynomial_hyperbolic_tanh_derivative_square_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x/cosh(x^2)^2, x)");

    assert_eq!(result, "tanh(x^2)");
    assert_eq!(
        required,
        Vec::<String>::new(),
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_linear_inverse_table_explains_internal_derivative_without_substitution() {
    for (
        input,
        expected_result,
        expected_required_display,
        expected_substep_title,
        expect_constant_adjustment,
    ) in [
        (
            "integrate(1/sqrt(9-(3*x-2)^2), x)",
            "1/3·arcsin((3·x - 2) / 3)",
            serde_json::json!(["-1/3 < x < 5/3"]),
            "Usar la regla de arcsin con derivada interna",
            true,
        ),
        (
            "integrate(-1/sqrt(9-(3*x-2)^2), x)",
            "-1/3·arcsin((3·x - 2) / 3)",
            serde_json::json!(["-1/3 < x < 5/3"]),
            "Usar la regla de arcsin con derivada interna",
            true,
        ),
        (
            "integrate(3/sqrt(9-(3*x-2)^2), x)",
            "arcsin((3·x - 2) / 3)",
            serde_json::json!(["-1/3 < x < 5/3"]),
            "Usar la regla de arcsin con derivada interna",
            false,
        ),
        (
            "integrate(1/(1+(2*x+1)^2), x)",
            "1/2·arctan(2·x + 1)",
            serde_json::json!([]),
            "Usar la regla de arctan con derivada interna",
            true,
        ),
        (
            "integrate(1/(1+(1-2*x)^2), x)",
            "1/2·arctan(2·x - 1)",
            serde_json::json!([]),
            "Usar la regla de arctan con derivada interna",
            true,
        ),
        (
            "integrate(2/(4-(2*x+1)^2), x)",
            "1/2·atanh((2·x + 1) / 2)",
            serde_json::json!(["-3/2 < x < 1/2"]),
            "Usar la regla de atanh con derivada interna",
            true,
        ),
        (
            "integrate(2/(4-(1-2*x)^2), x)",
            "1/2·atanh((2·x - 1) / 2)",
            serde_json::json!(["-1/2 < x < 3/2"]),
            "Usar la regla de atanh con derivada interna",
            true,
        ),
        (
            "integrate(1/sqrt((2*x+1)^2+1), x)",
            "1/2·asinh(2·x + 1)",
            serde_json::json!([]),
            "Usar la regla de asinh con derivada interna",
            true,
        ),
        (
            "integrate(1/sqrt((1-2*x)^2+1), x)",
            "1/2·asinh(2·x - 1)",
            serde_json::json!([]),
            "Usar la regla de asinh con derivada interna",
            true,
        ),
        (
            "integrate(2/(sqrt(2*x)*sqrt(2*x+2)), x)",
            "acosh(2·x + 1)",
            serde_json::json!(["x > 0"]),
            "Usar la regla de acosh con derivada interna",
            false,
        ),
        (
            "integrate(-1/(2*sqrt(x)*sqrt(x+1)), x)",
            "-1/2·acosh(2·x + 1)",
            serde_json::json!(["x > 0"]),
            "Usar la regla de acosh con derivada interna",
            true,
        ),
        (
            "integrate(2/sqrt((2*x-1)^2-1), x)",
            "acosh(2·x - 1)",
            serde_json::json!(["x > 1"]),
            "Usar la regla de acosh con derivada interna",
            false,
        ),
        (
            "integrate(-2/(sqrt(-2*x)*sqrt(2-2*x)), x)",
            "acosh(1 - 2·x)",
            serde_json::json!(["x < 0"]),
            "Usar la regla de acosh con derivada interna",
            false,
        ),
        (
            "integrate(-2/sqrt((2*x-1)^2-1), x)",
            "acosh(1 - 2·x)",
            serde_json::json!(["x < 0"]),
            "Usar la regla de acosh con derivada interna",
            false,
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(
            wire["required_display"], expected_required_display,
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "linear inverse table trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
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
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == expected_substep_title),
            "expected {expected_substep_title} substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .all(|substep| substep["title"] != "Usar sustitución"),
            "linear inverse table case should not use the polynomial-substitution substep for {input}: {substeps:?}"
        );
        assert_eq!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Ajustar el factor constante"),
            expect_constant_adjustment,
            "unexpected constant adjustment substep presence for {input}: {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }
}
#[test]
fn integrate_contract_linear_elementary_table_explains_internal_derivative_without_substitution() {
    for (input, expected_result, expected_substep_title, expect_constant_adjustment) in [
        (
            "integrate(sin(x+1), x)",
            "-cos(x + 1)",
            "Usar la regla de sin con derivada interna",
            false,
        ),
        (
            "integrate(sin(2*x+1), x)",
            "-1/2·cos(2·x + 1)",
            "Usar la regla de sin con derivada interna",
            true,
        ),
        (
            "integrate(sin(1-2*x), x)",
            "1/2·cos(1 - 2·x)",
            "Usar la regla de sin con derivada interna",
            true,
        ),
        (
            "integrate(cos(2*x+1), x)",
            "1/2·sin(2·x + 1)",
            "Usar la regla de cos con derivada interna",
            true,
        ),
        (
            "integrate(cos(1-2*x), x)",
            "-1/2·sin(1 - 2·x)",
            "Usar la regla de cos con derivada interna",
            true,
        ),
        (
            "integrate(exp(2*x+1), x)",
            "1/2·e^(2·x + 1)",
            "Usar la regla de exp con derivada interna",
            true,
        ),
        (
            "integrate(exp(1-2*x), x)",
            "-1/2·e^(1 - 2·x)",
            "Usar la regla de exp con derivada interna",
            true,
        ),
        (
            "integrate(sinh(2*x+1), x)",
            "1/2·cosh(2·x + 1)",
            "Usar la regla de sinh con derivada interna",
            true,
        ),
        (
            "integrate(sinh(1-2*x), x)",
            "-1/2·cosh(1 - 2·x)",
            "Usar la regla de sinh con derivada interna",
            true,
        ),
        (
            "integrate(cosh(2*x+1), x)",
            "1/2·sinh(2·x + 1)",
            "Usar la regla de cosh con derivada interna",
            true,
        ),
        (
            "integrate(cosh(1-2*x), x)",
            "-1/2·sinh(1 - 2·x)",
            "Usar la regla de cosh con derivada interna",
            true,
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(
            wire["required_display"],
            serde_json::json!([]),
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "linear elementary table trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
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
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == expected_substep_title),
            "expected {expected_substep_title} substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .all(|substep| substep["title"] != "Usar sustitución"),
            "linear elementary table case should not use the polynomial-substitution substep for {input}: {substeps:?}"
        );
        assert_eq!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Ajustar el factor constante"),
            expect_constant_adjustment,
            "unexpected constant adjustment substep presence for {input}: {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }
}
#[test]
fn integrate_contract_reciprocal_linear_log_table_explains_internal_derivative_without_substitution(
) {
    for (input, expected_result, expected_required_display, expect_constant_adjustment) in [
        (
            "integrate(1/(x+1), x)",
            "ln(|x + 1|)",
            serde_json::json!(["x ≠ -1"]),
            false,
        ),
        (
            "integrate(1/(2*x+1), x)",
            "1/2·ln(|2·x + 1|)",
            serde_json::json!(["x ≠ -1/2"]),
            true,
        ),
        (
            "integrate(1/(1-2*x), x)",
            "-1/2·ln(|1 - 2·x|)",
            serde_json::json!(["x ≠ 1/2"]),
            true,
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(
            wire["required_display"], expected_required_display,
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "linear log table trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
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
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Usar la regla de ln|u| con derivada interna"),
            "expected ln|u| substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Identificar el denominador afín"),
            "expected affine denominator substep for {input}, got {substeps:?}"
        );
        assert_eq!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Ajustar el factor constante"),
            expect_constant_adjustment,
            "unexpected constant adjustment substep presence for {input}: {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .all(|substep| substep["title"] != "Usar sustitución"),
            "linear log table case should not use the polynomial-substitution substep for {input}: {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }
}
#[test]
fn integrate_contract_linear_hyperbolic_coth_derivative_square_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(1/sinh(2*x + 1)^2, x)");

    assert_eq!(result, "-1 / (2 * tanh(2 * x + 1))");
    assert_eq!(
        required,
        vec!["sinh(2 * x + 1) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_polynomial_hyperbolic_coth_derivative_square_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x/sinh(x^2)^2, x)");

    assert_eq!(result, "-1 / tanh(x^2)");
    assert_eq!(
        required,
        vec!["sinh(x^2) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_linear_hyperbolic_cosh_reciprocal_derivative_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(sinh(2*x + 1)/cosh(2*x + 1)^2, x)");

    assert_eq!(result, "-1 / (2 * cosh(2 * x + 1))");
    assert_eq!(
        required,
        Vec::<String>::new(),
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_polynomial_hyperbolic_cosh_reciprocal_derivative_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x*sinh(x^2)/cosh(x^2)^2, x)");

    assert_eq!(result, "-1 / cosh(x^2)");
    assert_eq!(
        required,
        Vec::<String>::new(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(-x*sinh(x^2)/cosh(x^2)^2, x)");

    assert_eq!(result, "1 / (2 * cosh(x^2))");
    assert_eq!(
        required,
        Vec::<String>::new(),
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_linear_hyperbolic_sinh_reciprocal_derivative_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(cosh(2*x + 1)/sinh(2*x + 1)^2, x)");

    assert_eq!(result, "-1 / (2 * sinh(2 * x + 1))");
    assert_eq!(
        required,
        vec!["sinh(2 * x + 1) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_polynomial_hyperbolic_sinh_reciprocal_derivative_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x*cosh(x^2)/sinh(x^2)^2, x)");

    assert_eq!(result, "-1 / sinh(x^2)");
    assert_eq!(
        required,
        vec!["sinh(x^2) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(-x*cosh(x^2)/sinh(x^2)^2, x)");

    assert_eq!(result, "1 / (2 * sinh(x^2))");
    assert_eq!(
        required,
        vec!["sinh(x^2) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_constant_base_polynomial_log_substitution_preserves_domain() {
    for (input, expected_result, expected_required) in [
        (
            "integrate(2*x*log(2,x^2+1), x)",
            "(x^2 + 1) * log(2, x^2 + 1) - (x^2 + 1) / ln(2)",
            vec![],
        ),
        (
            "integrate(2*x*log(2,x^2-1), x)",
            "(x^2 - 1) * log(2, x^2 - 1) - (x^2 - 1) / ln(2)",
            vec!["x < -1 or x > 1".to_string()],
        ),
        (
            "integrate(2*x*log2(x^2+1), x)",
            "(x^2 + 1) * log2(x^2 + 1) - (x^2 + 1) / ln(2)",
            vec![],
        ),
        (
            "integrate(2*x*log(2,x^2+x+1)+log(2,x^2+x+1), x)",
            "(x^2 + x + 1) * log(2, x^2 + x + 1) - (x^2 + x + 1) / ln(2)",
            vec![],
        ),
    ] {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected_result, "input: {input}");
        assert_eq!(
            required, expected_required,
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert_antiderivative_verifies(input);
        assert_rendered_antiderivative_verifies(input, &result);
    }

    let (result, _required) =
        evaluated_integral_with_required_conditions("integrate(2*x*log(x,x^2+1), x)");
    assert!(
        result.starts_with("integrate("),
        "symbolic log base should remain residual for polynomial substitution, got {result}"
    );
}
#[test]
fn integrate_contract_constant_base_polynomial_log_trace_uses_substitution() {
    let input = "integrate(2*x*log(2,x^2+1), x)";
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

    assert!(
        stderr.is_empty(),
        "constant-base polynomial log integration should not emit stderr warnings: {stderr}"
    );
    assert_eq!(
        wire["result"],
        "(x^2 + 1)·log(2, x^2 + 1) - (x^2 + 1) / ln(2)"
    );
    assert!(wire["required_display"]
        .as_array()
        .expect("required_display array")
        .is_empty());
    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    assert_eq!(steps.len(), 1, "expected one integration step: {steps:?}");
    assert_eq!(steps[0]["rule"], "Calcular la integral");
    let substeps = steps[0]["substeps"]
        .as_array()
        .expect("substitution should be visible as a compact substep");
    assert_eq!(
        substeps.len(),
        1,
        "expected one substitution substep: {steps:?}"
    );
    assert_eq!(substeps[0]["title"], "Usar sustitución");
}
#[test]
fn integrate_contract_constant_base_additive_polynomial_log_trace_uses_common_factor_then_substitution(
) {
    let input = "integrate(2*x*log(2,x^2+x+1)+log(2,x^2+x+1), x)";
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

    assert!(
        stderr.is_empty(),
        "constant-base additive polynomial log integration should not emit stderr warnings: {stderr}"
    );
    assert_eq!(
        wire["result"],
        "(x^2 + x + 1)·log(2, x^2 + x + 1) - (x^2 + x + 1) / ln(2)"
    );
    assert!(wire["required_display"]
        .as_array()
        .expect("required_display array")
        .is_empty());
    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    assert_eq!(
        steps.len(),
        2,
        "expected factor then integration steps: {steps:?}"
    );
    assert_eq!(steps[0]["rule"], "Sacar factor común");
    assert_eq!(steps[1]["rule"], "Calcular la integral");
    let substeps = steps[1]["substeps"]
        .as_array()
        .expect("substitution should be visible as a compact substep");
    assert_eq!(
        substeps.len(),
        1,
        "expected one substitution substep: {steps:?}"
    );
    assert_eq!(substeps[0]["title"], "Usar sustitución");
}
#[test]
fn integrate_contract_linear_power_substitution() {
    assert_eq!(simplified_integral("integrate((3*x)^2, x)"), "3 * x^3");
}
#[test]
fn integrate_contract_polynomial_derivative_arctan_substitution() {
    assert_eq!(
        simplified_integral("integrate(2*x/(1+x^4), x)"),
        "arctan(x^2)"
    );
}
#[test]
fn integrate_contract_scaled_polynomial_derivative_arctan_substitution() {
    assert_eq!(
        simplified_integral("integrate(2*x/(4+x^4), x)"),
        "1/2 * arctan(x^2 / 2)"
    );
}
#[test]
fn integrate_contract_scaled_polynomial_derivative_atanh_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x/(4-x^4), x)");

    assert_eq!(result, "1/2 * atanh(x^2 / 2)");
    assert_eq!(
        required,
        vec!["4 - x^4 > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(2*x/(4-x^4), x)");
    let (nested_residual, nested_required) =
        evaluated_expr_with_required_conditions("diff(integrate(2*x/(4-x^4), x), x) - 2*x/(4-x^4)");
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["4 - x^4 > 0".to_string()],
        "atanh polynomial residual verification should preserve the open-interval condition"
    );

    let input = "integrate(-2*x/(1-x^4), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "-atanh(x^2)");
    assert_eq!(
        required,
        vec!["1 - x^4 > 0".to_string()],
        "negative atanh substitution should preserve its open-interval domain"
    );
    assert_antiderivative_verifies(input);
}
#[test]
fn integrate_contract_polynomial_derivative_acosh_substitution_preserves_real_domain() {
    let input = "integrate(2*x/sqrt(x^4-4), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "acosh(x^2 / 2)");
    assert_eq!(
        required,
        vec!["x < -sqrt(2) or x > sqrt(2)".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);

    let input = "integrate((2*x+1)/sqrt((x^2+x)^2-4), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "acosh((x^2 + x) / 2)");
    assert_eq!(
        required,
        vec!["x < -2 or x > 1".to_string()],
        "unexpected shifted acosh required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);

    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate((2*x+1)/sqrt((x^2+x)^2-4), x), x) - (2*x+1)/sqrt((x^2+x)^2-4)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["x < -2 or x > 1".to_string()],
        "nested acosh verification should preserve the real-domain conditions"
    );

    let input = "integrate((2*x+1)/sqrt((x^2+x)^2-5), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "acosh((x^2 + x) / sqrt(5))");
    assert_eq!(
        required,
        vec!["x^2 + x - sqrt(5) > 0".to_string()],
        "unexpected shifted surd-width acosh required_conditions: {required:?}"
    );
    assert_antiderivative_equiv_verifies(input);

    let (nested_equiv, nested_required) = evaluated_equiv_with_required_conditions(
        "diff(integrate((2*x+1)/sqrt((x^2+x)^2-5), x), x)",
        "(2*x+1)/sqrt((x^2+x)^2-5)",
    );
    assert!(nested_equiv);
    assert_eq!(
        nested_required,
        vec!["x^4 + 2 * x^3 + x^2 - 5 > 0".to_string()],
        "surd-width acosh equivalence verification should retain the direct radicand domain"
    );

    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate((2*x+1)/sqrt((x^2+x)^2-5), x), x) - (2*x+1)/sqrt((x^2+x)^2-5)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["x^2 + x - sqrt(5) > 0".to_string()],
        "nested surd-width acosh residual should keep the compact real-domain conditions"
    );
}
#[test]
fn integrate_contract_polynomial_derivative_square_minus_constant_log_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x/(x^4-4), x)");

    assert_eq!(result, "1/4 * ln(|(x^2 - 2) / (x^2 + 2)|)");
    assert_eq!(
        required,
        vec!["x^2 - 2 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_positive_log_derivative_power_substitution() {
    let input = "integrate((2*x)/(x^2+1)*ln(x^2+1)^2, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "1/3 * ln(x^2 + 1)^3");
    assert!(
        required.is_empty(),
        "positive log-power substitution should not add required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}
#[test]
fn integrate_contract_abs_log_derivative_power_substitution_preserves_nonzero_domain() {
    let input = "integrate((2*x+1)/(x^2+x-1)*ln(abs(x^2+x-1))^2, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "1/3 * ln(|x^2 + x - 1|)^3");
    assert_eq!(
        required,
        vec!["x^2 + x - 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}
#[test]
fn integrate_contract_scaled_polynomial_derivative_arcsin_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x/sqrt(4-x^4), x)");

    assert_eq!(result, "arcsin(x^2 / 2)");
    assert_eq!(
        required,
        vec!["4 - x^4 > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_shifted_linear_scaled_arcsin_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(1/sqrt(4-(x+1)^2), x)");

    assert_eq!(result, "arcsin((x + 1) / 2)");
    assert_eq!(
        required,
        vec!["-3 < x < 1".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(1/sqrt(4-(x+1)^2), x)");
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(1/sqrt(4-(x+1)^2), x), x) - 1/sqrt(4-(x+1)^2)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["-3 < x < 1".to_string()],
        "nested shifted arcsin verification should preserve its positive radicand condition"
    );
}
#[test]
fn integrate_contract_polynomial_derivative_asinh_substitution() {
    assert_eq!(
        simplified_integral("integrate(2*x/sqrt(1+x^4), x)"),
        "asinh(x^2)"
    );

    let input = "integrate(-2*x/sqrt(1+x^4), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "-asinh(x^2)");
    assert!(
        required.is_empty(),
        "negative asinh substitution should remain unconditional: {required:?}"
    );
    assert_antiderivative_verifies(input);
}
#[test]
fn integrate_contract_scaled_polynomial_derivative_asinh_substitution() {
    assert_eq!(
        simplified_integral("integrate(2*x/sqrt(4+x^4), x)"),
        "asinh(x^2 / 2)"
    );
}
#[test]
fn integrate_contract_shifted_linear_scaled_asinh_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(1/sqrt(4+(x+1)^2), x)");

    assert_eq!(result, "asinh((x + 1) / 2)");
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(1/sqrt(4+(x+1)^2), x)");
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(1/sqrt(4+(x+1)^2), x), x) - 1/sqrt(4+(x+1)^2)",
    );
    assert_eq!(nested_residual, "0");
    assert!(
        nested_required.is_empty(),
        "nested shifted asinh verification should remain unconditional: {nested_required:?}"
    );
}
#[test]
fn integrate_contract_polynomial_derivative_over_square_root_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(x/sqrt(x^2+1), x)");

    assert_eq!(result, "sqrt(x^2 + 1)");
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x/sqrt(x^2-1), x)");

    assert_eq!(result, "2 * sqrt(x^2 - 1)");
    assert_eq!(
        required,
        vec!["x < -1 or x > 1".to_string()],
        "unexpected required_conditions: {required:?}"
    );

    let input = "integrate((3*x^2+2*x+1)/sqrt(x^3+x^2+x+1), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "2 * sqrt(x^3 + x^2 + x + 1)");
    assert_eq!(
        required,
        vec!["x^3 + x^2 + x + 1 > 0".to_string()],
        "unexpected cubic radicand required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);

    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate((3*x^2+2*x+1)/sqrt(x^3+x^2+x+1), x), x) - (3*x^2+2*x+1)/sqrt(x^3+x^2+x+1)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["x^3 + x^2 + x + 1 > 0".to_string()],
        "cubic radicand residual verification should preserve positive-domain conditions"
    );

    let direct_diff = "diff(integrate((3*x^2+2*x+1)/sqrt(x^3+x^2+x+1), x), x)";
    let (direct_wire, direct_stderr) = cli_eval_json_with_stderr(direct_diff);
    assert!(
        direct_stderr.is_empty(),
        "direct square-root substitution diff/integrate should stay quiet: {direct_stderr}"
    );
    assert_eq!(
        direct_wire["result"],
        "(3·x^2 + 2·x + 1) / sqrt(x^3 + x^2 + x + 1)"
    );
    assert_eq!(
        direct_wire["required_display"],
        serde_json::json!(["x^3 + x^2 + x + 1 > 0"])
    );
}
#[test]
fn integrate_contract_polynomial_derivative_times_square_root_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(x*sqrt(x^2+1), x)");

    assert_eq!(result, "1/3 * sqrt(x^2 + 1) * (x^2 + 1)");
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(x*sqrt(x^2+1), x)");

    let input = "integrate(-x*sqrt(x^2+1), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "-(1/3 * (x^2 + 1) * sqrt(x^2 + 1))");
    assert!(
        required.is_empty(),
        "unexpected negative required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);

    let input = "integrate((2*x+1)*sqrt(x^2+x+1), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "2/3 * sqrt(x^2 + x + 1) * (x^2 + x + 1)");
    assert!(
        required.is_empty(),
        "unexpected affine required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);

    let input = "integrate(2*x*sqrt(x^2-1), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "2/3 * sqrt(x^2 - 1) * (x^2 - 1)");
    assert_eq!(
        required,
        vec!["x ≤ -1 or x ≥ 1".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);

    let input = "integrate(-2*x*sqrt(x^2-1), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "-(2/3 * sqrt(x^2 - 1) * (x^2 - 1))");
    assert_eq!(
        required,
        vec!["x ≤ -1 or x ≥ 1".to_string()],
        "negated square-root substitution should preserve the same nonnegative-base condition"
    );
    assert_antiderivative_verifies(input);
}
#[test]
fn integrate_contract_polynomial_derivative_times_power_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x*(x^2+1)^3, x)");

    assert_eq!(result, "1/4 * (x^2 + 1)^4");
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(2*x*(x^2+1)^3, x)");

    let input = "integrate(2*x*(x^2-1)^(3/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "2/5 * (x^2 - 1)^(5/2)");
    assert_eq!(
        required,
        vec!["x ≤ -1 or x ≥ 1".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);

    let input = "integrate(-2*x*(x^2-1)^(3/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "-2/5 * (x^2 - 1)^(5/2)");
    assert_eq!(
        required,
        vec!["x ≤ -1 or x ≥ 1".to_string()],
        "negated power substitution should preserve the same nonnegative-base condition"
    );
    assert_antiderivative_verifies(input);
}
#[test]
fn integrate_contract_polynomial_derivative_over_fractional_denominator_power_substitution() {
    let input = "integrate((2*x+1)/(x^2+x+1)^(3/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "-2 / sqrt(x^2 + x + 1)");
    assert!(
        required.is_empty(),
        "positive quadratic denominator should not emit redundant conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    let step_summaries = evaluated_expr_step_summaries(input);
    assert_eq!(
        step_summaries
            .iter()
            .filter(|(_, rule_name, _)| rule_name == "Symbolic Integration")
            .count(),
        1,
        "integration should stay as one compact didactic step: {step_summaries:?}"
    );
    assert!(
        !step_summaries.iter().any(|(_, rule_name, _)| {
            rule_name == "Rationalize Product Denominator"
                || rule_name == "Cancel Same Base Powers"
                || rule_name == "Present calculus result in compact form"
        }),
        "compact integration trace should not expose rationalize/cancel/post-presentation roundtrip: {step_summaries:?}"
    );
    assert_eq!(
        rationalize_rewrites_for_simplify(input),
        0,
        "held compact integration result should not take an internal rationalize route"
    );
    let direct_diff = "diff(integrate((2*x+1)/(x^2+x+1)^(3/2), x), x)";
    let (direct_wire, direct_stderr) = cli_eval_json_with_stderr(direct_diff);
    assert!(
        direct_stderr.is_empty(),
        "direct fractional denominator-power diff/integrate should stay quiet: {direct_stderr}"
    );
    assert_eq!(direct_wire["result"], "(2·x + 1) / (x^2 + x + 1)^(3 / 2)");
    assert_eq!(direct_wire["required_display"], serde_json::json!([]));

    let direct_diff = "diff(integrate((2*x+1)/(x^2+x+1)^(5/2), x), x)";
    let (direct_wire, direct_stderr) = cli_eval_json_with_stderr(direct_diff);
    assert!(
        direct_stderr.is_empty(),
        "direct higher fractional denominator-power diff/integrate should stay quiet: {direct_stderr}"
    );
    assert_eq!(direct_wire["result"], "(2·x + 1) / (x^2 + x + 1)^(5 / 2)");
    assert_eq!(direct_wire["required_display"], serde_json::json!([]));
    let input = "integrate((2*x+1)/(x^2+x+1)^(5/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    assert_eq!(result, "-2 / (3 * sqrt(x^2 + x + 1) * (x^2 + x + 1))");
    assert!(
        !result.contains("^(3/2)"),
        "post-integration presentation should prefer a polynomial-sqrt denominator: {result}"
    );
    assert!(
        required.is_empty(),
        "higher positive-quadratic denominator should not emit redundant conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    let residual = "diff(integrate((2*x+1)/(x^2+x+1)^(5/2), x), x) - (2*x+1)/(x^2+x+1)^(5/2)";
    let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual);
    assert!(
        residual_stderr.is_empty(),
        "higher fractional denominator-power residual should stay quiet: {residual_stderr}"
    );
    assert_eq!(residual_wire["result"], "0");
    assert_eq!(residual_wire["required_display"], serde_json::json!([]));

    let input = "integrate((2*x+1)/(x^2+x+1)^(7/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    assert_eq!(result, "-2 / (5 * sqrt(x^2 + x + 1) * (x^2 + x + 1)^2)");
    assert!(
        !result.contains("^(5/2)"),
        "deeper post-integration presentation should prefer a polynomial-sqrt denominator: {result}"
    );
    assert!(
        required.is_empty(),
        "deeper positive-quadratic denominator should not emit redundant conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    let residual = "diff(integrate((2*x+1)/(x^2+x+1)^(7/2), x), x) - (2*x+1)/(x^2+x+1)^(7/2)";
    let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual);
    assert!(
        residual_stderr.is_empty(),
        "deeper fractional denominator-power residual should stay quiet: {residual_stderr}"
    );
    assert_eq!(residual_wire["result"], "0");
    assert_eq!(residual_wire["required_display"], serde_json::json!([]));

    let input = "integrate((2*x+1)/(x^2+x+1)^(9/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    assert_eq!(result, "-2 / (7 * sqrt(x^2 + x + 1) * (x^2 + x + 1)^3)");
    assert!(
        !result.contains("^(7/2)"),
        "deepest post-integration presentation should prefer a polynomial-sqrt denominator: {result}"
    );
    assert!(
        required.is_empty(),
        "deepest positive-quadratic denominator should not emit redundant conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    let residual = "diff(integrate((2*x+1)/(x^2+x+1)^(9/2), x), x) - (2*x+1)/(x^2+x+1)^(9/2)";
    let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual);
    assert!(
        residual_stderr.is_empty(),
        "deepest fractional denominator-power residual should stay quiet: {residual_stderr}"
    );
    assert_eq!(residual_wire["result"], "0");
    assert_eq!(residual_wire["required_display"], serde_json::json!([]));

    let input = "integrate((2*x+1)/(sqrt(x^2+x+1)^3), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "-2 / sqrt(x^2 + x + 1)");
    assert!(
        required.is_empty(),
        "sqrt-denominator spelling should share the same positive-quadratic domain: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_eq!(
        rationalize_rewrites_for_simplify(input),
        0,
        "sqrt-denominator spelling should not take an internal rationalize route"
    );

    let input = "integrate(2*x/(x^2-1)^(3/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "-2 / sqrt(x^2 - 1)");
    assert_eq!(
        required,
        vec!["x < -1 or x > 1".to_string()],
        "fractional denominator power should require the base to be positive"
    );
    assert_antiderivative_verifies(input);
    assert_eq!(
        rationalize_rewrites_for_simplify(input),
        0,
        "conditional fractional denominator power should not take an internal rationalize route"
    );
    let input = "integrate(2*x/(x^2-1)^(5/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    assert_eq!(result, "-2 / (3 * (x^2 - 1) * sqrt(x^2 - 1))");
    assert_eq!(
        required,
        vec!["x < -1 or x > 1".to_string()],
        "conditional higher fractional denominator power should preserve the positive-base condition"
    );
    assert_antiderivative_verifies(input);

    let input = "integrate(2*x/(x^2-1)^(7/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    assert_eq!(result, "-2 / (5 * (x^2 - 1)^2 * sqrt(x^2 - 1))");
    assert_eq!(
        required,
        vec!["x < -1 or x > 1".to_string()],
        "conditional deeper fractional denominator power should preserve the positive-base condition"
    );
    assert_antiderivative_verifies(input);

    let input = "integrate(2*x/(x^2-1)^(11/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    assert_eq!(result, "-2 / (9 * (x^2 - 1)^4 * sqrt(x^2 - 1))");
    assert!(
        !result.contains("^(9/2)"),
        "conditional deepest denominator-power presentation should prefer a polynomial-sqrt denominator: {result}"
    );
    assert_eq!(
        required,
        vec!["x < -1 or x > 1".to_string()],
        "conditional deepest fractional denominator power should preserve the positive-base condition"
    );
    assert_antiderivative_verifies(input);
    let residual = "diff(integrate(2*x/(x^2-1)^(11/2), x), x) - 2*x/(x^2-1)^(11/2)";
    let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual);
    assert!(
        residual_stderr.is_empty(),
        "conditional deepest fractional denominator-power residual should stay quiet: {residual_stderr}"
    );
    assert_eq!(residual_wire["result"], "0");
    assert_eq!(
        residual_wire["required_display"],
        serde_json::json!(["x < -1 or x > 1"])
    );

    let input = "integrate(2*x/(x^2-1)^(13/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    assert_eq!(result, "-2 / (11 * (x^2 - 1)^5 * sqrt(x^2 - 1))");
    assert!(
        !result.contains("^(11/2)"),
        "conditional next-depth denominator-power presentation should prefer a polynomial-sqrt denominator: {result}"
    );
    assert_eq!(
        required,
        vec!["x < -1 or x > 1".to_string()],
        "conditional next-depth fractional denominator power should preserve the positive-base condition"
    );
    assert_antiderivative_verifies(input);
    let residual = "diff(integrate(2*x/(x^2-1)^(13/2), x), x) - 2*x/(x^2-1)^(13/2)";
    let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual);
    assert!(
        residual_stderr.is_empty(),
        "conditional next-depth fractional denominator-power residual should stay quiet: {residual_stderr}"
    );
    assert_eq!(residual_wire["result"], "0");
    assert_eq!(
        residual_wire["required_display"],
        serde_json::json!(["x < -1 or x > 1"])
    );

    let input = "integrate(2*x/(x^2-1)^(15/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    assert_eq!(result, "-2 / (13 * (x^2 - 1)^6 * sqrt(x^2 - 1))");
    assert!(
        !result.contains("^(13/2)"),
        "conditional odd-half denominator-power presentation should not depend on a manual exponent whitelist: {result}"
    );
    assert_eq!(
        required,
        vec!["x < -1 or x > 1".to_string()],
        "conditional odd-half fractional denominator power should preserve the positive-base condition"
    );
    assert_antiderivative_verifies(input);
    let residual = "diff(integrate(2*x/(x^2-1)^(15/2), x), x) - 2*x/(x^2-1)^(15/2)";
    let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual);
    assert!(
        residual_stderr.is_empty(),
        "conditional odd-half fractional denominator-power residual should stay quiet: {residual_stderr}"
    );
    assert_eq!(residual_wire["result"], "0");
    assert_eq!(
        residual_wire["required_display"],
        serde_json::json!(["x < -1 or x > 1"])
    );

    let direct_diff = "diff(integrate(2*x/(x^2-1)^(3/2), x), x)";
    let (direct_wire, direct_stderr) = cli_eval_json_with_stderr(direct_diff);
    assert!(
        direct_stderr.is_empty(),
        "direct conditional fractional denominator-power diff/integrate should stay quiet: {direct_stderr}"
    );
    assert_eq!(direct_wire["result"], "2·x / (x^2 - 1)^(3 / 2)");
    assert_eq!(
        direct_wire["required_display"],
        serde_json::json!(["x < -1 or x > 1"])
    );
    let residual = "diff(integrate(2*x/(x^2-1)^(5/2), x), x) - 2*x/(x^2-1)^(5/2)";
    let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual);
    assert!(
        residual_stderr.is_empty(),
        "higher conditional fractional denominator-power residual should stay quiet: {residual_stderr}"
    );
    assert_eq!(residual_wire["result"], "0");
    assert_eq!(
        residual_wire["required_display"],
        serde_json::json!(["x < -1 or x > 1"])
    );

    let input = "integrate((4*x+2)/(2*x^2+2*x-3)^(3/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "-2 / sqrt(2 * x^2 + 2 * x - 3)");
    assert_eq!(
        required,
        vec!["x < -1/2 - sqrt(7)/2 or x > -1/2 + sqrt(7)/2".to_string()],
        "scaled shifted fractional denominator power should preserve the positive-base condition"
    );
    assert_antiderivative_verifies(input);
    let input = "integrate((4*x+2)/(2*x^2+2*x-3)^(5/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    assert_eq!(
        result,
        "-2 / (3 * sqrt(2 * x^2 + 2 * x - 3) * (2 * x^2 + 2 * x - 3))"
    );
    assert_eq!(
        required,
        vec!["x < -1/2 - sqrt(7)/2 or x > -1/2 + sqrt(7)/2".to_string()],
        "higher scaled shifted fractional denominator power should preserve the positive-base condition"
    );
    assert_antiderivative_verifies(input);
    let input = "integrate((4*x+2)/(2*x^2+2*x-3)^(7/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    assert_eq!(
        result,
        "-2 / (5 * sqrt(2 * x^2 + 2 * x - 3) * (2 * x^2 + 2 * x - 3)^2)"
    );
    assert_eq!(
        required,
        vec!["x < -1/2 - sqrt(7)/2 or x > -1/2 + sqrt(7)/2".to_string()],
        "deeper scaled shifted fractional denominator power should preserve the positive-base condition"
    );
    assert_antiderivative_verifies(input);
    let residual =
        "diff(integrate((4*x+2)/(2*x^2+2*x-3)^(5/2), x), x) - (4*x+2)/(2*x^2+2*x-3)^(5/2)";
    let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual);
    assert!(
        residual_stderr.is_empty(),
        "higher scaled shifted fractional denominator-power residual should stay quiet: {residual_stderr}"
    );
    assert_eq!(residual_wire["result"], "0");
    assert_eq!(
        residual_wire["required_display"],
        serde_json::json!(["x < -1/2 - sqrt(7)/2 or x > -1/2 + sqrt(7)/2"])
    );
    let residual =
        "diff(integrate((4*x+2)/(2*x^2+2*x-3)^(7/2), x), x) - (4*x+2)/(2*x^2+2*x-3)^(7/2)";
    let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual);
    assert!(
        residual_stderr.is_empty(),
        "deeper scaled shifted fractional denominator-power residual should stay quiet: {residual_stderr}"
    );
    assert_eq!(residual_wire["result"], "0");
    assert_eq!(
        residual_wire["required_display"],
        serde_json::json!(["x < -1/2 - sqrt(7)/2 or x > -1/2 + sqrt(7)/2"])
    );
    assert_eq!(
        rationalize_rewrites_for_simplify(input),
        0,
        "scaled shifted fractional denominator power should not take an internal rationalize route"
    );

    let input = "integrate(-2*x/(x^2-1)^(3/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "2 / sqrt(x^2 - 1)");
    assert_eq!(
        required,
        vec!["x < -1 or x > 1".to_string()],
        "negated fractional denominator power should preserve the same positive-base condition"
    );
    assert_antiderivative_verifies(input);
    assert_eq!(
        rationalize_rewrites_for_simplify(input),
        0,
        "negated fractional denominator power should not take an internal rationalize route"
    );
}
#[test]
fn integrate_contract_reciprocal_cos_power_via_u_substitution() {
    // `sin(x)/cos(x)^n` for n >= 4 (and `sin^odd/cos^n`) declined: the polynomial-only odd-power owner
    // does not accept a NEGATIVE companion power. The u = cos substitution `∫ sin^p cos^q dx`
    // (p odd) now integrates the companion by the power rule for any integer q, closing the gap.
    // Certified by differentiating the antiderivative back to the integrand.
    let r = |input: &str| evaluated_integral_with_required_conditions(input).0;
    // Result-form pins for the newly-closed cases (the antiderivative differentiates back, checked by
    // hand and numerically): u = cos gives `1/((n-1)cos^(n-1))`. Affine argument scales by 1/a.
    assert_eq!(r("integrate(sin(x)/cos(x)^4, x)"), "1 / (3 * cos(x)^3)");
    assert_eq!(r("integrate(sin(x)/cos(x)^5, x)"), "1 / (4 * cos(x)^4)");
    assert_eq!(
        r("integrate(sin(2*x)/cos(2*x)^4, x)"),
        "1 / (6 * cos(2 * x)^3)"
    );
    // The odd-numerator-power case (the (1-u^2) expansion branch); form verified by hand/numerically
    // to differentiate back to sin(x)^3/cos(x)^4 (a result pin, not the slow simplify-and-diff path).
    assert_eq!(
        r("integrate(sin(x)^3/cos(x)^4, x)"),
        "(-3/2 * (2 * cos(x)^2 - 1) - 1/2) / (3 * cos(x)^3)"
    );
    // Existing owners keep their forms (fallback placement): pins guard against the u-substitution
    // accidentally taking over the canonical sec/tan/polynomial spellings.
    assert_eq!(r("integrate(sin(x)/cos(x)^2, x)"), "sec(x)");
    assert_eq!(r("integrate(sin(x)/cos(x)^3, x)"), "tan(x)^2 / 2");
    assert_eq!(
        r("integrate(sin(x)^3*cos(x)^2, x)"),
        "1/15 * (3 * cos(x)^5 - 5 * cos(x)^3)"
    );
}
#[test]
fn integrate_contract_linear_secant_squared_substitution_preserves_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(sec(2*x + 1)^2, x)");

    assert_eq!(result, "1/2 * tan(2 * x + 1)");
    assert_eq!(
        required,
        vec!["cos(2 * x + 1) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_linear_cosecant_squared_substitution_preserves_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(csc(2*x + 1)^2, x)");

    assert_eq!(result, "-cot(2 * x + 1) / 2");
    assert_eq!(
        required,
        vec!["sin(2 * x + 1) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_polynomial_secant_squared_substitution_preserves_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(x/(cos(x^2)^2), x)");

    assert_eq!(result, "tan(x^2) / 2");
    assert_eq!(
        required,
        vec!["cos(x^2) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_polynomial_cosecant_squared_substitution_preserves_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(x^2/(sin(x^3)^2), x)");

    assert_eq!(result, "-cot(x^3) / 3");
    assert_eq!(
        required,
        vec!["sin(x^3) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_polynomial_trig_derivative_substitution_preserves_compact_arg() {
    let input = "integrate((4*x^3-2*x)*sin(x^4-x^2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "-cos(x^4 - x^2)");
    assert!(
        required.is_empty(),
        "polynomial sine substitution should not add required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);

    let input = "integrate((4*x^3-2*x)*cos(x^4-x^2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "sin(x^4 - x^2)");
    assert!(
        required.is_empty(),
        "polynomial cosine substitution should not add required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}
#[test]
fn integrate_contract_polynomial_trig_log_substitution_explains_u_and_du() {
    for (
        input,
        expected_result,
        expected_required_display,
        expected_rule_title,
        expects_constant_adjustment,
    ) in [
        (
            "integrate(x*tan(x^2), x)",
            "-1/2·ln(|cos(x^2)|)",
            serde_json::json!(["cos(x^2) ≠ 0"]),
            "Usar la regla de tan(u) -> -ln|cos(u)|",
            true,
        ),
        (
            "integrate(x*cot(x^2), x)",
            "1/2·ln(|sin(x^2)|)",
            serde_json::json!(["sin(x^2) ≠ 0"]),
            "Usar la regla de cot(u) -> ln|sin(u)|",
            true,
        ),
        (
            "integrate(2*x*sec(x^2), x)",
            "ln(|tan(x^2) + sec(x^2)|)",
            serde_json::json!(["cos(x^2) ≠ 0"]),
            "Usar la regla de sec(u) -> ln|sec(u)+tan(u)|",
            false,
        ),
        (
            "integrate(x*sec(x^2), x)",
            "1/2·ln(|tan(x^2) + sec(x^2)|)",
            serde_json::json!(["cos(x^2) ≠ 0"]),
            "Usar la regla de sec(u) -> ln|sec(u)+tan(u)|",
            true,
        ),
        (
            "integrate(2*x*csc(x^2), x)",
            "ln(|csc(x^2) - cot(x^2)|)",
            serde_json::json!(["sin(x^2) ≠ 0"]),
            "Usar la regla de csc(u) -> ln|csc(u)-cot(u)|",
            false,
        ),
        (
            "integrate(x*csc(x^2), x)",
            "1/2·ln(|csc(x^2) - cot(x^2)|)",
            serde_json::json!(["sin(x^2) ≠ 0"]),
            "Usar la regla de csc(u) -> ln|csc(u)-cot(u)|",
            true,
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(
            wire["required_display"], expected_required_display,
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "polynomial trig log trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
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
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == expected_rule_title),
            "expected {expected_rule_title} substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Identificar u y du"),
            "expected concrete u/du substep for {input}, got {substeps:?}"
        );
        assert_u_du_substep_labels(substeps, input);
        assert_eq!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Ajustar el factor constante"),
            expects_constant_adjustment,
            "unexpected constant adjustment substep presence for {input}: {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .all(|substep| substep["title"] != "Usar sustitución"),
            "polynomial trig log table case should not use the generic substitution substep for {input}: {substeps:?}"
        );
    }
}
#[test]
fn integrate_contract_shifted_trig_power_u_du_keeps_composed_form() {
    // Regresión de L16-pista-(a): sin la vía u-du simbólica, este integrando
    // caía al carril Weierstrass y colgaba (>240 s tras cerrar el wrong
    // answer 7/3 de L15). La primitiva debe salir en forma compuesta.
    assert_eq!(
        simplified_integral("integrate(cos(x)*(sin(x)+1)^2, x)"),
        "1/3 * (sin(x) + 1)^3"
    );
}
#[test]
fn integrate_contract_shifted_trig_power_u_du_higher_power_and_offset() {
    assert_eq!(
        simplified_integral("integrate(cos(x)*(sin(x)+1)^3, x)"),
        "1/4 * (sin(x) + 1)^4"
    );
    assert_eq!(
        simplified_integral("integrate(cos(x)*(sin(x)+2)^2, x)"),
        "1/3 * (sin(x) + 2)^3"
    );
}
#[test]
fn integrate_contract_shifted_trig_reciprocal_u_du_textbook_forms() {
    // Extensión Div de la vía u-du simbólica: ∫s·u′/uᵐ con base compuesta no
    // polinómica. Antes: forma de medio ángulo ilegible (m=2), HANG (m=3 vía
    // exponente negativo) y forma no-libro para el logaritmo.
    assert_eq!(
        simplified_integral("integrate(cos(x)/(sin(x)+2)^2, x)"),
        "-1 / (sin(x) + 2)"
    );
    assert_eq!(
        simplified_integral("integrate(cos(x)*(sin(x)+2)^(-3), x)"),
        "-1 / (2 * (sin(x) + 2)^2)"
    );
}
#[test]
fn integrate_contract_shifted_log_u_du_abs_semantics() {
    // m=1 → s·ln(|u|), y el |·| queda en manos de la decidibilidad del signo:
    // sin(x)+2 ∈ [1,3] es positivo demostrable y el abs se pela; sinh(x)+3
    // puede anularse y el abs se conserva.
    assert_eq!(
        simplified_integral("integrate(cos(x)/(sin(x)+2), x)"),
        "ln(sin(x) + 2)"
    );
    assert_eq!(
        simplified_integral("integrate(cosh(x)/(sinh(x)+3), x)"),
        "ln(|sinh(x) + 3|)"
    );
}
#[test]
fn integrate_contract_symbolic_table_u_du_trig_outer() {
    // Tabla u-du simbólica (F ∈ {sin,cos,…} con u compuesta): antes, Werner
    // trataba sin(x) como ángulo independiente y el residual quedaba con el
    // integrando destrozado (∫cos(sin(x)+x)+…).
    assert_eq!(
        simplified_integral("integrate(cos(x)*cos(sin(x)), x)"),
        "sin(sin(x))"
    );
    assert_eq!(
        simplified_integral("integrate(cos(x)*sin(sin(x)), x)"),
        "-cos(sin(x))"
    );
    // regla de la cadena en el du: cos(2x) = (1/2)·d(sin(2x))/dx
    assert_eq!(
        simplified_integral("integrate(cos(2*x)*cos(sin(2*x)), x)"),
        "1/2 * sin(sin(2 * x))"
    );
}
#[test]
fn integrate_contract_werner_keeps_linear_angle_products() {
    // El gate de Werner declina solo con ángulos que contienen funciones; los
    // lineales siguen siendo suyos. Por el PIPELINE real (CLI): el harness
    // `simplified_integral` corre otra configuración (sin orquestador y con
    // Double Angle desactivada) donde la expansión múltiple-ángulo destroza
    // el par ANTES de Werner — preexistente y ajeno a este gate.
    let (wire, _) = cli_eval_json_with_stderr("integrate(sin(3*x)*cos(5*x), x)");
    let result = wire["result"].as_str().expect("result");
    assert!(
        !result.contains("integrate("),
        "Werner debe seguir integrando el par lineal, got {result}"
    );
    assert!(
        result.contains("cos(8"),
        "esperaba la forma Werner con cos(8x), got {result}"
    );
}
#[test]
fn integrate_contract_symbolic_table_nested_u() {
    // u COMPUESTA anidada: la ruta pliega la derivada cruda (u^(2-1)) con el
    // extractor plegador y reconoce exp canonizada como Pow(E, u).
    assert_eq!(
        simplified_integral("integrate(cos(x)*sin(x)*exp(sin(x)^2), x)"),
        "1/2 * e^(sin(x)^2)"
    );
    // composición TRIPLE: u = sin(x^2), du = 2x·cos(x^2) dx
    assert_eq!(
        simplified_integral("integrate(2*x*cos(x^2)*exp(sin(x^2)), x)"),
        "e^sin(x^2)"
    );
}
