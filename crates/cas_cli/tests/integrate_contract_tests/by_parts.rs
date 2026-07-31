use super::*;

#[test]
fn integrate_contract_antiderivative_verification_uses_bounded_public_residual_for_hyperbolic_by_parts(
) {
    for input in [
        "integrate(x^2*sinh(x), x)",
        "integrate(x^2*cosh(x), x)",
        "integrate((x^3+x)*sinh(2*x+1), x)",
        "integrate((x^3+x)*cosh(2*x+1), x)",
        "integrate(x^6*sinh(x), x)",
        "integrate(x^6*cosh(x), x)",
        "integrate((x^6+1)*sinh(2*x+1), x)",
        "integrate((x^6+1)*cosh(2*x+1), x)",
        "integrate((x+1)*sinh((3*x+2)/2), x)",
        "integrate((x+1)*cosh((3*x+2)/2), x)",
        "integrate((x+1)*sinh((2-3*x)/2), x)",
        "integrate((x+1)*cosh((2-3*x)/2), x)",
        "integrate(x^7*sinh(x), x)",
        "integrate(x^7*cosh(x), x)",
        "integrate((x^7+1)*sinh(2*x+1), x)",
        "integrate((x^7+1)*cosh(2*x+1), x)",
    ] {
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "{input} should verify through the bounded public residual route"
        );
    }
}
#[test]
fn integrate_contract_hyperbolic_by_parts_double_nested_residual_compacts_without_timeout() {
    for input in [
        "((((diff(integrate(x^5*sinh(2*x+1), x), x) - x^5*sinh(2*x+1)) + 1)/(x+2))/(x+3))/(x+4)",
        "((((diff(integrate(x^4*cosh(2*x+1), x), x) - x^4*cosh(2*x+1)) + 1)/(x+2))/(x+3))/(x+4)",
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr(input);
        assert!(
            stderr.is_empty(),
            "unexpected stderr for hyperbolic double-nested residual: {stderr}"
        );
        assert_eq!(wire["result"], "1 / ((x + 2)·(x + 3)·(x + 4))", "{input}");
        assert_eq!(
            wire["required_display"],
            serde_json::json!(["x ≠ -2", "x ≠ -3", "x ≠ -4"]),
            "{input}"
        );
    }
}
#[test]
fn integrate_contract_antiderivative_verification_uses_bounded_public_residual_for_exp_by_parts() {
    for input in [
        "integrate(x^2*exp(x), x)",
        "integrate(x^3*exp(x), x)",
        "integrate(x^5*exp(x), x)",
        "integrate((x^3+x)*exp(2*x+1), x)",
        "integrate((x^6+1)*exp(2*x+1), x)",
        "integrate(x^7*exp(x), x)",
        "integrate((x^7+1)*exp(2*x+1), x)",
    ] {
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "{input} should verify through the bounded public residual route"
        );
    }
}
#[test]
fn integrate_contract_antiderivative_verification_uses_bounded_public_residual_for_trig_by_parts() {
    for input in [
        "integrate(x*sin(x), x)",
        "integrate(x*cos(x), x)",
        "integrate(x^2*sin(x), x)",
        "integrate(x^2*cos(x), x)",
        "integrate(x^5*sin(x), x)",
        "integrate(x^5*cos(x), x)",
        "integrate(x^6*sin(x), x)",
        "integrate(x^6*cos(x), x)",
        "integrate((x^6+1)*sin(2*x+1), x)",
        "integrate((x^6+1)*cos(2*x+1), x)",
        "integrate(x^7*sin(x), x)",
        "integrate(x^7*cos(x), x)",
        "integrate((x^7+1)*sin(2*x+1), x)",
        "integrate((x^7+1)*cos(2*x+1), x)",
        "integrate((2*x+3)*sin(2*x+1), x)",
        "integrate((2*x+3)*cos(2*x+1), x)",
    ] {
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "{input} should verify through the bounded public residual route"
        );
    }
}
#[test]
fn integrate_contract_antiderivative_verification_uses_bounded_public_residual_for_log_by_parts() {
    for input in ["integrate(x*ln(x), x)", "integrate((2*x+1)*ln(2*x+1), x)"] {
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "{input} should verify through the bounded public residual route"
        );
    }
}
#[test]
fn integrate_contract_reciprocal_shifted_trig_by_parts_residual_keeps_compact_requires() {
    let input = "1/((diff(integrate(x^3*sin(x), x), x) - x^3*sin(x)) + x + 2) - 1/(x+2)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert!(
        stderr.is_empty(),
        "unexpected stderr for reciprocal shifted trig by-parts residual: {stderr}"
    );
    assert_eq!(wire["result"], "0");
    assert_eq!(wire["required_display"], serde_json::json!(["x ≠ -2"]));
}
#[test]
fn integrate_contract_product_zero_by_parts_residuals_compact_without_timeout() {
    for input in [
        "((diff(integrate(x^6*sin(x), x), x) - x^6*sin(x)) + x + 2)*(y-y)",
        "((diff(integrate((x^3+x)*cosh(2*x+1), x), x) - ((x^3+x)*cosh(2*x+1))) + x + 2)*(y-y)",
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr(input);

        assert!(
            stderr.is_empty(),
            "unexpected stderr for product-zero by-parts residual: {stderr}"
        );
        assert_eq!(wire["result"], "0", "{input}");
        assert_eq!(wire["required_display"], serde_json::json!([]), "{input}");
    }
}
#[test]
fn integrate_contract_negative_by_parts_primitives_keep_compact_public_form() {
    let cases = [
        (
            "integrate(-x*exp(x), x)",
            "e^x·(1 - x)",
            "{e}^{x}\\cdot (1 - x)",
            "diff(integrate(-x*exp(x), x), x) + x*exp(x)",
        ),
        (
            "integrate(-x^2*sin(x), x)",
            "-2·x·sin(x) + (x^2 - 2)·cos(x)",
            "-2\\cdot x\\cdot \\sin(x) + ({x}^{2} - 2)\\cdot \\cos(x)",
            "diff(integrate(-x^2*sin(x), x), x) + x^2*sin(x)",
        ),
    ];

    for (input, expected_result, expected_latex, residual) in cases {
        let (wire, stderr) = cli_eval_json_with_stderr(input);
        assert!(
            stderr.is_empty(),
            "unexpected stderr for negative by-parts primitive: {stderr}"
        );
        assert_eq!(wire["result"], expected_result);
        assert_eq!(wire["result_latex"], expected_latex);
        assert_eq!(wire["required_display"], serde_json::json!([]));

        let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual);
        assert!(
            residual_stderr.is_empty(),
            "unexpected stderr for negative by-parts residual: {residual_stderr}"
        );
        assert_eq!(residual_wire["result"], "0");
        assert_eq!(residual_wire["required_display"], serde_json::json!([]));
    }
}
#[test]
fn integrate_contract_negative_affine_trig_by_parts_keeps_compact_public_form() {
    let input = "integrate(-(2*x+3)*sin(2*x+1), x)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);
    assert!(
        stderr.is_empty(),
        "unexpected stderr for negative affine trig by-parts primitive: {stderr}"
    );
    assert_eq!(
        wire["result"],
        "1/2·(cos(2·x + 1)·(2·x + 3) - sin(2·x + 1))"
    );
    assert_eq!(
        wire["result_latex"],
        "\\frac{\\cos(2\\cdot x + 1)\\cdot (2\\cdot x + 3) - \\sin(2\\cdot x + 1)}{2}"
    );
    assert_eq!(wire["required_display"], serde_json::json!([]));

    let residual = "diff(integrate(-(2*x+3)*sin(2*x+1), x), x) + (2*x+3)*sin(2*x+1)";
    let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual);
    assert!(
        residual_stderr.is_empty(),
        "unexpected stderr for negative affine trig by-parts residual: {residual_stderr}"
    );
    assert_eq!(residual_wire["result"], "0");
    assert_eq!(residual_wire["required_display"], serde_json::json!([]));
}
#[test]
fn integrate_contract_quadratic_trig_by_parts_presents_without_blocked_hint() {
    let (sin_result, sin_required, sin_blocked) =
        evaluated_expr_with_required_conditions_and_blocked_count("integrate(x^2*sin(x), x)");
    assert_eq!(sin_result, "2 * x * sin(x) + (2 - x^2) * cos(x)");
    assert!(
        sin_required.is_empty(),
        "unexpected required conditions: {sin_required:?}"
    );
    assert_eq!(sin_blocked, 0, "unexpected blocked hints for x^2*sin(x)");
    assert_antiderivative_verifies("integrate(x^2*sin(x), x)");

    let (cos_result, cos_required, cos_blocked) =
        evaluated_expr_with_required_conditions_and_blocked_count("integrate(x^2*cos(x), x)");
    assert_eq!(cos_result, "2 * x * cos(x) + (x^2 - 2) * sin(x)");
    assert!(
        cos_required.is_empty(),
        "unexpected required conditions: {cos_required:?}"
    );
    assert_eq!(cos_blocked, 0, "unexpected blocked hints for x^2*cos(x)");
    assert_antiderivative_verifies("integrate(x^2*cos(x), x)");

    let expanded = "integrate(x^2*sin(2*x+1)+x*sin(2*x+1), x)";
    let (expanded_result, expanded_required, expanded_blocked) =
        evaluated_expr_with_required_conditions_and_blocked_count(expanded);
    assert!(
        !expanded_result.starts_with("integrate("),
        "expected additive common-trig by-parts primitive, got {expanded_result}"
    );
    assert!(
        expanded_required.is_empty(),
        "unexpected required conditions for additive common-trig by-parts: {expanded_required:?}"
    );
    assert_eq!(
        expanded_blocked, 0,
        "unexpected blocked hints for additive common-trig by-parts"
    );

    let residual =
        "diff(integrate(x^2*sin(2*x+1)+x*sin(2*x+1), x), x) - (x^2*sin(2*x+1)+x*sin(2*x+1))";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "unexpected residual required conditions for additive common-trig by-parts: {residual_required:?}"
    );

    let (cos_equiv, cos_equiv_required) = evaluated_equiv_with_required_conditions(
        "diff(integrate(x^2*cos(2*x+1)+x*cos(2*x+1), x), x)",
        "x^2*cos(2*x+1)+x*cos(2*x+1)",
    );
    assert!(
        cos_equiv,
        "public equivalence should reuse the direct residual proof for additive common-trig by-parts"
    );
    assert!(
        cos_equiv_required.is_empty(),
        "unexpected public equivalence required conditions for additive common-trig by-parts: {cos_equiv_required:?}"
    );
}
#[test]
fn integrate_contract_by_parts_residual_dedupes_integrate_denominator_condition() {
    let residual = "1/((integrate(x^2*cos(x),x))+c) - 1/((2*x*cos(x)+(x^2-2)*sin(x))+c)";
    let (wire, stderr) = cli_eval_json_with_stderr(residual);

    assert!(
        stderr.is_empty(),
        "unexpected stderr for by-parts denominator residual: {stderr}"
    );
    assert_eq!(wire["result"], "0");
    assert_eq!(
        wire["required_display"],
        serde_json::json!(["2·x·cos(x) + (x^2 - 2)·sin(x) + c ≠ 0"])
    );
}
#[test]
fn integrate_contract_cubic_trig_by_parts_presents_without_blocked_hint() {
    let (sin_result, sin_required, sin_blocked) =
        evaluated_expr_with_required_conditions_and_blocked_count("integrate(x^3*sin(x), x)");
    assert_eq!(
        sin_result,
        "(6 * x - x^3) * cos(x) + (3 * x^2 - 6) * sin(x)"
    );
    assert!(
        sin_required.is_empty(),
        "unexpected required conditions: {sin_required:?}"
    );
    assert_eq!(sin_blocked, 0, "unexpected blocked hints for x^3*sin(x)");
    assert_antiderivative_verifies("integrate(x^3*sin(x), x)");

    let (cos_result, cos_required, cos_blocked) =
        evaluated_expr_with_required_conditions_and_blocked_count("integrate(x^3*cos(x), x)");
    assert_eq!(
        cos_result,
        "(x^3 - 6 * x) * sin(x) + (3 * x^2 - 6) * cos(x)"
    );
    assert!(
        cos_required.is_empty(),
        "unexpected required conditions: {cos_required:?}"
    );
    assert_eq!(cos_blocked, 0, "unexpected blocked hints for x^3*cos(x)");
    assert_antiderivative_verifies("integrate(x^3*cos(x), x)");

    for (lhs, rhs) in [
        (
            "diff(integrate(x^3*sin(2*x+1)+x*sin(2*x+1), x), x)",
            "x^3*sin(2*x+1)+x*sin(2*x+1)",
        ),
        (
            "diff(integrate(x^3*cos(2*x+1)+x*cos(2*x+1), x), x)",
            "x^3*cos(2*x+1)+x*cos(2*x+1)",
        ),
    ] {
        let (equivalent, required) = evaluated_equiv_with_required_conditions(lhs, rhs);
        assert!(
            equivalent,
            "public equivalence should reuse the direct residual proof for {lhs} equiv {rhs}"
        );
        assert!(
            required.is_empty(),
            "unexpected public equivalence required conditions for {lhs} equiv {rhs}: {required:?}"
        );
    }
}
#[test]
fn integrate_contract_quintic_trig_by_parts_presents_without_blocked_hint() {
    let (sin_wire, sin_stderr) = cli_eval_json_with_stderr("integrate(x^5*sin(x), x)");
    assert_eq!(
        sin_wire["result"],
        "(-x^5 + 20·x^3 - 120·x)·cos(x) + (5·x^4 - 60·x^2 + 120)·sin(x)"
    );
    assert!(
        sin_wire["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .is_empty(),
        "unexpected required conditions for x^5*sin(x): {:?}",
        sin_wire["required_conditions"]
    );
    assert!(
        !sin_stderr.contains("depth_overflow"),
        "quintic sin by-parts presentation should not emit depth_overflow warning\nstderr:\n{sin_stderr}"
    );
    assert_antiderivative_verifies("integrate(x^5*sin(x), x)");

    let (cos_wire, cos_stderr) = cli_eval_json_with_stderr("integrate(x^5*cos(x), x)");
    assert_eq!(
        cos_wire["result"],
        "(x^5 - 20·x^3 + 120·x)·sin(x) + (5·x^4 - 60·x^2 + 120)·cos(x)"
    );
    assert!(
        cos_wire["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .is_empty(),
        "unexpected required conditions for x^5*cos(x): {:?}",
        cos_wire["required_conditions"]
    );
    assert!(
        !cos_stderr.contains("depth_overflow"),
        "quintic cos by-parts presentation should not emit depth_overflow warning\nstderr:\n{cos_stderr}"
    );
    assert_antiderivative_verifies("integrate(x^5*cos(x), x)");
}
#[test]
fn integrate_contract_quintic_trig_by_parts_nested_residual_verifies_publicly() {
    for residual in [
        "diff(integrate(x^5*sin(x), x), x) - x^5*sin(x)",
        "diff(integrate(x^5*cos(x), x), x) - x^5*cos(x)",
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr(residual);
        assert_eq!(wire["result"], "0");
        assert!(
            wire["required_conditions"]
                .as_array()
                .expect("required_conditions should be an array")
                .is_empty(),
            "unexpected required conditions for {residual}: {:?}",
            wire["required_conditions"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "quintic trig by-parts nested residual should not emit depth_overflow warning\nstderr:\n{stderr}"
        );
    }
}
#[test]
fn integrate_contract_sextic_trig_by_parts_verifies_publicly() {
    let (sin_wire, sin_stderr) =
        cli_eval_json_with_stderr_args("integrate(x^6*sin(x), x)", &["--steps", "on"]);
    assert_eq!(
        sin_wire["result"],
        "(6·x^5 - 120·x^3 + 720·x)·sin(x) + (-x^6 + 30·x^4 - 360·x^2 + 720)·cos(x)"
    );
    assert!(
        sin_wire["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .is_empty(),
        "unexpected required conditions for x^6*sin(x): {:?}",
        sin_wire["required_conditions"]
    );
    assert!(
        !sin_stderr.contains("depth_overflow"),
        "sextic sin by-parts presentation should not emit depth_overflow warning\nstderr:\n{sin_stderr}"
    );
    let steps = sin_wire["steps"]
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
            .any(|substep| substep["title"] == "Usar integración por partes repetida"),
        "expected repeated integration-by-parts substep for x^6*sin(x), got {substeps:?}"
    );
    assert_eq!(
        assert_antiderivative_verifies("integrate(x^6*sin(x), x)"),
        AntiderivativeVerificationRoute::PublicResidual
    );

    let (cos_wire, cos_stderr) =
        cli_eval_json_with_stderr_args("integrate(x^6*cos(x), x)", &["--steps", "on"]);
    assert_eq!(
        cos_wire["result"],
        "(6·x^5 - 120·x^3 + 720·x)·cos(x) + (x^6 - 30·x^4 + 360·x^2 - 720)·sin(x)"
    );
    assert!(
        cos_wire["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .is_empty(),
        "unexpected required conditions for x^6*cos(x): {:?}",
        cos_wire["required_conditions"]
    );
    assert!(
        !cos_stderr.contains("depth_overflow"),
        "sextic cos by-parts presentation should not emit depth_overflow warning\nstderr:\n{cos_stderr}"
    );
    assert_eq!(
        assert_antiderivative_verifies("integrate(x^6*cos(x), x)"),
        AntiderivativeVerificationRoute::PublicResidual
    );

    for residual in [
        "diff(integrate(x^6*sin(x), x), x) - x^6*sin(x)",
        "diff(integrate(x^6*cos(x), x), x) - x^6*cos(x)",
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr(residual);
        assert_eq!(wire["result"], "0");
        assert!(
            wire["required_conditions"]
                .as_array()
                .expect("required_conditions should be an array")
                .is_empty(),
            "unexpected required conditions for {residual}: {:?}",
            wire["required_conditions"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "sextic trig by-parts nested residual should not emit depth_overflow warning\nstderr:\n{stderr}"
        );
    }
}
#[test]
fn integrate_contract_sparse_affine_sextic_trig_by_parts_verifies_publicly() {
    for (input, expected_result) in [
        (
            "integrate((x^6+1)*sin(2*x+1), x)",
            "(3/2·x^5 - 15/2·x^3 + 45/4·x)·sin(2·x + 1) + (-1/2·x^6 + 15/4·x^4 - 45/4·x^2 + 41/8)·cos(2·x + 1)",
        ),
        (
            "integrate((x^6+1)*cos(2*x+1), x)",
            "(3/2·x^5 - 15/2·x^3 + 45/4·x)·cos(2·x + 1) + (1/2·x^6 - 15/4·x^4 + 45/4·x^2 - 41/8)·sin(2·x + 1)",
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result);
        assert!(
            wire["required_conditions"]
                .as_array()
                .expect("required_conditions should be an array")
                .is_empty(),
            "unexpected required conditions for {input}: {:?}",
            wire["required_conditions"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "sparse affine sextic trig by-parts presentation should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        assert_eq!(
            steps.len(),
            1,
            "expected direct integration trace without expansion noise for {input}, got {steps:?}"
        );
        assert!(
            steps
                .iter()
                .all(|step| step["rule"] != "Expandir la expresión"),
            "sparse affine sextic trig by-parts should not expand before integrating, got {steps:?}"
        );
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
                .any(|substep| substep["title"] == "Usar integración por partes repetida"),
            "expected repeated integration-by-parts substep for {input}, got {substeps:?}"
        );
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "{input} should verify through the bounded public residual route"
        );
    }

    for residual in [
        "diff(integrate((x^6+1)*sin(2*x+1), x), x) - (x^6+1)*sin(2*x+1)",
        "diff(integrate((x^6+1)*cos(2*x+1), x), x) - (x^6+1)*cos(2*x+1)",
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr(residual);
        assert_eq!(wire["result"], "0");
        assert!(
            wire["required_conditions"]
                .as_array()
                .expect("required_conditions should be an array")
                .is_empty(),
            "unexpected required conditions for {residual}: {:?}",
            wire["required_conditions"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "sparse affine sextic trig by-parts nested residual should not emit depth_overflow warning for {residual}\nstderr:\n{stderr}"
        );
    }
}
#[test]
fn integrate_contract_septic_trig_by_parts_verifies_publicly() {
    for (input, expected_result) in [
        (
            "integrate(x^7*sin(x), x)",
            "(-x^7 + 42·x^5 - 840·x^3 + 5040·x)·cos(x) + (7·x^6 - 210·x^4 + 2520·x^2 - 5040)·sin(x)",
        ),
        (
            "integrate(x^7*cos(x), x)",
            "(x^7 - 42·x^5 + 840·x^3 - 5040·x)·sin(x) + (7·x^6 - 210·x^4 + 2520·x^2 - 5040)·cos(x)",
        ),
        (
            "integrate((x^7+1)*sin(2*x+1), x)",
            "(7/4·x^6 - 105/8·x^4 + 315/8·x^2 - 315/16)·sin(2·x + 1) + (-1/2·x^7 + 21/4·x^5 - 105/4·x^3 + 315/8·x - 1/2)·cos(2·x + 1)",
        ),
        (
            "integrate((x^7+1)*cos(2*x+1), x)",
            "(7/4·x^6 - 105/8·x^4 + 315/8·x^2 - 315/16)·cos(2·x + 1) + (1/2·x^7 - 21/4·x^5 + 105/4·x^3 - 315/8·x + 1/2)·sin(2·x + 1)",
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result);
        assert!(
            wire["required_conditions"]
                .as_array()
                .expect("required_conditions should be an array")
                .is_empty(),
            "unexpected required conditions for {input}: {:?}",
            wire["required_conditions"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "septic trig by-parts presentation should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        assert_eq!(
            steps.len(),
            1,
            "expected direct integration trace without expansion noise for {input}, got {steps:?}"
        );
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
                .any(|substep| substep["title"] == "Usar integración por partes repetida"),
            "expected repeated integration-by-parts substep for {input}, got {substeps:?}"
        );
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "{input} should verify through the bounded public residual route"
        );
    }

    for residual in [
        "diff(integrate(x^7*sin(x), x), x) - x^7*sin(x)",
        "diff(integrate(x^7*cos(x), x), x) - x^7*cos(x)",
        "diff(integrate((x^7+1)*sin(2*x+1), x), x) - (x^7+1)*sin(2*x+1)",
        "diff(integrate((x^7+1)*cos(2*x+1), x), x) - (x^7+1)*cos(2*x+1)",
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr(residual);
        assert_eq!(wire["result"], "0");
        assert!(
            wire["required_conditions"]
                .as_array()
                .expect("required_conditions should be an array")
                .is_empty(),
            "unexpected required conditions for {residual}: {:?}",
            wire["required_conditions"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "septic trig by-parts nested residual should not emit depth_overflow warning for {residual}\nstderr:\n{stderr}"
        );
    }
}
#[test]
fn integrate_contract_octic_cos_by_parts_verifies_publicly() {
    let input = "integrate(x^8*cos(x), x)";
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

    assert_eq!(
        wire["result"],
        "(8·x^7 - 336·x^5 + 6720·x^3 - 40320·x)·cos(x) + (x^8 - 56·x^6 + 1680·x^4 - 20160·x^2 + 40320)·sin(x)"
    );
    assert!(
        wire["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .is_empty(),
        "unexpected required conditions for {input}: {:?}",
        wire["required_conditions"]
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "octic cosine by-parts presentation should not emit depth_overflow warning\nstderr:\n{stderr}"
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
            .any(|substep| substep["title"] == "Usar integración por partes repetida"),
        "expected repeated integration-by-parts substep for {input}, got {substeps:?}"
    );

    let residual = "diff(integrate(x^8*cos(x), x), x) - x^8*cos(x)";
    let (wire, stderr) = cli_eval_json_with_stderr(residual);
    assert_eq!(wire["result"], "0");
    assert!(
        wire["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .is_empty(),
        "unexpected residual required conditions for x^8*cos(x): {:?}",
        wire["required_conditions"]
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "nested octic cosine verification should not emit depth_overflow warning\nstderr:\n{stderr}"
    );
}
#[test]
fn integrate_contract_affine_octic_sin_by_parts_verifies_publicly() {
    let input = "integrate(x^8*sin(2*x+1), x)";
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

    assert_eq!(
        wire["result"],
        "(2·x^7 - 21·x^5 + 105·x^3 - 315/2·x)·sin(2·x + 1) + (-1/2·x^8 + 7·x^6 - 105/2·x^4 + 315/2·x^2 - 315/4)·cos(2·x + 1)"
    );
    assert!(
        wire["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .is_empty(),
        "unexpected required conditions for {input}: {:?}",
        wire["required_conditions"]
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "affine octic sine by-parts presentation should not emit depth_overflow warning\nstderr:\n{stderr}"
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
            .any(|substep| substep["title"] == "Usar integración por partes repetida"),
        "expected repeated integration-by-parts substep for {input}, got {substeps:?}"
    );

    let residual = "diff(integrate(x^8*sin(2*x+1), x), x) - x^8*sin(2*x+1)";
    let (wire, stderr) = cli_eval_json_with_stderr(residual);
    assert_eq!(wire["result"], "0");
    assert!(
        wire["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .is_empty(),
        "unexpected residual required conditions for affine octic sine: {:?}",
        wire["required_conditions"]
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "nested affine octic sine verification should not emit depth_overflow warning\nstderr:\n{stderr}"
    );
}
#[test]
fn integrate_contract_affine_octic_cos_by_parts_verifies_publicly() {
    let input = "integrate(x^8*cos(2*x+1), x)";
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

    assert_eq!(
        wire["result"],
        "(2·x^7 - 21·x^5 + 105·x^3 - 315/2·x)·cos(2·x + 1) + (1/2·x^8 - 7·x^6 + 105/2·x^4 - 315/2·x^2 + 315/4)·sin(2·x + 1)"
    );
    assert!(
        wire["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .is_empty(),
        "unexpected required conditions for {input}: {:?}",
        wire["required_conditions"]
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "affine octic cosine by-parts presentation should not emit depth_overflow warning\nstderr:\n{stderr}"
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
            .any(|substep| substep["title"] == "Usar integración por partes repetida"),
        "expected repeated integration-by-parts substep for {input}, got {substeps:?}"
    );

    let residual = "diff(integrate(x^8*cos(2*x+1), x), x) - x^8*cos(2*x+1)";
    let (wire, stderr) = cli_eval_json_with_stderr(residual);
    assert_eq!(wire["result"], "0");
    assert!(
        wire["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .is_empty(),
        "unexpected residual required conditions for affine octic cosine: {:?}",
        wire["required_conditions"]
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "nested affine octic cosine verification should not emit depth_overflow warning\nstderr:\n{stderr}"
    );
}
#[test]
fn integrate_contract_quadratic_exp_by_parts_presents_without_depth_overflow() {
    let input = "integrate((x^2+x+1)*exp(2*x+1), x)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert_eq!(wire["result"], "1/2·e^(2·x + 1)·(x^2 + 1)");
    assert!(
        !stderr.contains("depth_overflow"),
        "quadratic exp by-parts presentation should not emit depth_overflow warning\nstderr:\n{stderr}"
    );
    assert_antiderivative_verifies(input);

    let nested = "diff(integrate((x^2+x+1)*exp(2*x+1), x), x) - (x^2+x+1)*exp(2*x+1)";
    let (wire, stderr) = cli_eval_json_with_stderr(nested);
    assert_eq!(wire["result"], "0");
    assert!(
        !stderr.contains("depth_overflow"),
        "nested quadratic exp by-parts verification should not emit depth_overflow warning\nstderr:\n{stderr}"
    );
}
#[test]
fn integrate_contract_cubic_exp_by_parts_presents_without_depth_overflow() {
    for (input, expected_result, expected_substep_title) in [
        (
            "integrate(x^3*exp(x), x)",
            "e^x·(x^3 + 6·x - 3·x^2 - 6)",
            "Usar integración por partes repetida",
        ),
        (
            "integrate((x^3+x)*exp(2*x+1), x)",
            "1/8·e^(2·x + 1)·(4·x^3 + 10·x - 6·x^2 - 5)",
            "Usar integración por partes repetida",
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result);
        assert!(
            !stderr.contains("depth_overflow"),
            "cubic exp by-parts presentation should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
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
        assert_antiderivative_verifies(input);
    }
}
#[test]
fn integrate_contract_quartic_exp_by_parts_verifies() {
    let input = "integrate(x^4*exp(2*x+1), x)";
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

    assert_eq!(
        wire["result"],
        "1/4·e^(2·x + 1)·(2·x^4 + 6·x^2 + 3 - 4·x^3 - 6·x)"
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "quartic exp by-parts presentation should not emit depth_overflow warning\nstderr:\n{stderr}"
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
            .any(|substep| substep["title"] == "Usar integración por partes repetida"),
        "expected repeated integration-by-parts substep for {input}, got {substeps:?}"
    );

    let residual = "diff(integrate(x^4*exp(2*x+1), x), x) - x^4*exp(2*x+1)";
    let (wire, stderr) = cli_eval_json_with_stderr(residual);
    assert_eq!(wire["result"], "0");
    assert!(
        !stderr.contains("depth_overflow"),
        "nested quartic exp verification should not emit depth_overflow warning\nstderr:\n{stderr}"
    );
}
#[test]
fn integrate_contract_quintic_exp_by_parts_verifies() {
    let input = "integrate(x^5*exp(x), x)";
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

    assert_eq!(
        wire["result"],
        "e^x·(x^5 + 20·x^3 + 120·x - 5·x^4 - 60·x^2 - 120)"
    );
    assert!(
        wire["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .is_empty(),
        "unexpected required conditions for x^5*exp(x): {:?}",
        wire["required_conditions"]
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "quintic exp by-parts presentation should not emit depth_overflow warning\nstderr:\n{stderr}"
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
            .any(|substep| substep["title"] == "Usar integración por partes repetida"),
        "expected repeated integration-by-parts substep for {input}, got {substeps:?}"
    );

    let residual = "diff(integrate(x^5*exp(x), x), x) - x^5*exp(x)";
    let (wire, stderr) = cli_eval_json_with_stderr(residual);
    assert_eq!(wire["result"], "0");
    assert!(
        !stderr.contains("depth_overflow"),
        "nested quintic exp verification should not emit depth_overflow warning\nstderr:\n{stderr}"
    );
}
#[test]
fn integrate_contract_sextic_exp_by_parts_verifies() {
    let input = "integrate(x^6*exp(x), x)";
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

    assert_eq!(
        wire["result"],
        "e^x·(x^6 + 30·x^4 + 360·x^2 + 720 - 6·x^5 - 120·x^3 - 720·x)"
    );
    assert!(
        wire["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .is_empty(),
        "unexpected required conditions for x^6*exp(x): {:?}",
        wire["required_conditions"]
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "sextic exp by-parts presentation should not emit depth_overflow warning\nstderr:\n{stderr}"
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
            .any(|substep| substep["title"] == "Usar integración por partes repetida"),
        "expected repeated integration-by-parts substep for {input}, got {substeps:?}"
    );

    let residual = "diff(integrate(x^6*exp(x), x), x) - x^6*exp(x)";
    let (wire, stderr) = cli_eval_json_with_stderr(residual);
    assert_eq!(wire["result"], "0");
    assert!(
        !stderr.contains("depth_overflow"),
        "nested sextic exp verification should not emit depth_overflow warning\nstderr:\n{stderr}"
    );
}
#[test]
fn integrate_contract_septic_exp_by_parts_verifies() {
    let input = "integrate(x^7*exp(x), x)";
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

    assert_eq!(
        wire["result"],
        "e^x·(x^7 + 42·x^5 + 840·x^3 + 5040·x - 7·x^6 - 210·x^4 - 2520·x^2 - 5040)"
    );
    assert!(
        wire["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .is_empty(),
        "unexpected required conditions for x^7*exp(x): {:?}",
        wire["required_conditions"]
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "septic exp by-parts presentation should not emit depth_overflow warning\nstderr:\n{stderr}"
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
            .any(|substep| substep["title"] == "Usar integración por partes repetida"),
        "expected repeated integration-by-parts substep for {input}, got {substeps:?}"
    );

    for residual in [
        "diff(integrate(x^7*exp(x), x), x) - x^7*exp(x)",
        "diff(integrate((x^7+1)*exp(2*x+1), x), x) - (x^7+1)*exp(2*x+1)",
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr(residual);
        assert_eq!(wire["result"], "0");
        assert!(
            wire["required_conditions"]
                .as_array()
                .expect("required_conditions should be an array")
                .is_empty(),
            "unexpected required conditions for {residual}: {:?}",
            wire["required_conditions"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "nested septic exp verification should not emit depth_overflow warning for {residual}\nstderr:\n{stderr}"
        );
    }
}
#[test]
fn integrate_contract_octic_exp_by_parts_verifies() {
    let input = "integrate(x^8*exp(x), x)";
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

    assert_eq!(
        wire["result"],
        "e^x·(x^8 + 56·x^6 + 1680·x^4 + 20160·x^2 + 40320 - 8·x^7 - 336·x^5 - 6720·x^3 - 40320·x)"
    );
    assert!(
        wire["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .is_empty(),
        "unexpected required conditions for x^8*exp(x): {:?}",
        wire["required_conditions"]
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "octic exp by-parts presentation should not emit depth_overflow warning\nstderr:\n{stderr}"
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
            .any(|substep| substep["title"] == "Usar integración por partes repetida"),
        "expected repeated integration-by-parts substep for {input}, got {substeps:?}"
    );

    let residual = "diff(integrate(x^8*exp(x), x), x) - x^8*exp(x)";
    let (wire, stderr) = cli_eval_json_with_stderr(residual);
    assert_eq!(wire["result"], "0");
    assert!(
        wire["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .is_empty(),
        "unexpected residual required conditions for x^8*exp(x): {:?}",
        wire["required_conditions"]
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "nested octic exp verification should not emit depth_overflow warning\nstderr:\n{stderr}"
    );
}
#[test]
fn integrate_contract_sparse_quartic_exp_by_parts_keeps_direct_trace() {
    let input = "integrate((x^4+x^2)*exp(2*x+1), x)";
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

    assert_eq!(
        wire["result"],
        "1/2·e^(2·x + 1)·(x^4 + 4·x^2 + 2 - 2·x^3 - 4·x)"
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "sparse quartic exp by-parts presentation should not emit depth_overflow warning\nstderr:\n{stderr}"
    );

    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    assert_eq!(
        steps.len(),
        1,
        "expected direct integration trace without expansion or re-presentation noise for {input}, got {steps:?}"
    );
    assert!(
        steps
            .iter()
            .all(|step| step["rule"] != "Expandir la expresión"),
        "sparse quartic exp by-parts should not expand before integrating, got {steps:?}"
    );
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
            .any(|substep| substep["title"] == "Usar integración por partes repetida"),
        "expected repeated integration-by-parts substep for {input}, got {substeps:?}"
    );

    let residual = "diff(integrate((x^4+x^2)*exp(2*x+1), x), x) - (x^4+x^2)*exp(2*x+1)";
    let (wire, stderr) = cli_eval_json_with_stderr(residual);
    assert_eq!(wire["result"], "0");
    assert!(
        !stderr.contains("depth_overflow"),
        "nested sparse quartic exp verification should not emit depth_overflow warning\nstderr:\n{stderr}"
    );
}
#[test]
fn integrate_contract_sparse_affine_sextic_exp_by_parts_keeps_direct_trace() {
    let input = "integrate((x^6+1)*exp(2*x+1), x)";
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

    assert_eq!(
        wire["result"],
        "1/8·e^(2·x + 1)·(4·x^6 + 30·x^4 + 90·x^2 + 49 - 12·x^5 - 60·x^3 - 90·x)"
    );
    assert!(
        wire["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .is_empty(),
        "unexpected required conditions for {input}: {:?}",
        wire["required_conditions"]
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "sparse affine sextic exp by-parts presentation should not emit depth_overflow warning\nstderr:\n{stderr}"
    );

    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    assert!(
        steps.len() <= 2,
        "expected compact integration trace without expansion noise for {input}, got {steps:?}"
    );
    assert!(
        steps
            .iter()
            .all(|step| step["rule"] != "Expandir la expresión"),
        "sparse affine sextic exp by-parts should not expand before integrating, got {steps:?}"
    );
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
            .any(|substep| substep["title"] == "Usar integración por partes repetida"),
        "expected repeated integration-by-parts substep for {input}, got {substeps:?}"
    );
    assert_eq!(
        assert_antiderivative_verifies(input),
        AntiderivativeVerificationRoute::PublicResidual,
        "{input} should verify through the bounded public residual route"
    );

    let residual = "diff(integrate((x^6+1)*exp(2*x+1), x), x) - (x^6+1)*exp(2*x+1)";
    let (wire, stderr) = cli_eval_json_with_stderr(residual);
    assert_eq!(wire["result"], "0");
    assert!(
        wire["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .is_empty(),
        "unexpected required conditions for {residual}: {:?}",
        wire["required_conditions"]
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "nested sparse affine sextic exp verification should not emit depth_overflow warning\nstderr:\n{stderr}"
    );
}
#[test]
fn integrate_contract_sparse_quartic_trig_by_parts_keeps_direct_trace() {
    let input = "integrate((x^4+x^2)*sin(2*x+1), x)";
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

    assert_eq!(
        wire["result"],
        "(x^3 - x)·sin(2·x + 1) + (-1/2·x^4 + x^2 - 1/2)·cos(2·x + 1)"
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "sparse quartic trig by-parts presentation should not emit depth_overflow warning\nstderr:\n{stderr}"
    );

    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    assert_eq!(
        steps.len(),
        1,
        "expected direct integration trace without expansion noise for {input}, got {steps:?}"
    );
    assert!(
        steps
            .iter()
            .all(|step| step["rule"] != "Expandir la expresión"),
        "sparse quartic trig by-parts should not expand before integrating, got {steps:?}"
    );
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
            .any(|substep| substep["title"] == "Usar integración por partes repetida"),
        "expected repeated integration-by-parts substep for {input}, got {substeps:?}"
    );

    let residual = "diff(integrate((x^4+x^2)*sin(2*x+1), x), x) - (x^4+x^2)*sin(2*x+1)";
    let (wire, stderr) = cli_eval_json_with_stderr(residual);
    assert_eq!(wire["result"], "0");
    assert!(
        !stderr.contains("depth_overflow"),
        "nested sparse quartic trig verification should not emit depth_overflow warning\nstderr:\n{stderr}"
    );
}
#[test]
fn integrate_contract_affine_cubic_trig_by_parts_avoids_depth_overflow() {
    let input = "integrate((x^3+x)*sin(2*x+1), x)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert_eq!(
        wire["result"],
        "(1/4·x - 1/2·x^3)·cos(2·x + 1) + (3/4·x^2 - 1/8)·sin(2·x + 1)"
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "affine cubic trig by-parts presentation should not emit depth_overflow warning\nstderr:\n{stderr}"
    );

    let residual = "diff(integrate((x^3+x)*sin(2*x+1), x), x) - (x^3+x)*sin(2*x+1)";
    let (wire, stderr) = cli_eval_json_with_stderr(residual);
    assert_eq!(wire["result"], "0");
    assert!(
        !stderr.contains("depth_overflow"),
        "nested affine cubic trig verification should not emit depth_overflow warning\nstderr:\n{stderr}"
    );
}
#[test]
fn integrate_contract_affine_quartic_trig_by_parts_verifies() {
    for (input, expected_result, residual) in [
        (
            "integrate(x^4*sin(2*x+1), x)",
            "(x^3 - 3/2·x)·sin(2·x + 1) + (-1/2·x^4 + 3/2·x^2 - 3/4)·cos(2·x + 1)",
            "diff(integrate(x^4*sin(2*x+1), x), x) - x^4*sin(2*x+1)",
        ),
        (
            "integrate(x^4*cos(2*x+1), x)",
            "(x^3 - 3/2·x)·cos(2·x + 1) + (1/2·x^4 - 3/2·x^2 + 3/4)·sin(2·x + 1)",
            "diff(integrate(x^4*cos(2*x+1), x), x) - x^4*cos(2*x+1)",
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr(input);

        assert_eq!(wire["result"], expected_result);
        assert!(
            !stderr.contains("depth_overflow"),
            "affine quartic trig by-parts presentation should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let (wire, stderr) = cli_eval_json_with_stderr(residual);
        assert_eq!(wire["result"], "0");
        assert!(
            !stderr.contains("depth_overflow"),
            "nested affine quartic trig verification should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );
    }
}
#[test]
fn integrate_contract_linear_exp_by_parts_steps_keep_compact_presentation() {
    let input = "integrate(x*exp(x), x)";
    let (wire, _stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);
    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    let rules: Vec<_> = steps.iter().map(|step| step["rule"].as_str()).collect();

    assert_eq!(wire["result"], "(x - 1)·e^x");
    assert!(
        !rules.contains(&Some("Expandir la expresión")),
        "linear exp by-parts should not expand the compact antiderivative: {rules:?}"
    );
    assert!(
        !rules.contains(&Some("Sacar factor común")),
        "linear exp by-parts should not refactor immediately after expansion: {rules:?}"
    );
}
#[test]
fn integrate_contract_linear_times_exp_linear_by_parts() {
    let (result, required) = evaluated_integral_with_required_conditions("integrate(x*exp(x), x)");
    assert_eq!(result, "(x - 1) * e^x");
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((2*x+3)*exp(2*x+1), x)");
    assert_eq!(result, "(x + 1) * e^(2 * x + 1)");
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((x+1)*exp((3*x+2)/2), x)");
    assert_eq!(result, "(2/3 * x + 2/9) * e^((3 * x + 2) / 2)");
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((x+1)*exp((2-3*x)/2), x)");
    assert_eq!(result, "(-2/3 * x - 10/9) * e^((2 - 3 * x) / 2)");
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_linear_times_trig_linear_by_parts() {
    let (result, required) = evaluated_integral_with_required_conditions("integrate(x*sin(x), x)");
    assert_eq!(result, "sin(x) - x * cos(x)");
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) = evaluated_integral_with_required_conditions("integrate(x*cos(x), x)");
    assert_eq!(result, "cos(x) + x * sin(x)");
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((2*x+3)*sin(2*x+1), x)");
    assert_eq!(
        result,
        "1/2 * sin(2 * x + 1) - (cos(2 * x + 1) * (2 * x + 3))/2"
    );
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((2*x+3)*cos(2*x+1), x)");
    assert_eq!(
        result,
        "1/2 * cos(2 * x + 1) + (sin(2 * x + 1) * (2 * x + 3))/2"
    );
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((x+1)*sin((3*x+2)/2), x)");
    assert_eq!(
        result,
        "4/9 * sin((3 * x + 2) / 2) - 2/3 * (x + 1) * cos((3 * x + 2) / 2)"
    );
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((x+1)*cos((3*x+2)/2), x)");
    assert_eq!(
        result,
        "4/9 * cos((3 * x + 2) / 2) + 2/3 * (x + 1) * sin((3 * x + 2) / 2)"
    );
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((x+1)*sin((2-3*x)/2), x)");
    assert_eq!(
        result,
        "4/9 * sin((2 - 3 * x) / 2) + 2/3 * (x + 1) * cos((2 - 3 * x) / 2)"
    );
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((x+1)*cos((2-3*x)/2), x)");
    assert_eq!(
        result,
        "4/9 * cos((2 - 3 * x) / 2) - 2/3 * (x + 1) * sin((2 - 3 * x) / 2)"
    );
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_linear_times_hyperbolic_linear_by_parts() {
    let (result, required) = evaluated_integral_with_required_conditions("integrate(x*sinh(x), x)");
    assert_eq!(result, "x * cosh(x) - sinh(x)");
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) = evaluated_integral_with_required_conditions("integrate(x*cosh(x), x)");
    assert_eq!(result, "x * sinh(x) - cosh(x)");
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((2*x+3)*sinh(2*x+1), x)");
    assert_eq!(
        result,
        "(cosh(2 * x + 1) * (2 * x + 3))/2 - 1/2 * sinh(2 * x + 1)"
    );
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((2*x+3)*cosh(2*x+1), x)");
    assert_eq!(
        result,
        "(sinh(2 * x + 1) * (2 * x + 3))/2 - 1/2 * cosh(2 * x + 1)"
    );
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    for (input, expected) in [
        (
            "integrate((2*x+3)*sinh(1-2*x), x)",
            "1/2·(-sinh(1 - 2·x) - cosh(1 - 2·x)·(2·x + 3))",
        ),
        (
            "integrate((2*x+3)*cosh(1-2*x), x)",
            "1/2·(-cosh(1 - 2·x) - sinh(1 - 2·x)·(2·x + 3))",
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr(input);
        assert!(stderr.is_empty(), "unexpected stderr for {input}: {stderr}");
        assert_eq!(wire["result"], expected, "{input}");
        assert_eq!(wire["required_display"], serde_json::json!([]), "{input}");
    }

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((x+1)*sinh(2*x+1), x)");
    assert_eq!(
        result,
        "(cosh(2 * x + 1) * (x + 1))/2 - 1/4 * sinh(2 * x + 1)"
    );
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let residual = "diff(integrate((x+1)*sinh(2*x+1), x), x) - (x+1)*sinh(2*x+1)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "linear hyperbolic residual should not add conditions: {residual_required:?}"
    );

    for residual in [
        "diff(integrate((2*x+3)*sinh(1-2*x), x), x) - (2*x+3)*sinh(1-2*x)",
        "diff(integrate((2*x+3)*cosh(1-2*x), x), x) - (2*x+3)*cosh(1-2*x)",
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr(residual);
        assert!(
            stderr.is_empty(),
            "negative-affine hyperbolic residual should not emit stderr: {stderr}"
        );
        assert_eq!(wire["result"], "0", "{residual}");
        assert_eq!(
            wire["required_display"],
            serde_json::json!([]),
            "{residual}"
        );
    }
}
#[test]
fn integrate_contract_polynomial_times_hyperbolic_linear_by_parts_verifies() {
    for (input, expected_result, residual) in [
        (
            "integrate(x^2*sinh(x), x)",
            "(x^2 + 2)·cosh(x) - 2·x·sinh(x)",
            "diff(integrate(x^2*sinh(x), x), x) - x^2*sinh(x)",
        ),
        (
            "integrate(x^2*cosh(x), x)",
            "(x^2 + 2)·sinh(x) - 2·x·cosh(x)",
            "diff(integrate(x^2*cosh(x), x), x) - x^2*cosh(x)",
        ),
        (
            "integrate((x^2+x)*sinh(2*x+1), x)",
            "(1/2·x^2 + 1/2·x + 1/4)·cosh(2·x + 1) - (1/2·x + 1/4)·sinh(2·x + 1)",
            "diff(integrate((x^2+x)*sinh(2*x+1), x), x) - (x^2+x)*sinh(2*x+1)",
        ),
        (
            "integrate((x^3+x)*sinh(2*x+1), x)",
            "(1/2·x^3 + 5/4·x)·cosh(2·x + 1) - (3/4·x^2 + 5/8)·sinh(2·x + 1)",
            "diff(integrate((x^3+x)*sinh(2*x+1), x), x) - (x^3+x)*sinh(2*x+1)",
        ),
        (
            "integrate((x^3+x)*cosh(2*x+1), x)",
            "(1/2·x^3 + 5/4·x)·sinh(2·x + 1) - (3/4·x^2 + 5/8)·cosh(2·x + 1)",
            "diff(integrate((x^3+x)*cosh(2*x+1), x), x) - (x^3+x)*cosh(2*x+1)",
        ),
        (
            "integrate(x^2*sinh(2*x+1)+x*sinh(2*x+1), x)",
            "(1/2·x^2 + 1/2·x + 1/4)·cosh(2·x + 1) - (1/2·x + 1/4)·sinh(2·x + 1)",
            "diff(integrate(x^2*sinh(2*x+1)+x*sinh(2*x+1), x), x) - (x^2*sinh(2*x+1)+x*sinh(2*x+1))",
        ),
        (
            "integrate(x^2*cosh(2*x+1)+x*cosh(2*x+1), x)",
            "(1/2·x^2 + 1/2·x + 1/4)·sinh(2·x + 1) - (1/2·x + 1/4)·cosh(2·x + 1)",
            "diff(integrate(x^2*cosh(2*x+1)+x*cosh(2*x+1), x), x) - (x^2*cosh(2*x+1)+x*cosh(2*x+1))",
        ),
        (
            "integrate(x^6*sinh(x), x)",
            "(x^6 + 30·x^4 + 360·x^2 + 720)·cosh(x) - (6·x^5 + 120·x^3 + 720·x)·sinh(x)",
            "diff(integrate(x^6*sinh(x), x), x) - x^6*sinh(x)",
        ),
        (
            "integrate(x^6*cosh(x), x)",
            "(x^6 + 30·x^4 + 360·x^2 + 720)·sinh(x) - (6·x^5 + 120·x^3 + 720·x)·cosh(x)",
            "diff(integrate(x^6*cosh(x), x), x) - x^6*cosh(x)",
        ),
        (
            "integrate((x^6+1)*sinh(2*x+1), x)",
            "(1/2·x^6 + 15/4·x^4 + 45/4·x^2 + 49/8)·cosh(2·x + 1) - (3/2·x^5 + 15/2·x^3 + 45/4·x)·sinh(2·x + 1)",
            "diff(integrate((x^6+1)*sinh(2*x+1), x), x) - (x^6+1)*sinh(2*x+1)",
        ),
        (
            "integrate((x^6+1)*cosh(2*x+1), x)",
            "(1/2·x^6 + 15/4·x^4 + 45/4·x^2 + 49/8)·sinh(2·x + 1) - (3/2·x^5 + 15/2·x^3 + 45/4·x)·cosh(2·x + 1)",
            "diff(integrate((x^6+1)*cosh(2*x+1), x), x) - (x^6+1)*cosh(2*x+1)",
        ),
        (
            "integrate(x^7*sinh(x), x)",
            "(x^7 + 42·x^5 + 840·x^3 + 5040·x)·cosh(x) - (7·x^6 + 210·x^4 + 2520·x^2 + 5040)·sinh(x)",
            "diff(integrate(x^7*sinh(x), x), x) - x^7*sinh(x)",
        ),
        (
            "integrate(x^7*cosh(x), x)",
            "(x^7 + 42·x^5 + 840·x^3 + 5040·x)·sinh(x) - (7·x^6 + 210·x^4 + 2520·x^2 + 5040)·cosh(x)",
            "diff(integrate(x^7*cosh(x), x), x) - x^7*cosh(x)",
        ),
        (
            "integrate((x^7+1)*sinh(2*x+1), x)",
            "(1/2·x^7 + 21/4·x^5 + 105/4·x^3 + 315/8·x + 1/2)·cosh(2·x + 1) - (7/4·x^6 + 105/8·x^4 + 315/8·x^2 + 315/16)·sinh(2·x + 1)",
            "diff(integrate((x^7+1)*sinh(2*x+1), x), x) - (x^7+1)*sinh(2*x+1)",
        ),
        (
            "integrate((x^7+1)*cosh(2*x+1), x)",
            "(1/2·x^7 + 21/4·x^5 + 105/4·x^3 + 315/8·x + 1/2)·sinh(2·x + 1) - (7/4·x^6 + 105/8·x^4 + 315/8·x^2 + 315/16)·cosh(2·x + 1)",
            "diff(integrate((x^7+1)*cosh(2*x+1), x), x) - (x^7+1)*cosh(2*x+1)",
        ),
        (
            "integrate(x^8*cosh(x), x)",
            "(x^8 + 56·x^6 + 1680·x^4 + 20160·x^2)·sinh(x) - (8·x^7 + 336·x^5 + 6720·x^3 + 40320·x)·cosh(x)",
            "diff(integrate(x^8*cosh(x), x), x) - x^8*cosh(x)",
        ),
        (
            "integrate(x^8*cosh(2*x+1), x)",
            "(1/2·x^8 + 7·x^6 + 105/2·x^4 + 315/2·x^2)·sinh(2·x + 1) - (2·x^7 + 21·x^5 + 105·x^3 + 315/2·x)·cosh(2·x + 1)",
            "diff(integrate(x^8*cosh(2*x+1), x), x) - x^8*cosh(2*x+1)",
        ),
        (
            "integrate(x^8*sinh(2*x+1), x)",
            "(1/2·x^8 + 7·x^6 + 105/2·x^4 + 315/2·x^2)·cosh(2·x + 1) - (2·x^7 + 21·x^5 + 105·x^3 + 315/2·x)·sinh(2·x + 1)",
            "diff(integrate(x^8*sinh(2*x+1), x), x) - x^8*sinh(2*x+1)",
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result);
        assert_eq!(
            wire["required_conditions"]
                .as_array()
                .expect("required_conditions array")
                .len(),
            0,
            "unexpected required_conditions for {input}: {:?}",
            wire["required_conditions"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "quadratic hyperbolic by-parts presentation should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );
        assert!(
            wire["steps"]
                .as_array()
                .expect("steps array")
                .iter()
                .flat_map(|step| step["substeps"].as_array().into_iter().flatten())
                .any(|substep| substep["title"] == "Usar integración por partes repetida"),
            "quadratic hyperbolic by-parts should expose repeated integration-by-parts substep: {wire:?}"
        );

        let (wire, stderr) = cli_eval_json_with_stderr(residual);
        assert_eq!(wire["result"], "0");
        assert!(
            !stderr.contains("depth_overflow"),
            "quadratic hyperbolic by-parts verification should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );
    }

    for (lhs, rhs) in [
        (
            "diff(integrate(x^2*sinh(2*x+1)+x*sinh(2*x+1), x), x)",
            "x^2*sinh(2*x+1)+x*sinh(2*x+1)",
        ),
        (
            "diff(integrate(x^2*cosh(2*x+1)+x*cosh(2*x+1), x), x)",
            "x^2*cosh(2*x+1)+x*cosh(2*x+1)",
        ),
    ] {
        let (equivalent, required) = evaluated_equiv_with_required_conditions(lhs, rhs);
        assert!(
            equivalent,
            "public equivalence should reuse the direct residual proof for {lhs} equiv {rhs}"
        );
        assert!(
            required.is_empty(),
            "unexpected public equivalence required conditions for {lhs} equiv {rhs}: {required:?}"
        );
    }
}
#[test]
fn integrate_contract_linear_times_hyperbolic_rational_linear_by_parts() {
    for (input, expected_result, residual) in [
        (
            "integrate((x+1)*sinh((3*x+2)/2), x)",
            "2/3 * cosh((3 * x + 2) / 2) * (x + 1) - 4/9 * sinh((3 * x + 2) / 2)",
            "diff(integrate((x+1)*sinh((3*x+2)/2), x), x) - (x+1)*sinh((3*x+2)/2)",
        ),
        (
            "integrate((x+1)*cosh((3*x+2)/2), x)",
            "2/3 * sinh((3 * x + 2) / 2) * (x + 1) - 4/9 * cosh((3 * x + 2) / 2)",
            "diff(integrate((x+1)*cosh((3*x+2)/2), x), x) - (x+1)*cosh((3*x+2)/2)",
        ),
        (
            "integrate((x+1)*sinh((2-3*x)/2), x)",
            "-2/3 * cosh((2 - 3 * x) / 2) * (x + 1) - 4/9 * sinh((2 - 3 * x) / 2)",
            "diff(integrate((x+1)*sinh((2-3*x)/2), x), x) - (x+1)*sinh((2-3*x)/2)",
        ),
        (
            "integrate((x+1)*cosh((2-3*x)/2), x)",
            "-2/3 * sinh((2 - 3 * x) / 2) * (x + 1) - 4/9 * cosh((2 - 3 * x) / 2)",
            "diff(integrate((x+1)*cosh((2-3*x)/2), x), x) - (x+1)*cosh((2-3*x)/2)",
        ),
    ] {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected_result);
        assert!(
            required.is_empty(),
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "{input} should verify through the bounded public residual route"
        );

        let (residual_result, residual_required) =
            evaluated_expr_with_required_conditions(residual);
        assert_eq!(residual_result, "0");
        assert!(
            residual_required.is_empty(),
            "unexpected residual required_conditions for {input}: {residual_required:?}"
        );
    }
}
#[test]
fn integrate_contract_monomial_times_log_by_parts_preserves_positive_domain() {
    let cases = [
        ("integrate(x*ln(x), x)", "1/4 * x^2 * (2 * ln(x) - 1)"),
        ("integrate(x^2*ln(x), x)", "1/9 * x^3 * (3 * ln(x) - 1)"),
        (
            "integrate(x*ln(x)^2, x)",
            "1/8 * x^2 * (4 * ln(x)^2 - 4 * ln(x) + 2)",
        ),
        (
            "integrate(x^2*ln(x)^2, x)",
            "1/27 * x^3 * (9 * ln(x)^2 - 6 * ln(x) + 2)",
        ),
        (
            "integrate(x*ln(x)^3, x)",
            "1/16 * x^2 * (8 * ln(x)^3 - 12 * ln(x)^2 + 12 * ln(x) - 6)",
        ),
        (
            "integrate(x^2*ln(x)^3, x)",
            "1/81 * x^3 * (27 * ln(x)^3 - 27 * ln(x)^2 + 18 * ln(x) - 6)",
        ),
        (
            "integrate(x*ln(x)^4, x)",
            "1/32 * x^2 * (16 * ln(x)^4 - 32 * ln(x)^3 + 48 * ln(x)^2 - 48 * ln(x) + 24)",
        ),
        (
            "integrate(x^2*ln(x)^4, x)",
            "1/243 * x^3 * (81 * ln(x)^4 - 108 * ln(x)^3 + 108 * ln(x)^2 - 72 * ln(x) + 24)",
        ),
        (
            "integrate(x*ln(x)^5, x)",
            "1/64 * x^2 * (32 * ln(x)^5 - 80 * ln(x)^4 + 160 * ln(x)^3 - 240 * ln(x)^2 + 240 * ln(x) - 120)",
        ),
        (
            "integrate(x^2*ln(x)^5, x)",
            "1/729 * x^3 * (243 * ln(x)^5 - 405 * ln(x)^4 + 540 * ln(x)^3 - 540 * ln(x)^2 + 360 * ln(x) - 120)",
        ),
    ];

    for (input, expected) in cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);

        assert_eq!(result, expected, "input: {input}");
        assert_eq!(
            required,
            vec!["x > 0".to_string()],
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert_antiderivative_verifies(input);
        assert_rendered_antiderivative_verifies(input, &result);
    }
}
#[test]
fn integrate_contract_positive_quadratic_log_by_parts_keeps_compact_trace() {
    let input = "integrate((x^2+1)*ln(x^2+1), x)";
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);
    assert!(
        stderr.is_empty(),
        "positive quadratic log by-parts presentation should stay quiet\nstderr:\n{stderr}"
    );
    assert_eq!(
        wire["result"],
        "(1/3·x^3 + x)·ln(x^2 + 1) - 2/9·x^3 - 4/3·x + 4/3·arctan(x)"
    );
    assert_eq!(
        wire["required_display"],
        serde_json::json!([]),
        "positive quadratic log by-parts should not add domain requirements"
    );

    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    assert_eq!(
        steps.len(),
        1,
        "expected compact direct integration trace, got {steps:?}"
    );
    assert!(
        steps
            .iter()
            .all(|step| step["rule"] != "Expandir la expresión"),
        "positive quadratic log by-parts trace should not expand before integrating, got {steps:?}"
    );
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
            .any(|substep| substep["title"] == "Usar integración por partes"),
        "expected integration-by-parts substep, got {substeps:?}"
    );

    assert_eq!(
        assert_antiderivative_verifies(input),
        AntiderivativeVerificationRoute::PublicResidual
    );
}
#[test]
fn integrate_contract_positive_quadratic_log_by_parts_collects_repeated_log_factor() {
    let input = "integrate((x^2+x+1)*ln(x^2+1), x)";
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);
    assert!(
        stderr.is_empty(),
        "positive quadratic log by-parts presentation should stay quiet\nstderr:\n{stderr}"
    );
    assert_eq!(
        wire["result"],
        "ln(x^2 + 1)·(1/3·x^3 + 1/2·x^2 + x + 1/2) + 4/3·arctan(x) - 2/9·x^3 - 1/2·x^2 - 4/3·x"
    );
    assert_eq!(wire["required_display"], serde_json::json!([]));

    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    assert_eq!(
        steps.len(),
        1,
        "expected compact direct integration trace, got {steps:?}"
    );
    assert!(
        steps
            .iter()
            .all(|step| step["rule"] != "Expandir la expresión"),
        "positive quadratic log by-parts trace should not expand before integrating, got {steps:?}"
    );
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
            .any(|substep| substep["title"] == "Usar integración por partes"),
        "expected integration-by-parts substep, got {substeps:?}"
    );

    assert_eq!(
        assert_antiderivative_verifies(input),
        AntiderivativeVerificationRoute::PublicResidual
    );
    let residual = "diff(integrate((x^2+x+1)*ln(x^2+1), x), x) - (x^2+x+1)*ln(x^2+1)";
    let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual);
    assert!(
        residual_stderr.is_empty(),
        "positive quadratic log by-parts residual should stay quiet\nstderr:\n{residual_stderr}"
    );
    assert_eq!(residual_wire["result"], "0");
    assert_eq!(residual_wire["required_display"], serde_json::json!([]));
}
#[test]
fn integrate_contract_linear_monomial_times_affine_log_by_parts_preserves_positive_domain() {
    let cases = [
        (
            "integrate(x*ln(2*x+1), x)",
            "(x^2 / 2 - 1/8) * ln(2 * x + 1) + 1/4 * x - 1/4 * x^2",
            "x > -1/2",
        ),
        (
            "integrate(3*x*ln(2*x+1), x)",
            "3 * ((x^2 / 2 - 1/8) * ln(2 * x + 1) + 1/4 * x - 1/4 * x^2)",
            "x > -1/2",
        ),
        (
            "integrate(x*ln(x+1), x)",
            "(x^2 / 2 - 1/2) * ln(x + 1) + 1/2 * x - 1/4 * x^2",
            "x > -1",
        ),
        (
            "integrate((x+1)*ln(2*x+1), x)",
            "1/8 * (ln(2 * x + 1) * (2 * x + 1) * (2 * x + 3) - 2 * x^2 - 6 * x)",
            "x > -1/2",
        ),
        (
            "integrate((2*x+3)*ln(2*x+1), x)",
            "1/4 * (ln(2 * x + 1) * (2 * x + 1) * (2 * x + 5) - 2 * x^2 - 10 * x)",
            "x > -1/2",
        ),
        (
            "integrate((1-2*x)*ln(1-2*x), x)",
            "1/4 * (2 * x^2 - ln(1 - 2 * x) * (1 - 2 * x)^2 - 2 * x)",
            "x < 1/2",
        ),
    ];

    for (input, expected, required_condition) in cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);

        assert_eq!(result, expected, "input: {input}");
        assert_eq!(
            required,
            vec![required_condition.to_string()],
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert_antiderivative_verifies(input);
        assert_rendered_antiderivative_verifies(input, &result);
    }
}
#[test]
fn integrate_contract_quadratic_monomial_times_affine_log_by_parts_verifies() {
    let input = "integrate(x^2*ln(2*x+1), x)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert!(
        stderr.is_empty(),
        "quadratic affine-log by-parts integration should stay quiet\nstderr:\n{stderr}"
    );
    assert_eq!(
        wire["result"],
        "1/3·x^3·ln(2·x + 1) - 1/9·x^3 - 1/12·x + 1/24·ln(2·x + 1) + 1/12·x^2"
    );
    assert_eq!(wire["required_display"], serde_json::json!(["x > -1/2"]));

    let residual = "diff(integrate(x^2*ln(2*x+1), x), x) - x^2*ln(2*x+1)";
    let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual);

    assert!(
        residual_stderr.is_empty(),
        "quadratic affine-log by-parts residual should stay quiet\nstderr:\n{residual_stderr}"
    );
    assert_eq!(residual_wire["result"], "0");
    assert_eq!(
        residual_wire["required_display"],
        serde_json::json!(["x > -1/2"])
    );

    let negative_input = "integrate(x^2*ln(1-2*x), x)";
    let (negative_wire, negative_stderr) = cli_eval_json_with_stderr_args(
        negative_input,
        &["--budget", "small", "--time-budget-ms", "100"],
    );

    assert!(
        negative_stderr.is_empty(),
        "negative-slope quadratic affine-log integration should stay quiet\nstderr:\n{negative_stderr}"
    );
    assert_eq!(
        negative_wire["result"],
        "1/3·x^3·ln(1 - 2·x) - (1/24·ln(1 - 2·x) + 1/9·x^3 + 1/12·x^2 + 1/12·x)"
    );
    assert_eq!(
        negative_wire["required_display"],
        serde_json::json!(["x < 1/2"])
    );

    let negative_residual = "diff(integrate(x^2*ln(1-2*x), x), x) - x^2*ln(1-2*x)";
    let (negative_residual_wire, negative_residual_stderr) = cli_eval_json_with_stderr_args(
        negative_residual,
        &["--budget", "small", "--time-budget-ms", "100"],
    );

    assert!(
        negative_residual_stderr.is_empty(),
        "negative-slope quadratic affine-log residual should stay quiet\nstderr:\n{negative_residual_stderr}"
    );
    assert_eq!(negative_residual_wire["result"], "0");
    assert_eq!(
        negative_residual_wire["required_display"],
        serde_json::json!(["x < 1/2"])
    );
}
#[test]
fn integrate_contract_cubic_monomial_times_affine_log_by_parts_verifies() {
    let input = "integrate(x^3*ln(2*x+1), x)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert!(
        stderr.is_empty(),
        "cubic affine-log by-parts integration should stay quiet\nstderr:\n{stderr}"
    );
    assert_eq!(
        wire["result"],
        "ln(2·x + 1)·(1/4·x^4 - 1/64) + 1/24·x^3 + 1/32·x - 1/16·x^4 - 1/32·x^2"
    );
    assert_eq!(wire["required_display"], serde_json::json!(["x > -1/2"]));
    assert!(wire.get("blocked_hints").is_none());

    let residual = "diff(integrate(x^3*ln(2*x+1), x), x) - x^3*ln(2*x+1)";
    let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual);

    assert!(
        residual_stderr.is_empty(),
        "cubic affine-log by-parts residual should stay quiet\nstderr:\n{residual_stderr}"
    );
    assert_eq!(residual_wire["result"], "0");
    assert_eq!(
        residual_wire["required_display"],
        serde_json::json!(["x > -1/2"])
    );
}
#[test]
fn integrate_contract_sparse_cubic_affine_log_by_parts_stays_compact() {
    let input = "integrate((x^3+x)*ln(x+1), x)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert!(
        stderr.is_empty(),
        "sparse cubic affine-log by-parts integration should stay quiet\nstderr:\n{stderr}"
    );
    assert_eq!(
        wire["result"],
        "ln(x + 1)·(1/4·x^4 + 1/2·x^2 - 3/4) + 1/12·x^3 + 3/4·x - 1/16·x^4 - 3/8·x^2"
    );
    assert_eq!(wire["required_display"], serde_json::json!(["x > -1"]));
    assert!(wire.get("blocked_hints").is_none());

    let residual = "diff(integrate((x^3+x)*ln(x+1), x), x) - (x^3+x)*ln(x+1)";
    let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual);

    assert!(
        residual_stderr.is_empty(),
        "sparse cubic affine-log by-parts residual should stay quiet\nstderr:\n{residual_stderr}"
    );
    assert_eq!(residual_wire["result"], "0");
    assert_eq!(
        residual_wire["required_display"],
        serde_json::json!(["x > -1"])
    );
}
#[test]
fn integrate_contract_quadratic_times_positive_quadratic_log_by_parts_verifies_shifted_argument() {
    let cases = [
        (
            "integrate(ln(x^2+x+1), x)",
            "1/2·ln(x^2 + x + 1) + ln(x^2 + x + 1)·x + 3·arctan((2·x + 1) / sqrt(3)) / sqrt(3) - 2·x",
            "diff(integrate(ln(x^2+x+1), x), x) - ln(x^2+x+1)",
            true,
        ),
        (
            "integrate(x*ln(x^2+x+1), x)",
            "1/4·ln(x^2 + x + 1) + 1/2·ln(x^2 + x + 1)·x^2 - 3/2·arctan((2·x + 1) / sqrt(3)) / sqrt(3) + 1/2·x - 1/2·x^2",
            "diff(integrate(x*ln(x^2+x+1), x), x) - x*ln(x^2+x+1)",
            true,
        ),
        (
            "integrate(x^2*ln(x^2+x+1), x)",
            "1/3·x^3·ln(x^2 + x + 1) - 1/3·ln(x^2 + x + 1) - 2/9·x^3 + 1/6·x^2 + 1/3·x",
            "diff(integrate(x^2*ln(x^2+x+1), x), x) - x^2*ln(x^2+x+1)",
            true,
        ),
        (
            "integrate(x^3*ln(x^2+1), x)",
            "1/4·x^4·ln(x^2 + 1) - 1/4·ln(x^2 + 1) - 1/8·x^4 + 1/4·x^2",
            "diff(integrate(x^3*ln(x^2+1), x), x) - x^3*ln(x^2+1)",
            true,
        ),
        (
            "integrate(x^4*ln(x^2+1), x)",
            "1/5·x^5·ln(x^2 + 1) - 2/25·x^5 - 2/5·x + 2/5·arctan(x) + 2/15·x^3",
            "diff(integrate(x^4*ln(x^2+1), x), x) - x^4*ln(x^2+1)",
            true,
        ),
        (
            "integrate(x^5*ln(x^2+1), x)",
            "1/6·x^6·ln(x^2 + 1) - 1/18·x^6 - 1/6·x^2 + 1/6·ln(x^2 + 1) + 1/12·x^4",
            "diff(integrate(x^5*ln(x^2+1), x), x) - x^5*ln(x^2+1)",
            true,
        ),
        (
            "integrate(x^6*ln(x^2+1), x)",
            "1/7·x^7·ln(x^2 + 1) - 2/7·arctan(x) - 2/49·x^7 - 2/21·x^3 + 2/35·x^5 + 2/7·x",
            "diff(integrate(x^6*ln(x^2+1), x), x) - x^6*ln(x^2+1)",
            true,
        ),
        (
            "integrate(x^7*ln(x^2+1), x)",
            "1/8·x^8·ln(x^2 + 1) - 1/8·ln(x^2 + 1) - 1/32·x^8 - 1/16·x^4 + 1/24·x^6 + 1/8·x^2",
            "diff(integrate(x^7*ln(x^2+1), x), x) - x^7*ln(x^2+1)",
            true,
        ),
        (
            "integrate(x^8*ln(x^2+1), x)",
            "1/9·x^9·ln(x^2 + 1) - 2/81·x^9 - 2/45·x^5 - 2/9·x + 2/9·arctan(x) + 2/63·x^7 + 2/27·x^3",
            "diff(integrate(x^8*ln(x^2+1), x), x) - x^8*ln(x^2+1)",
            true,
        ),
    ];

    for (input, expected, residual, verify_rendered) in cases {
        let (wire, stderr) = cli_eval_json_with_stderr(input);

        assert!(
            stderr.is_empty(),
            "positive-quadratic log by-parts integration should stay quiet for {input}\nstderr:\n{stderr}"
        );
        assert_eq!(wire["result"], expected, "input: {input}");
        assert_eq!(wire["required_display"], serde_json::json!([]));
        assert_antiderivative_verifies(input);
        if verify_rendered {
            assert_rendered_antiderivative_verifies(input, expected);
        }

        let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual);

        assert!(
            residual_stderr.is_empty(),
            "positive-quadratic log by-parts residual should stay quiet for {input}\nstderr:\n{residual_stderr}"
        );
        assert_eq!(residual_wire["result"], "0", "input: {input}");
        assert_eq!(residual_wire["required_display"], serde_json::json!([]));
    }

    let explicit_quintic_residual = "diff(1/6*x^6*ln(x^2+1) - 1/18*x^6 - 1/6*x^2 + 1/6*ln(x^2+1) + 1/12*x^4, x) - x^5*ln(x^2+1)";
    let (wire, stderr) = cli_eval_json_with_stderr(explicit_quintic_residual);
    assert!(
        stderr.is_empty(),
        "positive-quadratic log rendered quintic residual should stay quiet\nstderr:\n{stderr}"
    );
    assert_eq!(wire["result"], "0");
    assert_eq!(wire["required_display"], serde_json::json!([]));

    let explicit_sextic_residual = "diff(1/7*x^7*ln(x^2+1) - 2/7*arctan(x) - 2/49*x^7 - 2/21*x^3 + 2/35*x^5 + 2/7*x, x) - x^6*ln(x^2+1)";
    let (wire, stderr) = cli_eval_json_with_stderr(explicit_sextic_residual);
    assert!(
        stderr.is_empty(),
        "positive-quadratic log rendered sextic residual should stay quiet\nstderr:\n{stderr}"
    );
    assert_eq!(wire["result"], "0");
    assert_eq!(wire["required_display"], serde_json::json!([]));

    let explicit_septic_residual = "diff(1/8*x^8*ln(x^2+1) - 1/8*ln(x^2+1) - 1/32*x^8 - 1/16*x^4 + 1/24*x^6 + 1/8*x^2, x) - x^7*ln(x^2+1)";
    let (wire, stderr) = cli_eval_json_with_stderr(explicit_septic_residual);
    assert!(
        stderr.is_empty(),
        "positive-quadratic log rendered septic residual should stay quiet\nstderr:\n{stderr}"
    );
    assert_eq!(wire["result"], "0");
    assert_eq!(wire["required_display"], serde_json::json!([]));

    let explicit_octic_residual = "diff(1/9*x^9*ln(x^2+1) - 2/81*x^9 - 2/45*x^5 - 2/9*x + 2/9*arctan(x) + 2/63*x^7 + 2/27*x^3, x) - x^8*ln(x^2+1)";
    let (wire, stderr) = cli_eval_json_with_stderr(explicit_octic_residual);
    assert!(
        stderr.is_empty(),
        "positive-quadratic log rendered octic residual should stay quiet\nstderr:\n{stderr}"
    );
    assert_eq!(wire["result"], "0");
    assert_eq!(wire["required_display"], serde_json::json!([]));

    let (wire, stderr) = cli_eval_json_with_stderr("integrate(x^9*ln(x^2+1), x)");
    assert!(
        stderr.is_empty(),
        "positive-quadratic log by-parts budget boundary should stay quiet\nstderr:\n{stderr}"
    );
    assert_eq!(wire["result"], "integrate(ln(x^2 + 1)·x^9, x)");
    assert_eq!(wire["required_display"], serde_json::json!([]));
}
#[test]
fn integrate_contract_positive_quadratic_log_by_parts_flattens_compound_remainder() {
    let input = "integrate((x+1)*ln(x^2+x+1), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "ln(x^2 + x + 1) * (1/2 * x^2 + x + 3/4) + 3/2 * arctan((2 * x + 1) / sqrt(3)) / sqrt(3) - 1/2 * x^2 - 3/2 * x"
    );
    assert!(
        !result.contains(" - ("),
        "compound positive-quadratic log by-parts presentation should flatten subtracting a remainder group, got {result}"
    );
    assert!(
        required.is_empty(),
        "positive quadratic log argument should not add conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);

    let residual = "diff(integrate((x+1)*ln(x^2+x+1), x), x) - (x+1)*ln(x^2+x+1)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "positive-quadratic log by-parts residual should not add conditions: {residual_required:?}"
    );
}
#[test]
fn integrate_contract_positive_quadratic_log_by_parts_recombines_expanded_orientation() {
    let input = "integrate((x+1)*ln(x^2-x+1), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "ln(x^2 + 1 - x) * (1/2 * x^2 + x - 1/4) + 9/2 * arctan((2 * x - 1) / sqrt(3)) / sqrt(3) - 1/2 * x^2 - 5/2 * x"
    );
    assert!(
        !result.contains(" - (") && !result.contains("1/2 * ("),
        "expanded-orientation positive-quadratic log by-parts presentation should stay flat, got {result}"
    );
    assert!(
        required.is_empty(),
        "positive quadratic log argument should not add conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);

    let residual = "diff(integrate((x+1)*ln(x^2-x+1), x), x) - (x+1)*ln(x^2-x+1)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "expanded-orientation positive-quadratic log by-parts residual should not add conditions: {residual_required:?}"
    );
}
#[test]
fn integrate_contract_positive_quadratic_self_log_by_parts_flattens_remainder() {
    let input = "integrate((x^2+x+1)*ln(x^2+x+1), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "ln(x^2 + x + 1) * (1/3 * x^3 + 1/2 * x^2 + x + 5/12) + 3/2 * arctan((2 * x + 1) / sqrt(3)) / sqrt(3) - 2/9 * x^3 - 1/3 * x^2 - 7/6 * x"
    );
    assert!(
        !result.contains(" - ("),
        "self positive-quadratic log by-parts presentation should flatten subtracting a remainder group, got {result}"
    );
    assert!(
        required.is_empty(),
        "positive quadratic self-log argument should not add conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);

    let residual = "diff(integrate((x^2+x+1)*ln(x^2+x+1), x), x) - (x^2+x+1)*ln(x^2+x+1)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "positive-quadratic self-log by-parts residual should not add conditions: {residual_required:?}"
    );
}
#[test]
fn integrate_contract_affine_log_by_parts_presentation_stays_quiet() {
    for input in ["integrate(x*ln(2*x+1), x)", "integrate((x+1)*ln(2*x+1), x)"] {
        let (_wire, stderr) = cli_eval_json_with_stderr(input);
        assert!(
            !stderr.contains("depth_overflow"),
            "affine log by-parts presentation should not emit depth_overflow for {input}\nstderr:\n{stderr}"
        );
    }
}
#[test]
fn integrate_contract_affine_log_by_parts_offset_residual_stays_quiet() {
    for input in [
        "diff(integrate((x+1)*ln(2*x+1), x), x) - (x+1)*ln(2*x+1)",
        "diff(integrate((1-2*x)*ln(1-2*x), x), x) - (1-2*x)*ln(1-2*x)",
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr(input);

        assert_eq!(wire["result"], "0", "input: {input}");
        assert!(
            !stderr.contains("depth_overflow"),
            "offset affine log by-parts residual should not emit depth_overflow for {input}\nstderr:\n{stderr}"
        );
    }
}
#[test]
fn integrate_contract_quadratic_times_positive_quadratic_log_by_parts_verifies() {
    let input = "integrate(x^2*ln(x^2+1), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "1/3 * x^3 * ln(x^2 + 1) - (2/3 * arctan(x) + 2/9 * x^3 - 2/3 * x)"
    );
    assert!(
        required.is_empty(),
        "positive quadratic log argument should not add conditions: {required:?}"
    );
    assert_rendered_antiderivative_verifies(input, &result);

    let residual = "diff(integrate(x^2*ln(x^2+1), x), x) - x^2*ln(x^2+1)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "positive quadratic log residual should not add conditions: {residual_required:?}"
    );

    assert_eq!(
        simplified_integral("integrate(x^2*ln(x^2-1), x)"),
        "integrate(ln(x^2 - 1) * x^2, x)",
        "indefinite-sign quadratic log arguments must stay unsupported"
    );
}
#[test]
fn integrate_contract_arctan_scaled_variable_by_parts() {
    let (result, required) = evaluated_integral_with_required_conditions("integrate(arctan(x), x)");
    assert_eq!(result, "-1/2 * ln(x^2 + 1) + x * arctan(x)");
    assert!(
        required.is_empty(),
        "arctan integration should not add required conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(arctan(x), x)");

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(arctan(2*x), x)");
    assert_eq!(result, "-1/4 * ln((2 * x)^2 + 1) + x * arctan(2 * x)");
    assert!(
        required.is_empty(),
        "scaled arctan integration should not add required conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(arctan(2*x), x)");
    assert_rendered_antiderivative_verifies("integrate(arctan(2*x), x)", &result);

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(arctan(2*x+1), x)");
    assert_eq!(
        result,
        "-1/4 * ln((2 * x + 1)^2 + 1) + 1/2 * (2 * x + 1) * arctan(2 * x + 1)"
    );
    assert!(
        required.is_empty(),
        "shifted arctan integration should not add required conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(arctan(2*x+1), x)");
    assert_rendered_antiderivative_verifies("integrate(arctan(2*x+1), x)", &result);

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(arctan(1-2*x), x)");
    assert_eq!(
        result,
        "1/4 * ln((1 - 2 * x)^2 + 1) + -1/2 * (1 - 2 * x) * arctan(1 - 2 * x)"
    );
    assert!(
        required.is_empty(),
        "negative-slope shifted arctan integration should not add required conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(arctan(1-2*x), x)");
    assert_rendered_antiderivative_verifies("integrate(arctan(1-2*x), x)", &result);

    let (wire, _stderr) = cli_eval_json_with_stderr("integrate(arctan(1-2*x), x)");
    assert_eq!(
        wire["result"],
        "1/4·ln((1 - 2·x)^2 + 1) - 1/2·(1 - 2·x)·arctan(1 - 2·x)"
    );
    assert!(
        !wire["result"].as_str().unwrap_or_default().contains("+ -"),
        "public negative-slope shifted arctan integration should compact adjacent signs: {}",
        wire["result"]
    );
}
#[test]
fn integrate_contract_polynomial_times_arctan_affine_by_parts() {
    for (input, fragments) in [
        (
            "integrate(x*arctan(x), x)",
            vec!["arctan(x) * (x^2 + 1)", "- x"],
        ),
        (
            "integrate(x^2*arctan(x), x)",
            vec!["x^3 * arctan(x)", "ln(x^2 + 1)"],
        ),
        (
            "integrate(x^3*arctan(x), x)",
            vec!["arctan(x) * (3 * x^4 - 3)", "3 * x - x^3"],
        ),
        (
            "integrate(x^4*arctan(x), x)",
            vec!["x^5 * arctan(x)", "x^4", "ln(x^2 + 1)"],
        ),
        (
            "integrate(x^5*arctan(x), x)",
            vec!["arctan(x) * (15 * x^6 + 15)", "5 * x^3 - 3 * x^5 - 15 * x"],
        ),
        (
            "integrate(x^6*arctan(x), x)",
            vec!["x^7 * arctan(x)", "x^6", "ln(x^2 + 1)"],
        ),
        (
            "integrate(x^2*arctan(x+1), x)",
            vec!["arctan(x + 1) * (2 * x^3 - 4)", "ln(x^2 + 2 * x + 2)"],
        ),
        (
            "integrate(x^3*arctan(x+1), x)",
            vec!["arctan(x + 1) * (3 * x^4 + 12)", "3 * x^2 - x^3 - 6 * x"],
        ),
        (
            "integrate(x^2*arctan(1-x), x)",
            vec!["arctan(1 - x) * (1/3 * x^3 + 2/3)", "ln(x^2 + 2 - 2 * x)"],
        ),
    ] {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert!(
            !result.starts_with("integrate("),
            "expected polynomial-arctan affine product to integrate, got {result}"
        );
        for fragment in fragments {
            assert!(
                result.contains(fragment),
                "expected `{fragment}` in polynomial-arctan antiderivative for {input}, got {result}"
            );
        }
        assert!(
            !result.contains("1/3 * x^2 / 2"),
            "polynomial-arctan presentation should not keep nested polynomial fractions, got {result}"
        );
        assert!(
            required.is_empty(),
            "polynomial-arctan integration should not add required conditions: {required:?}"
        );
        assert_antiderivative_verifies(input);
    }

    let (linear_arctan_result, _) =
        evaluated_integral_with_required_conditions("integrate(x*arctan(x), x)");
    assert!(
        linear_arctan_result.matches("arctan(x)").nth(1).is_none(),
        "linear polynomial-arctan by-parts presentation should collect repeated arctan terms, got {linear_arctan_result}"
    );
    let (cubic_arctan_result, _) =
        evaluated_integral_with_required_conditions("integrate(x^3*arctan(x), x)");
    assert!(
        cubic_arctan_result.matches("arctan(x)").nth(1).is_none(),
        "cubic polynomial-arctan by-parts presentation should collect repeated arctan terms, got {cubic_arctan_result}"
    );
    let (quintic_arctan_result, _) =
        evaluated_integral_with_required_conditions("integrate(x^5*arctan(x), x)");
    assert!(
        quintic_arctan_result.matches("arctan(x)").nth(1).is_none(),
        "quintic polynomial-arctan by-parts presentation should collect repeated arctan terms, got {quintic_arctan_result}"
    );
    let (shifted_quadratic_arctan_result, _) =
        evaluated_integral_with_required_conditions("integrate(x^2*arctan(x+1), x)");
    assert!(
        shifted_quadratic_arctan_result
            .matches("arctan(x + 1)")
            .nth(1)
            .is_none(),
        "shifted quadratic polynomial-arctan by-parts presentation should collect repeated arctan terms, got {shifted_quadratic_arctan_result}"
    );
    let (shifted_cubic_arctan_result, _) =
        evaluated_integral_with_required_conditions("integrate(x^3*arctan(x+1), x)");
    assert!(
        shifted_cubic_arctan_result
            .matches("arctan(x + 1)")
            .nth(1)
            .is_none(),
        "shifted cubic polynomial-arctan by-parts presentation should collect repeated arctan terms, got {shifted_cubic_arctan_result}"
    );

    for input in [
        "integrate(x*arctan(x), x)",
        "integrate(x^2*arctan(x), x)",
        "integrate(x^3*arctan(x), x)",
        "integrate(x^4*arctan(x), x)",
        "integrate(x^5*arctan(x), x)",
        "integrate(x^6*arctan(x), x)",
        "integrate(x^2*arctan(x+1), x)",
        "integrate(x^3*arctan(x+1), x)",
        "integrate(x^2*arctan(1-x), x)",
    ] {
        let (wire, _) = cli_eval_json_with_stderr(input);
        let result = wire["result"].as_str().expect("result string");
        assert!(
            !result.contains(" - ("),
            "public polynomial-arctan by-parts presentation should flatten subtracting a difference, got {result}"
        );
        let result_latex = wire["result_latex"].as_str().expect("result_latex string");
        assert!(
            !result_latex.contains(" - ("),
            "public polynomial-arctan by-parts LaTeX should flatten subtracting a difference, got {result_latex}"
        );
    }

    let (negative_shifted_result, _) =
        evaluated_integral_with_required_conditions("integrate(x^2*arctan(1-x), x)");
    assert_eq!(
        negative_shifted_result.matches("arctan(1 - x)").count(),
        1,
        "negative-shifted polynomial-arctan presentation should collect repeated arctan terms, got {negative_shifted_result}"
    );
    assert!(
        negative_shifted_result.contains("1/3 * x^3 + 2/3")
            && !negative_shifted_result.contains("arctan(x - 1)"),
        "negative-shifted polynomial-arctan presentation should orient correction terms toward the input argument, got {negative_shifted_result}"
    );

    let (negative_expanded_result, _) =
        evaluated_integral_with_required_conditions("integrate((x^2+x)*arctan(1-x), x)");
    assert_eq!(
        negative_expanded_result.matches("arctan(1 - x)").count(),
        1,
        "expanded negative-shifted polynomial-arctan presentation should collect repeated arctan terms, got {negative_expanded_result}"
    );
    assert_eq!(
        negative_expanded_result.matches("ln(x^2 + 2 - 2 * x)").count(),
        1,
        "expanded negative-shifted polynomial-arctan presentation should collect repeated log companions, got {negative_expanded_result}"
    );
    assert!(
        negative_expanded_result.contains("5/6 * ln(x^2 + 2 - 2 * x)")
            && negative_expanded_result.contains("7/6 * x")
            && !negative_expanded_result.contains("arctan(x - 1)"),
        "expanded negative-shifted polynomial-arctan presentation should keep compact companions and input orientation, got {negative_expanded_result}"
    );

    let residual = "diff(integrate(x^6*arctan(x), x), x) - x^6*arctan(x)";
    let (wire, stderr) = cli_eval_json_with_stderr(residual);
    assert_eq!(wire["result"].as_str().unwrap_or_default(), "0");
    assert!(
        !stderr.contains("depth_overflow"),
        "degree-six arctan by-parts residual should not emit depth_overflow warning\nstderr:\n{stderr}"
    );

    let shifted_residual = "diff(integrate(x^2*arctan(x+1), x), x) - x^2*arctan(x+1)";
    let (wire, stderr) = cli_eval_json_with_stderr(shifted_residual);
    assert_eq!(wire["result"], "0");
    assert!(
        stderr.is_empty(),
        "shifted polynomial-arctan residual should stay quiet\nstderr:\n{stderr}"
    );
    assert_eq!(wire["required_display"], serde_json::json!([]));

    let shifted_cubic_residual = "diff(integrate(x^3*arctan(x+1), x), x) - x^3*arctan(x+1)";
    let (wire, stderr) = cli_eval_json_with_stderr(shifted_cubic_residual);
    assert_eq!(wire["result"], "0");
    assert!(
        stderr.is_empty(),
        "shifted cubic polynomial-arctan residual should stay quiet\nstderr:\n{stderr}"
    );
    assert_eq!(wire["required_display"], serde_json::json!([]));

    let expanded_residual =
        "diff(integrate(x^2*arctan(x+1)+x*arctan(x+1), x), x) - (x^2*arctan(x+1)+x*arctan(x+1))";
    let (wire, stderr) = cli_eval_json_with_stderr(expanded_residual);
    assert_eq!(wire["result"], "0");
    assert!(
        stderr.is_empty(),
        "expanded shifted polynomial-arctan residual should stay quiet\nstderr:\n{stderr}"
    );
    assert_eq!(wire["required_display"], serde_json::json!([]));

    let negative_shifted_residual = "diff(integrate(x^2*arctan(1-x), x), x) - x^2*arctan(1-x)";
    let (wire, stderr) = cli_eval_json_with_stderr(negative_shifted_residual);
    assert_eq!(wire["result"], "0");
    assert!(
        stderr.is_empty(),
        "negative-shifted polynomial-arctan residual should stay quiet\nstderr:\n{stderr}"
    );
    assert_eq!(wire["required_display"], serde_json::json!([]));

    let compact_negative_shifted_residual =
        "diff(((x^3+2)*arctan(1-x))/3 + ln(x^2+2-2*x)/3 + x^2/6 + 2*x/3, x) - x^2*arctan(1-x)";
    let (wire, stderr) = cli_eval_json_with_stderr(compact_negative_shifted_residual);
    assert_eq!(wire["result"], "0");
    assert!(
        stderr.is_empty(),
        "explicit compact negative-shifted polynomial-arctan residual should stay quiet\nstderr:\n{stderr}"
    );
    assert_eq!(wire["required_display"], serde_json::json!([]));

    let negative_expanded_residual =
        "diff(integrate(x^2*arctan(1-x)+x*arctan(1-x), x), x) - (x^2*arctan(1-x)+x*arctan(1-x))";
    let (wire, stderr) = cli_eval_json_with_stderr(negative_expanded_residual);
    assert_eq!(wire["result"], "0");
    assert!(
        stderr.is_empty(),
        "expanded negative-shifted polynomial-arctan residual should stay quiet\nstderr:\n{stderr}"
    );
    assert_eq!(wire["required_display"], serde_json::json!([]));
}
#[test]
fn integrate_contract_bounded_inverse_trig_variable_by_parts() {
    let (result, required) = evaluated_integral_with_required_conditions("integrate(arcsin(x), x)");
    assert_eq!(result, "sqrt(1 - x^2) + x * arcsin(x)");
    assert_eq!(
        required,
        vec!["-1 < x < 1".to_string()],
        "arcsin integration should publish its open-domain condition"
    );
    assert_antiderivative_verifies("integrate(arcsin(x), x)");
    let (nested_residual, nested_required) =
        evaluated_expr_with_required_conditions("diff(integrate(arcsin(x), x), x) - arcsin(x)");
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["-1 < x < 1".to_string()],
        "nested arcsin verification should preserve the open-domain condition"
    );

    let (result, required) = evaluated_integral_with_required_conditions("integrate(arccos(x), x)");
    assert_eq!(result, "x * arccos(x) - sqrt(1 - x^2)");
    assert_eq!(
        required,
        vec!["-1 < x < 1".to_string()],
        "arccos integration should publish its open-domain condition"
    );
    assert_antiderivative_verifies("integrate(arccos(x), x)");
    let (nested_residual, nested_required) =
        evaluated_expr_with_required_conditions("diff(integrate(arccos(x), x), x) - arccos(x)");
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["-1 < x < 1".to_string()],
        "nested arccos verification should preserve the open-domain condition"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(arcsin(2*x), x)");
    assert_eq!(result, "sqrt(1/4 - x^2) + x * arcsin(2 * x)");
    assert_eq!(
        required,
        vec!["-1/2 < x < 1/2".to_string()],
        "scaled arcsin integration should publish its open-domain condition"
    );
    assert_rendered_antiderivative_verifies("integrate(arcsin(2*x), x)", &result);
    let (nested_residual, nested_required) =
        evaluated_expr_with_required_conditions("diff(integrate(arcsin(2*x), x), x) - arcsin(2*x)");
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["-1/2 < x < 1/2".to_string()],
        "nested scaled arcsin verification should preserve the open-domain condition"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(arccos(2*x), x)");
    assert_eq!(result, "x * arccos(2 * x) - sqrt(1/4 - x^2)");
    assert_eq!(
        required,
        vec!["-1/2 < x < 1/2".to_string()],
        "scaled arccos integration should publish its open-domain condition"
    );
    assert_rendered_antiderivative_verifies("integrate(arccos(2*x), x)", &result);
    let (nested_residual, nested_required) =
        evaluated_expr_with_required_conditions("diff(integrate(arccos(2*x), x), x) - arccos(2*x)");
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["-1/2 < x < 1/2".to_string()],
        "nested scaled arccos verification should preserve the open-domain condition"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(arcsin(-2*x), x)");
    assert_eq!(result, "x * arcsin(-2 * x) - 1/2 * sqrt(1 - (-2 * x)^2)");
    assert_eq!(
        required,
        vec!["-1/2 < x < 1/2".to_string()],
        "negative-scale arcsin integration should publish its open-domain condition"
    );
    assert_rendered_antiderivative_verifies("integrate(arcsin(-2*x), x)", &result);
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(arcsin(-2*x), x), x) - arcsin(-2*x)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["-1/2 < x < 1/2".to_string()],
        "nested negative-scale arcsin verification should preserve the open-domain condition"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(arcsin(2*x+1), x)");
    assert_eq!(
        result,
        "1/2 * sqrt(1 - (2 * x + 1)^2) + 1/2 * (2 * x + 1) * arcsin(2 * x + 1)"
    );
    assert_eq!(
        required,
        vec!["-1 < x < 0".to_string()],
        "shifted positive-slope arcsin integration should publish its open-domain condition"
    );
    assert_antiderivative_verifies("integrate(arcsin(2*x+1), x)");
    assert_rendered_antiderivative_verifies("integrate(arcsin(2*x+1), x)", &result);
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(arcsin(2*x+1), x), x) - arcsin(2*x+1)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["-1 < x < 0".to_string()],
        "nested shifted arcsin verification should preserve the open-domain condition"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(arccos(2*x+1), x)");
    assert_eq!(
        result,
        "1/2 * (2 * x + 1) * arccos(2 * x + 1) - 1/2 * sqrt(1 - (2 * x + 1)^2)"
    );
    assert_eq!(
        required,
        vec!["-1 < x < 0".to_string()],
        "shifted positive-slope arccos integration should publish its open-domain condition"
    );
    assert_antiderivative_verifies("integrate(arccos(2*x+1), x)");
    assert_rendered_antiderivative_verifies("integrate(arccos(2*x+1), x)", &result);
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(arccos(2*x+1), x), x) - arccos(2*x+1)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["-1 < x < 0".to_string()],
        "nested shifted arccos verification should preserve the open-domain condition"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(arcsin(2*x-1), x)");
    assert_eq!(
        result,
        "1/2 * sqrt(1 - (2 * x - 1)^2) + 1/2 * (2 * x - 1) * arcsin(2 * x - 1)"
    );
    assert_eq!(
        required,
        vec!["0 < x < 1".to_string()],
        "opposite-offset positive-slope arcsin integration should publish its open-domain condition"
    );
    assert_antiderivative_verifies("integrate(arcsin(2*x-1), x)");
    assert_rendered_antiderivative_verifies("integrate(arcsin(2*x-1), x)", &result);

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(arcsin(1-2*x), x)");
    assert_eq!(
        result,
        "-1/2 * (4 * (x - x^2))^(1/2) - 1/2 * (1 - 2 * x) * arcsin(1 - 2 * x)"
    );
    assert_eq!(
        required,
        vec!["0 < x < 1".to_string()],
        "negative-slope shifted arcsin integration should publish its open-domain condition"
    );
    assert_antiderivative_verifies("integrate(arcsin(1-2*x), x)");
    assert_rendered_antiderivative_verifies("integrate(arcsin(1-2*x), x)", &result);
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(arcsin(1-2*x), x), x) - arcsin(1-2*x)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["0 < x < 1".to_string()],
        "nested negative-slope arcsin verification should preserve the open-domain condition"
    );

    assert_eq!(
        simplified_integral("integrate(arcsin(a*x+1), x)"),
        "integrate(arcsin(a * x + 1), x)",
        "symbolic-scale bounded inverse-trig integration remains intentionally deferred"
    );
}
#[test]
fn integrate_contract_arccos_negative_slope_preserves_compact_by_parts_form() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(arccos(1-2*x), x)");
    assert_eq!(
        result,
        "1/2 * (4 * (x - x^2))^(1/2) - 1/2 * (1 - 2 * x) * arccos(1 - 2 * x)"
    );
    assert_eq!(
        required,
        vec!["0 < x < 1".to_string()],
        "negative-slope shifted arccos integration should publish its open-domain condition"
    );
    assert_rendered_antiderivative_verifies("integrate(arccos(1-2*x), x)", &result);
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(arccos(1-2*x), x), x) - arccos(1-2*x)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["0 < x < 1".to_string()],
        "nested negative-slope arccos verification should preserve the open-domain condition"
    );
}
#[test]
fn integrate_contract_arctan_reciprocal_scaled_variable_by_parts() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(arctan(1/x), x)");
    assert_eq!(result, "1/2 * (ln(x^2 + 1) + 2 * x * arctan(1 / x))");
    assert_eq!(
        required,
        vec!["x ≠ 0".to_string()],
        "reciprocal arctan integration should preserve the reciprocal nonzero condition"
    );
    assert_antiderivative_verifies("integrate(arctan(1/x), x)");
    assert_rendered_antiderivative_verifies("integrate(arctan(1/x), x)", &result);

    let (result, required) = evaluated_integral_with_required_conditions("integrate(arccot(x), x)");
    assert_eq!(result, "1/2 * (ln(x^2 + 1) + 2 * x * arctan(1 / x))");
    assert_eq!(
        required,
        vec!["x ≠ 0".to_string()],
        "arccot canonicalization should keep the explicit reciprocal condition"
    );
    assert_antiderivative_verifies("integrate(arccot(x), x)");
    assert_rendered_antiderivative_verifies("integrate(arccot(x), x)", &result);

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(arccot(2*x), x)");
    assert_eq!(
        result,
        "1/4 * (ln(4 * x^2 + 1) + 4 * x * arctan(1 / (2 * x)))"
    );
    assert_eq!(
        required,
        vec!["x ≠ 0".to_string()],
        "scaled arccot canonicalization should keep the explicit reciprocal condition"
    );
    assert_antiderivative_verifies("integrate(arccot(2*x), x)");
    assert_rendered_antiderivative_verifies("integrate(arccot(2*x), x)", &result);

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(arctan(1/(2*x+1)), x)");
    assert_eq!(
        result,
        "1/4 * ln((2 * x + 1)^2 + 1) + 1/2 * (2 * x + 1) * arctan(1 / (2 * x + 1))"
    );
    assert_eq!(
        required,
        vec!["x ≠ -1/2".to_string()],
        "shifted reciprocal arctan integration should preserve the affine reciprocal condition"
    );
    assert_antiderivative_verifies("integrate(arctan(1/(2*x+1)), x)");
    assert_rendered_antiderivative_verifies("integrate(arctan(1/(2*x+1)), x)", &result);
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(arctan(1/(2*x+1)), x), x) - arctan(1/(2*x+1))",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["x ≠ -1/2".to_string()],
        "nested integrate/diff verification should keep the affine reciprocal condition"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(arccot(2*x+1), x)");
    assert_eq!(
        result,
        "1/4 * ln((2 * x + 1)^2 + 1) + 1/2 * (2 * x + 1) * arctan(1 / (2 * x + 1))"
    );
    assert_eq!(
        required,
        vec!["x ≠ -1/2".to_string()],
        "shifted arccot canonicalization should keep the explicit affine reciprocal condition"
    );
    assert_antiderivative_verifies("integrate(arccot(2*x+1), x)");
    assert_rendered_antiderivative_verifies("integrate(arccot(2*x+1), x)", &result);

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(arccot(1-2*x), x)");
    assert_eq!(
        result,
        "-1/2 * (1 - 2 * x) * arctan(1 / (1 - 2 * x)) - 1/4 * ln((1 - 2 * x)^2 + 1)"
    );
    assert_eq!(
        required,
        vec!["x ≠ 1/2".to_string()],
        "negative-slope shifted arccot canonicalization should keep the explicit affine reciprocal condition"
    );
    assert_antiderivative_verifies("integrate(arccot(1-2*x), x)");
    assert_rendered_antiderivative_verifies("integrate(arccot(1-2*x), x)", &result);
}
#[test]
fn integrate_contract_asinh_affine_by_parts() {
    let (result, required) = evaluated_integral_with_required_conditions("integrate(asinh(x), x)");
    assert_eq!(result, "x * asinh(x) - sqrt(x^2 + 1)");
    assert!(
        required.is_empty(),
        "asinh integration should not add required conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(asinh(x), x)");
    assert_rendered_antiderivative_verifies("integrate(asinh(x), x)", &result);

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(asinh(2*x), x)");
    assert_eq!(result, "x * asinh(2 * x) - 1/2 * sqrt((2 * x)^2 + 1)");
    assert!(
        required.is_empty(),
        "scaled asinh integration should not add required conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(asinh(2*x), x)");
    assert_rendered_antiderivative_verifies("integrate(asinh(2*x), x)", &result);

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(asinh(2*x+1), x)");
    assert_eq!(
        result,
        "1/2 * (2 * x + 1) * asinh(2 * x + 1) - 1/2 * sqrt((2 * x + 1)^2 + 1)"
    );
    assert!(
        required.is_empty(),
        "affine asinh integration should not add required conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(asinh(2*x+1), x)");
    assert_rendered_antiderivative_verifies("integrate(asinh(2*x+1), x)", &result);
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(asinh(2*x+1), x), x) - asinh(2*x+1)",
    );
    assert_eq!(nested_residual, "0");
    assert!(
        nested_required.is_empty(),
        "nested shifted asinh verification should not add required conditions: {nested_required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(asinh(1-2*x), x)");
    assert_eq!(
        result,
        "1/2 * sqrt((1 - 2 * x)^2 + 1) - 1/2 * (1 - 2 * x) * asinh(1 - 2 * x)"
    );
    assert!(
        required.is_empty(),
        "negative-slope affine asinh integration should not add required conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(asinh(1-2*x), x)");
    assert_rendered_antiderivative_verifies("integrate(asinh(1-2*x), x)", &result);
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(asinh(1-2*x), x), x) - asinh(1-2*x)",
    );
    assert_eq!(nested_residual, "0");
    assert!(
        nested_required.is_empty(),
        "nested negative-slope asinh verification should not add required conditions: {nested_required:?}"
    );
}
#[test]
fn integrate_contract_atanh_affine_by_parts_preserves_open_interval_domain() {
    let (result, required) = evaluated_integral_with_required_conditions("integrate(atanh(x), x)");
    assert_eq!(result, "1/2 * ln(1 - x^2) + x * atanh(x)");
    assert_eq!(
        required,
        vec!["-1 < x < 1".to_string()],
        "atanh integration should publish its open-interval condition"
    );
    assert_antiderivative_verifies("integrate(atanh(x), x)");
    assert_rendered_antiderivative_verifies("integrate(atanh(x), x)", &result);
    let (nested_residual, nested_required) =
        evaluated_expr_with_required_conditions("diff(integrate(atanh(x), x), x) - atanh(x)");
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["-1 < x < 1".to_string()],
        "nested atanh verification should preserve the open-interval condition"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(atanh(2*x), x)");
    assert_eq!(result, "1/4 * ln(1 - (2 * x)^2) + x * atanh(2 * x)");
    assert_eq!(
        required,
        vec!["-1/2 < x < 1/2".to_string()],
        "scaled atanh integration should publish its normalized open-interval condition"
    );
    assert_antiderivative_verifies("integrate(atanh(2*x), x)");
    assert_rendered_antiderivative_verifies("integrate(atanh(2*x), x)", &result);

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(atanh(2*x+1), x)");
    assert_eq!(
        result,
        "1/4 * ln(1 - (2 * x + 1)^2) + 1/2 * (2 * x + 1) * atanh(2 * x + 1)"
    );
    assert_eq!(
        required,
        vec!["-1 < x < 0".to_string()],
        "shifted atanh integration should publish its normalized open-interval condition"
    );
    assert_antiderivative_verifies("integrate(atanh(2*x+1), x)");
    assert_rendered_antiderivative_verifies("integrate(atanh(2*x+1), x)", &result);
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(atanh(2*x+1), x), x) - atanh(2*x+1)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["-1 < x < 0".to_string()],
        "nested shifted atanh verification should preserve the normalized condition"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(atanh(1-2*x), x)");
    assert_eq!(
        result,
        "-1/2 * (1 - 2 * x) * atanh(1 - 2 * x) - 1/4 * ln(1 - (1 - 2 * x)^2)"
    );
    assert_eq!(
        required,
        vec!["0 < x < 1".to_string()],
        "negative-slope atanh integration should publish its normalized open-interval condition"
    );
    let (wire, _stderr) = cli_eval_json_with_stderr("integrate(atanh(1-2*x), x)");
    assert_eq!(
        wire["result"],
        "-1/2·(1 - 2·x)·atanh(1 - 2·x) - 1/4·ln(1 - (1 - 2·x)^2)"
    );
    assert!(
        !wire["result"]
            .as_str()
            .unwrap_or_default()
            .contains("ln(1 - 1 +"),
        "public negative-slope atanh presentation must not rewrite inside ln: {}",
        wire["result"]
    );
    assert_antiderivative_verifies("integrate(atanh(1-2*x), x)");
    assert_rendered_antiderivative_verifies("integrate(atanh(1-2*x), x)", &result);
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(atanh(1-2*x), x), x) - atanh(1-2*x)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["0 < x < 1".to_string()],
        "nested negative-slope atanh verification should preserve the normalized condition"
    );
}
#[test]
fn integrate_contract_acosh_affine_by_parts_preserves_real_radical_domain() {
    let (result, required) = evaluated_integral_with_required_conditions("integrate(acosh(x), x)");
    assert_eq!(result, "x * acosh(x) - sqrt(x - 1) * sqrt(x + 1)");
    assert_eq!(
        required,
        vec!["x > 1".to_string()],
        "acosh integration should publish the real radical conditions"
    );
    assert_rendered_antiderivative_verifies("integrate(acosh(x), x)", &result);
    let (nested_residual, nested_required) =
        evaluated_expr_with_required_conditions("diff(integrate(acosh(x), x), x) - acosh(x)");
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["x > 1".to_string()],
        "nested acosh verification should preserve the real radical conditions"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(acosh(2*x), x)");
    assert_eq!(
        result,
        "x * acosh(2 * x) - 1/2 * sqrt(2 * x - 1) * sqrt(2 * x + 1)"
    );
    assert_eq!(
        required,
        vec!["x > 1/2".to_string()],
        "scaled acosh integration should publish the real radical conditions"
    );
    assert_rendered_antiderivative_verifies("integrate(acosh(2*x), x)", &result);

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(acosh(2*x+1), x)");
    assert_eq!(
        result,
        "1/2 * (2 * x + 1) * acosh(2 * x + 1) - 1/2 * sqrt(2 * x) * sqrt(2 * x + 2)"
    );
    assert_eq!(
        required,
        vec!["x > 0".to_string()],
        "shifted acosh integration should publish its normalized real-domain condition"
    );
    assert_rendered_antiderivative_verifies("integrate(acosh(2*x+1), x)", &result);
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(acosh(2*x+1), x), x) - acosh(2*x+1)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["x > 0".to_string()],
        "nested shifted acosh verification should preserve the normalized condition"
    );
}
#[test]
fn integrate_contract_acosh_negative_slope_preserves_compact_by_parts_form() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(acosh(1-2*x), x)");
    assert_eq!(
        result,
        "1/2 * sqrt(-2 * x) * sqrt(2 - 2 * x) - 1/2 * (1 - 2 * x) * acosh(1 - 2 * x)"
    );
    assert_eq!(
        required,
        vec!["x < 0".to_string()],
        "negative-slope acosh integration should publish its normalized real-domain condition"
    );
    assert_rendered_antiderivative_verifies("integrate(acosh(1-2*x), x)", &result);
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(acosh(1-2*x), x), x) - acosh(1-2*x)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["x < 0".to_string()],
        "nested negative-slope acosh verification should preserve the normalized condition"
    );
}
#[test]
fn integrate_contract_positive_log_square_product_by_parts() {
    let input = "integrate(2*x*ln(x^2+1)^2, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "(x^2 + 1) * (ln(x^2 + 1)^2 - 2 * ln(x^2 + 1) + 2)");
    assert!(
        required.is_empty(),
        "positive log-square product integration should not add required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}
#[test]
fn integrate_contract_positive_log_cube_product_by_parts_verifies() {
    let input = "integrate(2*x*ln(x^2+1)^3, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "(ln(x^2 + 1)^3 - 3 * ln(x^2 + 1)^2 + 6 * ln(x^2 + 1) - 6) * (x^2 + 1)"
    );
    assert!(
        required.is_empty(),
        "positive log-cube product integration should not add required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);

    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(2*x*ln(x^2+1)^3, x), x) - 2*x*ln(x^2+1)^3",
    );
    assert_eq!(nested_residual, "0");
    assert!(
        nested_required.is_empty(),
        "nested log-cube verification should not add required conditions: {nested_required:?}"
    );
}
#[test]
fn integrate_contract_shifted_quadratic_log_cube_product_by_parts_verifies() {
    let input = "integrate((2*x+1)*ln(x^2+x+1)^3, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "(ln(x^2 + x + 1)^3 - 3 * ln(x^2 + x + 1)^2 + 6 * ln(x^2 + x + 1) - 6) * (x^2 + x + 1)"
    );
    assert!(
        required.is_empty(),
        "shifted quadratic log-cube product integration should not add required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);

    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate((2*x+1)*ln(x^2+x+1)^3, x), x) - (2*x+1)*ln(x^2+x+1)^3",
    );
    assert_eq!(nested_residual, "0");
    assert!(
        nested_required.is_empty(),
        "nested shifted quadratic log-cube verification should not add required conditions: {nested_required:?}"
    );
}
#[test]
fn integrate_contract_conditional_log_cube_product_by_parts_verifies() {
    let input = "integrate(2*x*ln(x^2-1)^3, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "(ln(x^2 - 1)^3 - 3 * ln(x^2 - 1)^2 + 6 * ln(x^2 - 1) - 6) * (x^2 - 1)"
    );
    assert_eq!(
        required,
        vec!["x < -1 or x > 1".to_string()],
        "conditional log-cube product integration should publish its positive-domain condition"
    );
    assert_antiderivative_verifies(input);

    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(2*x*ln(x^2-1)^3, x), x) - 2*x*ln(x^2-1)^3",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["x < -1 or x > 1".to_string()],
        "nested conditional log-cube verification should preserve the positive-domain condition"
    );
}
#[test]
fn integrate_contract_log_high_power_product_by_parts_verifies() {
    let cases: [(&str, &str, &[&str], &str); 4] = [
        (
            "integrate(2*x*ln(x^2+1)^4, x)",
            "(x^2 + 1) * (ln(x^2 + 1)^4 - 4 * ln(x^2 + 1)^3 + 12 * ln(x^2 + 1)^2 - 24 * ln(x^2 + 1) + 24)",
            &[],
            "diff(integrate(2*x*ln(x^2+1)^4, x), x) - 2*x*ln(x^2+1)^4",
        ),
        (
            "integrate(2*x*ln(x^2+1)^5, x)",
            "(x^2 + 1) * (ln(x^2 + 1)^5 - 5 * ln(x^2 + 1)^4 + 20 * ln(x^2 + 1)^3 - 60 * ln(x^2 + 1)^2 + 120 * ln(x^2 + 1) - 120)",
            &[],
            "diff(integrate(2*x*ln(x^2+1)^5, x), x) - 2*x*ln(x^2+1)^5",
        ),
        (
            "integrate((2*x+1)*ln(x^2+x+1)^4, x)",
            "(x^2 + x + 1) * (ln(x^2 + x + 1)^4 - 4 * ln(x^2 + x + 1)^3 + 12 * ln(x^2 + x + 1)^2 - 24 * ln(x^2 + x + 1) + 24)",
            &[],
            "diff(integrate((2*x+1)*ln(x^2+x+1)^4, x), x) - (2*x+1)*ln(x^2+x+1)^4",
        ),
        (
            "integrate((2*x+1)*ln(x^2+x+1)^5, x)",
            "(x^2 + x + 1) * (ln(x^2 + x + 1)^5 - 5 * ln(x^2 + x + 1)^4 + 20 * ln(x^2 + x + 1)^3 - 60 * ln(x^2 + x + 1)^2 + 120 * ln(x^2 + x + 1) - 120)",
            &[],
            "diff(integrate((2*x+1)*ln(x^2+x+1)^5, x), x) - (2*x+1)*ln(x^2+x+1)^5",
        ),
    ];

    for (input, expected, expected_required, residual_input) in cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected);
        let expected_required: Vec<String> = expected_required
            .iter()
            .map(|condition| condition.to_string())
            .collect();
        assert_eq!(
            required, expected_required,
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert_antiderivative_verifies(input);

        let (residual, residual_required) = evaluated_expr_with_required_conditions(residual_input);
        assert_eq!(residual, "0");
        assert_eq!(
            residual_required, expected_required,
            "unexpected nested required_conditions for {residual_input}: {residual_required:?}"
        );
    }
}
#[test]
fn integrate_contract_conditional_high_log_power_product_by_parts_verifies() {
    let cases: [(&str, &str, &[&str], &str); 3] = [
        (
            "integrate(2*x*ln(x^2-1)^4, x)",
            "(x^2 - 1) * (ln(x^2 - 1)^4 - 4 * ln(x^2 - 1)^3 + 12 * ln(x^2 - 1)^2 - 24 * ln(x^2 - 1) + 24)",
            &["x < -1 or x > 1"],
            "diff(integrate(2*x*ln(x^2-1)^4, x), x) - 2*x*ln(x^2-1)^4",
        ),
        (
            "integrate(2*x*ln(x^2-1)^5, x)",
            "(x^2 - 1) * (ln(x^2 - 1)^5 - 5 * ln(x^2 - 1)^4 + 20 * ln(x^2 - 1)^3 - 60 * ln(x^2 - 1)^2 + 120 * ln(x^2 - 1) - 120)",
            &["x < -1 or x > 1"],
            "diff(integrate(2*x*ln(x^2-1)^5, x), x) - 2*x*ln(x^2-1)^5",
        ),
        (
            "integrate((2*x+1)*ln(x^2+x-1)^4, x)",
            "(x^2 + x - 1) * (ln(x^2 + x - 1)^4 - 4 * ln(x^2 + x - 1)^3 + 12 * ln(x^2 + x - 1)^2 - 24 * ln(x^2 + x - 1) + 24)",
            &["x < -1/2 - sqrt(5)/2 or x > -1/2 + sqrt(5)/2"],
            "diff(integrate((2*x+1)*ln(x^2+x-1)^4, x), x) - (2*x+1)*ln(x^2+x-1)^4",
        ),
    ];

    for (input, expected, expected_required, residual_input) in cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected, "input: {input}");
        let expected_required: Vec<String> = expected_required
            .iter()
            .map(|condition| condition.to_string())
            .collect();
        assert_eq!(
            required, expected_required,
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert_antiderivative_verifies(input);

        let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual_input);
        assert!(
            residual_stderr.is_empty(),
            "conditional high-log-power residual should stay quiet for {input}\nstderr:\n{residual_stderr}"
        );
        assert_eq!(residual_wire["result"], "0", "input: {input}");
        assert_eq!(
            residual_wire["required_display"],
            serde_json::json!(expected_required),
            "unexpected residual required_display for {input}: {:?}",
            residual_wire["required_display"]
        );
    }
}
#[test]
fn integrate_contract_linear_log_square_product_by_parts_preserves_positive_domain() {
    let input = "integrate(ln(2*x+1)^2, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "1/2 * (2 * x + 1) * (ln(2 * x + 1)^2 - 2 * ln(2 * x + 1) + 2)"
    );
    assert_eq!(
        required,
        vec!["x > -1/2".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}
#[test]
fn integrate_contract_quadratic_log_square_product_by_parts() {
    let input = "integrate((2*x+1)*ln(x^2+x+1)^2, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "(x^2 + x + 1) * (ln(x^2 + x + 1)^2 - 2 * ln(x^2 + x + 1) + 2)"
    );
    assert!(
        required.is_empty(),
        "positive quadratic log-square integration should not add required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}
#[test]
fn integrate_contract_conditional_monomial_log_square_product_by_parts_verifies() {
    let input = "integrate(2*x*ln(x^2-1)^2, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "(x^2 - 1) * (ln(x^2 - 1)^2 - 2 * ln(x^2 - 1) + 2)");
    assert_eq!(
        required,
        vec!["x < -1 or x > 1".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}
#[test]
fn integrate_contract_conditional_quadratic_log_square_product_by_parts_verifies() {
    let input = "integrate((2*x+1)*ln(x^2+x-1)^2, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "(ln(x^2 + x - 1)^2 - 2 * ln(x^2 + x - 1) + 2) * (x^2 + x - 1)"
    );
    assert_eq!(
        required,
        vec!["x < -1/2 - sqrt(5)/2 or x > -1/2 + sqrt(5)/2".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}
#[test]
fn integrate_contract_conditional_cubic_log_square_product_by_parts_verifies() {
    let input = "integrate((3*x^2-1)*ln(x^3-x)^2, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "(x^3 - x) * (ln(x^3 - x)^2 - 2 * ln(x^3 - x) + 2)");
    assert_eq!(
        required,
        vec!["x^3 - x > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}
#[test]
fn integrate_contract_conditional_quartic_log_square_product_by_parts_verifies() {
    let input = "integrate((4*x^3-2*x)*ln(x^4-x^2-1)^2, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "(ln(x^4 - x^2 - 1)^2 - 2 * ln(x^4 - x^2 - 1) + 2) * (x^4 - x^2 - 1)"
    );
    assert_eq!(
        required,
        vec!["x^4 - x^2 - 1 > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}
#[test]
fn integrate_contract_quadratic_affine_log_by_parts_nested_residual_collapses() {
    let input = "integrate((x^2+x)*ln(x+1), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert!(
        result.contains("ln(x + 1)"),
        "quadratic affine-log primitive should retain the log factor: {result}"
    );
    assert_eq!(
        required,
        vec!["x > -1".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);

    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate((x^2+x)*ln(x+1), x), x) - (x^2+x)*ln(x+1)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["x > -1".to_string()],
        "nested residual should preserve the log domain condition"
    );

    let (expanded_nested_residual, expanded_nested_required) =
        evaluated_expr_with_required_conditions(
            "diff(integrate(x^2*ln(x+1)+x*ln(x+1), x), x) - (x^2*ln(x+1)+x*ln(x+1))",
        );
    assert_eq!(expanded_nested_residual, "0");
    assert_eq!(
        expanded_nested_required,
        vec!["x > -1".to_string()],
        "expanded nested residual should preserve the log domain condition"
    );

    let (negative_nested_residual, negative_nested_required) =
        evaluated_expr_with_required_conditions(
            "diff(integrate(x^2*ln(x+1)-x*ln(x+1), x), x) - (x^2*ln(x+1)-x*ln(x+1))",
        );
    assert_eq!(negative_nested_residual, "0");
    assert_eq!(
        negative_nested_required,
        vec!["x > -1".to_string()],
        "negative nested residual should preserve the log domain condition"
    );
}
