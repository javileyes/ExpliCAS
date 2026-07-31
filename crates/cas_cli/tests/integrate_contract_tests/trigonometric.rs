use super::*;

#[test]
fn integrate_contract_exp_trig_same_linear_antiderivatives_verify_by_differentiation() {
    for (input, expected) in [
        (
            "integrate(exp(x)*sin(x), x)",
            "1/2 * e^x * (sin(x) - cos(x))",
        ),
        (
            "integrate(exp(x)*cos(x), x)",
            "1/2 * e^x * (sin(x) + cos(x))",
        ),
        (
            "integrate(exp(2*x+1)*sin(2*x+1), x)",
            "1/4 * e^(2 * x + 1) * (sin(2 * x + 1) - cos(2 * x + 1))",
        ),
        (
            "integrate(exp(2*x+1)*cos(2*x+1), x)",
            "1/4 * e^(2 * x + 1) * (sin(2 * x + 1) + cos(2 * x + 1))",
        ),
        (
            "integrate(3*exp(2*x+1)*sin(2*x+1), x)",
            "3/4 * e^(2 * x + 1) * (sin(2 * x + 1) - cos(2 * x + 1))",
        ),
        (
            "integrate(exp(2*x)*sin(3*x), x)",
            "1/13 * e^(2 * x) * (2 * sin(3 * x) - 3 * cos(3 * x))",
        ),
        (
            "integrate(exp(2*x)*sin((3*x+1)/2), x)",
            "4/25 * e^(2 * x) * (2 * sin((3 * x + 1) / 2) - 3/2 * cos((3 * x + 1) / 2))",
        ),
        (
            "integrate(exp(2*x)*cos((3*x+1)/2), x)",
            "4/25 * e^(2 * x) * (3/2 * sin((3 * x + 1) / 2) + 2 * cos((3 * x + 1) / 2))",
        ),
    ] {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected, "input: {input}");
        assert!(
            required.is_empty(),
            "exp-trig same-linear antiderivative should not add domain conditions for {input}: {required:?}"
        );
        assert_rendered_antiderivative_verifies(input, expected);
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "{input} should verify through the bounded public residual route"
        );
    }
}
#[test]
fn integrate_contract_exp_trig_integer_multiple_cos_verifies_by_public_residual() {
    let input = "integrate(exp(2*x)*cos(3*x), x)";
    let expected = "1/13 * e^(2 * x) * (2 * cos(3 * x) + 3 * sin(3 * x))";

    let (result, required) = evaluated_integral_with_required_conditions(input);
    assert_eq!(result, expected);
    assert!(
        required.is_empty(),
        "exp-trig integer-multiple cosine should not add domain conditions: {required:?}"
    );
    assert_eq!(
        assert_antiderivative_verifies(input),
        AntiderivativeVerificationRoute::PublicResidual,
        "{input} should verify through the bounded public residual route"
    );
    assert_rendered_antiderivative_verifies(input, expected);
    assert_rendered_antiderivative_verifies(input, &format!("{expected} + 7"));
    assert_rendered_antiderivative_verifies(input, &format!("{expected} + C"));
}
#[test]
fn integrate_contract_exp_trig_wrong_sign_antiderivative_residual_compacts_without_depth_warning() {
    let primitive = "1/13*exp(2*x)*(2*cos(3*x)+3*sin(3*x))";
    let integrand = "exp(2*x)*cos(3*x)";
    let input = format!("diff(7 - {primitive}, x) - {integrand}");

    let (wire, stderr) = cli_eval_json_with_stderr(&input);
    assert_eq!(wire["result"], "-2·cos(3·x)·e^(2·x)");
    assert!(
        wire["warnings"]
            .as_array()
            .is_some_and(|warnings| warnings.is_empty()),
        "wrong-sign exp-trig residual should not emit warnings: {wire:#}"
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "wrong-sign exp-trig residual should not emit depth_overflow\nstderr:\n{stderr}"
    );
}
#[test]
fn integrate_contract_trig_of_logarithm_resolves_and_respects_boundary() {
    // cos/sin of ln(affine) integrate via the cyclic substitution u = ln(inner),
    // carrying the ln-positivity domain. The result round-trips to the integrand.
    for (input, expected_result, expected_required) in [
        (
            "integrate(cos(ln(x)), x)",
            "1/2 * x * (sin(ln(x)) + cos(ln(x)))",
            vec!["x > 0".to_string()],
        ),
        (
            "integrate(sin(ln(x)), x)",
            "1/2 * x * (sin(ln(x)) - cos(ln(x)))",
            vec!["x > 0".to_string()],
        ),
        (
            "integrate(cos(ln(2*x+1)), x)",
            "1/4 * (2 * x + 1) * (sin(ln(2 * x + 1)) + cos(ln(2 * x + 1)))",
            vec!["x > -1/2".to_string()],
        ),
    ] {
        let (result, required) = evaluated_expr_with_required_conditions(input);
        assert_eq!(result, expected_result, "result for {input}");
        assert_eq!(required, expected_required, "required for {input}");
        let (rt, _) = evaluated_expr_with_required_conditions(&format!(
            "diff(integrate({}, x), x) - ({})",
            &input[10..input.len() - 4],
            &input[10..input.len() - 4]
        ));
        assert_eq!(rt, "0", "round-trip for {input}");
    }

    // Boundary: only a bare cos/sin of ln(affine) fires. A non-logarithmic or
    // non-affine inner stays an honest residual, and the u=ln/x and bare-ln
    // owners keep their results (the new detector must not hijack them).
    for input in [
        "integrate(cos(ln(x) + 1), x)",
        "integrate(cos(x*ln(x)), x)",
        "integrate(cos(ln(x^2)), x)",
    ] {
        let (result, _) = evaluated_expr_with_required_conditions(input);
        assert!(
            result.starts_with("integrate("),
            "{input} should stay an honest residual, got {result}"
        );
    }
    assert_eq!(
        evaluated_expr_with_required_conditions("integrate(ln(x)/x, x)").0,
        "1/2 * ln(x)^2"
    );
    assert_eq!(
        evaluated_expr_with_required_conditions("integrate(ln(x), x)").0,
        "x * ln(x) - x"
    );
    assert_eq!(
        evaluated_expr_with_required_conditions("integrate(cos(x), x)").0,
        "sin(x)"
    );
}
#[test]
fn integrate_contract_reciprocal_shifted_csc_residual_compacts_without_timeout() {
    let input = "1/((diff(integrate(csc(2*x+1), x), x) - csc(2*x+1)) + x + 2) - 1/(x+2)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert!(
        stderr.is_empty(),
        "unexpected stderr for reciprocal shifted csc residual: {stderr}"
    );
    assert_eq!(wire["result"], "0");
    assert_eq!(
        wire["required_display"],
        serde_json::json!(["sin(2·x + 1) ≠ 0", "x ≠ -2"])
    );
}
#[test]
fn integrate_contract_shifted_quotient_csc_residual_compacts_without_timeout() {
    let input = "1 - (x+2)/((diff(integrate(csc(2*x+1), x), x) - csc(2*x+1)) + x + 2)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert!(
        stderr.is_empty(),
        "unexpected stderr for shifted quotient csc residual: {stderr}"
    );
    assert_eq!(wire["result"], "0");
    assert_eq!(
        wire["required_display"],
        serde_json::json!(["sin(2·x + 1) ≠ 0", "x ≠ -2"])
    );
}
#[test]
fn integrate_contract_root_product_residual_constant_passthrough_quotient_compacts_without_timeout()
{
    let input =
        "((diff(integrate(1/(sqrt(2*x)*sqrt(2*x+6)), x), x) - 1/(sqrt(2*x)*sqrt(2*x+6))) + x + 2)/(x+2)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert!(
        stderr.is_empty(),
        "unexpected stderr for root-product residual passthrough quotient: {stderr}"
    );
    assert_eq!(wire["result"], "1");
    assert_eq!(wire["required_display"], serde_json::json!(["x > 0"]));
}
#[test]
fn integrate_contract_product_zero_csc_residual_compacts_without_timeout() {
    for input in [
        "((diff(integrate(csc(2*x+1), x), x) - csc(2*x+1)) + x + 2)*(y-y)",
        "(y-y)*((diff(integrate(csc(2*x+1), x), x) - csc(2*x+1)) + x + 2)",
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr(input);

        assert!(
            stderr.is_empty(),
            "unexpected stderr for product-zero csc residual: {stderr}"
        );
        assert_eq!(wire["result"], "0", "{input}");
        assert_eq!(
            wire["required_display"],
            serde_json::json!(["sin(2·x + 1) ≠ 0"]),
            "{input}"
        );
    }
}
#[test]
fn integrate_contract_symbolic_constant_verification_preserves_independent_domain_conditions() {
    let input = "integrate(ln(y)*(z+1)^(-2), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);
    required.sort();

    assert_eq!(result, "x * ln(y) / (z + 1)^2");
    assert_eq!(
        required,
        vec!["y > 0".to_string(), "z ≠ -1".to_string()],
        "symbolic constant integration should publish independent domain conditions"
    );

    let (nested_residual, mut nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(ln(y)*(z+1)^(-2), x), x) - ln(y)*(z+1)^(-2)",
    );
    nested_required.sort();
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["y > 0".to_string(), "z ≠ -1".to_string()],
        "nested antiderivative verification should preserve independent domain conditions"
    );
}
#[test]
fn integrate_contract_affine_trig_square_power_reduction() {
    assert_eq!(
        simplified_integral("integrate(sin(x)^2, x)"),
        "1/4 * (2 * x - sin(2 * x))"
    );
    assert_eq!(
        simplified_integral("integrate(cos(x)^2, x)"),
        "1/4 * (sin(2 * x) + 2 * x)"
    );
    assert_eq!(
        simplified_integral("integrate(sin(2*x + 1)^2, x)"),
        "1/8 * (4 * x - sin(4 * x + 2))"
    );
    assert_eq!(
        simplified_integral("integrate(cos(2*x + 1)^2, x)"),
        "1/8 * (sin(4 * x + 2) + 4 * x)"
    );

    let public_cases = [
        ("integrate(sin(x)^2, x)", "1/2 * x - 1/4 * sin(2 * x)"),
        ("integrate(cos(x)^2, x)", "1/4 * sin(2 * x) + 1/2 * x"),
        (
            "integrate(sin(2*x + 1)^2, x)",
            "1/2 * x - 1/8 * sin(4 * x + 2)",
        ),
        (
            "integrate(cos(2*x + 1)^2, x)",
            "1/8 * sin(4 * x + 2) + 1/2 * x",
        ),
    ];

    for (input, expected) in public_cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected, "input: {input}");
        assert!(
            required.is_empty(),
            "input: {input}, required: {required:?}"
        );
        assert_rendered_antiderivative_verifies(input, expected);
    }
}
#[test]
fn integrate_contract_affine_trig_ratio_square_power_reduction() {
    let public_cases = [
        ("integrate(tan(x)^2, x)", "tan(x) - x", vec!["cos(x) ≠ 0"]),
        (
            "integrate(tan(2*x + 1)^2, x)",
            "1/2 * (tan(2 * x + 1) - 2 * x)",
            vec!["cos(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(tan(x)^4, x)",
            "tan(x)^3 / 3 + x - tan(x)",
            vec!["cos(x) ≠ 0"],
        ),
        (
            "integrate(tan(2*x + 1)^4, x)",
            "-tan(2 * x + 1) / 2 + tan(2 * x + 1)^3 / 6 + x",
            vec!["cos(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(sin(2*x + 1)^4/cos(2*x + 1)^4, x)",
            "-tan(2 * x + 1) / 2 + tan(2 * x + 1)^3 / 6 + x",
            vec!["cos(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(tan(x)^6, x)",
            "tan(x) + -tan(x)^3 / 3 + tan(x)^5 / 5 - x",
            vec!["cos(x) ≠ 0"],
        ),
        (
            "integrate(tan(2*x + 1)^6, x)",
            "tan(2 * x + 1) / 2 + -tan(2 * x + 1)^3 / 6 + tan(2 * x + 1)^5 / 10 - x",
            vec!["cos(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(tan(1-2*x)^6, x)",
            "-tan(1 - 2 * x) / 2 + -tan(1 - 2 * x)^5 / 10 + tan(1 - 2 * x)^3 / 6 - x",
            vec!["cos(1 - 2 * x) ≠ 0"],
        ),
        (
            "integrate(sin(2*x + 1)^6/cos(2*x + 1)^6, x)",
            "tan(2 * x + 1) / 2 + -tan(2 * x + 1)^3 / 6 + tan(2 * x + 1)^5 / 10 - x",
            vec!["cos(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(tan(x)^8, x)",
            "-tan(x)^5 / 5 + tan(x)^3 / 3 + tan(x)^7 / 7 + x - tan(x)",
            vec!["cos(x) ≠ 0"],
        ),
        (
            "integrate(tan(2*x + 1)^8, x)",
            "-tan(2 * x + 1) / 2 + -tan(2 * x + 1)^5 / 10 + tan(2 * x + 1)^3 / 6 + tan(2 * x + 1)^7 / 14 + x",
            vec!["cos(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(tan(1-2*x)^8, x)",
            "tan(1 - 2 * x) / 2 + -tan(1 - 2 * x)^3 / 6 + -tan(1 - 2 * x)^7 / 14 + tan(1 - 2 * x)^5 / 10 + x",
            vec!["cos(1 - 2 * x) ≠ 0"],
        ),
        (
            "integrate(sin(2*x + 1)^8/cos(2*x + 1)^8, x)",
            "-tan(2 * x + 1) / 2 + -tan(2 * x + 1)^5 / 10 + tan(2 * x + 1)^3 / 6 + tan(2 * x + 1)^7 / 14 + x",
            vec!["cos(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(sec(x)^4, x)",
            "tan(x) + tan(x)^3 / 3",
            vec!["cos(x) ≠ 0"],
        ),
        (
            "integrate(1/cos(2*x + 1)^4, x)",
            "tan(2 * x + 1) / 2 + tan(2 * x + 1)^3 / 6",
            vec!["cos(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(sec(x)^6, x)",
            "tan(x) + tan(x)^5 / 5 + 2 * tan(x)^3 / 3",
            vec!["cos(x) ≠ 0"],
        ),
        (
            "integrate(1/cos(2*x + 1)^6, x)",
            "tan(2 * x + 1) / 2 + tan(2 * x + 1)^3 / 3 + tan(2 * x + 1)^5 / 10",
            vec!["cos(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(sec(1-2*x)^6, x)",
            "-tan(1 - 2 * x) / 2 + -tan(1 - 2 * x)^3 / 3 + -tan(1 - 2 * x)^5 / 10",
            vec!["cos(1 - 2 * x) ≠ 0"],
        ),
        (
            "integrate(sec(x)^8, x)",
            "tan(x) + tan(x)^7 / 7 + 3 * tan(x)^5 / 5 + tan(x)^3",
            vec!["cos(x) ≠ 0"],
        ),
        (
            "integrate(1/cos(2*x + 1)^8, x)",
            "tan(2 * x + 1) / 2 + tan(2 * x + 1)^3 / 2 + tan(2 * x + 1)^7 / 14 + 3 * tan(2 * x + 1)^5 / 10",
            vec!["cos(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(sec(1-2*x)^8, x)",
            "-tan(1 - 2 * x) / 2 + -tan(1 - 2 * x)^3 / 2 + -tan(1 - 2 * x)^7 / 14 + -3 * tan(1 - 2 * x)^5 / 10",
            vec!["cos(1 - 2 * x) ≠ 0"],
        ),
        (
            "integrate(csc(x)^4, x)",
            "-cot(x)^3 / 3 - cot(x)",
            vec!["sin(x) ≠ 0"],
        ),
        (
            "integrate(1/sin(2*x + 1)^4, x)",
            "-cot(2 * x + 1)^3 / 6 - cot(2 * x + 1) / 2",
            vec!["sin(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(csc(x)^6, x)",
            "-cot(x)^5 / 5 + -2 * cot(x)^3 / 3 - cot(x)",
            vec!["sin(x) ≠ 0"],
        ),
        (
            "integrate(1/sin(2*x + 1)^6, x)",
            "-cot(2 * x + 1) / 2 + -cot(2 * x + 1)^3 / 3 + -cot(2 * x + 1)^5 / 10",
            vec!["sin(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(csc(1-2*x)^6, x)",
            "cot(1 - 2 * x) / 2 + cot(1 - 2 * x)^3 / 3 + cot(1 - 2 * x)^5 / 10",
            vec!["sin(1 - 2 * x) ≠ 0"],
        ),
        (
            "integrate(csc(x)^8, x)",
            "-cot(x)^7 / 7 + -3 * cot(x)^5 / 5 - cot(x) - cot(x)^3",
            vec!["sin(x) ≠ 0"],
        ),
        (
            "integrate(1/sin(2*x + 1)^8, x)",
            "-cot(2 * x + 1) / 2 + -cot(2 * x + 1)^3 / 2 + -cot(2 * x + 1)^7 / 14 + -3 * cot(2 * x + 1)^5 / 10",
            vec!["sin(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(csc(1-2*x)^8, x)",
            "cot(1 - 2 * x) / 2 + cot(1 - 2 * x)^3 / 2 + cot(1 - 2 * x)^7 / 14 + 3 * cot(1 - 2 * x)^5 / 10",
            vec!["sin(1 - 2 * x) ≠ 0"],
        ),
        (
            "integrate(cot(x)^4, x)",
            "cot(x) + x - cot(x)^3 / 3",
            vec!["sin(x) ≠ 0"],
        ),
        (
            "integrate(cos(2*x + 1)^4/sin(2*x + 1)^4, x)",
            "cot(2 * x + 1) / 2 + x - cot(2 * x + 1)^3 / 6",
            vec!["sin(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(cot(x)^6, x)",
            "-cot(x)^5 / 5 + cot(x)^3 / 3 - cot(x) - x",
            vec!["sin(x) ≠ 0"],
        ),
        (
            "integrate(cot(2*x + 1)^6, x)",
            "-cot(2 * x + 1) / 2 + -cot(2 * x + 1)^5 / 10 + cot(2 * x + 1)^3 / 6 - x",
            vec!["sin(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(cot(1-2*x)^6, x)",
            "cot(1 - 2 * x) / 2 + -cot(1 - 2 * x)^3 / 6 + cot(1 - 2 * x)^5 / 10 - x",
            vec!["sin(1 - 2 * x) ≠ 0"],
        ),
        (
            "integrate(cos(2*x + 1)^6/sin(2*x + 1)^6, x)",
            "-cot(2 * x + 1) / 2 + -cot(2 * x + 1)^5 / 10 + cot(2 * x + 1)^3 / 6 - x",
            vec!["sin(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(cot(x)^8, x)",
            "cot(x) + -cot(x)^3 / 3 + -cot(x)^7 / 7 + cot(x)^5 / 5 + x",
            vec!["sin(x) ≠ 0"],
        ),
        (
            "integrate(cot(2*x + 1)^8, x)",
            "cot(2 * x + 1) / 2 + -cot(2 * x + 1)^3 / 6 + -cot(2 * x + 1)^7 / 14 + cot(2 * x + 1)^5 / 10 + x",
            vec!["sin(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(cot(1-2*x)^8, x)",
            "-cot(1 - 2 * x) / 2 + -cot(1 - 2 * x)^5 / 10 + cot(1 - 2 * x)^3 / 6 + cot(1 - 2 * x)^7 / 14 + x",
            vec!["sin(1 - 2 * x) ≠ 0"],
        ),
        (
            "integrate(cos(2*x + 1)^8/sin(2*x + 1)^8, x)",
            "cot(2 * x + 1) / 2 + -cot(2 * x + 1)^3 / 6 + -cot(2 * x + 1)^7 / 14 + cot(2 * x + 1)^5 / 10 + x",
            vec!["sin(2 * x + 1) ≠ 0"],
        ),
        ("integrate(cot(x)^2, x)", "-cot(x) - x", vec!["sin(x) ≠ 0"]),
        (
            "integrate(cot(2*x + 1)^2, x)",
            "1/2 * (-cot(2 * x + 1) - 2 * x)",
            vec!["sin(2 * x + 1) ≠ 0"],
        ),
    ];

    for (input, expected, expected_required) in public_cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected, "input: {input}");
        assert_eq!(required, expected_required, "input: {input}");
        assert_rendered_antiderivative_verifies(input, expected);
    }

    for input in [
        "integrate(tan(x)^4, x)",
        "integrate(tan(2*x + 1)^4, x)",
        "integrate(sin(2*x + 1)^4/cos(2*x + 1)^4, x)",
        "integrate(tan(x)^6, x)",
        "integrate(tan(2*x + 1)^6, x)",
        "integrate(tan(1-2*x)^6, x)",
        "integrate(sin(2*x + 1)^6/cos(2*x + 1)^6, x)",
        "integrate(tan(x)^8, x)",
        "integrate(tan(2*x + 1)^8, x)",
        "integrate(tan(1-2*x)^8, x)",
        "integrate(sin(2*x + 1)^8/cos(2*x + 1)^8, x)",
        "integrate(sec(x)^4, x)",
        "integrate(1/cos(2*x + 1)^4, x)",
        "integrate(sec(x)^6, x)",
        "integrate(1/cos(2*x + 1)^6, x)",
        "integrate(sec(1-2*x)^6, x)",
        "integrate(sec(x)^8, x)",
        "integrate(1/cos(2*x + 1)^8, x)",
        "integrate(sec(1-2*x)^8, x)",
        "integrate(csc(x)^4, x)",
        "integrate(1/sin(2*x + 1)^4, x)",
        "integrate(csc(x)^6, x)",
        "integrate(1/sin(2*x + 1)^6, x)",
        "integrate(csc(1-2*x)^6, x)",
        "integrate(csc(x)^8, x)",
        "integrate(1/sin(2*x + 1)^8, x)",
        "integrate(csc(1-2*x)^8, x)",
        "integrate(cot(x)^4, x)",
        "integrate(cos(2*x + 1)^4/sin(2*x + 1)^4, x)",
        "integrate(cot(x)^6, x)",
        "integrate(cot(2*x + 1)^6, x)",
        "integrate(cot(1-2*x)^6, x)",
        "integrate(cos(2*x + 1)^6/sin(2*x + 1)^6, x)",
        "integrate(cot(x)^8, x)",
        "integrate(cot(2*x + 1)^8, x)",
        "integrate(cot(1-2*x)^8, x)",
        "integrate(cos(2*x + 1)^8/sin(2*x + 1)^8, x)",
    ] {
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "{input} should verify through the bounded public residual route"
        );
    }
}
#[test]
fn integrate_contract_affine_sine_cosine_product() {
    let public_cases = [
        ("integrate(sin(x)*cos(x), x)", "1/2 * sin(x)^2"),
        (
            "integrate(3*sin(2*x + 1)*cos(2*x + 1), x)",
            "3/4 * sin(2 * x + 1)^2",
        ),
        (
            "integrate(3*cos(2*x + 1)*sin(2*x + 1), x)",
            "3/4 * sin(2 * x + 1)^2",
        ),
        (
            "integrate(-3*sin(2*x + 1)*cos(2*x + 1), x)",
            "-3/4 * sin(2 * x + 1)^2",
        ),
    ];

    for (input, expected) in public_cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected, "input: {input}");
        assert!(
            required.is_empty(),
            "input: {input}, required: {required:?}"
        );
        assert_rendered_antiderivative_verifies(input, expected);
    }
}
#[test]
fn integrate_contract_affine_trig_power_times_derivative_product() {
    let public_cases = [
        ("integrate(sin(x)^2*cos(x), x)", "1/3 * sin(x)^3"),
        (
            "integrate(2*cos(2*x + 1)*sin(2*x + 1)^2, x)",
            "1/3 * sin(2 * x + 1)^3",
        ),
        (
            "integrate(2*sin(2*x + 1)^2*cos(2*x + 1), x)",
            "1/3 * sin(2 * x + 1)^3",
        ),
        (
            "integrate(-2*cos(2*x + 1)*sin(2*x + 1)^2, x)",
            "-1/3 * sin(2 * x + 1)^3",
        ),
        ("integrate(sin(x)*cos(x)^2, x)", "-1/3 * cos(x)^3"),
        ("integrate(-sin(x)*cos(x)^2, x)", "1/3 * cos(x)^3"),
    ];

    for (input, expected) in public_cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected, "input: {input}");
        assert!(
            required.is_empty(),
            "input: {input}, required: {required:?}"
        );
        assert_rendered_antiderivative_verifies(input, expected);
    }
}
#[test]
fn integrate_contract_affine_trig_ratio_power_reciprocal_square_product() {
    let public_cases = [
        (
            "integrate(sec(x)^2*tan(x), x)",
            "tan(x)^2 / 2",
            vec!["cos(x) ≠ 0"],
        ),
        (
            "integrate(2*sec(2*x + 1)^2*tan(2*x + 1), x)",
            "tan(2 * x + 1)^2 / 2",
            vec!["cos(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(sin(x)/cos(x)^3, x)",
            "tan(x)^2 / 2",
            vec!["cos(x) ≠ 0"],
        ),
        (
            "integrate(tan(x)^2/cos(x)^2, x)",
            "tan(x)^3 / 3",
            vec!["cos(x) ≠ 0"],
        ),
        (
            "integrate(2*tan(2*x + 1)^2/cos(2*x + 1)^2, x)",
            "tan(2 * x + 1)^3 / 3",
            vec!["cos(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(sin(x)^2/cos(x)^4, x)",
            "tan(x)^3 / 3",
            vec!["cos(x) ≠ 0"],
        ),
        (
            "integrate(csc(x)^2*cot(x), x)",
            "-cot(x)^2 / 2",
            vec!["sin(x) ≠ 0"],
        ),
        (
            "integrate(2*csc(2*x + 1)^2*cot(2*x + 1), x)",
            "-cot(2 * x + 1)^2 / 2",
            vec!["sin(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(cos(x)/sin(x)^3, x)",
            "-cot(x)^2 / 2",
            vec!["sin(x) ≠ 0"],
        ),
        (
            "integrate(cot(x)^2/sin(x)^2, x)",
            "-cot(x)^3 / 3",
            vec!["sin(x) ≠ 0"],
        ),
        (
            "integrate(2*cot(2*x + 1)^2/sin(2*x + 1)^2, x)",
            "-cot(2 * x + 1)^3 / 3",
            vec!["sin(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(cos(x)^2/sin(x)^4, x)",
            "-cot(x)^3 / 3",
            vec!["sin(x) ≠ 0"],
        ),
    ];

    for (input, expected, expected_required) in public_cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected, "input: {input}");
        assert_eq!(
            required, expected_required,
            "input: {input}, required: {required:?}"
        );
        assert_rendered_antiderivative_verifies(input, expected);
    }
}
#[test]
fn integrate_contract_reciprocal_trig_power_verification_avoids_depth_overflow() {
    for input in [
        "diff(tan(x)^2/2, x) - sec(x)^2*tan(x)",
        "diff(-cot(x)^2/2, x) - csc(x)^2*cot(x)",
        "diff(tan(2*x+1)^2/2, x) - 2*sec(2*x+1)^2*tan(2*x+1)",
        "diff(-cot(2*x+1)^2/2, x) - 2*csc(2*x+1)^2*cot(2*x+1)",
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr(input);
        assert_eq!(wire["result"], "0", "unexpected residual for {input}");
        assert!(
            !stderr.contains("depth_overflow"),
            "reciprocal trig power verification should not emit depth_overflow for {input}\nstderr:\n{stderr}"
        );
    }
}
#[test]
fn integrate_contract_affine_trig_cube_power_reduction() {
    assert_eq!(
        simplified_integral("integrate(sin(x)^3, x)"),
        "1/3 * (cos(x)^3 - 3 * cos(x))"
    );
    assert_eq!(
        simplified_integral("integrate(cos(x)^3, x)"),
        "1/3 * (3 * sin(x) - sin(x)^3)"
    );
    assert_eq!(
        simplified_integral("integrate(sin(2*x + 1)^3, x)"),
        "1/6 * (cos(2 * x + 1)^3 - 3 * cos(2 * x + 1))"
    );
    assert_eq!(
        simplified_integral("integrate(cos(2*x + 1)^3, x)"),
        "1/6 * (3 * sin(2 * x + 1) - sin(2 * x + 1)^3)"
    );

    let public_cases = [
        ("integrate(sin(x)^3, x)", "1/3 * cos(x)^3 - cos(x)"),
        ("integrate(cos(x)^3, x)", "sin(x) - 1/3 * sin(x)^3"),
        (
            "integrate(sin(2*x + 1)^3, x)",
            "1/6 * cos(2 * x + 1)^3 - 1/2 * cos(2 * x + 1)",
        ),
        (
            "integrate(cos(2*x + 1)^3, x)",
            "1/2 * sin(2 * x + 1) - 1/6 * sin(2 * x + 1)^3",
        ),
    ];

    for (input, expected) in public_cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected, "input: {input}");
        assert!(
            required.is_empty(),
            "input: {input}, required: {required:?}"
        );
        assert_antiderivative_verifies(input);
    }
}
#[test]
fn integrate_contract_affine_trig_fifth_power_reduction() {
    let cases = [
        (
            "integrate(sin(x)^5, x)",
            "sin(x)^5",
            "2/3 * cos(x)^3 - cos(x) - 1/5 * cos(x)^5",
        ),
        (
            "integrate(cos(x)^5, x)",
            "cos(x)^5",
            "sin(x) + 1/5 * sin(x)^5 - 2/3 * sin(x)^3",
        ),
        (
            "integrate(sin(2*x + 1)^5, x)",
            "sin(2*x + 1)^5",
            "1/3 * cos(2 * x + 1)^3 - 1/2 * cos(2 * x + 1) - 1/10 * cos(2 * x + 1)^5",
        ),
        (
            "integrate(cos(2*x + 1)^5, x)",
            "cos(2*x + 1)^5",
            "1/10 * sin(2 * x + 1)^5 + 1/2 * sin(2 * x + 1) - 1/3 * sin(2 * x + 1)^3",
        ),
    ];

    for (input, integrand, expected) in cases {
        let (antiderivative, required) = evaluated_integral_with_required_conditions(input);
        assert!(
            required.is_empty(),
            "trig fifth primitive should not add domain conditions for {input}: {required:?}"
        );
        assert_eq!(antiderivative, expected, "{input}");
        assert!(
            !antiderivative.contains("10/3"),
            "post-calculus presentation should distribute nested fifth-power primitive coefficients: {antiderivative}"
        );
        assert_antiderivative_verifies(input);
        assert_rendered_antiderivative_verifies(input, &antiderivative);

        let residual = format!("diff({input}, x) - {integrand}");
        let (wire, _stderr) = cli_eval_json_with_stderr(&residual);
        assert_eq!(wire["result"], "0", "{residual}");
    }
}
#[test]
fn integrate_contract_affine_trig_seventh_power_reduction() {
    let cases = [
        (
            "integrate(sin(x)^7, x)",
            "sin(x)^7",
            "1/7 * (cos(x)^7 + 7 * cos(x)^3 - 7 * cos(x) - 21/5 * cos(x)^5)",
        ),
        (
            "integrate(cos(x)^7, x)",
            "cos(x)^7",
            "1/7 * (21/5 * sin(x)^5 + 7 * sin(x) - sin(x)^7 - 7 * sin(x)^3)",
        ),
        (
            "integrate(sin(2*x + 1)^7, x)",
            "sin(2*x + 1)^7",
            "(cos(2 * x + 1)^3 + 1/7 * cos(2 * x + 1)^7 - cos(2 * x + 1) - 3/5 * cos(2 * x + 1)^5) / 2",
        ),
        (
            "integrate(cos(2*x + 1)^7, x)",
            "cos(2*x + 1)^7",
            "(sin(2 * x + 1) + 3/5 * sin(2 * x + 1)^5 - sin(2 * x + 1)^3 - 1/7 * sin(2 * x + 1)^7) / 2",
        ),
        (
            "integrate(sin(1 - 2*x)^7, x)",
            "sin(1 - 2*x)^7",
            "-(cos(1 - 2 * x)^3 + 1/7 * cos(1 - 2 * x)^7 - cos(1 - 2 * x) - 3/5 * cos(1 - 2 * x)^5) / 2",
        ),
        (
            "integrate(cos(1 - 2*x)^7, x)",
            "cos(1 - 2*x)^7",
            "-(sin(1 - 2 * x) + 3/5 * sin(1 - 2 * x)^5 - sin(1 - 2 * x)^3 - 1/7 * sin(1 - 2 * x)^7) / 2",
        ),
    ];

    for (input, integrand, expected) in cases {
        let (antiderivative, required) = evaluated_integral_with_required_conditions(input);
        assert!(
            required.is_empty(),
            "trig seventh primitive should not add domain conditions for {input}: {required:?}"
        );
        assert_eq!(antiderivative, expected, "{input}");
        assert_antiderivative_verifies(input);
        assert_rendered_antiderivative_verifies(input, &antiderivative);

        let residual = format!("diff({input}, x) - {integrand}");
        let (wire, stderr) = cli_eval_json_with_stderr(&residual);
        assert!(
            stderr.is_empty(),
            "trig seventh residual should stay quiet for {input}: {stderr}"
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
fn integrate_contract_affine_trig_fourth_power_reduction() {
    let cases = [
        (
            "integrate(sin(x)^4, x)",
            "sin(x)^4",
            "1/32 * sin(4 * x) + 3/8 * x - 1/4 * sin(2 * x)",
        ),
        (
            "integrate(cos(x)^4, x)",
            "cos(x)^4",
            "1/32 * sin(4 * x) + 1/4 * sin(2 * x) + 3/8 * x",
        ),
        (
            "integrate(sin(2*x + 1)^4, x)",
            "sin(2*x + 1)^4",
            "1/64 * sin(4 * (2 * x + 1)) + 3/8 * x - 1/8 * sin(2 * (2 * x + 1))",
        ),
        (
            "integrate(cos(2*x + 1)^4, x)",
            "cos(2*x + 1)^4",
            "1/64 * sin(4 * (2 * x + 1)) + 1/8 * sin(2 * (2 * x + 1)) + 3/8 * x",
        ),
    ];

    for (input, integrand, expected) in cases {
        let (antiderivative, required) = evaluated_integral_with_required_conditions(input);
        assert!(
            required.is_empty(),
            "trig fourth primitive should not add domain conditions for {input}: {required:?}"
        );
        assert_eq!(antiderivative, expected, "{input}");
        assert_rendered_antiderivative_verifies(input, &antiderivative);

        let residual = format!("diff({input}, x) - {integrand}");
        let (wire, stderr) = cli_eval_json_with_stderr(&residual);
        assert_eq!(wire["result"], "0", "{residual}");
        assert_eq!(wire["required_display"], serde_json::json!([]));
        assert!(
            !stderr.contains("depth_overflow"),
            "trig fourth nested residual should not emit depth_overflow for {residual}\nstderr:\n{stderr}"
        );
    }
}
#[test]
fn integrate_contract_affine_trig_sixth_power_reduction() {
    let cases = [
        (
            "integrate(sin(x)^6, x)",
            "sin(x)^6",
            "3/64 * sin(4 * x) + 5/16 * x - 15/64 * sin(2 * x) - 1/192 * sin(6 * x)",
        ),
        (
            "integrate(cos(x)^6, x)",
            "cos(x)^6",
            "1/192 * sin(6 * x) + 3/64 * sin(4 * x) + 15/64 * sin(2 * x) + 5/16 * x",
        ),
        (
            "integrate(sin(2*x + 1)^6, x)",
            "sin(2*x + 1)^6",
            "3/128 * sin(4 * (2 * x + 1)) + 5/16 * x - 15/128 * sin(2 * (2 * x + 1)) - 1/384 * sin(6 * (2 * x + 1))",
        ),
        (
            "integrate(cos(2*x + 1)^6, x)",
            "cos(2*x + 1)^6",
            "1/384 * sin(6 * (2 * x + 1)) + 3/128 * sin(4 * (2 * x + 1)) + 15/128 * sin(2 * (2 * x + 1)) + 5/16 * x",
        ),
    ];

    for (input, integrand, expected) in cases {
        let (antiderivative, required) = evaluated_integral_with_required_conditions(input);
        assert!(
            required.is_empty(),
            "trig sixth primitive should not add domain conditions for {input}: {required:?}"
        );
        assert_eq!(antiderivative, expected, "{input}");
        assert_rendered_antiderivative_verifies(input, &antiderivative);

        let residual = format!("diff({input}, x) - {integrand}");
        let (wire, stderr) = cli_eval_json_with_stderr(&residual);
        assert_eq!(wire["result"], "0", "{residual}");
        assert_eq!(wire["required_display"], serde_json::json!([]));
        assert!(
            !stderr.contains("depth_overflow"),
            "trig sixth nested residual should not emit depth_overflow for {residual}\nstderr:\n{stderr}"
        );
    }
}
#[test]
fn integrate_contract_affine_trig_eighth_power_reduction() {
    let cases = [
        (
            "integrate(sin(x)^8, x)",
            "sin(x)^8",
            "1/1024 * sin(8 * x) + 7/128 * sin(4 * x) + 35/128 * x - 7/32 * sin(2 * x) - 1/96 * sin(6 * x)",
        ),
        (
            "integrate(cos(x)^8, x)",
            "cos(x)^8",
            "1/1024 * sin(8 * x) + 1/96 * sin(6 * x) + 7/128 * sin(4 * x) + 7/32 * sin(2 * x) + 35/128 * x",
        ),
        (
            "integrate(sin(2*x + 1)^8, x)",
            "sin(2*x + 1)^8",
            "1/2048 * sin(8 * (2 * x + 1)) + 7/256 * sin(4 * (2 * x + 1)) + 35/128 * x - 7/64 * sin(2 * (2 * x + 1)) - 1/192 * sin(6 * (2 * x + 1))",
        ),
        (
            "integrate(cos(2*x + 1)^8, x)",
            "cos(2*x + 1)^8",
            "1/2048 * sin(8 * (2 * x + 1)) + 1/192 * sin(6 * (2 * x + 1)) + 7/256 * sin(4 * (2 * x + 1)) + 7/64 * sin(2 * (2 * x + 1)) + 35/128 * x",
        ),
    ];

    for (input, integrand, expected) in cases {
        let (antiderivative, required) = evaluated_integral_with_required_conditions(input);
        assert!(
            required.is_empty(),
            "trig eighth primitive should not add domain conditions for {input}: {required:?}"
        );
        assert_eq!(antiderivative, expected, "{input}");
        assert_rendered_antiderivative_verifies(input, &antiderivative);

        let residual = format!("diff({input}, x) - {integrand}");
        let (wire, stderr) = cli_eval_json_with_stderr(&residual);
        assert_eq!(wire["result"], "0", "{residual}");
        assert_eq!(wire["required_display"], serde_json::json!([]));
        assert!(
            !stderr.contains("depth_overflow"),
            "trig eighth nested residual should not emit depth_overflow for {residual}\nstderr:\n{stderr}"
        );
    }
}
#[test]
fn integrate_contract_explicit_trig_fourth_power_antiderivative_residual_verifies() {
    for residual in [
        "diff(3*x/8 - sin(2*x)/4 + sin(4*x)/32, x) - sin(x)^4",
        "diff(3*x/8 + sin(2*x)/4 + sin(4*x)/32, x) - cos(x)^4",
        "diff(3*x/8 - sin(2*(2*x+1))/8 + sin(4*(2*x+1))/64, x) - sin(2*x+1)^4",
        "diff(3*x/8 + sin(2*(2*x+1))/8 + sin(4*(2*x+1))/64, x) - cos(2*x+1)^4",
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr(residual);
        assert_eq!(wire["result"], "0", "{residual}");
        assert_eq!(wire["required_display"], serde_json::json!([]));
        assert!(
            !stderr.contains("depth_overflow"),
            "fourth-power trig residual should not emit depth_overflow for {residual}\nstderr:\n{stderr}"
        );
    }
}
#[test]
fn integrate_contract_explicit_trig_sixth_power_antiderivative_residual_verifies() {
    for residual in [
        "diff(5*x/16 - 15*sin(2*x)/64 + 3*sin(4*x)/64 - sin(6*x)/192, x) - sin(x)^6",
        "diff(5*x/16 + 15*sin(2*x)/64 + 3*sin(4*x)/64 + sin(6*x)/192, x) - cos(x)^6",
        "diff(5*x/16 - 15*sin(2*(2*x+1))/128 + 3*sin(4*(2*x+1))/128 - sin(6*(2*x+1))/384, x) - sin(2*x+1)^6",
        "diff(5*x/16 + 15*sin(2*(2*x+1))/128 + 3*sin(4*(2*x+1))/128 + sin(6*(2*x+1))/384, x) - cos(2*x+1)^6",
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr(residual);
        assert_eq!(wire["result"], "0", "{residual}");
        assert_eq!(wire["required_display"], serde_json::json!([]));
        assert!(
            !stderr.contains("depth_overflow"),
            "sixth-power trig residual should not emit depth_overflow for {residual}\nstderr:\n{stderr}"
        );
    }
}
#[test]
fn integrate_contract_explicit_trig_eighth_power_antiderivative_residual_verifies() {
    for residual in [
        "diff(35*x/128 - 7*sin(2*x)/32 + 7*sin(4*x)/128 - sin(6*x)/96 + sin(8*x)/1024, x) - sin(x)^8",
        "diff(35*x/128 + 7*sin(2*x)/32 + 7*sin(4*x)/128 + sin(6*x)/96 + sin(8*x)/1024, x) - cos(x)^8",
        "diff(35*x/128 - 7*sin(2*(2*x+1))/64 + 7*sin(4*(2*x+1))/256 - sin(6*(2*x+1))/192 + sin(8*(2*x+1))/2048, x) - sin(2*x+1)^8",
        "diff(35*x/128 + 7*sin(2*(2*x+1))/64 + 7*sin(4*(2*x+1))/256 + sin(6*(2*x+1))/192 + sin(8*(2*x+1))/2048, x) - cos(2*x+1)^8",
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr(residual);
        assert_eq!(wire["result"], "0", "{residual}");
        assert_eq!(wire["required_display"], serde_json::json!([]));
        assert!(
            !stderr.contains("depth_overflow"),
            "eighth-power trig residual should not emit depth_overflow for {residual}\nstderr:\n{stderr}"
        );
    }
}
#[test]
fn integrate_contract_explicit_negated_sine_uses_linearity() {
    assert_eq!(simplified_integral("integrate(-(sin(x)), x)"), "cos(x)");
}
#[test]
fn integrate_contract_direct_cos_table() {
    assert_eq!(simplified_integral("integrate(cos(x), x)"), "sin(x)");
}
#[test]
fn integrate_contract_constant_base_affine_logs_use_table_and_preserve_domain() {
    for (input, expected_result, expected_required) in [
        (
            "integrate(log(2, x), x)",
            "x * (log(2, x) - 1 / ln(2))",
            vec!["x > 0".to_string()],
        ),
        (
            "integrate(log(2, 3*x+2), x)",
            "1/3 * (3 * x + 2) * (log(2, 3 * x + 2) - 1 / ln(2))",
            vec!["x > -2/3".to_string()],
        ),
        (
            "integrate(log(1/2, x), x)",
            "x * (log(1/2, x) - 1 / ln(1/2))",
            vec!["x > 0".to_string()],
        ),
        (
            "integrate(log(e, x), x)",
            "x * (ln(x) - 1)",
            vec!["x > 0".to_string()],
        ),
        (
            "integrate(log2(x), x)",
            "x * (log2(x) - 1 / ln(2))",
            vec!["x > 0".to_string()],
        ),
        (
            "integrate(log10(3*x+2), x)",
            "1/3 * (3 * x + 2) * (log10(3 * x + 2) - 1 / ln(10))",
            vec!["x > -2/3".to_string()],
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
}
#[test]
fn integrate_contract_constant_base_log_handles_invalid_and_symbolic_bases() {
    for input in [
        "integrate(log(1, x), x)",
        "integrate(log(-2, x), x)",
        "integrate(log(0, x), x)",
    ] {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(
            result, "undefined",
            "invalid log base should make the real-domain integrand undefined for {input}"
        );
        assert!(
            required.is_empty(),
            "invalid log base should not add conditional assumptions for {input}: {required:?}"
        );
    }

    let input = "integrate(log(y, x), x)";
    let (result, _required) = evaluated_integral_with_required_conditions(input);
    assert!(
        result.starts_with("integrate("),
        "symbolic log base should remain residual for {input}, got {result}"
    );
}
#[test]
fn integrate_contract_constant_base_affine_log_trace_stays_compact() {
    let input = "integrate(log(2, 3*x+2), x)";
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

    assert!(
        stderr.is_empty(),
        "constant-base affine log integration should not emit stderr warnings: {stderr}"
    );
    assert_eq!(
        wire["result"],
        "1/3·(3·x + 2)·(log(2, 3·x + 2) - 1 / ln(2))"
    );
    assert_eq!(wire["required_display"], serde_json::json!(["x > -2/3"]));
    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    assert_eq!(steps.len(), 1, "expected one integration step: {steps:?}");
    assert_eq!(steps[0]["rule"], "Calcular la integral");
    assert!(
        steps[0]["substeps"].is_null(),
        "constant-base affine log table should not invent didactic substeps: {steps:?}"
    );
}
#[test]
fn integrate_contract_square_minus_constant_uses_abs_log_ratio_and_nonzero_domain() {
    let (result, required) = evaluated_integral_with_required_conditions("integrate(1/(x^2-1), x)");

    assert_eq!(result, "1/2 * ln(|(x - 1) / (x + 1)|)");
    assert_eq!(
        required,
        vec!["x ≠ 1".to_string(), "x ≠ -1".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_secant_squared_kernel_uses_tangent_and_nonzero_domain() {
    let (result, required) = evaluated_integral_with_required_conditions("integrate(sec(x)^2, x)");

    assert_eq!(result, "tan(x)");
    assert_eq!(
        required,
        vec!["cos(x) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_cosecant_squared_kernel_uses_cotangent_and_nonzero_domain() {
    let (result, required) = evaluated_integral_with_required_conditions("integrate(csc(x)^2, x)");

    assert_eq!(result, "-cot(x)");
    assert_eq!(
        required,
        vec!["sin(x) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_raw_reciprocal_trig_derivative_quotients_render_compactly() {
    let cases = [
        (
            "integrate(x*sin(x^2)/cos(x^2)^2, x)",
            "sec(x^2) / 2",
            "cos(x^2) ≠ 0",
            "diff(integrate(x*sin(x^2)/cos(x^2)^2, x), x) - x*sin(x^2)/cos(x^2)^2",
        ),
        (
            "integrate(x^2*cos(x^3)/sin(x^3)^2, x)",
            "-csc(x^3) / 3",
            "sin(x^3) ≠ 0",
            "diff(integrate(x^2*cos(x^3)/sin(x^3)^2, x), x) - x^2*cos(x^3)/sin(x^3)^2",
        ),
        (
            "integrate((2*x+1)*sin(x^2+x)/cos(x^2+x)^2, x)",
            "sec(x^2 + x)",
            "cos(x^2 + x) ≠ 0",
            "diff(integrate((2*x+1)*sin(x^2+x)/cos(x^2+x)^2, x), x) - (2*x+1)*sin(x^2+x)/cos(x^2+x)^2",
        ),
        (
            "integrate((2*x+1)*cos(x^2+x)/sin(x^2+x)^2, x)",
            "-csc(x^2 + x)",
            "sin(x^2 + x) ≠ 0",
            "diff(integrate((2*x+1)*cos(x^2+x)/sin(x^2+x)^2, x), x) - (2*x+1)*cos(x^2+x)/sin(x^2+x)^2",
        ),
        (
            "integrate((3*sin(x^2+x)+6*x*sin(x^2+x))/cos(x^2+x)^2, x)",
            "3 * sec(x^2 + x)",
            "cos(x^2 + x) ≠ 0",
            "diff(integrate(3*(2*x+1)*sin(x^2+x)/cos(x^2+x)^2, x), x) - 3*(2*x+1)*sin(x^2+x)/cos(x^2+x)^2",
        ),
        (
            "integrate((3*cos(x^2+x)+6*x*cos(x^2+x))/sin(x^2+x)^2, x)",
            "-3 * csc(x^2 + x)",
            "sin(x^2 + x) ≠ 0",
            "diff(integrate(3*(2*x+1)*cos(x^2+x)/sin(x^2+x)^2, x), x) - 3*(2*x+1)*cos(x^2+x)/sin(x^2+x)^2",
        ),
    ];

    for (input, expected_result, expected_condition, residual_input) in cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);

        assert_eq!(result, expected_result, "unexpected result for {input}");
        assert_eq!(
            required,
            vec![expected_condition.to_string()],
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert_rendered_antiderivative_verifies(input, &result);

        let (residual_result, residual_required) =
            evaluated_expr_with_required_conditions(residual_input);
        assert_eq!(
            residual_result, "0",
            "unexpected antiderivative residual for {input}"
        );
        assert_eq!(
            residual_required,
            vec![expected_condition.to_string()],
            "residual should preserve required domain for {input}: {residual_required:?}"
        );
    }
}
#[test]
fn integrate_contract_expanded_polynomial_tangent_cotangent_preserves_domain() {
    let input = "integrate((4*x^3-2*x)*tan(x^4-x^2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "-ln(|cos(x^4 - x^2)|)");
    assert_eq!(
        required,
        vec!["cos(x^4 - x^2) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_eq!(integrate_call_antiderivative_residual_result(input), "0");

    let input = "integrate((4*x^3-2*x)*cot(x^4-x^2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "ln(|sin(x^4 - x^2)|)");
    assert_eq!(
        required,
        vec!["sin(x^4 - x^2) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_eq!(integrate_call_antiderivative_residual_result(input), "0");
}
#[test]
fn integrate_contract_linear_secant_uses_abs_log_and_nonzero_domain() {
    let (result, _required) =
        evaluated_integral_with_required_conditions("integrate(sec(2*x + 1), x)");

    assert_eq!(result, "1/2 * ln(|tan(2 * x + 1) + sec(2 * x + 1)|)");
    let (wire, stderr) = cli_eval_json_with_stderr("integrate(sec(2*x + 1), x)");
    assert!(stderr.is_empty(), "unexpected stderr: {stderr}");
    assert_eq!(
        wire["required_display"],
        serde_json::json!(["cos(2·x + 1) ≠ 0"]),
        "unexpected public required_display: {:?}",
        wire["required_display"]
    );
    assert_antiderivative_verifies("integrate(sec(2*x + 1), x)");
    let (nested_wire, nested_stderr) =
        cli_eval_json_with_stderr("diff(integrate(sec(2*x+1), x), x) - sec(2*x+1)");
    assert!(
        nested_stderr.is_empty(),
        "unexpected stderr: {nested_stderr}"
    );
    assert_eq!(nested_wire["result"], "0");
    assert_eq!(
        nested_wire["required_display"],
        serde_json::json!(["cos(2·x + 1) ≠ 0"]),
        "secant log primitive verification should preserve the trig pole condition"
    );
}
#[test]
fn integrate_contract_linear_cosecant_uses_abs_log_and_nonzero_domain() {
    let (result, _required) =
        evaluated_integral_with_required_conditions("integrate(csc(2*x + 1), x)");

    assert_eq!(result, "1/2 * ln(|csc(2 * x + 1) - cot(2 * x + 1)|)");
    let (wire, stderr) = cli_eval_json_with_stderr("integrate(csc(2*x + 1), x)");
    assert!(stderr.is_empty(), "unexpected stderr: {stderr}");
    assert_eq!(
        wire["required_display"],
        serde_json::json!(["sin(2·x + 1) ≠ 0"]),
        "unexpected public required_display: {:?}",
        wire["required_display"]
    );
    let (nested_wire, nested_stderr) =
        cli_eval_json_with_stderr("diff(integrate(csc(2*x+1), x), x) - csc(2*x+1)");
    assert!(
        nested_stderr.is_empty(),
        "unexpected stderr: {nested_stderr}"
    );
    assert_eq!(nested_wire["result"], "0");
    assert_eq!(
        nested_wire["required_display"],
        serde_json::json!(["sin(2·x + 1) ≠ 0"]),
        "cosecant log primitive verification should preserve the trig pole condition"
    );
}
#[test]
fn integrate_contract_scaled_affine_secant_cosecant_uses_abs_log_and_nonzero_domain() {
    let cases = [
        (
            "integrate(sec((3*x+2)/2), x)",
            "2/3 * ln(|tan((3 * x + 2) / 2) + sec((3 * x + 2) / 2)|)",
            "cos((3 * x + 2) / 2) ≠ 0",
        ),
        (
            "integrate(csc((2-3*x)/2), x)",
            "-2/3 * ln(|csc((2 - 3 * x) / 2) - cot((2 - 3 * x) / 2)|)",
            "sin((2 - 3 * x) / 2) ≠ 0",
        ),
    ];

    for (input, expected_result, expected_condition) in cases {
        let (result, _required) = evaluated_integral_with_required_conditions(input);

        assert_eq!(result, expected_result, "unexpected result for {input}");
        let (wire, stderr) = cli_eval_json_with_stderr(input);
        assert!(stderr.is_empty(), "unexpected stderr for {input}: {stderr}");
        assert_eq!(
            wire["required_display"],
            serde_json::json!([expected_condition.replace(" * ", "·")]),
            "unexpected public required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert_rendered_antiderivative_verifies(input, &result);
    }
}
#[test]
fn integrate_contract_nested_reciprocal_trig_residual_verifies_antiderivative() {
    let cases = [
        (
            "diff(integrate(sec((3*x+2)/2), x), x) - sec((3*x+2)/2)",
            "cos((3 * x + 2) / 2) ≠ 0",
        ),
        (
            "diff(integrate(csc((2-3*x)/2), x), x) - csc((2-3*x)/2)",
            "sin((2 - 3 * x) / 2) ≠ 0",
        ),
        (
            "diff(integrate(1/sin((2-3*x)/2), x), x) - 1/sin((2-3*x)/2)",
            "sin((2 - 3 * x) / 2) ≠ 0",
        ),
        (
            "diff(integrate(2*x*sec(x^2)*tan(x^2), x), x) - 2*x*sec(x^2)*tan(x^2)",
            "cos(x^2) ≠ 0",
        ),
        (
            "diff(integrate(2*x*csc(x^2)*cot(x^2), x), x) - 2*x*csc(x^2)*cot(x^2)",
            "sin(x^2) ≠ 0",
        ),
        (
            "diff(integrate((4*x^3-2*x)*sec(x^4-x^2)*tan(x^4-x^2), x), x) - (4*x^3-2*x)*sec(x^4-x^2)*tan(x^4-x^2)",
            "cos(x^4 - x^2) ≠ 0",
        ),
        (
            "diff(integrate((4*x^3-2*x)*csc(x^4-x^2)*cot(x^4-x^2), x), x) - (4*x^3-2*x)*csc(x^4-x^2)*cot(x^4-x^2)",
            "sin(x^4 - x^2) ≠ 0",
        ),
    ];

    for (input, expected_condition) in cases {
        let (wire, stderr) = cli_eval_json_with_stderr(input);
        assert!(stderr.is_empty(), "unexpected stderr for {input}: {stderr}");
        assert_eq!(
            wire["result"], "0",
            "unexpected nested residual for {input}"
        );
        assert_eq!(
            wire["required_display"],
            serde_json::json!([expected_condition.replace(" * ", "·")]),
            "unexpected required_display for nested residual {input}: {:?}",
            wire["required_display"]
        );
    }
}
#[test]
fn integrate_contract_wrapped_nested_reciprocal_trig_residual_verifies_antiderivative() {
    let cases = [
        (
            "(diff(integrate((4*x^3-2*x)*sec(x^4-x^2)*tan(x^4-x^2), x), x) - (4*x^3-2*x)*sec(x^4-x^2)*tan(x^4-x^2)) + 0",
            vec!["cos(x^4 - x^2) ≠ 0"],
        ),
        (
            "2*(diff(integrate((4*x^3-2*x)*csc(x^4-x^2)*cot(x^4-x^2), x), x) - (4*x^3-2*x)*csc(x^4-x^2)*cot(x^4-x^2))",
            vec!["sin(x^4 - x^2) ≠ 0"],
        ),
        (
            "(diff(integrate((4*x^3-2*x)*sec(x^4-x^2)*tan(x^4-x^2), x), x) - (4*x^3-2*x)*sec(x^4-x^2)*tan(x^4-x^2))/(x+1)",
            vec!["cos(x^4 - x^2) ≠ 0", "x ≠ -1"],
        ),
    ];

    for (input, expected_conditions) in cases {
        let (result, required) = evaluated_expr_with_required_conditions(input);
        assert_eq!(result, "0", "unexpected wrapped residual for {input}");
        assert_eq!(
            required, expected_conditions,
            "wrapped residual should preserve required domain for {input}: {required:?}"
        );
    }
}
#[test]
fn integrate_contract_linear_tangent_uses_abs_log_and_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(tan(2*x + 1), x)");

    assert_eq!(result, "-1/2 * ln(|cos(2 * x + 1)|)");
    assert_eq!(
        required,
        vec!["cos(2 * x + 1) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_linear_cotangent_uses_abs_log_and_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(cot(2*x + 1), x)");

    assert_eq!(result, "1/2 * ln(|sin(2 * x + 1)|)");
    assert_eq!(
        required,
        vec!["sin(2 * x + 1) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_nested_tangent_cotangent_residual_verifies_antiderivative() {
    let cases = [
        (
            "diff(integrate(tan(2*x+1), x), x) - tan(2*x+1)",
            "cos(2 * x + 1) ≠ 0",
        ),
        (
            "diff(integrate(cot(2*x+1), x), x) - cot(2*x+1)",
            "sin(2 * x + 1) ≠ 0",
        ),
        (
            "diff(integrate(cos(2*x+1)/sin(2*x+1), x), x) - cos(2*x+1)/sin(2*x+1)",
            "sin(2 * x + 1) ≠ 0",
        ),
        (
            "diff(integrate(2*x*tan(x^2), x), x) - 2*x*tan(x^2)",
            "cos(x^2) ≠ 0",
        ),
        (
            "diff(integrate(3*x^2*cot(x^3), x), x) - 3*x^2*cot(x^3)",
            "sin(x^3) ≠ 0",
        ),
        (
            "diff(integrate((4*x^3-2*x)*tan(x^4-x^2), x), x) - (4*x^3-2*x)*tan(x^4-x^2)",
            "cos(x^4 - x^2) ≠ 0",
        ),
        (
            "diff(integrate((4*x^3-2*x)*cot(x^4-x^2), x), x) - (4*x^3-2*x)*cot(x^4-x^2)",
            "sin(x^4 - x^2) ≠ 0",
        ),
        (
            "diff(integrate(2*x*sec(x^2), x), x) - 2*x*sec(x^2)",
            "cos(x^2) ≠ 0",
        ),
        (
            "diff(integrate(2*x*csc(x^2), x), x) - 2*x*csc(x^2)",
            "sin(x^2) ≠ 0",
        ),
    ];

    for (input, expected_condition) in cases {
        let (wire, stderr) = cli_eval_json_with_stderr(input);
        assert!(stderr.is_empty(), "unexpected stderr for {input}: {stderr}");
        assert_eq!(
            wire["result"], "0",
            "unexpected nested residual for {input}"
        );
        assert_eq!(
            wire["required_display"],
            serde_json::json!([expected_condition.replace(" * ", "·")]),
            "unexpected required_display for nested residual {input}: {:?}",
            wire["required_display"]
        );
    }
}
#[test]
fn integrate_contract_wrapped_nested_tangent_cotangent_residual_verifies_antiderivative() {
    let cases = [
        (
            "(diff(integrate((4*x^3-2*x)*tan(x^4-x^2), x), x) - (4*x^3-2*x)*tan(x^4-x^2)) + 0",
            vec!["cos(x^4 - x^2) ≠ 0"],
        ),
        (
            "2*(diff(integrate((4*x^3-2*x)*cot(x^4-x^2), x), x) - (4*x^3-2*x)*cot(x^4-x^2))",
            vec!["sin(x^4 - x^2) ≠ 0"],
        ),
        (
            "(diff(integrate((4*x^3-2*x)*cot(x^4-x^2), x), x) - (4*x^3-2*x)*cot(x^4-x^2))/(x+1)",
            vec!["sin(x^4 - x^2) ≠ 0", "x ≠ -1"],
        ),
    ];

    for (input, expected_conditions) in cases {
        let (result, required) = evaluated_expr_with_required_conditions(input);
        assert_eq!(result, "0", "unexpected wrapped residual for {input}");
        assert_eq!(
            required, expected_conditions,
            "wrapped residual should preserve required domain for {input}: {required:?}"
        );
    }
}
#[test]
fn integrate_contract_polynomial_tangent_uses_abs_log_and_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x*tan(x^2), x)");

    assert_eq!(result, "-ln(|cos(x^2)|)");
    assert_eq!(
        required,
        vec!["cos(x^2) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_polynomial_cotangent_uses_abs_log_and_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(3*x^2*cot(x^3), x)");

    assert_eq!(result, "ln(|sin(x^3)|)");
    assert_eq!(
        required,
        vec!["sin(x^3) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_polynomial_secant_uses_abs_log_and_nonzero_domain() {
    let (result, _required) =
        evaluated_integral_with_required_conditions("integrate(2*x*sec(x^2), x)");

    assert_eq!(result, "ln(|tan(x^2) + sec(x^2)|)");
    let (wire, stderr) = cli_eval_json_with_stderr("integrate(2*x*sec(x^2), x)");
    assert!(stderr.is_empty(), "unexpected stderr: {stderr}");
    assert_eq!(
        wire["required_display"],
        serde_json::json!(["cos(x^2) ≠ 0"]),
        "unexpected public required_display: {:?}",
        wire["required_display"]
    );
}
#[test]
fn integrate_contract_polynomial_cosecant_uses_abs_log_and_nonzero_domain() {
    let (result, _required) =
        evaluated_integral_with_required_conditions("integrate(2*x*csc(x^2), x)");

    assert_eq!(result, "ln(|csc(x^2) - cot(x^2)|)");
    let (wire, stderr) = cli_eval_json_with_stderr("integrate(2*x*csc(x^2), x)");
    assert!(stderr.is_empty(), "unexpected stderr: {stderr}");
    assert_eq!(
        wire["required_display"],
        serde_json::json!(["sin(x^2) ≠ 0"]),
        "unexpected public required_display: {:?}",
        wire["required_display"]
    );
}
#[test]
fn integrate_contract_presimplified_reciprocal_secant_and_cosecant_use_source_domain() {
    let (sec_wire, sec_stderr) = cli_eval_json_with_stderr("integrate(2*x/sec(x^2+0), x)");
    assert!(sec_stderr.is_empty(), "unexpected stderr: {sec_stderr}");
    assert_eq!(sec_wire["result"], serde_json::json!("sin(x^2)"));
    assert_eq!(
        sec_wire["required_display"],
        serde_json::json!(["cos(x^2) ≠ 0"]),
        "unexpected secant reciprocal required_display: {:?}",
        sec_wire["required_display"]
    );

    let (csc_wire, csc_stderr) = cli_eval_json_with_stderr("integrate(2*x/csc(x^2+0), x)");
    assert!(csc_stderr.is_empty(), "unexpected stderr: {csc_stderr}");
    assert_eq!(csc_wire["result"], serde_json::json!("-cos(x^2)"));
    assert_eq!(
        csc_wire["required_display"],
        serde_json::json!(["sin(x^2) ≠ 0"]),
        "unexpected cosecant reciprocal required_display: {:?}",
        csc_wire["required_display"]
    );
}
#[test]
fn integrate_contract_presimplified_reciprocal_tangent_and_cotangent_use_source_domain() {
    let (tan_wire, tan_stderr) = cli_eval_json_with_stderr("integrate(2*x/tan(x^2+0), x)");
    assert!(tan_stderr.is_empty(), "unexpected stderr: {tan_stderr}");
    assert_eq!(tan_wire["result"], serde_json::json!("ln(|sin(x^2)|)"));
    assert_eq!(
        tan_wire["required_display"],
        serde_json::json!(["sin(x^2) ≠ 0", "cos(x^2) ≠ 0"]),
        "unexpected tangent reciprocal required_display: {:?}",
        tan_wire["required_display"]
    );

    let (cot_wire, cot_stderr) = cli_eval_json_with_stderr("integrate(2*x/cot(x^2+0), x)");
    assert!(cot_stderr.is_empty(), "unexpected stderr: {cot_stderr}");
    assert_eq!(cot_wire["result"], serde_json::json!("-ln(|cos(x^2)|)"));
    assert_eq!(
        cot_wire["required_display"],
        serde_json::json!(["cos(x^2) ≠ 0", "sin(x^2) ≠ 0"]),
        "unexpected cotangent reciprocal required_display: {:?}",
        cot_wire["required_display"]
    );
}
#[test]
fn integrate_contract_reciprocal_trig_derivative_product_explains_u_and_du() {
    for (
        input,
        expected_result,
        expected_required_display,
        expected_rule_title,
        expects_constant_adjustment,
    ) in [
        (
            "integrate(x*sec(x^2)*tan(x^2), x)",
            "sec(x^2) / 2",
            serde_json::json!(["cos(x^2) ≠ 0"]),
            "Usar la regla de sec(u)·tan(u) -> sec(u)",
            true,
        ),
        (
            "integrate(2*x*sec(x^2)*tan(x^2), x)",
            "sec(x^2)",
            serde_json::json!(["cos(x^2) ≠ 0"]),
            "Usar la regla de sec(u)·tan(u) -> sec(u)",
            false,
        ),
        (
            "integrate(x*csc(x^2)*cot(x^2), x)",
            "-csc(x^2) / 2",
            serde_json::json!(["sin(x^2) ≠ 0"]),
            "Usar la regla de csc(u)·cot(u) -> -csc(u)",
            true,
        ),
        (
            "integrate(3*x^2*csc(x^3)*cot(x^3), x)",
            "-csc(x^3)",
            serde_json::json!(["sin(x^3) ≠ 0"]),
            "Usar la regla de csc(u)·cot(u) -> -csc(u)",
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
            "reciprocal trig derivative product trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
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
            "reciprocal trig derivative product should not use the generic substitution substep for {input}: {substeps:?}"
        );
    }
}
#[test]
fn integrate_contract_polynomial_trig_log_explicit_ratios_preserve_source_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*(x*sin(x^2)/cos(x^2)), x)");

    assert_eq!(result, "-ln(|cos(x^2)|)");
    assert_eq!(
        required,
        vec!["cos(x^2) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(3*(x^2*cos(x^3)/sin(x^3)), x)");

    assert_eq!(result, "ln(|sin(x^3)|)");
    assert_eq!(
        required,
        vec!["sin(x^3) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_linear_secant_tangent_product_preserves_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(sec(2*x + 1)*tan(2*x + 1), x)");

    assert_eq!(result, "sec(2 * x + 1) / 2");
    assert_eq!(
        required,
        vec!["cos(2 * x + 1) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_linear_cosecant_cotangent_product_preserves_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(csc(2*x + 1)*cot(2*x + 1), x)");

    assert_eq!(result, "-csc(2 * x + 1) / 2");
    assert_eq!(
        required,
        vec!["sin(2 * x + 1) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_polynomial_secant_tangent_product_preserves_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(x*sec(x^2)*tan(x^2), x)");

    assert_eq!(result, "sec(x^2) / 2");
    assert_eq!(
        required,
        vec!["cos(x^2) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_polynomial_cosecant_cotangent_product_preserves_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(x^2*csc(x^3)*cot(x^3), x)");

    assert_eq!(result, "-csc(x^3) / 3");
    assert_eq!(
        required,
        vec!["sin(x^3) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_exact_polynomial_secant_tangent_product_uses_clean_antiderivative() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x*sec(x^2)*tan(x^2), x)");

    assert_eq!(result, "sec(x^2)");
    assert_eq!(
        required,
        vec!["cos(x^2) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_exact_polynomial_cosecant_cotangent_product_uses_clean_antiderivative() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(3*x^2*csc(x^3)*cot(x^3), x)");

    assert_eq!(result, "-csc(x^3)");
    assert_eq!(
        required,
        vec!["sin(x^3) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_exact_polynomial_secant_squared_preserves_nonzero_domain() {
    let input = "integrate(2*x*sec(x^2)^2, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "tan(x^2)");
    assert_eq!(
        required,
        vec!["cos(x^2) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}
#[test]
fn integrate_contract_exact_polynomial_cosecant_squared_preserves_nonzero_domain() {
    let input = "integrate(2*x*csc(x^2)^2, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "-cot(x^2)");
    assert_eq!(
        required,
        vec!["sin(x^2) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}
#[test]
fn integrate_contract_sqrt_chain_secant_cosecant_products_verify() {
    let cases = [
        (
            "integrate(sec(sqrt(x))*tan(sqrt(x))/(2*sqrt(x)), x)",
            "sec(sqrt(x))",
            "tan(sqrt(x)) * sec(sqrt(x)) / (2 * sqrt(x))",
            vec!["cos(sqrt(x)) ≠ 0", "x > 0"],
        ),
        (
            "integrate(-sec(sqrt(x))*tan(sqrt(x))/(2*sqrt(x)), x)",
            "-sec(sqrt(x))",
            "-tan(sqrt(x)) * sec(sqrt(x)) / (2 * sqrt(x))",
            vec!["cos(sqrt(x)) ≠ 0", "x > 0"],
        ),
        (
            "integrate(csc(sqrt(x))*cot(sqrt(x))/(2*sqrt(x)), x)",
            "-csc(sqrt(x))",
            "csc(sqrt(x)) * cot(sqrt(x)) / (2 * sqrt(x))",
            vec!["sin(sqrt(x)) ≠ 0", "x > 0"],
        ),
        (
            "integrate(-csc(sqrt(x))*cot(sqrt(x))/(2*sqrt(x)), x)",
            "csc(sqrt(x))",
            "-cot(sqrt(x)) * csc(sqrt(x)) / (2 * sqrt(x))",
            vec!["sin(sqrt(x)) ≠ 0", "x > 0"],
        ),
        (
            "integrate(sec(sqrt(2*x))*tan(sqrt(2*x))/sqrt(2*x), x)",
            "sec(sqrt(2 * x))",
            "tan(sqrt(2 * x)) * sec(sqrt(2 * x)) / sqrt(2 * x)",
            vec!["cos(sqrt(2 * x)) ≠ 0", "x > 0"],
        ),
        (
            "integrate(csc(sqrt(2*x))*cot(sqrt(2*x))/sqrt(2*x), x)",
            "-csc(sqrt(2 * x))",
            "csc(sqrt(2 * x)) * cot(sqrt(2 * x)) / sqrt(2 * x)",
            vec!["sin(sqrt(2 * x)) ≠ 0", "x > 0"],
        ),
        (
            "integrate(-sec(sqrt(2*x))*tan(sqrt(2*x))/sqrt(2*x), x)",
            "-sec(sqrt(2 * x))",
            "-tan(sqrt(2 * x)) * sec(sqrt(2 * x)) / sqrt(2 * x)",
            vec!["cos(sqrt(2 * x)) ≠ 0", "x > 0"],
        ),
        (
            "integrate(-csc(sqrt(2*x))*cot(sqrt(2*x))/sqrt(2*x), x)",
            "csc(sqrt(2 * x))",
            "-cot(sqrt(2 * x)) * csc(sqrt(2 * x)) / sqrt(2 * x)",
            vec!["sin(sqrt(2 * x)) ≠ 0", "x > 0"],
        ),
        (
            "integrate(sec(sqrt(3*x+1))*tan(sqrt(3*x+1))*3/(2*sqrt(3*x+1)), x)",
            "sec(sqrt(3 * x + 1))",
            "3 * tan(sqrt(3 * x + 1)) * sec(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))",
            vec!["cos(sqrt(3 * x + 1)) ≠ 0", "x > -1/3"],
        ),
        (
            "integrate(-sec(sqrt(3-2*x))*tan(sqrt(3-2*x))/sqrt(3-2*x), x)",
            "sec(sqrt(3 - 2 * x))",
            "-tan(sqrt(3 - 2 * x)) * sec(sqrt(3 - 2 * x)) / sqrt(3 - 2 * x)",
            vec!["cos(sqrt(3 - 2 * x)) ≠ 0", "x < 3/2"],
        ),
        (
            "integrate(sec(sqrt(3-2*x))*tan(sqrt(3-2*x))/sqrt(3-2*x), x)",
            "-sec(sqrt(3 - 2 * x))",
            "tan(sqrt(3 - 2 * x)) * sec(sqrt(3 - 2 * x)) / sqrt(3 - 2 * x)",
            vec!["cos(sqrt(3 - 2 * x)) ≠ 0", "x < 3/2"],
        ),
        (
            "integrate(csc(sqrt(3*x+1))*cot(sqrt(3*x+1))*3/(2*sqrt(3*x+1)), x)",
            "-csc(sqrt(3 * x + 1))",
            "3 * csc(sqrt(3 * x + 1)) * cot(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))",
            vec!["sin(sqrt(3 * x + 1)) ≠ 0", "x > -1/3"],
        ),
        (
            "integrate(-csc(sqrt(3-2*x))*cot(sqrt(3-2*x))/sqrt(3-2*x), x)",
            "-csc(sqrt(3 - 2 * x))",
            "-cot(sqrt(3 - 2 * x)) * csc(sqrt(3 - 2 * x)) / sqrt(3 - 2 * x)",
            vec!["sin(sqrt(3 - 2 * x)) ≠ 0", "x < 3/2"],
        ),
        (
            "integrate(csc(sqrt(3-2*x))*cot(sqrt(3-2*x))/sqrt(3-2*x), x)",
            "csc(sqrt(3 - 2 * x))",
            "csc(sqrt(3 - 2 * x)) * cot(sqrt(3 - 2 * x)) / sqrt(3 - 2 * x)",
            vec!["sin(sqrt(3 - 2 * x)) ≠ 0", "x < 3/2"],
        ),
        (
            "integrate(-csc(sqrt(3*x+1))*cot(sqrt(3*x+1))*3/(2*sqrt(3*x+1)), x)",
            "csc(sqrt(3 * x + 1))",
            "-3 * cot(sqrt(3 * x + 1)) * csc(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))",
            vec!["x > -1/3", "sin(sqrt(3 * x + 1)) ≠ 0"],
        ),
    ];

    for (input, expected_result, expected_nested_result, expected_conditions) in cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);

        assert_eq!(result, expected_result, "unexpected result for {input}");
        assert_eq!(
            required, expected_conditions,
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert_antiderivative_verifies(input);

        let nested_input = format!("diff({input}, x)");
        let (nested_result, nested_required) =
            evaluated_expr_with_required_conditions(&nested_input);
        assert_eq!(
            nested_result, expected_nested_result,
            "unexpected nested diff/integrate result for {input}"
        );
        assert_eq!(
            nested_required, expected_conditions,
            "unexpected nested required_conditions for {input}: {nested_required:?}"
        );
    }
}
#[test]
fn integrate_contract_sqrt_chain_raw_reciprocal_trig_derivative_quotients_render_compactly() {
    let cases = [
        (
            "integrate(sin(sqrt(x))*sqrt(x)/(2*x*cos(sqrt(x))^2), x)",
            "sec(sqrt(x))",
            vec!["cos(sqrt(x)) ≠ 0", "x > 0"],
            "diff(integrate(sin(sqrt(x))*sqrt(x)/(2*x*cos(sqrt(x))^2), x), x) - sin(sqrt(x))*sqrt(x)/(2*x*cos(sqrt(x))^2)",
        ),
        (
            "integrate(cos(sqrt(2*x))*(2*x)^(-1/2)/sin(sqrt(2*x))^2, x)",
            "-csc(sqrt(2 * x))",
            vec!["sin(sqrt(2 * x)) ≠ 0", "x > 0"],
            "diff(integrate(cos(sqrt(2*x))*(2*x)^(-1/2)/sin(sqrt(2*x))^2, x), x) - cos(sqrt(2*x))*(2*x)^(-1/2)/sin(sqrt(2*x))^2",
        ),
        (
            "integrate(3*sin(sqrt(3*x+1))/(2*sqrt(3*x+1)*cos(sqrt(3*x+1))^2), x)",
            "sec(sqrt(3 * x + 1))",
            vec!["cos(sqrt(3 * x + 1)) ≠ 0", "x > -1/3"],
            "diff(integrate(3*sin(sqrt(3*x+1))/(2*sqrt(3*x+1)*cos(sqrt(3*x+1))^2), x), x) - 3*sin(sqrt(3*x+1))/(2*sqrt(3*x+1)*cos(sqrt(3*x+1))^2)",
        ),
        (
            "integrate(-2*cos(sqrt(3-2*x))/(sqrt(3-2*x)*sin(sqrt(3-2*x))^2), x)",
            "-2 * csc(sqrt(3 - 2 * x))",
            vec!["sin(sqrt(3 - 2 * x)) ≠ 0", "x < 3/2"],
            "diff(integrate(-2*cos(sqrt(3-2*x))/(sqrt(3-2*x)*sin(sqrt(3-2*x))^2), x), x) + 2*cos(sqrt(3-2*x))/(sqrt(3-2*x)*sin(sqrt(3-2*x))^2)",
        ),
    ];

    for (input, expected_result, expected_conditions, residual_input) in cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);

        assert_eq!(result, expected_result, "unexpected result for {input}");
        assert_eq!(
            required, expected_conditions,
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert_rendered_antiderivative_verifies(input, &result);

        let (residual_result, residual_required) =
            evaluated_expr_with_required_conditions(residual_input);
        assert_eq!(
            residual_result, "0",
            "unexpected antiderivative residual for {input}"
        );
        assert_eq!(
            residual_required, expected_conditions,
            "residual should preserve required domain for {input}: {residual_required:?}"
        );
    }
}
#[test]
fn integrate_contract_sqrt_chain_reciprocal_trig_products_explain_u_and_du() {
    for (
        input,
        expected_result,
        expected_required_display,
        expected_rule_title,
        expects_constant_adjustment,
    ) in [
        (
            "integrate(sec(sqrt(x))*tan(sqrt(x))/(2*sqrt(x)), x)",
            "sec(sqrt(x))",
            serde_json::json!(["cos(sqrt(x)) ≠ 0", "x > 0"]),
            "Usar la regla de sec(u)·tan(u) -> sec(u)",
            false,
        ),
        (
            "integrate(csc(sqrt(x))*cot(sqrt(x))/(2*sqrt(x)), x)",
            "-csc(sqrt(x))",
            serde_json::json!(["sin(sqrt(x)) ≠ 0", "x > 0"]),
            "Usar la regla de csc(u)·cot(u) -> -csc(u)",
            false,
        ),
        (
            "integrate(sec(sqrt(2*x))*tan(sqrt(2*x))/sqrt(2*x), x)",
            "sec(sqrt(2·x))",
            serde_json::json!(["cos(sqrt(2·x)) ≠ 0", "x > 0"]),
            "Usar la regla de sec(u)·tan(u) -> sec(u)",
            false,
        ),
        (
            "integrate(sin(sqrt(x))*sqrt(x)/(2*x*cos(sqrt(x))^2), x)",
            "sec(sqrt(x))",
            serde_json::json!(["cos(sqrt(x)) ≠ 0", "x > 0"]),
            "Usar la regla de sec(u)·tan(u) -> sec(u)",
            false,
        ),
        (
            "integrate(-sec(sqrt(x))*tan(sqrt(x))/(2*sqrt(x)), x)",
            "-sec(sqrt(x))",
            serde_json::json!(["cos(sqrt(x)) ≠ 0", "x > 0"]),
            "Usar la regla de sec(u)·tan(u) -> sec(u)",
            true,
        ),
        (
            "integrate(-csc(sqrt(x))*cot(sqrt(x))/(2*sqrt(x)), x)",
            "csc(sqrt(x))",
            serde_json::json!(["sin(sqrt(x)) ≠ 0", "x > 0"]),
            "Usar la regla de csc(u)·cot(u) -> -csc(u)",
            true,
        ),
        (
            "integrate(-3*sec(sqrt(3*x+1))*tan(sqrt(3*x+1))/(2*sqrt(3*x+1)), x)",
            "-sec(sqrt(3·x + 1))",
            serde_json::json!(["cos(sqrt(3·x + 1)) ≠ 0", "x > -1/3"]),
            "Usar la regla de sec(u)·tan(u) -> sec(u)",
            true,
        ),
        (
            "integrate(-3*csc(sqrt(3*x+1))*cot(sqrt(3*x+1))/(2*sqrt(3*x+1)), x)",
            "csc(sqrt(3·x + 1))",
            serde_json::json!(["sin(sqrt(3·x + 1)) ≠ 0", "x > -1/3"]),
            "Usar la regla de csc(u)·cot(u) -> -csc(u)",
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
            "sqrt-chain reciprocal trig product trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
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
            "sqrt-chain reciprocal trig product should not use the generic substitution substep for {input}: {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }
}
#[test]
fn integrate_contract_negative_scaled_cosecant_result_latex_keeps_sign_on_coefficient() {
    let input = "integrate(-2*cos(sqrt(3-2*x))/(sqrt(3-2*x)*sin(sqrt(3-2*x))^2), x)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);
    assert!(
        stderr.is_empty(),
        "unexpected stderr for negative scaled cosecant integral: {stderr}"
    );
    assert_eq!(wire["result"], "-2·csc(sqrt(3 - 2·x))");
    assert_eq!(
        wire["result_latex"],
        "-2\\cdot \\csc(\\sqrt{3 - 2\\cdot x})"
    );
    assert_ne!(
        wire["result_latex"], "2\\cdot -\\csc(\\sqrt{3 - 2\\cdot x})",
        "negative sign should not remain inside the multiplicative factor"
    );
}
#[test]
fn integrate_contract_negative_scaled_raw_reciprocal_trig_primitives_keep_sign_compact() {
    let cases = [
        (
            "integrate(-4*x*sin(x^2)/cos(x^2)^2, x)",
            "-2·sec(x^2)",
            "-2\\cdot \\sec({x}^{2})",
            "cos(x^2) ≠ 0",
            "diff(integrate(-4*x*sin(x^2)/cos(x^2)^2, x), x) + 4*x*sin(x^2)/cos(x^2)^2",
        ),
        (
            "integrate(-4*x*cos(x^2)/sin(x^2)^2, x)",
            "2·csc(x^2)",
            "2\\cdot \\csc({x}^{2})",
            "sin(x^2) ≠ 0",
            "diff(integrate(-4*x*cos(x^2)/sin(x^2)^2, x), x) + 4*x*cos(x^2)/sin(x^2)^2",
        ),
    ];

    for (input, expected_result, expected_latex, expected_condition, residual) in cases {
        let (wire, stderr) = cli_eval_json_with_stderr(input);
        assert!(
            stderr.is_empty(),
            "unexpected stderr for negative scaled raw reciprocal trig primitive: {stderr}"
        );
        assert_eq!(wire["result"], expected_result);
        assert_eq!(wire["result_latex"], expected_latex);
        assert_eq!(
            wire["required_display"],
            serde_json::json!([expected_condition])
        );

        let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual);
        assert!(
            residual_stderr.is_empty(),
            "unexpected stderr for negative scaled raw reciprocal trig residual: {residual_stderr}"
        );
        assert_eq!(residual_wire["result"], "0");
        assert_eq!(
            residual_wire["required_display"],
            serde_json::json!([expected_condition])
        );
    }
}
#[test]
fn integrate_contract_sqrt_chain_secant_cosecant_products_integrate_directly() {
    let cases = [
        "integrate(sec(sqrt(x))*tan(sqrt(x))/(2*sqrt(x)), x)",
        "integrate(csc(sqrt(x))*cot(sqrt(x))/(2*sqrt(x)), x)",
    ];

    for input in cases {
        let step_rules = evaluated_integral_step_rules(input);
        assert_eq!(
            step_rules,
            vec!["Symbolic Integration".to_string()],
            "sqrt-chain reciprocal trig products should integrate directly: {step_rules:?}"
        );
        assert_antiderivative_verifies(input);
    }
}
#[test]
fn integrate_contract_sqrt_chain_tangent_cotangent_logs_integrate_directly() {
    let cases = [
        "integrate(tan(sqrt(x))/(2*sqrt(x)), x)",
        "integrate(cot(sqrt(x))/(2*sqrt(x)), x)",
        "integrate(tan(sqrt(3*x+1))*3/(2*sqrt(3*x+1)), x)",
        "integrate(cot(sqrt(3*x+1))*3/(2*sqrt(3*x+1)), x)",
        "integrate(-tan(sqrt(3-2*x))/sqrt(3-2*x), x)",
        "integrate(-cot(sqrt(3-2*x))/sqrt(3-2*x), x)",
    ];

    for input in cases {
        let step_rules = evaluated_integral_step_rules(input);
        assert_eq!(
            step_rules,
            vec!["Symbolic Integration".to_string()],
            "sqrt-chain trig log derivatives should integrate directly: {step_rules:?}"
        );
        assert_antiderivative_verifies(input);
    }
}
#[test]
fn integrate_contract_sqrt_chain_tangent_cotangent_logs_explain_u_and_du() {
    for (
        input,
        expected_result,
        expected_required_display,
        expected_rule_title,
        expects_adjustment,
    ) in [
        (
            "integrate(tan(sqrt(x))/(2*sqrt(x)), x)",
            "-ln(|cos(sqrt(x))|)",
            serde_json::json!(["cos(sqrt(x)) ≠ 0", "x > 0"]),
            "Usar la regla de tan(u) -> -ln|cos(u)|",
            false,
        ),
        (
            "integrate(cot(sqrt(x))/(2*sqrt(x)), x)",
            "ln(|sin(sqrt(x))|)",
            serde_json::json!(["sin(sqrt(x)) ≠ 0", "x > 0"]),
            "Usar la regla de cot(u) -> ln|sin(u)|",
            false,
        ),
        (
            "integrate(tan(sqrt(x))/sqrt(x), x)",
            "-2·ln(|cos(sqrt(x))|)",
            serde_json::json!(["cos(sqrt(x)) ≠ 0", "x > 0"]),
            "Usar la regla de tan(u) -> -ln|cos(u)|",
            true,
        ),
        (
            "integrate(cot(sqrt(x))/sqrt(x), x)",
            "2·ln(|sin(sqrt(x))|)",
            serde_json::json!(["sin(sqrt(x)) ≠ 0", "x > 0"]),
            "Usar la regla de cot(u) -> ln|sin(u)|",
            true,
        ),
        (
            "integrate(tan(sqrt(3*x+1))*3/(2*sqrt(3*x+1)), x)",
            "-ln(|cos(sqrt(3·x + 1))|)",
            serde_json::json!(["cos(sqrt(3·x + 1)) ≠ 0", "x > -1/3"]),
            "Usar la regla de tan(u) -> -ln|cos(u)|",
            false,
        ),
        (
            "integrate(-cot(sqrt(3-2*x))/sqrt(3-2*x), x)",
            "ln(|sin(sqrt(3 - 2·x))|)",
            serde_json::json!(["sin(sqrt(3 - 2·x)) ≠ 0", "x < 3/2"]),
            "Usar la regla de cot(u) -> ln|sin(u)|",
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
            "sqrt-chain trig log trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
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
            expects_adjustment,
            "unexpected constant adjustment substep presence for {input}: {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .all(|substep| substep["title"] != "Usar sustitución"),
            "sqrt-chain trig log table should not use the generic substitution substep for {input}: {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }
}
#[test]
fn integrate_contract_sqrt_chain_tangent_cotangent_logs_verify() {
    let cases = [
        (
            "integrate(tan(sqrt(x))/(2*sqrt(x)), x)",
            "-ln(|cos(sqrt(x))|)",
            "tan(sqrt(x)) / (2 * sqrt(x))",
            vec!["cos(sqrt(x)) ≠ 0", "x > 0"],
            vec!["cos(sqrt(x)) ≠ 0", "x > 0"],
        ),
        (
            "integrate(cot(sqrt(x))/(2*sqrt(x)), x)",
            "ln(|sin(sqrt(x))|)",
            "cot(sqrt(x)) / (2 * sqrt(x))",
            vec!["sin(sqrt(x)) ≠ 0", "x > 0"],
            vec!["sin(sqrt(x)) ≠ 0", "x > 0"],
        ),
        (
            "integrate(-tan(sqrt(x))/(2*sqrt(x)), x)",
            "ln(|cos(sqrt(x))|)",
            "-tan(sqrt(x)) / (2 * sqrt(x))",
            vec!["cos(sqrt(x)) ≠ 0", "x > 0"],
            vec!["cos(sqrt(x)) ≠ 0", "x > 0"],
        ),
        (
            "integrate(-cot(sqrt(x))/(2*sqrt(x)), x)",
            "-ln(|sin(sqrt(x))|)",
            "-cot(sqrt(x)) / (2 * sqrt(x))",
            vec!["sin(sqrt(x)) ≠ 0", "x > 0"],
            vec!["sin(sqrt(x)) ≠ 0", "x > 0"],
        ),
        (
            "integrate(tan(sqrt(2*x))/sqrt(2*x), x)",
            "-ln(|cos(sqrt(2 * x))|)",
            "tan(sqrt(2 * x)) / sqrt(2 * x)",
            vec!["cos(sqrt(2 * x)) ≠ 0", "x > 0"],
            vec!["cos(sqrt(2 * x)) ≠ 0", "x > 0"],
        ),
        (
            "integrate(-tan(sqrt(2*x))/sqrt(2*x), x)",
            "ln(|cos(sqrt(2 * x))|)",
            "-tan(sqrt(2 * x)) / sqrt(2 * x)",
            vec!["cos(sqrt(2 * x)) ≠ 0", "x > 0"],
            vec!["cos(sqrt(2 * x)) ≠ 0", "x > 0"],
        ),
        (
            "integrate(-cot(sqrt(2*x))/sqrt(2*x), x)",
            "-ln(|sin(sqrt(2 * x))|)",
            "-cot(sqrt(2 * x)) / sqrt(2 * x)",
            vec!["sin(sqrt(2 * x)) ≠ 0", "x > 0"],
            vec!["sin(sqrt(2 * x)) ≠ 0", "x > 0"],
        ),
        (
            "integrate(tan(sqrt(3*x+1))*3/(2*sqrt(3*x+1)), x)",
            "-ln(|cos(sqrt(3 * x + 1))|)",
            "3 * tan(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))",
            vec!["cos(sqrt(3 * x + 1)) ≠ 0", "x > -1/3"],
            vec!["cos(sqrt(3 * x + 1)) ≠ 0", "x > -1/3"],
        ),
        (
            "integrate(cot(sqrt(3*x+1))*3/(2*sqrt(3*x+1)), x)",
            "ln(|sin(sqrt(3 * x + 1))|)",
            "3 * cot(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))",
            vec!["sin(sqrt(3 * x + 1)) ≠ 0", "x > -1/3"],
            vec!["sin(sqrt(3 * x + 1)) ≠ 0", "x > -1/3"],
        ),
        (
            "integrate(-tan(sqrt(3*x+1))*3/(2*sqrt(3*x+1)), x)",
            "ln(|cos(sqrt(3 * x + 1))|)",
            "-3 * tan(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))",
            vec!["cos(sqrt(3 * x + 1)) ≠ 0", "x > -1/3"],
            vec!["cos(sqrt(3 * x + 1)) ≠ 0", "x > -1/3"],
        ),
        (
            "integrate(-tan(sqrt(3-2*x))/sqrt(3-2*x), x)",
            "-ln(|cos(sqrt(3 - 2 * x))|)",
            "-tan(sqrt(3 - 2 * x)) / sqrt(3 - 2 * x)",
            vec!["cos(sqrt(3 - 2 * x)) ≠ 0", "x < 3/2"],
            vec!["cos(sqrt(3 - 2 * x)) ≠ 0", "x < 3/2"],
        ),
        (
            "integrate(-cot(sqrt(3-2*x))/sqrt(3-2*x), x)",
            "ln(|sin(sqrt(3 - 2 * x))|)",
            "-cot(sqrt(3 - 2 * x)) / sqrt(3 - 2 * x)",
            vec!["sin(sqrt(3 - 2 * x)) ≠ 0", "x < 3/2"],
            vec!["sin(sqrt(3 - 2 * x)) ≠ 0", "x < 3/2"],
        ),
        (
            "integrate(-cot(sqrt(3*x+1))*3/(2*sqrt(3*x+1)), x)",
            "-ln(|sin(sqrt(3 * x + 1))|)",
            "-3 * cot(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))",
            vec!["sin(sqrt(3 * x + 1)) ≠ 0", "x > -1/3"],
            vec!["sin(sqrt(3 * x + 1)) ≠ 0", "x > -1/3"],
        ),
    ];

    for (
        input,
        expected_result,
        expected_nested_result,
        expected_conditions,
        expected_nested_conditions,
    ) in cases
    {
        let (result, required) = evaluated_integral_with_required_conditions(input);

        assert_eq!(result, expected_result, "unexpected result for {input}");
        assert_eq!(
            required, expected_conditions,
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert_antiderivative_verifies(input);

        let nested_input = format!("diff({input}, x)");
        let (nested_result, nested_required) =
            evaluated_expr_with_required_conditions(&nested_input);
        assert_eq!(
            nested_result, expected_nested_result,
            "unexpected nested diff/integrate result for {input}"
        );
        assert_eq!(
            nested_required, expected_nested_conditions,
            "unexpected nested required_conditions for {input}: {nested_required:?}"
        );
    }
}
#[test]
fn integrate_contract_sqrt_chain_reciprocal_trig_logs_verify_by_diff() {
    let cases = [
        (
            "integrate(3/(2*sqrt(3*x+1)*cos(sqrt(3*x+1))), x)",
            "ln(|tan(sqrt(3 * x + 1)) + sec(sqrt(3 * x + 1))|)",
            vec!["cos(sqrt(3 * x + 1)) ≠ 0", "x > -1/3"],
        ),
        (
            "integrate(3/(2*sqrt(3*x+1)*sin(sqrt(3*x+1))), x)",
            "ln(|csc(sqrt(3 * x + 1)) - cot(sqrt(3 * x + 1))|)",
            vec!["sin(sqrt(3 * x + 1)) ≠ 0", "x > -1/3"],
        ),
    ];

    for (input, expected_result, expected_conditions) in cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected_result, "input: {input}");
        assert_eq!(required, expected_conditions, "input: {input}");
        assert_antiderivative_verifies(input);
    }
}
#[test]
fn integrate_contract_sqrt_chain_reciprocal_trig_logs_explain_u_and_du() {
    for (input, expected_result, expected_required_display, expected_rule_title) in [
        (
            "integrate(3/(2*sqrt(3*x+1)*cos(sqrt(3*x+1))), x)",
            "ln(|tan(sqrt(3·x + 1)) + sec(sqrt(3·x + 1))|)",
            serde_json::json!(["cos(sqrt(3·x + 1)) ≠ 0", "x > -1/3"]),
            "Usar la regla de sec(u) -> ln|sec(u)+tan(u)|",
        ),
        (
            "integrate(3/(2*sqrt(3*x+1)*sin(sqrt(3*x+1))), x)",
            "ln(|csc(sqrt(3·x + 1)) - cot(sqrt(3·x + 1))|)",
            serde_json::json!(["sin(sqrt(3·x + 1)) ≠ 0", "x > -1/3"]),
            "Usar la regla de csc(u) -> ln|csc(u)-cot(u)|",
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
            "sqrt-chain reciprocal trig log trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
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
        assert!(
            substeps
                .iter()
                .all(|substep| substep["title"] != "Usar sustitución"),
            "sqrt-chain reciprocal trig log table should not use the generic substitution substep for {input}: {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }
}
#[test]
fn integrate_contract_negated_polynomial_secant_tangent_product_preserves_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(-x*sec(x^2)*tan(x^2), x)");

    assert_eq!(result, "-sec(x^2) / 2");
    assert_eq!(
        required,
        vec!["cos(x^2) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_negated_polynomial_cosecant_cotangent_product_preserves_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(-x^2*csc(x^3)*cot(x^3), x)");

    assert_eq!(result, "csc(x^3) / 3");
    assert_eq!(
        required,
        vec!["sin(x^3) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_secant_tangent_non_linear_argument_remains_residual() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(sec(x^2)*tan(x^2), x)");

    assert_eq!(result, "integrate(sin(x^2) / cos(x^2)^2, x)");
    assert_eq!(
        required,
        vec!["cos(x^2) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_tangent_non_linear_argument_remains_residual_with_pole_condition() {
    let (result, required) = evaluated_integral_with_required_conditions("integrate(tan(x^2), x)");

    assert_eq!(result, "integrate(tan(x^2), x)");
    assert_eq!(
        required,
        vec!["cos(x^2) ≠ 0".to_string()],
        "unsupported tangent residual should preserve pole domain: {required:?}"
    );
}
#[test]
fn integrate_contract_presimplified_tangent_residual_preserves_pole_condition() {
    for (input, expected_result) in [
        ("integrate(tan(x^2+0), x)", "integrate(tan(x^2), x)"),
        (
            "integrate(tan(x^2+0)+sin(x^2), x)",
            "integrate(sin(x^2) + tan(x^2), x)",
        ),
    ] {
        let (result, required) = evaluated_integral_with_required_conditions(input);

        assert_eq!(result, expected_result, "unexpected result for {input}");
        assert_eq!(
            required,
            vec!["cos(x^2) ≠ 0".to_string()],
            "pre-simplified tangent residual should preserve pole domain for {input}: {required:?}"
        );
    }
}
#[test]
fn integrate_contract_presimplified_cosecant_residual_preserves_pole_condition() {
    for (input, expected_result) in [
        ("integrate(csc(x^2+0), x)", "integrate(csc(x^2), x)"),
        ("integrate(cot(x^2+0), x)", "integrate(cot(x^2), x)"),
    ] {
        let (result, required) = evaluated_integral_with_required_conditions(input);

        assert_eq!(result, expected_result, "unexpected result for {input}");
        assert_eq!(
            required,
            vec!["sin(x^2) ≠ 0".to_string()],
            "pre-simplified sine-pole residual should preserve domain for {input}: {required:?}"
        );
    }
}
