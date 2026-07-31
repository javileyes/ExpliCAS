use super::*;

#[test]
fn integrate_contract_antiderivative_verification_uses_bounded_public_residual_for_stable_hyperbolic_reciprocal_square_subset(
) {
    for input in [
        "integrate(1/cosh(2*x+1)^2, x)",
        "integrate(1/sinh(2*x+1)^2, x)",
        "integrate(sinh(2*x+1)/cosh(2*x+1)^2, x)",
        "integrate(cosh(2*x+1)/sinh(2*x+1)^2, x)",
    ] {
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "{input} should verify through the bounded public residual route"
        );
    }
}
#[test]
fn integrate_contract_reciprocal_hyperbolic_first_power_resolves_sech_csch() {
    // n == 1: sech -> arctan(sinh) (no domain condition, cosh > 0), csch ->
    // ln|tanh(x/2)| (the generic sinh != 0 domain carries over). Affine arguments
    // get the 1/u' scale; a constant numerator scales the primitive.
    for (input, expected_result, expected_required) in [
        (
            "integrate(1/cosh(x), x)",
            "arctan(sinh(x))",
            Vec::<String>::new(),
        ),
        ("integrate(sech(x), x)", "arctan(sinh(x))", Vec::new()),
        (
            "integrate(1/cosh(2*x+1), x)",
            "1/2 * arctan(sinh(2 * x + 1))",
            Vec::new(),
        ),
        ("integrate(3/cosh(x), x)", "3 * arctan(sinh(x))", Vec::new()),
        (
            "integrate(1/sinh(x), x)",
            "ln(|tanh(x / 2)|)",
            vec!["sinh(x) ≠ 0".to_string()],
        ),
    ] {
        let (result, required) = evaluated_expr_with_required_conditions(input);
        assert_eq!(result, expected_result, "result for {input}");
        assert_eq!(required, expected_required, "required for {input}");
    }

    // Soundness: each emitted antiderivative differentiates back to the integrand.
    for integrand in ["1/cosh(x)", "1/sinh(x)", "1/cosh(2*x+1)"] {
        let (result, _) = evaluated_expr_with_required_conditions(&format!(
            "diff(integrate({integrand}, x), x) - ({integrand})"
        ));
        assert_eq!(result, "0", "round-trip for {integrand}");
    }

    // The n=1 branch must NOT disturb the n>=2 table route nor hijack tanh/coth,
    // and must leave a constant-argument denominator as a constant multiple.
    assert_eq!(
        evaluated_expr_with_required_conditions("integrate(1/cosh(x)^2, x)").0,
        "tanh(x)"
    );
    assert_eq!(
        evaluated_expr_with_required_conditions("integrate(1/sinh(x)^2, x)").0,
        "-1 / tanh(x)"
    );
    assert_eq!(
        evaluated_expr_with_required_conditions("integrate(1/tanh(x), x)").0,
        "ln(|sinh(x)|)"
    );
    assert_eq!(
        evaluated_expr_with_required_conditions("integrate(1/cosh(5), x)").0,
        "x / cosh(5)"
    );
}
#[test]
fn integrate_contract_antiderivative_verification_uses_bounded_public_residual_for_nonlinear_hyperbolic_reciprocal_square_subset(
) {
    for input in [
        "integrate(2*x/cosh(x^2)^2, x)",
        "integrate(2*x/sinh(x^2)^2, x)",
        "integrate(2*x*sinh(x^2)/cosh(x^2)^2, x)",
        "integrate(2*x*cosh(x^2)/sinh(x^2)^2, x)",
    ] {
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "{input} should verify through the bounded public residual route"
        );
    }
}
#[test]
fn integrate_contract_antiderivative_verification_uses_bounded_public_residual_for_hyperbolic_reciprocal_fourth_subset(
) {
    for input in [
        "integrate(1/cosh(2*x+1)^4, x)",
        "integrate(2*x/cosh(x^2)^4, x)",
        "integrate(1/sinh(2*x+1)^4, x)",
        "integrate(2*x/sinh(x^2)^4, x)",
        "integrate(2*k*x/sinh(x^2+b)^4, x)",
    ] {
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "{input} should verify through the bounded public residual route"
        );
    }
}
#[test]
fn integrate_contract_nonlinear_hyperbolic_reciprocal_square_residual_survives_wrappers() {
    for (input, expected_result, expected_required_display) in [
        (
            "(diff(integrate(2*x/sinh(x^2)^2, x), x) - 2*x/sinh(x^2)^2) + y - y",
            "0",
            serde_json::json!(["sinh(x^2) ≠ 0"]),
        ),
        (
            "(diff(integrate(2*x/sinh(x^2)^2, x), x) - 2*x/sinh(x^2)^2)/(x+1)",
            "0",
            serde_json::json!(["sinh(x^2) ≠ 0", "x ≠ -1"]),
        ),
        (
            "((diff(integrate(2*x/sinh(x^2)^2, x), x) - 2*x/sinh(x^2)^2) + x + 1)/(x+1)",
            "1",
            serde_json::json!(["sinh(x^2) ≠ 0", "x ≠ -1"]),
        ),
        (
            "1/((diff(integrate(2*x/sinh(x^2)^2, x), x) - 2*x/sinh(x^2)^2) + x + 1) - 1/(x+1)",
            "0",
            serde_json::json!(["sinh(x^2) ≠ 0", "x ≠ -1"]),
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr(input);
        assert!(
            stderr.is_empty(),
            "unexpected stderr for wrapped nonlinear hyperbolic reciprocal-square residual: {stderr}"
        );
        assert_eq!(wire["result"], expected_result);
        assert_eq!(
            wire["required_display"], expected_required_display,
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
    }
}
#[test]
fn integrate_contract_affine_hyperbolic_square_power_reduction() {
    let public_cases = [
        ("integrate(sinh(x)^2, x)", "1/2 * sinh(x) * cosh(x) - x / 2"),
        ("integrate(cosh(x)^2, x)", "1/2 * sinh(x) * cosh(x) + x / 2"),
        ("integrate(sinh(2*x)^2, x)", "1/8 * sinh(4 * x) - 1/2 * x"),
        (
            "integrate(sinh(2*x + 1)^2, x)",
            "1/4 * sinh(2 * x + 1) * cosh(2 * x + 1) - x / 2",
        ),
        (
            "integrate(cosh(2*x + 1)^2, x)",
            "1/4 * sinh(2 * x + 1) * cosh(2 * x + 1) + x / 2",
        ),
        ("integrate(tanh(x)^2, x)", "x - tanh(x)"),
        ("integrate(tanh(2*x + 1)^2, x)", "x - 1/2 * tanh(2 * x + 1)"),
        ("integrate(tanh(1-2*x)^2, x)", "1/2 * tanh(1 - 2 * x) + x"),
        (
            "integrate(tanh(x)^4, x)",
            "1/3 * (3 * x - tanh(x)^3 - 3 * tanh(x))",
        ),
        (
            "integrate(tanh(2*x + 1)^4, x)",
            "1/6 * (6 * x - tanh(2 * x + 1)^3 - 3 * tanh(2 * x + 1))",
        ),
        (
            "integrate(tanh(1-2*x)^4, x)",
            "1/6 * (tanh(1 - 2 * x)^3 + 3 * tanh(1 - 2 * x) + 6 * x)",
        ),
        (
            "integrate(tanh(x)^6, x)",
            "1/15 * (15 * x - 15 * tanh(x) - 5 * tanh(x)^3 - 3 * tanh(x)^5)",
        ),
        (
            "integrate(tanh(2*x + 1)^6, x)",
            "1/30 * (30 * x - 15 * tanh(2 * x + 1) - 5 * tanh(2 * x + 1)^3 - 3 * tanh(2 * x + 1)^5)",
        ),
        (
            "integrate(tanh(1-2*x)^6, x)",
            "1/30 * (3 * tanh(1 - 2 * x)^5 + 5 * tanh(1 - 2 * x)^3 + 15 * tanh(1 - 2 * x) + 30 * x)",
        ),
        (
            "integrate(tanh(x)^8, x)",
            "x - (tanh(x) + 1/7 * tanh(x)^7 + 1/5 * tanh(x)^5 + 1/3 * tanh(x)^3)",
        ),
        (
            "integrate(tanh(2*x + 1)^8, x)",
            "x - (tanh(2 * x + 1) + 1/7 * tanh(2 * x + 1)^7 + 1/5 * tanh(2 * x + 1)^5 + 1/3 * tanh(2 * x + 1)^3) / 2",
        ),
        (
            "integrate(tanh(1-2*x)^8, x)",
            "(tanh(1 - 2 * x) + 1/7 * tanh(1 - 2 * x)^7 + 1/5 * tanh(1 - 2 * x)^5 + 1/3 * tanh(1 - 2 * x)^3) / 2 + x",
        ),
        (
            "integrate(4*sinh(x)^2*cosh(x)^2, x)",
            "1/8 * sinh(4 * x) - 1/2 * x",
        ),
        (
            "integrate(sinh(2*x + 1)^2*cosh(2*x + 1)^2, x)",
            "1/64 * sinh(4 * (2 * x + 1)) - 1/8 * x",
        ),
    ];

    for (input, expected) in public_cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected, "input: {input}");
        assert!(
            required.is_empty(),
            "input: {input}, required: {required:?}"
        );
        assert_eq!(integrate_call_antiderivative_residual_result(input), "0");
    }
}
#[test]
fn integrate_contract_affine_tanh_eight_explicit_primitives_verify_publicly() {
    let residuals = [
        "diff(integrate(tanh(x)^8, x), x) - tanh(x)^8",
        "diff(integrate(tanh(2*x+1)^8, x), x) - tanh(2*x+1)^8",
        "diff(integrate(tanh(1-2*x)^8, x), x) - tanh(1-2*x)^8",
        "diff(x - 1/2*(tanh(2*x+1) + tanh(2*x+1)^3/3 + tanh(2*x+1)^5/5 + tanh(2*x+1)^7/7), x) - tanh(2*x+1)^8",
        "diff(x + 1/2*(tanh(1-2*x) + tanh(1-2*x)^3/3 + tanh(1-2*x)^5/5 + tanh(1-2*x)^7/7), x) - tanh(1-2*x)^8",
    ];

    for input in residuals {
        let (wire, stderr) = cli_eval_json_with_stderr(input);
        assert!(
            stderr.is_empty(),
            "tanh eighth primitive residual should stay quiet for {input}: {stderr}"
        );
        assert_eq!(wire["result"], "0", "{input}");
        assert_eq!(wire["required_display"], serde_json::json!([]), "{input}");
        assert!(
            wire["warnings"]
                .as_array()
                .is_some_and(|warnings| warnings.is_empty()),
            "tanh eighth primitive residual should not warn for {input}: {wire:#}"
        );
    }
}
#[test]
fn integrate_contract_affine_tanh_six_explicit_primitives_verify_publicly() {
    let residuals = [
        "diff(x - tanh(x) - tanh(x)^3/3 - tanh(x)^5/5, x) - tanh(x)^6",
        "diff(x - 1/2*(tanh(2*x+1) + tanh(2*x+1)^3/3 + tanh(2*x+1)^5/5), x) - tanh(2*x+1)^6",
        "diff(x + 1/2*(tanh(1-2*x) + tanh(1-2*x)^3/3 + tanh(1-2*x)^5/5), x) - tanh(1-2*x)^6",
    ];

    for input in residuals {
        let (wire, stderr) = cli_eval_json_with_stderr(input);
        assert!(
            stderr.is_empty(),
            "tanh sixth primitive residual should stay quiet for {input}: {stderr}"
        );
        assert_eq!(wire["result"], "0", "{input}");
        assert_eq!(wire["required_display"], serde_json::json!([]), "{input}");
        assert!(
            wire["warnings"]
                .as_array()
                .is_some_and(|warnings| warnings.is_empty()),
            "tanh sixth primitive residual should not warn for {input}: {wire:#}"
        );
    }
}
#[test]
fn integrate_contract_affine_hyperbolic_cubic_power_reduction() {
    let public_cases = [
        (
            "integrate(sinh(2*x + 1)^3, x)",
            "1/2 * (1/3 * cosh(2 * x + 1)^3 - cosh(2 * x + 1))",
        ),
        (
            "integrate(cosh(2*x + 1)^3, x)",
            "1/2 * (sinh(2 * x + 1) + 1/3 * sinh(2 * x + 1)^3)",
        ),
        (
            "integrate(sinh(1 - 2*x)^3, x)",
            "1/6 * (3 * cosh(1 - 2 * x) - cosh(1 - 2 * x)^3)",
        ),
        (
            "integrate(cosh(1 - 2*x)^3, x)",
            "-1/2 * (sinh(1 - 2 * x) + 1/3 * sinh(1 - 2 * x)^3)",
        ),
    ];

    for (input, expected) in public_cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected, "input: {input}");
        assert!(
            required.is_empty(),
            "input: {input}, required: {required:?}"
        );
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "{input} should verify through the bounded public residual route"
        );
    }
}
#[test]
fn integrate_contract_hyperbolic_odd_power_reduction() {
    let cases = [
        ("integrate(sinh(x)^3, x)", "1/3 * cosh(x)^3 - cosh(x)"),
        ("integrate(cosh(x)^3, x)", "sinh(x) + 1/3 * sinh(x)^3"),
        (
            "integrate(sinh(x)^5, x)",
            "1/5 * (cosh(x)^5 + 5 * cosh(x) - 10/3 * cosh(x)^3)",
        ),
        (
            "integrate(cosh(x)^5, x)",
            "1/5 * (sinh(x)^5 + 10/3 * sinh(x)^3 + 5 * sinh(x))",
        ),
        (
            "integrate(sinh(x)^7, x)",
            "-cosh(x) + cosh(x)^3 + 1/7 * cosh(x)^7 - 3/5 * cosh(x)^5",
        ),
        (
            "integrate(cosh(x)^7, x)",
            "sinh(x) + sinh(x)^3 + 1/7 * sinh(x)^7 + 3/5 * sinh(x)^5",
        ),
    ];

    for (input, expected) in cases {
        let (antiderivative, required) = evaluated_integral_with_required_conditions(input);
        assert!(
            required.is_empty(),
            "hyperbolic odd-power primitive should not add domain conditions for {input}: {required:?}"
        );
        assert_eq!(antiderivative, expected, "{input}");
        assert!(
            !antiderivative.contains("integrate("),
            "expected a closed-form primitive for {input}, got {antiderivative}"
        );
    }

    for input in ["integrate(sinh(x)^3, x)", "integrate(cosh(x)^7, x)"] {
        assert_antiderivative_equiv_verifies(input);
    }

    for input in ["integrate(sinh(x)^7, x)", "integrate(cosh(x)^7, x)"] {
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "{input} should verify through the bounded public residual route"
        );
    }
}
#[test]
fn integrate_contract_affine_hyperbolic_fifth_explicit_primitives_verify_publicly() {
    let residuals = [
        "diff(1/2*(1/5*cosh(2*x+1)^5 - 2/3*cosh(2*x+1)^3 + cosh(2*x+1)), x) - sinh(2*x+1)^5",
        "diff(1/2*(sinh(2*x+1) + 2/3*sinh(2*x+1)^3 + 1/5*sinh(2*x+1)^5), x) - cosh(2*x+1)^5",
        "diff(-1/2*(1/5*cosh(1-2*x)^5 - 2/3*cosh(1-2*x)^3 + cosh(1-2*x)), x) - sinh(1-2*x)^5",
        "diff(-1/2*(sinh(1-2*x) + 2/3*sinh(1-2*x)^3 + 1/5*sinh(1-2*x)^5), x) - cosh(1-2*x)^5",
    ];

    for input in residuals {
        let (wire, stderr) = cli_eval_json_with_stderr(input);
        assert!(stderr.is_empty(), "unexpected stderr for {input}: {stderr}");
        assert_eq!(wire["result"], "0", "{input}");
        assert_eq!(wire["required_display"], serde_json::json!([]), "{input}");
        assert!(
            wire["warnings"]
                .as_array()
                .is_some_and(|warnings| warnings.is_empty()),
            "affine fifth primitive residual should not warn for {input}: {wire:#}"
        );
    }
}
#[test]
fn integrate_contract_affine_hyperbolic_fifth_power_reduction() {
    let public_cases = [
        (
            "integrate(sinh(2*x + 1)^5, x)",
            "1/2 * (cosh(2 * x + 1) + 1/5 * cosh(2 * x + 1)^5 - 2/3 * cosh(2 * x + 1)^3)",
        ),
        (
            "integrate(cosh(2*x + 1)^5, x)",
            "1/2 * (sinh(2 * x + 1) + 1/5 * sinh(2 * x + 1)^5 + 2/3 * sinh(2 * x + 1)^3)",
        ),
        (
            "integrate(sinh(1 - 2*x)^5, x)",
            "-1/2 * (cosh(1 - 2 * x) + 1/5 * cosh(1 - 2 * x)^5 - 2/3 * cosh(1 - 2 * x)^3)",
        ),
        (
            "integrate(cosh(1 - 2*x)^5, x)",
            "-1/2 * (sinh(1 - 2 * x) + 1/5 * sinh(1 - 2 * x)^5 + 2/3 * sinh(1 - 2 * x)^3)",
        ),
    ];

    for (input, expected) in public_cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected, "input: {input}");
        assert!(
            required.is_empty(),
            "input: {input}, required: {required:?}"
        );
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "{input} should verify through the bounded public residual route"
        );
    }
}
#[test]
fn integrate_contract_affine_hyperbolic_seventh_explicit_primitives_verify_publicly() {
    let residuals = [
        "diff(1/2*(1/7*cosh(2*x+1)^7 - 3/5*cosh(2*x+1)^5 + cosh(2*x+1)^3 - cosh(2*x+1)), x) - sinh(2*x+1)^7",
        "diff(1/2*(sinh(2*x+1) + sinh(2*x+1)^3 + 3/5*sinh(2*x+1)^5 + 1/7*sinh(2*x+1)^7), x) - cosh(2*x+1)^7",
        "diff(-1/2*(1/7*cosh(1-2*x)^7 - 3/5*cosh(1-2*x)^5 + cosh(1-2*x)^3 - cosh(1-2*x)), x) - sinh(1-2*x)^7",
        "diff(-1/2*(sinh(1-2*x) + sinh(1-2*x)^3 + 3/5*sinh(1-2*x)^5 + 1/7*sinh(1-2*x)^7), x) - cosh(1-2*x)^7",
    ];

    for input in residuals {
        let (wire, stderr) = cli_eval_json_with_stderr(input);
        assert!(stderr.is_empty(), "unexpected stderr for {input}: {stderr}");
        assert_eq!(wire["result"], "0", "{input}");
        assert_eq!(wire["required_display"], serde_json::json!([]), "{input}");
        assert!(
            wire["warnings"]
                .as_array()
                .is_some_and(|warnings| warnings.is_empty()),
            "affine seventh primitive residual should not warn for {input}: {wire:#}"
        );
    }
}
#[test]
fn integrate_contract_affine_hyperbolic_seventh_power_reduction() {
    let public_cases = [
        (
            "integrate(sinh(2*x + 1)^7, x)",
            "1/2 * (-cosh(2 * x + 1) + cosh(2 * x + 1)^3 + 1/7 * cosh(2 * x + 1)^7 - 3/5 * cosh(2 * x + 1)^5)",
        ),
        (
            "integrate(cosh(2*x + 1)^7, x)",
            "1/2 * (sinh(2 * x + 1) + sinh(2 * x + 1)^3 + 1/7 * sinh(2 * x + 1)^7 + 3/5 * sinh(2 * x + 1)^5)",
        ),
        (
            "integrate(sinh(1 - 2*x)^7, x)",
            "-1/2 * (-cosh(1 - 2 * x) + cosh(1 - 2 * x)^3 + 1/7 * cosh(1 - 2 * x)^7 - 3/5 * cosh(1 - 2 * x)^5)",
        ),
        (
            "integrate(cosh(1 - 2*x)^7, x)",
            "-1/2 * (sinh(1 - 2 * x) + sinh(1 - 2 * x)^3 + 1/7 * sinh(1 - 2 * x)^7 + 3/5 * sinh(1 - 2 * x)^5)",
        ),
    ];

    for (input, expected) in public_cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected, "input: {input}");
        assert!(
            required.is_empty(),
            "input: {input}, required: {required:?}"
        );
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "{input} should verify through the bounded public residual route"
        );
    }
}
#[test]
fn integrate_contract_linear_tanh_uses_abs_log_cosh_and_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(tanh(2*x + 1), x)");

    assert_eq!(result, "1/2 * ln(cosh(2 * x + 1))");
    assert_eq!(
        required,
        Vec::<String>::new(),
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_affine_hyperbolic_power_times_derivative_product() {
    let public_cases = [
        ("integrate(sinh(x)^2*cosh(x), x)", "1/3 * sinh(x)^3"),
        (
            "integrate(2*cosh(2*x + 1)*sinh(2*x + 1)^2, x)",
            "1/3 * sinh(2 * x + 1)^3",
        ),
        (
            "integrate(-2*cosh(2*x + 1)*sinh(2*x + 1)^2, x)",
            "-1/3 * sinh(2 * x + 1)^3",
        ),
        ("integrate(sinh(x)*cosh(x)^2, x)", "1/3 * cosh(x)^3"),
        ("integrate(-sinh(x)*cosh(x)^2, x)", "-1/3 * cosh(x)^3"),
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
fn integrate_contract_hyperbolic_non_linear_argument_without_cofactor_remains_residual() {
    let (result, required) = evaluated_integral_with_required_conditions("integrate(sinh(x^2), x)");

    assert_eq!(result, "integrate(sinh(x^2), x)");
    assert!(
        required.is_empty(),
        "unsupported hyperbolic integral should not invent conditions: {required:?}"
    );

    let (result, required) = evaluated_integral_with_required_conditions("integrate(tanh(x^2), x)");

    assert_eq!(result, "integrate(tanh(x^2), x)");
    assert!(
        required.is_empty(),
        "unsupported tanh integral should not invent conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_hyperbolic_ratio_uses_tanh_kernel_and_preserves_source_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(sinh(2*x + 1)/cosh(2*x + 1), x)");

    assert_eq!(result, "1/2 * ln(cosh(2 * x + 1))");
    assert_eq!(
        required,
        Vec::<String>::new(),
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_hyperbolic_coth_ratio_uses_log_sinh_and_preserves_source_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(cosh(2*x + 1)/sinh(2*x + 1), x)");

    assert_eq!(result, "1/2 * ln(|sinh(2 * x + 1)|)");
    assert_eq!(
        required,
        vec!["sinh(2 * x + 1) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_hyperbolic_tanh_reciprocal_uses_log_sinh_and_preserves_domains() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(1/tanh(2*x + 1), x)");

    assert_eq!(result, "1/2 * ln(|sinh(2 * x + 1)|)");
    assert_eq!(
        required,
        vec!["sinh(2 * x + 1) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn integrate_contract_negative_scaled_hyperbolic_reciprocal_primitives_keep_domain_signal() {
    let cases = [
        (
            "integrate(-2*x*sinh(x^2)/cosh(x^2)^2, x)",
            "1 / cosh(x^2)",
            "\\frac{1}{\\cosh({x}^{2})}",
            serde_json::json!([]),
            "diff(integrate(-2*x*sinh(x^2)/cosh(x^2)^2, x), x) + 2*x*sinh(x^2)/cosh(x^2)^2",
        ),
        (
            "integrate(-2*x*cosh(x^2)/sinh(x^2)^2, x)",
            "1 / sinh(x^2)",
            "\\frac{1}{\\sinh({x}^{2})}",
            serde_json::json!(["sinh(x^2) ≠ 0"]),
            "diff(integrate(-2*x*cosh(x^2)/sinh(x^2)^2, x), x) + 2*x*cosh(x^2)/sinh(x^2)^2",
        ),
    ];

    for (input, expected_result, expected_latex, expected_required, residual) in cases {
        let (wire, stderr) = cli_eval_json_with_stderr(input);
        assert!(
            stderr.is_empty(),
            "unexpected stderr for negative scaled hyperbolic reciprocal primitive: {stderr}"
        );
        assert_eq!(wire["result"], expected_result);
        assert_eq!(wire["result_latex"], expected_latex);
        assert_eq!(wire["required_display"], expected_required);

        let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual);
        assert!(
            residual_stderr.is_empty(),
            "unexpected stderr for negative scaled hyperbolic reciprocal residual: {residual_stderr}"
        );
        assert_eq!(residual_wire["result"], "0");
        assert_eq!(residual_wire["required_display"], expected_required);
    }
}
#[test]
fn integrate_contract_inverse_hyperbolic_sqrt_reciprocal_kernels_invert_diff_output() {
    for (input, expected, expected_required) in [
        (
            "integrate(-1/(2*x*sqrt(x+1)), x)",
            "asinh(sqrt(1 / x))",
            vec!["x > 0".to_string()],
        ),
        (
            "integrate(-1/(x*sqrt(x+4)), x)",
            "asinh(sqrt(4 / x))",
            vec!["x > 0".to_string()],
        ),
        (
            "integrate(-1/(2*(x+1)*sqrt(x+2)), x)",
            "asinh(sqrt(1 / (x + 1)))",
            vec!["x > -1".to_string()],
        ),
        (
            "integrate(-1/(2*sqrt(x)*(x-1)), x)",
            "atanh(sqrt(1 / x))",
            vec!["x > 1".to_string()],
        ),
        (
            "integrate(-1/(sqrt(x)*(x-4)), x)",
            "atanh(sqrt(4 / x))",
            vec!["x > 4".to_string()],
        ),
        (
            "integrate(3/(2*sqrt(3*x)*(3-x)), x)",
            "atanh(sqrt(3 / x))",
            vec!["x > 3".to_string()],
        ),
        (
            "integrate(-3/(2*sqrt(3*x+1)*(3*x)), x)",
            "atanh(sqrt(1 / (3 * x + 1)))",
            vec!["x > 0".to_string()],
        ),
        (
            "integrate(-2/((2*x+1)*sqrt(2*x+5)), x)",
            "asinh(sqrt(4 / (2 * x + 1)))",
            vec!["x > -1/2".to_string()],
        ),
        (
            "integrate(-1/(sqrt(2)*(x+3)*sqrt(x+5)), x)",
            "asinh(sqrt(2 / (x + 3)))",
            vec!["x > -3".to_string()],
        ),
        (
            "integrate(-2/(sqrt(2)*(2*x+1)*sqrt(2*x+3)), x)",
            "asinh(sqrt(2 / (2 * x + 1)))",
            vec!["x > -1/2".to_string()],
        ),
        (
            "integrate(1/((6-2*x)*sqrt(8-2*x)), x)",
            "1/2 * asinh(sqrt(1 / (3 - x))) * sqrt(2)",
            vec!["x < 3".to_string()],
        ),
        (
            "integrate(-1/(x*sqrt(2*x+4)), x)",
            "atanh(sqrt(2 / (x + 2)))",
            vec!["x > 0".to_string()],
        ),
    ] {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected, "input: {input}");
        assert_eq!(
            required, expected_required,
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert_antiderivative_verifies(input);

        let nested = format!("diff({input}, x)");
        let (nested_derivative, nested_required) = evaluated_expr_with_required_conditions(&nested);
        assert!(
            !nested_derivative.contains("integrate("),
            "nested derivative should not leave an integration residual for {input}: {nested_derivative}"
        );
        assert_eq!(
            nested_required, expected_required,
            "nested derivative should preserve domain conditions for {input}"
        );
    }
}
#[test]
fn integrate_contract_inverse_hyperbolic_sqrt_reciprocal_kernels_integrate_directly_without_denominator_expansion(
) {
    for input in [
        "integrate(-1/(2*sqrt(x)*(x-1)), x)",
        "integrate(-1/(sqrt(x)*(x-4)), x)",
    ] {
        let step_rules = evaluated_integral_step_rules(input);
        assert_eq!(
            step_rules,
            vec!["Symbolic Integration".to_string()],
            "inverse-hyperbolic sqrt reciprocal kernel should integrate directly without pre-expanding the denominator: {step_rules:?}"
        );
        assert_antiderivative_verifies(input);
    }
}
#[test]
fn integrate_contract_ambiguous_inverse_hyperbolic_family_selection_verifies_by_diff() {
    let input = "integrate(3/(sqrt(5-3*x)*(1-3*x)), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "asinh(sqrt(4 / (1 - 3 * x)))");
    assert_eq!(
        required,
        vec!["x < 1/3".to_string()],
        "ambiguous inverse-hyperbolic primitive should keep the witnessed positive denominator"
    );
    assert_antiderivative_verifies(input);
    assert_antiderivative_equiv_verifies(input);

    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(3/(sqrt(5-3*x)*(1-3*x)), x), x) - 3/(sqrt(5-3*x)*(1-3*x))",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["x < 1/3".to_string()],
        "nested ambiguous primitive verification should preserve the selected-domain condition"
    );
}
#[test]
fn integrate_contract_sqrt_chain_hyperbolic_tangent_logs_verify() {
    let cases = [
        (
            "integrate(tanh(sqrt(x))/(2*sqrt(x)), x)",
            "ln(cosh(sqrt(x)))",
            "tanh(sqrt(x)) / (2 * sqrt(x))",
            vec!["x > 0"],
        ),
        (
            "integrate(1/(2*sqrt(x)*tanh(sqrt(x))), x)",
            "ln(|sinh(sqrt(x))|)",
            "1 / (2 * tanh(sqrt(x)) * sqrt(x))",
            vec!["sinh(sqrt(x)) ≠ 0", "x > 0"],
        ),
        (
            "integrate(tanh(sqrt(2*x))/sqrt(2*x), x)",
            "ln(cosh(sqrt(2 * x)))",
            "tanh(sqrt(2 * x)) / sqrt(2 * x)",
            vec!["x > 0"],
        ),
        (
            "integrate(tanh(sqrt(3*x+1))*3/(2*sqrt(3*x+1)), x)",
            "ln(cosh(sqrt(3 * x + 1)))",
            "3 * tanh(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))",
            vec!["x > -1/3"],
        ),
        (
            "integrate(3/(2*sqrt(3*x+1)*tanh(sqrt(3*x+1))), x)",
            "ln(|sinh(sqrt(3 * x + 1))|)",
            "3 / (2 * tanh(sqrt(3 * x + 1)) * sqrt(3 * x + 1))",
            vec!["sinh(sqrt(3 * x + 1)) ≠ 0", "x > -1/3"],
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
        assert_rendered_antiderivative_verifies(input, &result);

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
fn integrate_contract_sqrt_chain_hyperbolic_tangent_logs_explain_u_and_du() {
    for (input, expected_result, expected_required_display, expected_rule_title) in [
        (
            "integrate(tanh(sqrt(3*x+1))*3/(2*sqrt(3*x+1)), x)",
            "ln(cosh(sqrt(3·x + 1)))",
            serde_json::json!(["x > -1/3"]),
            "Usar la regla de tanh(u) -> ln(cosh(u))",
        ),
        (
            "integrate(3/(2*sqrt(3*x+1)*tanh(sqrt(3*x+1))), x)",
            "ln(|sinh(sqrt(3·x + 1))|)",
            serde_json::json!(["sinh(sqrt(3·x + 1)) ≠ 0", "x > -1/3"]),
            "Usar la regla de 1/tanh(u) -> ln|sinh(u)|",
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
            "sqrt-chain hyperbolic log trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
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
            "sqrt-chain hyperbolic log table should not use only the generic substitution substep for {input}: {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }
}
#[test]
fn integrate_contract_sqrt_chain_hyperbolic_reciprocal_squares_verify() {
    let cases = [
        (
            "integrate(1/(2*sqrt(x)*cosh(sqrt(x))^2), x)",
            "tanh(sqrt(x))",
            vec!["x > 0"],
        ),
        (
            "integrate(1/(2*sqrt(x)*sinh(sqrt(x))^2), x)",
            "-1 / tanh(sqrt(x))",
            vec!["x > 0", "sinh(sqrt(x)) ≠ 0"],
        ),
        (
            "integrate((2*x)^(-1/2)/cosh(sqrt(2*x))^2, x)",
            "tanh(sqrt(2 * x))",
            vec!["x > 0"],
        ),
        (
            "integrate(3/(2*sqrt(3*x+1)*cosh(sqrt(3*x+1))^2), x)",
            "tanh(sqrt(3 * x + 1))",
            vec!["x > -1/3"],
        ),
        (
            "integrate(3/(2*sqrt(3*x+1)*sinh(sqrt(3*x+1))^2), x)",
            "-1 / tanh(sqrt(3 * x + 1))",
            vec!["x > -1/3", "sinh(sqrt(3 * x + 1)) ≠ 0"],
        ),
        (
            "integrate(k/(2*sqrt(x)*cosh(sqrt(x)-b)^2), x)",
            "tanh(sqrt(x) - b) * k",
            vec!["x > 0"],
        ),
        (
            "integrate(k/(2*sqrt(x)*sinh(sqrt(x)-b)^2), x)",
            "-k / tanh(sqrt(x) - b)",
            vec!["x > 0", "sinh(sqrt(x) - b) ≠ 0"],
        ),
    ];

    for (input, expected_result, expected_conditions) in cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);

        assert_eq!(result, expected_result, "unexpected result for {input}");
        assert_eq!(
            required, expected_conditions,
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert_antiderivative_verifies(input);
    }

    let nested_cases = [
        (
            "diff(integrate(3/(2*sqrt(3*x+1)*cosh(sqrt(3*x+1))^2), x), x)",
            "3 / (2 * sqrt(3 * x + 1) * cosh(sqrt(3 * x + 1))^2)",
            vec!["x > -1/3"],
        ),
        (
            "diff(integrate(3/(2*sqrt(3*x+1)*sinh(sqrt(3*x+1))^2), x), x)",
            "3 / (2 * sqrt(3 * x + 1) * sinh(sqrt(3 * x + 1))^2)",
            vec!["x > -1/3", "sinh(sqrt(3 * x + 1)) ≠ 0"],
        ),
    ];

    for (input, expected_result, expected_conditions) in nested_cases {
        let (result, required) = evaluated_expr_with_required_conditions(input);

        assert_eq!(
            result, expected_result,
            "unexpected nested result for {input}"
        );
        assert!(
            !result.contains("^(-1/2)") && !result.contains("^(1/2)"),
            "nested diff/integrate presentation should keep explicit sqrt forms, got: {result}"
        );
        assert_eq!(
            required, expected_conditions,
            "unexpected nested required_conditions for {input}: {required:?}"
        );
    }

    let step_summaries = evaluated_expr_step_summaries(
        "diff(integrate(3/(2*sqrt(3*x+1)*cosh(sqrt(3*x+1))^2), x), x)",
    );
    assert!(
        step_summaries
            .iter()
            .any(|(description, rule_name, importance)| description
                == "Post-calculus presentation"
                && rule_name == "Present calculus result in compact form"
                && *importance >= ImportanceLevel::Medium),
        "post-calculus presentation should be visible in normal step output: {step_summaries:?}"
    );
    assert!(
        !step_summaries.iter().any(|(_, rule_name, _)| {
            rule_name == "Pull Constant From Fraction"
                || rule_name == "Simplify Multiplication with Division"
        }),
        "post-calculus presentation should hide mechanical fraction cleanup before the compact result: {step_summaries:?}"
    );
}
#[test]
fn integrate_contract_sqrt_chain_hyperbolic_reciprocal_derivatives_verify() {
    let cases = [
        (
            "integrate(sinh(sqrt(x))/(2*sqrt(x)*cosh(sqrt(x))^2), x)",
            "-1 / cosh(sqrt(x))",
            vec!["x > 0"],
        ),
        (
            "integrate(cosh(sqrt(x))/(2*sqrt(x)*sinh(sqrt(x))^2), x)",
            "-1 / sinh(sqrt(x))",
            vec!["x > 0", "sinh(sqrt(x)) ≠ 0"],
        ),
        (
            "integrate(sinh(sqrt(2*x))/(sqrt(2*x)*cosh(sqrt(2*x))^2), x)",
            "-1 / cosh(sqrt(2 * x))",
            vec!["x > 0"],
        ),
        (
            "integrate(3*sinh(sqrt(3*x+1))/(2*sqrt(3*x+1)*cosh(sqrt(3*x+1))^2), x)",
            "-1 / cosh(sqrt(3 * x + 1))",
            vec!["x > -1/3"],
        ),
        (
            "integrate(3*cosh(sqrt(3*x+1))/(2*sqrt(3*x+1)*sinh(sqrt(3*x+1))^2), x)",
            "-1 / sinh(sqrt(3 * x + 1))",
            vec!["x > -1/3", "sinh(sqrt(3 * x + 1)) ≠ 0"],
        ),
        (
            "integrate(k*sinh(sqrt(x)-b)/(2*sqrt(x)*cosh(sqrt(x)-b)^2), x)",
            "-k / cosh(sqrt(x) - b)",
            vec!["x > 0"],
        ),
        (
            "integrate(k*cosh(sqrt(x)-b)/(2*sqrt(x)*sinh(sqrt(x)-b)^2), x)",
            "-k / sinh(sqrt(x) - b)",
            vec!["x > 0", "sinh(sqrt(x) - b) ≠ 0"],
        ),
    ];

    for (input, expected_result, expected_conditions) in cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);

        assert_eq!(result, expected_result, "unexpected result for {input}");
        assert_eq!(
            required, expected_conditions,
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert_antiderivative_verifies(input);
    }

    let nested_cases = [
        (
            "diff(integrate(3*sinh(sqrt(3*x+1))/(2*sqrt(3*x+1)*cosh(sqrt(3*x+1))^2), x), x)",
            "3 * sinh(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1) * cosh(sqrt(3 * x + 1))^2)",
            vec!["x > -1/3"],
        ),
        (
            "diff(integrate(3*cosh(sqrt(3*x+1))/(2*sqrt(3*x+1)*sinh(sqrt(3*x+1))^2), x), x)",
            "3 * cosh(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1) * sinh(sqrt(3 * x + 1))^2)",
            vec!["x > -1/3", "sinh(sqrt(3 * x + 1)) ≠ 0"],
        ),
    ];

    for (input, expected_result, expected_conditions) in nested_cases {
        let (result, required) = evaluated_expr_with_required_conditions(input);

        assert_eq!(
            result, expected_result,
            "unexpected nested result for {input}"
        );
        assert!(
            !result.contains("^(-1/2)") && !result.contains("^(1/2)"),
            "nested diff/integrate presentation should keep explicit sqrt forms, got: {result}"
        );
        assert_eq!(
            required, expected_conditions,
            "unexpected nested required_conditions for {input}: {required:?}"
        );
    }
}
#[test]
fn integrate_contract_sqrt_chain_hyperbolic_reciprocal_tables_explain_u_and_du() {
    for (input, expected_result, expected_required_display, expected_rule_title) in [
        (
            "integrate(3/(2*sqrt(3*x+1)*cosh(sqrt(3*x+1))^2), x)",
            "tanh(sqrt(3·x + 1))",
            serde_json::json!(["x > -1/3"]),
            "Usar la regla de 1/cosh(u)^2 -> tanh(u)",
        ),
        (
            "integrate(3/(2*sqrt(3*x+1)*sinh(sqrt(3*x+1))^2), x)",
            "-1 / tanh(sqrt(3·x + 1))",
            serde_json::json!(["x > -1/3", "sinh(sqrt(3·x + 1)) ≠ 0"]),
            "Usar la regla de 1/sinh(u)^2 -> -1/tanh(u)",
        ),
        (
            "integrate(3*sinh(sqrt(3*x+1))/(2*sqrt(3*x+1)*cosh(sqrt(3*x+1))^2), x)",
            "-1 / cosh(sqrt(3·x + 1))",
            serde_json::json!(["x > -1/3"]),
            "Usar la regla de sinh(u)/cosh(u)^2 -> -1/cosh(u)",
        ),
        (
            "integrate(3*cosh(sqrt(3*x+1))/(2*sqrt(3*x+1)*sinh(sqrt(3*x+1))^2), x)",
            "-1 / sinh(sqrt(3·x + 1))",
            serde_json::json!(["x > -1/3", "sinh(sqrt(3·x + 1)) ≠ 0"]),
            "Usar la regla de cosh(u)/sinh(u)^2 -> -1/sinh(u)",
        ),
        (
            "integrate(k/(2*sqrt(x)*sinh(sqrt(x)-b)^2), x)",
            "-k / tanh(sqrt(x) - b)",
            serde_json::json!(["x > 0", "sinh(sqrt(x) - b) ≠ 0"]),
            "Usar la regla de 1/sinh(u)^2 -> -1/tanh(u)",
        ),
        (
            "integrate(k*sinh(sqrt(x)-b)/(2*sqrt(x)*cosh(sqrt(x)-b)^2), x)",
            "-k / cosh(sqrt(x) - b)",
            serde_json::json!(["x > 0"]),
            "Usar la regla de sinh(u)/cosh(u)^2 -> -1/cosh(u)",
        ),
        (
            "integrate(k*cosh(sqrt(x)-b)/(2*sqrt(x)*sinh(sqrt(x)-b)^2), x)",
            "-k / sinh(sqrt(x) - b)",
            serde_json::json!(["x > 0", "sinh(sqrt(x) - b) ≠ 0"]),
            "Usar la regla de cosh(u)/sinh(u)^2 -> -1/sinh(u)",
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
            "sqrt-chain hyperbolic reciprocal trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
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
            "sqrt-chain hyperbolic reciprocal table should not use only the generic substitution substep for {input}: {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }
}
