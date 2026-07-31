use super::*;

#[test]
fn integrate_contract_invalid_real_domain_integrands_return_undefined() {
    for input in [
        "integrate(log(1,2), x)",
        "integrate(log(1,x), x)",
        "integrate(log(-2,x), x)",
        "integrate(ln(0), x)",
        "integrate(sqrt(-1), x)",
        "integrate(infinity, x)",
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);
        assert!(
            stderr.is_empty(),
            "invalid real-domain integrand should not emit stderr for {input}: {stderr}"
        );
        assert_eq!(
            wire["result"].as_str(),
            Some("undefined"),
            "invalid real-domain integrand should not produce a formal primitive for {input}: {wire:?}"
        );
        assert_eq!(
            wire["required_display"]
                .as_array()
                .expect("required_display should be an array")
                .len(),
            0,
            "undefined integrands should not introduce conditional assumptions for {input}"
        );

        let step_text = wire["steps"].to_string();
        assert!(
            step_text.contains("undefined"),
            "integration trace should expose the undefined result for {input}: {step_text}"
        );
        assert!(
            !step_text.contains("x·log") && !step_text.contains("infinity·x"),
            "integration trace should not present a false formal primitive for {input}: {step_text}"
        );
    }

    let (result, required) = evaluated_integral_with_required_conditions("integrate(2, x)");
    assert_eq!(result, "2 * x");
    assert!(
        required.is_empty(),
        "finite constants should still integrate without domain assumptions: {required:?}"
    );
}
#[test]
fn integrate_contract_supported_antiderivatives_verify_by_differentiation() {
    for input in REPRESENTATIVE_ANTIDERIVATIVE_VERIFICATION_CASES {
        assert_antiderivative_verifies(input);
    }
}
#[test]
#[ignore = "exhaustive debug verification is intentionally slower; CI runs the representative smoke test"]
fn integrate_contract_supported_antiderivatives_verify_by_differentiation_exhaustive() {
    let public_residual_inputs: Vec<_> = [
        "integrate(2*x + 3, x)",
        "integrate((3*x)^2, x)",
        "integrate(sin(2*x), x)",
        "integrate(-(sin(x)), x)",
        "integrate(cos(x), x)",
        "integrate(exp(3*x + 1), x)",
        "integrate(x*exp(x), x)",
        "integrate(x^2*exp(x), x)",
        "integrate(x^3*exp(x), x)",
        "integrate((2*x+3)*exp(2*x+1), x)",
        "integrate((x^2+x+1)*exp(2*x+1), x)",
        "integrate((x^3+x)*exp(2*x+1), x)",
        "integrate((x+1)*exp((3*x+2)/2), x)",
        "integrate((x+1)*exp((2-3*x)/2), x)",
        "integrate(x*sin(x), x)",
        "integrate(x*cos(x), x)",
        "integrate(x^2*sin(x), x)",
        "integrate(x^2*cos(x), x)",
        "integrate((2*x+3)*sin(2*x+1), x)",
        "integrate((2*x+3)*cos(2*x+1), x)",
        "integrate((x+1)*sin((3*x+2)/2), x)",
        "integrate((x+1)*cos((3*x+2)/2), x)",
        "integrate((x+1)*sin((2-3*x)/2), x)",
        "integrate((x+1)*cos((2-3*x)/2), x)",
        "integrate(x*sinh(x), x)",
        "integrate(x*cosh(x), x)",
        "integrate((2*x+3)*sinh(2*x+1), x)",
        "integrate((2*x+3)*cosh(2*x+1), x)",
        "integrate((x+1)*sinh((3*x+2)/2), x)",
        "integrate((x+1)*cosh((3*x+2)/2), x)",
        "integrate((x+1)*sinh((2-3*x)/2), x)",
        "integrate((x+1)*cosh((2-3*x)/2), x)",
        "integrate(x^2*sinh(x), x)",
        "integrate(x^2*cosh(x), x)",
        "integrate(2*x*exp(x^2), x)",
        "integrate(2*x*cos(x^2), x)",
        "integrate(2*x*sin(x^2), x)",
        "integrate(sinh(2*x + 1), x)",
        "integrate(cosh(2*x + 1), x)",
        "integrate(tanh(2*x + 1), x)",
        "integrate(2*x*sinh(x^2), x)",
        "integrate(2*x*cosh(x^2), x)",
        "integrate(2*x*tanh(x^2), x)",
        "integrate(cosh(x)/(1+sinh(x)^2), x)",
        "integrate(2*cosh(2*x+1)/(1+sinh(2*x+1)^2), x)",
        "integrate(sinh(x)/(1+cosh(x)^2), x)",
        "integrate(sinh(2*x + 1)/cosh(2*x + 1), x)",
        "integrate(cosh(2*x + 1)/sinh(2*x + 1), x)",
        "integrate(1/tanh(2*x + 1), x)",
        "integrate(2*x*cosh(x^2)/sinh(x^2), x)",
        "integrate(2*x/tanh(x^2), x)",
        "integrate(1/cosh(2*x + 1)^2, x)",
        "integrate(1/sinh(2*x + 1)^2, x)",
        "integrate(2*x/cosh(x^2)^2, x)",
        "integrate(2*x/sinh(x^2)^2, x)",
        "integrate(sinh(2*x + 1)/cosh(2*x + 1)^2, x)",
        "integrate(2*x*sinh(x^2)/cosh(x^2)^2, x)",
        "integrate(-x*sinh(x^2)/cosh(x^2)^2, x)",
        "integrate(cosh(2*x + 1)/sinh(2*x + 1)^2, x)",
        "integrate(2*x*cosh(x^2)/sinh(x^2)^2, x)",
        "integrate(-x*cosh(x^2)/sinh(x^2)^2, x)",
        "integrate(ln(x), x)",
        "integrate(ln(2*x+1), x)",
        "integrate(2*x*ln(x^2+1), x)",
        "integrate((2*x+1)*ln(x^2+x+1), x)",
        "integrate(2*x*ln(x^2-1), x)",
        "integrate(1/x, x)",
        "integrate(1/(2*x + 1), x)",
        "integrate(1/(x^2+1), x)",
        "integrate(2*x/(1+x^4), x)",
        "integrate(2*x/(4+x^4), x)",
        "integrate(2*x/(4-x^4), x)",
        "integrate(arcsin(x), x)",
        "integrate(arccos(x), x)",
        "integrate(arcsin(2*x+1), x)",
        "integrate(arccos(2*x+1), x)",
        "integrate(arcsin(1-2*x), x)",
        "integrate(arccos(1-2*x), x)",
        "integrate(arctan(x), x)",
        "integrate(arctan(2*x), x)",
        "integrate(arctan(1/x), x)",
        "integrate(arctan(1/(2*x+1)), x)",
        "integrate(arccot(x), x)",
        "integrate(arccot(2*x), x)",
        "integrate(arccot(2*x+1), x)",
        "integrate(arccot(1-2*x), x)",
        "integrate(asinh(x), x)",
        "integrate(asinh(2*x), x)",
        "integrate(asinh(2*x+1), x)",
        "integrate(asinh(1-2*x), x)",
        "integrate(atanh(x), x)",
        "integrate(atanh(2*x), x)",
        "integrate(atanh(2*x+1), x)",
        "integrate(atanh(1-2*x), x)",
        "integrate(1/(x^2-1), x)",
        "integrate(2*x/(x^2-1)^2, x)",
        "integrate((2*x+1)/(x^2+x-1)^2, x)",
        "integrate((2*x+1)/(3*(x^2+x-1)^2), x)",
        "integrate((2*x+1)/(x^2+x-1)^3, x)",
        "integrate((3*x+3/2)/(x^2+x-1)^4, x)",
        "integrate((8*x+2)/(3*(2*x^2+x-1)^3), x)",
        "integrate((2*x+1)/(x^4+2*x^3-x^2-2*x+1), x)",
        "integrate((2*x+1)/(x^6+3*x^5-5*x^3+3*x-1), x)",
        "integrate((2*x+1)/(4*x^6+12*x^5-20*x^3+12*x-4), x)",
        "integrate((2*x+1)/(-4*x^6-12*x^5+20*x^3-12*x+4), x)",
        "integrate((3*x+3/2)/(x^8+4*x^7+2*x^6-8*x^5-5*x^4+8*x^3+2*x^2-4*x+1), x)",
        "integrate(1/(x^5+5*x^4+10*x^3+10*x^2+5*x+1), x)",
        "integrate(2*x/(x^4-4), x)",
        "integrate((2*x + 1)/(x^2 + x - 1), x)",
        "integrate((4*x + 2)/(x^2 + x + 1), x)",
        "integrate(2*x/sqrt(4-x^4), x)",
        "integrate(2*x/sqrt(3-x^4), x)",
        "integrate(1/sqrt(4-(x+1)^2), x)",
        "integrate(2*x/sqrt(1+x^4), x)",
        "integrate(2*x/sqrt(4+x^4), x)",
        "integrate(2*x/sqrt(3+x^4), x)",
        "integrate(1/sqrt(4+(x+1)^2), x)",
        "integrate(3/(sqrt(5-3*x)*(1-3*x)), x)",
        "integrate(x/sqrt(x^2+1), x)",
        "integrate((3*x+5)/(2*sqrt(x+2)), x)",
        "integrate(2*x/sqrt(x^2-1), x)",
        "integrate(x*sqrt(x^2+1), x)",
        "integrate(2*x*sqrt(x^2-1), x)",
        "integrate(2*x*(x^2+1)^3, x)",
        "integrate(2*x*(x^2-1)^(3/2), x)",
        "integrate((x^2+1)^(-1/2), x)",
        "integrate(1/(x^2+1)^2, x)",
        "integrate(1/((x+1)^2+1)^2, x)",
        "integrate(1/(4*x^2+1)^2, x)",
        "integrate(sin(x)^2, x)",
        "integrate(cos(x)^2, x)",
        "integrate(sin(2*x + 1)^2, x)",
        "integrate(cos(2*x + 1)^2, x)",
        "integrate(sin(x)^3, x)",
        "integrate(cos(x)^3, x)",
        "integrate(sin(2*x + 1)^3, x)",
        "integrate(cos(2*x + 1)^3, x)",
        "integrate(sec(x)^2, x)",
        "integrate(csc(x)^2, x)",
        "integrate(sec(2*x + 1)^2, x)",
        "integrate(csc(2*x + 1)^2, x)",
        "integrate(x/(cos(x^2)^2), x)",
        "integrate(x^2/(sin(x^3)^2), x)",
        "integrate(sec(x), x)",
        "integrate(csc(x), x)",
        "integrate(csc(x)*cot(x), x)",
        "integrate(tan(2*x + 1), x)",
        "integrate(cot(2*x + 1), x)",
        "integrate(2*x*tan(x^2), x)",
        "integrate(3*x^2*cot(x^3), x)",
        "integrate(2*(x*sin(x^2)/cos(x^2)), x)",
        "integrate(3*(x^2*cos(x^3)/sin(x^3)), x)",
        "integrate(sec((3*x+2)/2), x)",
        "integrate(csc((2-3*x)/2), x)",
        "integrate(sec(2*x + 1)*tan(2*x + 1), x)",
        "integrate(x*sec(x^2)*tan(x^2), x)",
        "integrate(2*x*sec(x^2)*tan(x^2), x)",
        "integrate(-x*sec(x^2)*tan(x^2), x)",
        "integrate(csc(2*x + 1)*cot(2*x + 1), x)",
        "integrate(x^2*csc(x^3)*cot(x^3), x)",
        "integrate(3*x^2*csc(x^3)*cot(x^3), x)",
        "integrate(-x^2*csc(x^3)*cot(x^3), x)",
    ]
    .into_iter()
    .filter(|input| {
        assert_antiderivative_verifies(input) == AntiderivativeVerificationRoute::PublicResidual
    })
    .collect();

    assert_eq!(
        public_residual_inputs,
        vec![
            "integrate(x^2*exp(x), x)",
            "integrate(x^3*exp(x), x)",
            "integrate((x^2+x+1)*exp(2*x+1), x)",
            "integrate((x^3+x)*exp(2*x+1), x)",
            "integrate(x*sin(x), x)",
            "integrate(x*cos(x), x)",
            "integrate(x^2*sin(x), x)",
            "integrate(x^2*cos(x), x)",
            "integrate((2*x+3)*sin(2*x+1), x)",
            "integrate((2*x+3)*cos(2*x+1), x)",
            "integrate((x+1)*sin((3*x+2)/2), x)",
            "integrate((x+1)*cos((3*x+2)/2), x)",
            "integrate((x+1)*sin((2-3*x)/2), x)",
            "integrate((x+1)*cos((2-3*x)/2), x)",
            "integrate(x*sinh(x), x)",
            "integrate(x*cosh(x), x)",
            "integrate((2*x+3)*sinh(2*x+1), x)",
            "integrate((2*x+3)*cosh(2*x+1), x)",
            "integrate((x+1)*sinh((3*x+2)/2), x)",
            "integrate((x+1)*cosh((3*x+2)/2), x)",
            "integrate((x+1)*sinh((2-3*x)/2), x)",
            "integrate((x+1)*cosh((2-3*x)/2), x)",
            "integrate(x^2*sinh(x), x)",
            "integrate(x^2*cosh(x), x)",
            "integrate(sinh(2*x + 1)/cosh(2*x + 1), x)",
            "integrate(cosh(2*x + 1)/sinh(2*x + 1), x)",
            "integrate(1/tanh(2*x + 1), x)",
            "integrate(2*x*cosh(x^2)/sinh(x^2), x)",
            "integrate(2*x/tanh(x^2), x)",
            "integrate(1/cosh(2*x + 1)^2, x)",
            "integrate(1/sinh(2*x + 1)^2, x)",
            "integrate(2*x/cosh(x^2)^2, x)",
            "integrate(2*x/sinh(x^2)^2, x)",
            "integrate(sinh(2*x + 1)/cosh(2*x + 1)^2, x)",
            "integrate(2*x*sinh(x^2)/cosh(x^2)^2, x)",
            "integrate(-x*sinh(x^2)/cosh(x^2)^2, x)",
            "integrate(cosh(2*x + 1)/sinh(2*x + 1)^2, x)",
            "integrate(2*x*cosh(x^2)/sinh(x^2)^2, x)",
            "integrate(-x*cosh(x^2)/sinh(x^2)^2, x)",
            "integrate(2*x*ln(x^2+1), x)",
            "integrate((2*x+1)*ln(x^2+x+1), x)",
            "integrate((3*x+5)/(2*sqrt(x+2)), x)",
            "integrate(sec(x), x)",
            "integrate(csc(x), x)",
            "integrate(tan(2*x + 1), x)",
            "integrate(cot(2*x + 1), x)",
            "integrate(sec((3*x+2)/2), x)",
            "integrate(csc((2-3*x)/2), x)",
        ],
        "exhaustive debug antiderivative verification should only use the bounded public residual route for known public-residual families"
    );
}
#[test]
fn integrate_contract_unsupported_non_elementary_residual() {
    assert_eq!(
        simplified_integral("integrate(sin(x^2), x)"),
        "integrate(sin(x^2), x)"
    );
}
/// `equiv` answers a bare `true`/`false` with no steps to qualify it, so a
/// `false` that is only false IN THE REAL DOMAIN reads as "this identity is
/// wrong". `equiv(e^(i*pi), -1)` printed `false` with no warning at all: the
/// student reads that Euler's identity does not hold. (With
/// `--value-domain complex` the same call answers `true`.)
#[test]
fn equiv_false_over_the_reals_says_so_when_the_argument_is_complex() {
    for input in [
        "equiv(e^(i*pi), -1)",
        "equiv(i^2, -1)",
        "equiv(sin(i), i*sinh(1))",
    ] {
        let (wire, _) = cli_eval_json_with_stderr(input);
        assert_eq!(wire["result"], "false", "{input}");
        let warnings = wire["warnings"]
            .as_array()
            .expect("warnings array")
            .iter()
            .filter_map(|w| w["rule"].as_str())
            .collect::<Vec<_>>();
        assert!(
            warnings.contains(&"Imaginary Usage Warning"),
            "{input} answered false over the reals without saying why: {warnings:?}"
        );
    }

    // A real-domain equivalence must not gain noise.
    for input in ["equiv(x+x, 2*x)", "equiv(ln(x*y), ln(x)+ln(y))"] {
        let (wire, _) = cli_eval_json_with_stderr(input);
        assert_eq!(wire["result"], "true", "{input}");
        assert!(
            wire["warnings"].as_array().expect("warnings").is_empty(),
            "{input} should carry no imaginary warning: {:?}",
            wire["warnings"]
        );
    }
}
