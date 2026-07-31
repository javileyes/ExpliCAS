use super::*;

/// G1 sub-cycle Cap. D (2026-07-15): the pure cube-root cubic `x^3 - k`
/// (rational `k > 0`, `∛k` irrational) is kept WHOLE so the residue solve stays
/// over ℚ; the real split `(x - c)(x^2 + cx + c^2)` with `c = ∛k` and residues
/// in ℚ(c) (exact `[1, c, c²]` triples) appears only in the render. Every
/// radical in the result is FLAT (`∛k` and `√3` — the arctan radius is written
/// `√3·c`), so the degree-aware relation tower (`t³ = k`, `s² = 3`) confirms
/// the differentiate-back and gates emission. All emissions below were also
/// verified numerically against an independent oracle (sympy, 30 digits).
/// This graduates the LAST named probe of Phase-1 exit criterion #1.
#[test]
fn integrate_contract_cbrt_cubic_integrates_x3_minus_2_family() {
    for input in [
        "integrate(1/(x^3-2), x)",
        "integrate(x/(x^3-2), x)",
        "integrate((x^2+1)/(x^3-2), x)",
        "integrate(1/(x^3-5), x)",
        "integrate(1/((x-1)*(x^3-2)), x)",
        // Negative k (the sign-invariant branch): c = cbrt(-2) < 0.
        "integrate(1/(x^3+2), x)",
        "integrate(x/(x^3+2), x)",
    ] {
        let (result, _required) = evaluated_integral_with_required_conditions(input);
        assert!(
            !result.contains("integrate("),
            "should integrate via the cube-root split: {input} -> {result}"
        );
        assert!(
            result.contains("cbrt("),
            "expected the cube-root surd in the render: {input} -> {result}"
        );
    }
    // The named G1 exit probe carries the real pole and the √3·∛2 arctan radius.
    let (probe, required) = evaluated_integral_with_required_conditions("integrate(1/(x^3-2), x)");
    assert!(
        probe.contains("ln(|x - cbrt(2)|)"),
        "expected the real-pole log term: {probe}"
    );
    assert!(
        probe.contains("sqrt(3)"),
        "expected the √3 arctan radius factor: {probe}"
    );
    assert!(
        required.iter().any(|c| c.contains("cbrt(2)")),
        "the irrational real pole x ≠ ∛2 must surface as a required condition: {required:?}"
    );
}
/// G1 sub-cycle Cap. B (2026-07-14): an irreducible-over-ℝ even quartic
/// `x^4 + p*x^2 + r` appearing as a FACTOR (e.g. `x^4-x^2+1` in `x^6+1`,
/// `x^4+1` in `x^8-1`) is kept whole so the partial-fraction residue solve stays
/// over ℚ; the surd split `(x^2+a*x+s)(x^2-a*x+s)` appears only in the render.
/// See docs/G1_RATIONAL_INTEGRATION_SCOPING.md. The named constant-numerator
/// targets are confirmed by differentiate-back (they are also in
/// REPRESENTATIVE_ANTIDERIVATIVE_VERIFICATION_CASES); odd/even numerator variants
/// are verified numerically here by support + render form (the combined-surd
/// differentiate-back is verifier-limited, a documented residual, never a wrong
/// answer).
#[test]
fn integrate_contract_irreducible_even_quartic_factor_integrates_via_surd_split() {
    // Named G1 exit probes plus odd-numerator variants routed through the surd
    // even-quartic render (x^2/(x^6+1) takes the cleaner u=x^3 substitution and
    // is covered elsewhere).
    for input in [
        "integrate(1/(x^6+1), x)",
        "integrate(1/(x^8-1), x)",
        "integrate(x^3/(x^4-x^2+1), x)",
        "integrate(x/(x^6+1), x)",
    ] {
        let (result, _required) = evaluated_integral_with_required_conditions(input);
        assert!(
            !result.contains("integrate("),
            "should integrate via the even-quartic surd split: {input} -> {result}"
        );
        assert!(
            result.contains("sqrt(3)") || result.contains("sqrt(2)"),
            "expected a surd coefficient in the even-quartic render: {input} -> {result}"
        );
    }

    // The standalone constant-numerator even quartic stays owned by the earlier
    // symmetric-surd closed form (byte-identical), not the general route.
    let (standalone, _) =
        evaluated_integral_with_required_conditions("integrate(1/(x^4-x^2+1), x)");
    assert!(
        standalone.contains("arctan(sqrt(3) + 2") || standalone.contains("arctan(sqrt(3)+2"),
        "standalone even quartic keeps its symmetric-surd render: {standalone}"
    );

    // Differentiate-back for the surd targets through the PUBLIC pipeline (the
    // full simplifier folds the sqrt(3)/sqrt(2) normal forms; the in-process
    // sweep in REPRESENTATIVE_ANTIDERIVATIVE_VERIFICATION_CASES disables
    // "Double Angle Identity", under which these surd forms cannot reach 0 and
    // the rationalization rules grind unboundedly — the same harness weakness
    // G1 Cap. A documented, which is why only sqrt(2) cases live in that sweep).
    for input in ["1/(x^6+1)", "1/(x^8-1)"] {
        let residual =
            integrate_call_antiderivative_residual_result(&format!("integrate({input}, x)"));
        assert_eq!(
            residual, "0",
            "public differentiate-back must confirm integrate({input}, x)"
        );
    }
}
/// G1 Cap. E-iv (2026-07-16): the UNIVERSAL closure. Denominators whose
/// Rothstein-Trager resultant R(t) is squarefree emit the clean parameterized
/// form `root_sum(R(t), t, t·ln(x − w(t)))` — including the Galois-obstructed
/// class where NO radical closed form exists (`1/(x^5-x-1)` is the canonical
/// S₅ case, and SymPy expands the "solvable" `1/(x^3-x-1)` into an unreadable
/// Cardano nested-radical monster; the clean RootSum strictly beats it).
/// Rational roots of R peel into elementary real logs (`1/(x^7-1)` →
/// `(1/7)·ln|x−1| + root_sum(...)`). Emission is gated on the EXACT
/// rational-identity proof (Newton traces at 2·bound+1 exact points), and the
/// only public conditions are the integrand's own pole conditions — the
/// root_sum node is a binder, opaque to domain inference (no bound-variable
/// leaks).
#[test]
fn integrate_contract_rootsum_universal_closure_emits() {
    for (input, fragment) in [
        ("integrate(1/(x^3-x-1), x)", "root_sum("),
        ("integrate(1/(x^5-x-1), x)", "root_sum("),
        ("integrate(x/(x^4+x+1), x)", "root_sum("),
        (
            "integrate(1/(x^5-2), x)",
            "root_sum(1 - 50000 * t^5, t, t * ln(x - 10 * t))",
        ),
        ("integrate(1/(x^3-3*x-1), x)", "root_sum("),
        // Mixed: the rational root of R peels into an elementary log.
        ("integrate(1/(x^7-1), x)", "ln(|x - 1|)"),
    ] {
        let (result, _required) = evaluated_integral_with_required_conditions(input);
        assert!(
            !result.contains("integrate("),
            "should emit via the RootSum closure: {input} -> {result}"
        );
        assert!(
            result.contains(fragment),
            "expected `{fragment}` in: {input} -> {result}"
        );
        assert!(
            !result.contains("root_sum") || result.contains("t * ln("),
            "the summand must be the clean t·ln(x − w(t)) form: {result}"
        );
    }

    // The bound variable must never leak into public conditions.
    let (_, required) = evaluated_integral_with_required_conditions("integrate(1/(x^3-x-1), x)");
    assert_eq!(
        required,
        vec!["x^3 - x - 1 ≠ 0"],
        "only the integrand pole condition may surface: {required:?}"
    );
}
#[test]
fn integrate_contract_positive_half_power_antiderivatives_render_as_sqrt() {
    for (input, expected) in [
        ("integrate(x/sqrt(x^2+3), x)", "sqrt(x^2 + 3)"),
        (
            "integrate((2*x+1)/sqrt(x^2+x+1), x)",
            "2 * sqrt(x^2 + x + 1)",
        ),
    ] {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected, "input: {input}");
        assert!(
            required.is_empty(),
            "positive half-power presentation should not add domain conditions for {input}: {required:?}"
        );
        assert!(
            !result.contains("^(1/2)"),
            "post-integration presentation should prefer sqrt notation: {result}"
        );
        assert_antiderivative_verifies(input);
    }
}
#[test]
fn integrate_contract_reciprocal_sqrt_antiderivative_rationalized_residual_collapses() {
    let input = "integrate((2*x+1)/(x^2+x+1)^(3/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    assert_eq!(result, "-2 / sqrt(x^2 + x + 1)");
    assert!(
        required.is_empty(),
        "positive quadratic reciprocal-root primitive should not add domain conditions: {required:?}"
    );

    let residual = "integrate((2*x+1)/(x^2+x+1)^(3/2), x) - (-2*sqrt(x^2+x+1)/(x^2+x+1))";
    let (residual_result, residual_required) =
        evaluated_integral_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "positive quadratic rationalized residual should not add domain conditions: {residual_required:?}"
    );

    let step_summaries = evaluated_expr_step_summaries(residual);
    assert!(
        step_summaries
            .iter()
            .any(
                |(description, rule, _)| description == "Post-calculus residual simplification"
                    || rule == "Post-calculus residual simplification"
            ),
        "expected a visible post-calculus residual simplification step, got {step_summaries:?}"
    );
}
#[test]
fn integrate_contract_positive_sqrt_antiderivative_rationalized_residual_collapses() {
    let input = "integrate((2*x+1)/sqrt(x^2+x+1), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    assert_eq!(result, "2 * sqrt(x^2 + x + 1)");
    assert!(
        required.is_empty(),
        "positive quadratic sqrt primitive should not add domain conditions: {required:?}"
    );

    let residual = "integrate((2*x+1)/sqrt(x^2+x+1), x) - 2*(x^2+x+1)/sqrt(x^2+x+1)";
    let (residual_result, residual_required) =
        evaluated_integral_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "positive quadratic rationalized sqrt residual should not add domain conditions: {residual_required:?}"
    );

    let mismatch = "integrate((2*x+1)/sqrt(x^2+x+1), x) - 3*(x^2+x+1)/sqrt(x^2+x+1)";
    let (mismatch_result, _) = evaluated_integral_with_required_conditions(mismatch);
    assert_ne!(
        mismatch_result, "0",
        "mismatched rationalized sqrt scale must not collapse"
    );
}
#[test]
fn integrate_contract_positive_sqrt_antiderivative_rationalized_residual_survives_quotient_wrapper()
{
    let residual = "(integrate((2*x+1)/sqrt(x^2+x+1), x) - 2*(x^2+x+1)/sqrt(x^2+x+1))/(x+2)";
    let (residual_result, residual_required) =
        evaluated_integral_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -2".to_string()],
        "quotient-wrapped rationalized sqrt residual should preserve denominator domain"
    );

    let step_summaries = evaluated_expr_step_summaries(residual);
    assert!(
        step_summaries
            .iter()
            .any(
                |(description, rule, _)| description == "Post-calculus residual simplification"
                    || rule == "Post-calculus residual simplification"
            ),
        "expected a visible post-calculus residual simplification step, got {step_summaries:?}"
    );
}
#[test]
fn integrate_contract_positive_sqrt_antiderivative_rationalized_residual_survives_shifted_reciprocal_difference(
) {
    let residual =
        "1/((integrate((2*x+1)/sqrt(x^2+x+1), x) - 2*(x^2+x+1)/sqrt(x^2+x+1))+x+2)-1/(x+2)";
    let (residual_result, residual_required) =
        evaluated_integral_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -2".to_string()],
        "shifted reciprocal residual should preserve the compact denominator domain"
    );

    let mismatch =
        "1/((integrate((2*x+1)/sqrt(x^2+x+1), x) - 3*(x^2+x+1)/sqrt(x^2+x+1))+x+2)-1/(x+2)";
    let (mismatch_result, _) = evaluated_integral_with_required_conditions(mismatch);
    assert_ne!(
        mismatch_result, "0",
        "mismatched rationalized sqrt scale must not collapse under reciprocal shift"
    );

    let step_summaries = evaluated_expr_step_summaries(residual);
    assert!(
        step_summaries
            .iter()
            .any(
                |(description, rule, _)| description == "Post-calculus residual simplification"
                    || rule == "Post-calculus residual simplification"
            ),
        "expected a visible post-calculus residual simplification step, got {step_summaries:?}"
    );
}
#[test]
fn integrate_contract_reciprocal_shifted_root_product_residual_compacts_without_timeout() {
    let input =
        "1/((diff(integrate(1/(sqrt(2*x)*sqrt(2*x+6)), x), x) - 1/(sqrt(2*x)*sqrt(2*x+6))) + x + 2) - 1/(x+2)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert!(
        stderr.is_empty(),
        "unexpected stderr for reciprocal shifted root-product residual: {stderr}"
    );
    assert_eq!(wire["result"], "0");
    assert_eq!(wire["required_display"], serde_json::json!(["x > 0"]));
}
#[test]
fn integrate_contract_reciprocal_shifted_root_product_residual_additive_noise_compacts_without_timeout(
) {
    let input =
        "1/((diff(integrate(1/(sqrt(2*x)*sqrt(2*x+6)), x), x) - 1/(sqrt(2*x)*sqrt(2*x+6))) + x + 2) - 1/(x+2) + y - y";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert!(
        stderr.is_empty(),
        "unexpected stderr for reciprocal shifted root-product residual with additive noise: {stderr}"
    );
    assert_eq!(wire["result"], "0");
    assert_eq!(wire["required_display"], serde_json::json!(["x > 0"]));
}
#[test]
fn integrate_contract_root_product_residual_reciprocal_shifted_quotient_compacts_without_timeout() {
    let input =
        "1/(((diff(integrate(1/(sqrt(2*x)*sqrt(2*x+6)), x), x) - 1/(sqrt(2*x)*sqrt(2*x+6))) + x + 2)/(x+2)) - 1";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert!(
        stderr.is_empty(),
        "unexpected stderr for root-product residual reciprocal shifted quotient: {stderr}"
    );
    assert_eq!(wire["result"], "0");
    assert_eq!(wire["required_display"], serde_json::json!(["x > 0"]));
}
#[test]
fn integrate_contract_root_product_residual_reciprocal_shifted_quotient_additive_noise_compacts_without_timeout(
) {
    let input =
        "1/(((diff(integrate(1/(sqrt(2*x)*sqrt(2*x+6)), x), x) - 1/(sqrt(2*x)*sqrt(2*x+6))) + x + 2)/(x+2)) + y - (1+y)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert!(
        stderr.is_empty(),
        "unexpected stderr for root-product residual reciprocal shifted quotient with additive noise: {stderr}"
    );
    assert_eq!(wire["result"], "0");
    assert_eq!(wire["required_display"], serde_json::json!(["x > 0"]));
}
#[test]
fn integrate_contract_root_product_residual_squared_shifted_quotient_compacts_without_timeout() {
    let input =
        "(((diff(integrate(1/(sqrt(2*x)*sqrt(2*x+6)), x), x) - 1/(sqrt(2*x)*sqrt(2*x+6))) + x + 2)/(x+2))^2 - 1";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert!(
        stderr.is_empty(),
        "unexpected stderr for root-product residual squared shifted quotient: {stderr}"
    );
    assert_eq!(wire["result"], "0");
    assert_eq!(wire["required_display"], serde_json::json!(["x > 0"]));
}
#[test]
fn integrate_contract_cbrt_elementary_calculus_differentiates_and_integrates() {
    // cbrt is now first-class elementary calculus, like sqrt: it differentiates
    // and integrates as x^(1/3) (Pow form, no domain condition -- cbrt is defined
    // on all reals). cbrt stays Function(Cbrt) for display and the limit rules.
    // The derivative 1/(3 x^(2/3)) is undefined at x=0 (cbrt has a vertical
    // tangent there), so diff carries x != 0; the antiderivative x^(4/3) is
    // defined everywhere, so integrate carries no condition.
    for (input, expected, expected_required) in [
        (
            "diff(cbrt(x), x)",
            "1/3 * x^(-2/3)",
            vec!["x ≠ 0".to_string()],
        ),
        (
            "diff(5*cbrt(x), x)",
            "5/3 * x^(-2/3)",
            vec!["x ≠ 0".to_string()],
        ),
        ("integrate(cbrt(x), x)", "3/4 * x^(4/3)", Vec::new()),
        ("integrate(cbrt(x)^2, x)", "3/5 * x^(5/3)", Vec::new()),
        ("integrate(2*cbrt(x), x)", "3/2 * x^(4/3)", Vec::new()),
    ] {
        let (result, required) = evaluated_expr_with_required_conditions(input);
        assert_eq!(result, expected, "result for {input}");
        assert_eq!(required, expected_required, "required for {input}");
    }

    // Soundness in Pow form: diff(antiderivative) - x^(1/3) and the derivative
    // minus its closed form both reduce to 0 (the engine does not yet simplify
    // x^(1/3) == cbrt(x), so the cbrt-form round-trip stays open -- a separate
    // simplification gap, not an integration error).
    for (expr, want) in [
        ("diff(integrate(cbrt(x), x), x) - x^(1/3)", "0"),
        ("diff(integrate(cbrt(x)^2, x), x) - x^(2/3)", "0"),
        ("diff(cbrt(x), x) - 1/3*x^(-2/3)", "0"),
    ] {
        assert_eq!(
            evaluated_expr_with_required_conditions(expr).0,
            want,
            "soundness {expr}"
        );
    }

    // Boundary: a non-linear radicand stays an honest residual (no power-rule
    // target), bare cbrt(x) still displays as cbrt(x) (not lowered), and the
    // cube-root LIMIT rule (Function(Cbrt)) is untouched.
    for residual in ["integrate(cbrt(x^2+1), x)", "integrate(cbrt(x^2), x)"] {
        assert!(
            evaluated_expr_with_required_conditions(residual)
                .0
                .starts_with("integrate("),
            "{residual} should stay an honest residual"
        );
    }
    assert_eq!(
        evaluated_expr_with_required_conditions("cbrt(x)").0,
        "cbrt(x)"
    );
    // sqrt and x^(1/3) are untouched.
    assert_eq!(
        evaluated_expr_with_required_conditions("integrate(sqrt(x), x)").0,
        "2/3 * sqrt(x) * x"
    );
}
#[test]
fn integrate_contract_affine_sqrt_product_public_diff_verifies_antiderivative() {
    for (input, expected_diff, expected_required) in [
        (
            "integrate(1/(sqrt(x)*sqrt(x+5)), x)",
            "1 / (sqrt(x) * sqrt(x + 5))",
            vec!["x > 0".to_string()],
        ),
        (
            "integrate(1/(sqrt(x-1)*sqrt(x+2)), x)",
            "1 / (sqrt(x + 2) * sqrt(x - 1))",
            vec!["x > 1".to_string()],
        ),
        (
            "integrate(1/(sqrt(x)*sqrt(2*x+4)), x)",
            "1 / (sqrt(x) * sqrt(2 * x + 4))",
            vec!["x > 0".to_string()],
        ),
        (
            "integrate(1/(sqrt(2*x)*sqrt(2*x+6)), x)",
            "1 / (sqrt(2) * sqrt(x) * sqrt(2 * x + 6))",
            vec!["x > 0".to_string()],
        ),
    ] {
        let diff_input = format!("diff({input}, x)");
        let (result, required) = evaluated_expr_with_required_conditions(&diff_input);

        assert_eq!(result, expected_diff, "input: {diff_input}");
        assert_eq!(required, expected_required, "input: {diff_input}");
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "affine sqrt-product antiderivative should verify through the public residual"
        );
    }
}
#[test]
fn integrate_contract_polynomial_inverse_sqrt_public_diff_keeps_sqrt_presentation() {
    for (input, expected_diff, expected_required) in [
        (
            "integrate(1/sqrt(1-x^2), x)",
            "1 / sqrt(1 - x^2)",
            vec!["-1 < x < 1".to_string()],
        ),
        (
            "integrate(1/sqrt(4-(x+1)^2), x)",
            "1 / sqrt(4 - (x + 1)^2)",
            vec!["-3 < x < 1".to_string()],
        ),
        (
            "integrate(2*x/sqrt(1+x^4), x)",
            "2 * x / sqrt(x^4 + 1)",
            Vec::new(),
        ),
    ] {
        let direct_diff = format!("diff({input}, x)");
        let (result, required) = evaluated_expr_with_required_conditions(&direct_diff);

        assert_eq!(result, expected_diff, "input: {direct_diff}");
        assert_eq!(required, expected_required, "input: {direct_diff}");

        let residual = format!("{direct_diff} - {expected_diff}");
        let (residual_result, residual_required) =
            evaluated_expr_with_required_conditions(&residual);
        assert_eq!(residual_result, "0", "input: {residual}");
        assert_eq!(residual_required, expected_required, "input: {residual}");
    }
}
#[test]
fn integrate_contract_beta_sqrt_product_kernel_preserves_open_domain_and_verifies() {
    for (input, expected_result, expected_derivative) in [
        (
            "integrate(1/(sqrt(x)*sqrt(1-x)), x)",
            "arcsin(2 * x - 1)",
            "1 / (sqrt(x) * sqrt(1 - x))",
        ),
        (
            "integrate(1/(2*sqrt(x)*sqrt(1-x)), x)",
            "1/2 * arcsin(2 * x - 1)",
            "1 / (2 * sqrt(x) * sqrt(1 - x))",
        ),
    ] {
        let (result, mut required) = evaluated_integral_with_required_conditions(input);
        let mut expected_required = vec!["x < 1".to_string(), "x > 0".to_string()];
        required.sort();
        expected_required.sort();

        assert_eq!(result, expected_result, "input: {input}");
        assert_eq!(
            required, expected_required,
            "sqrt-product beta kernel should preserve both open denominator conditions"
        );
        assert_antiderivative_verifies(input);

        let rendered_derivative = format!("diff({result}, x)");
        let (derivative_result, mut nested_required) =
            evaluated_expr_with_required_conditions(&rendered_derivative);
        nested_required.sort();
        assert_eq!(derivative_result, expected_derivative, "input: {input}");
        assert_eq!(
            nested_required, expected_required,
            "rendered beta-kernel derivative should preserve both open denominator conditions"
        );

        let direct_diff = format!("diff({input}, x)");
        let (direct_result, mut direct_required) =
            evaluated_expr_with_required_conditions(&direct_diff);
        direct_required.sort();
        assert_eq!(direct_result, expected_derivative, "input: {direct_diff}");
        assert_eq!(
            direct_required, expected_required,
            "direct diff(integrate(...)) beta-kernel presentation should preserve both open denominator conditions"
        );
    }

    let direct_affine = "diff(integrate(1/(sqrt(2*x+1)*sqrt(3-2*x)), x), x)";
    let (direct_affine_result, mut direct_affine_required) =
        evaluated_expr_with_required_conditions(direct_affine);
    let mut expected_affine_required = vec!["x < 3/2".to_string(), "x > -1/2".to_string()];
    direct_affine_required.sort();
    expected_affine_required.sort();
    assert_eq!(
        direct_affine_result,
        "1 / (sqrt(2 * x + 1) * sqrt(3 - 2 * x))"
    );
    assert_eq!(
        direct_affine_required, expected_affine_required,
        "affine beta-kernel presentation should preserve both open denominator conditions"
    );

    let direct_symbolic = "diff(integrate(a/(2*sqrt(x)*sqrt(1-x)), x), x)";
    let (direct_symbolic_result, mut direct_symbolic_required) =
        evaluated_expr_with_required_conditions(direct_symbolic);
    let mut expected_required = vec!["x < 1".to_string(), "x > 0".to_string()];
    direct_symbolic_required.sort();
    expected_required.sort();
    assert_eq!(direct_symbolic_result, "a / (2 * sqrt(x) * sqrt(1 - x))");
    assert_eq!(
        direct_symbolic_required, expected_required,
        "symbolic beta-kernel presentation should preserve both open denominator conditions"
    );

    let direct_symbolic_affine = "diff(integrate(a/(sqrt(2*x+1)*sqrt(3-2*x)), x), x)";
    let (direct_symbolic_affine_result, mut direct_symbolic_affine_required) =
        evaluated_expr_with_required_conditions(direct_symbolic_affine);
    direct_symbolic_affine_required.sort();
    assert_eq!(
        direct_symbolic_affine_result,
        "a / (sqrt(2 * x + 1) * sqrt(3 - 2 * x))"
    );
    assert_eq!(
        direct_symbolic_affine_required, expected_affine_required,
        "symbolic affine beta-kernel presentation should preserve both open denominator conditions"
    );

    let (nested_residual, mut nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(a/(2*sqrt(x)*sqrt(1-x)), x), x) - a/(2*sqrt(x)*sqrt(1-x))",
    );
    nested_required.sort();
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required, expected_required,
        "nested symbolic beta-kernel verification should preserve both open denominator conditions"
    );
}
#[test]
fn integrate_contract_affine_sqrt_product_derivative_inverse() {
    let input = "integrate((3*x+5)/(2*sqrt(x+2)), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "sqrt(x + 2) * (x + 1)");
    assert_eq!(
        required,
        vec!["x > -2".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_antiderivative_equiv_verifies(input);

    let input = "integrate((3*x+1)/(2*sqrt(x)), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "sqrt(x) * (x + 1)");
    assert_eq!(
        required,
        vec!["x > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_equiv_verifies(input);

    let input = "integrate((1/2)*(3*x+1)*x^(-1/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "sqrt(x) * (x + 1)");
    assert_eq!(
        required,
        vec!["x > 0".to_string()],
        "unexpected product-form required_conditions: {required:?}"
    );
    assert_antiderivative_equiv_verifies(input);

    let input = "integrate((2-3*x)*(3-2*x)^(-1/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "sqrt(3 - 2 * x) * (x + 1)");
    assert_eq!(
        required,
        vec!["x < 3/2".to_string()],
        "unexpected negative-slope product-form required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}
#[test]
fn integrate_contract_affine_numerator_over_sqrt_quadratic_splits_by_linearity() {
    // `(a·x+b)/√(quadratic)` used to decline even though both pieces work (`x/√q → √q`,
    // `c/√q → asinh/arcsin/acosh`): the integrand normalizes to `(q)^(-1/2)·(a·x+b)` (a product, not
    // a Div) and `expand` will not distribute over the fractional-power factor. Distributing the sum
    // over the radical before integration now closes the whole asinh/arcsin/acosh family. Each is
    // certified by differentiating the antiderivative back to the integrand.
    for input in [
        "integrate((x+1)/sqrt(x^2+1), x)",     // asinh(x) + sqrt(x^2+1)
        "integrate((2*x+3)/sqrt(x^2+1), x)",   // 2 sqrt(x^2+1) + 3 asinh(x)
        "integrate((x+1)/sqrt(1-x^2), x)",     // arcsin(x) - sqrt(1-x^2)
        "integrate((x-2)/sqrt(x^2-1), x)",     // acosh family
        "integrate((x^2+x+1)/sqrt(x^2+1), x)", // higher-degree numerator also splits
    ] {
        assert_antiderivative_verifies(input);
    }
    // The single-term owners are unchanged (still verify), and a product WITH sqrt (positive half) is
    // untouched by the reciprocal-only split.
    for input in [
        "integrate(x/sqrt(x^2+1), x)",
        "integrate(1/sqrt(x^2+1), x)",
        "integrate((x+1)*sqrt(x^2+1), x)",
    ] {
        assert_antiderivative_verifies(input);
    }
}
/// The RootSum frontier narrated from the RESULT itself — no engine signature
/// touched. `integrate(1/(x^5-x-1), x)` answered with a correct RootSum and
/// published ZERO substeps: not even the method's name, let alone why no closed
/// form in radicals exists. These are the rows the corpus advertises as the
/// differentiator against sympy.
#[test]
fn integrate_contract_root_sum_names_its_method_and_its_resolvent() {
    let subs = |input: &str| -> Vec<(String, String)> {
        let (wire, _) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);
        wire["steps"]
            .as_array()
            .expect("steps with --steps on")
            .iter()
            .filter_map(|step| step["substeps"].as_array())
            .flatten()
            .map(|s| {
                (
                    s["title"].as_str().unwrap_or_default().to_string(),
                    s["after"].as_str().unwrap_or_default().to_string(),
                )
            })
            .collect()
    };

    let bare = subs("integrate(1/(x^5-x-1), x)");
    assert!(
        bare.iter()
            .any(|(t, _)| t.contains("no son expresables por radicales")),
        "the method must be named: {bare:?}"
    );
    assert!(
        bare.iter().any(|(_, a)| a.starts_with("R(t) = ")),
        "and the concrete resolvent published: {bare:?}"
    );

    // With an elementary part, the split comes first.
    let split = subs("integrate(1/(x^7-1), x)");
    assert_eq!(
        split.first().map(|(t, _)| t.as_str()),
        Some("Separar la parte de raíces racionales"),
        "{split:?}"
    );
}
