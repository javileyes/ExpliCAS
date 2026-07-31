use super::*;

/// G1 Cap. E-iv-d3 (2026-07-16): DEFINITE integrals over algorithmic-backend
/// antiderivatives emit the EXACT bound-substituted difference — including
/// the root_sum closure (the S₅ definite!) — certified pole-free on the
/// interval by the exact Sturm count. A pole strictly inside gives an
/// honest `undefined` (the integral diverges); the substituted difference
/// keeps the backend's __hold protection (without it the simplifier mangled
/// the mixed log/arctan constants — a wrong value caught in-cycle).
#[test]
fn integrate_contract_definite_over_backend_antiderivatives() {
    for (input, fragment) in [
        ("integrate(1/(x^5-x-1), x, 2, 3)", "root_sum("),
        ("integrate(1/(x^3-x-1), x, 2, 3)", "root_sum("),
        ("integrate(1/(x^4-4), x, 3, 5)", "arctan("),
        ("integrate(1/(x^7-1), x, 2, 3)", "root_sum("),
    ] {
        let (result, _required) = evaluated_integral_with_required_conditions(input);
        assert!(
            !result.contains("integrate("),
            "definite must emit the exact difference: {input} -> {result}"
        );
        assert!(
            result.contains(fragment),
            "expected `{fragment}` in: {input} -> {result}"
        );
    }
    // Algebraic-constant pole conditions (x − ∛2, x − 5^(1/4)) are located
    // against the interval by the exact constant-bounds oracle: outside the
    // interval emits, strictly inside reports divergence (E-iv-d4).
    for (input, fragment) in [
        ("integrate(1/(x^3-2), x, 2, 3)", "cbrt("),
        ("integrate(1/(x^4-5), x, 3, 4)", "arctan("),
    ] {
        let (result, _required) = evaluated_integral_with_required_conditions(input);
        assert!(
            !result.contains("integrate(") && result.contains(fragment),
            "algebraic pole outside the interval must emit: {input} -> {result}"
        );
    }
    // A pole strictly inside the interval: honest divergence, never a value.
    for input in [
        "integrate(1/(x^3-x-1), x, 1, 2)",
        "integrate(1/(x^4-5), x, 1, 2)", // pole at 5^(1/4) ≈ 1.495
    ] {
        let (result, _required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(
            result, "undefined",
            "pole inside the interval must report divergence: {input}"
        );
    }
}
#[test]
fn integrate_contract_linear_times_definite_quadratic_handles_negative_orientation() {
    let input = "integrate(1/((-x-1)*(x^2+1)), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert!(
        !result.starts_with("integrate("),
        "expected a proper antiderivative for negatively oriented linear times definite quadratic, got {result}"
    );
    assert!(
        result.contains("ln(|x + 1|)")
            && result.contains("ln(x^2 + 1)")
            && result.contains("arctan(x)"),
        "expected negatively oriented log-linear plus quadratic log/arctan terms, got {result}"
    );
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string()],
        "unexpected negative-orientation linear-definite-quadratic required_conditions: {required:?}"
    );

    let residual = "diff(integrate(1/((-x-1)*(x^2+1)), x), x) - 1/((-x-1)*(x^2+1))";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string()],
        "negative-orientation verification should keep only the linear-pole domain"
    );
    assert_rendered_antiderivative_verifies(input, &result);
}
#[test]
fn integrate_contract_repeated_linear_times_definite_quadratic_partial_fraction() {
    let input = "integrate((x+2)/((x+1)^2*(x^2+1)), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert!(
        !result.starts_with("integrate("),
        "expected a proper antiderivative for repeated-linear times definite quadratic, got {result}"
    );
    assert!(
        result.contains("ln(|x + 1|)")
            && result.contains("ln(x^2 + 1)")
            && result.contains("arctan(x)")
            && result.contains("1 / (2 * (x + 1))"),
        "expected repeated-pole reciprocal plus quadratic log/arctan terms, got {result}"
    );
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string()],
        "unexpected repeated-linear-definite-quadratic required_conditions: {required:?}"
    );

    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);
    assert!(
        stderr.is_empty(),
        "repeated-pole partial-fraction trace should stay quiet\nstderr:\n{stderr}"
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
    let decomposition_latex = substeps
        .iter()
        .find(|substep| substep["title"] == "Descomponer en fracciones parciales")
        .and_then(|substep| substep["after_latex"].as_str())
        .expect("expected concrete partial-fraction decomposition substep");
    assert!(
        decomposition_latex.contains("- \\frac{x - \\frac{1}{2}}{{x}^{2} + 1}")
            && !decomposition_latex.contains("+ \\frac{-"),
        "negative quadratic numerator should render as subtraction, got {decomposition_latex}"
    );

    let residual = "diff(integrate((x+2)/((x+1)^2*(x^2+1)), x), x) - (x+2)/((x+1)^2*(x^2+1))";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string()],
        "repeated-linear-definite-quadratic verification should keep only the linear-pole domain"
    );
    assert_rendered_antiderivative_verifies(input, &result);
}
#[test]
fn integrate_contract_cubic_repeated_linear_times_definite_quadratic_partial_fraction() {
    let input = "integrate((x+2)/((x+1)^3*(x^2+1)), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert!(
        !result.starts_with("integrate("),
        "expected a proper antiderivative for cubic repeated-linear times definite quadratic, got {result}"
    );
    assert!(
        result.contains("ln(|x + 1|)")
            && result.contains("ln(x^2 + 1)")
            && result.contains("arctan(x)")
            && result.contains("1 / (x + 1)")
            && result.contains("1 / (4 * (x + 1)^2)"),
        "expected cubic repeated-pole reciprocal plus quadratic log/arctan terms, got {result}"
    );
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string()],
        "unexpected cubic repeated-linear-definite-quadratic required_conditions: {required:?}"
    );

    assert_antiderivative_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);
}
#[test]
fn integrate_contract_shifted_definite_quadratic_cubic_repeated_pole_verifies_by_diff() {
    let input = "integrate(1/((x+1)^3*(x^2+2*x+2)), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert!(
        !result.starts_with("integrate("),
        "expected a proper antiderivative for shifted quadratic cubic repeated pole, got {result}"
    );
    assert!(
        result.contains("ln(|x + 1|)")
            && result.contains("ln(x^2 + 2 * x + 2)")
            && result.contains("1 / (2 * (x + 1)^2)"),
        "expected shifted quadratic log and compact reciprocal terms, got {result}"
    );
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string()],
        "unexpected shifted-quadratic repeated-pole required_conditions: {required:?}"
    );

    let residual = "diff(integrate(1/((x+1)^3*(x^2+2*x+2)), x), x) - 1/((x+1)^3*(x^2+2*x+2))";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string()],
        "shifted-quadratic repeated-pole verification should keep only the linear-pole domain"
    );
    assert_rendered_antiderivative_verifies(input, &result);
}
#[test]
fn integrate_contract_improper_rational_partial_fractions_use_polynomial_division() {
    let input = "integrate(x^2/(x+1), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert!(
        !result.starts_with("integrate("),
        "expected a proper antiderivative for linear improper rational integrand, got {result}"
    );
    assert!(
        (result.contains("x^2 / 2") || result.contains("1/2 * x^2"))
            && result.contains("- x")
            && result.contains("ln(|x + 1|)")
            && !result.contains("+ -"),
        "expected polynomial division plus linear-log remainder, got {result}"
    );
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string()],
        "unexpected linear improper rational required_conditions: {required:?}"
    );

    let residual = "diff(integrate(x^2/(x+1), x), x) - x^2/(x+1)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string()],
        "linear improper rational nested verification should preserve the source denominator domain"
    );

    let input = "integrate(x^2/(2*x+2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result, "1/2 * ln(|x + 1|) + 1/4 * x^2 - 1/2 * x",
        "scaled linear improper rational should fold nested rational factors"
    );
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string()],
        "unexpected scaled linear improper rational required_conditions: {required:?}"
    );

    let residual = "diff(integrate(x^2/(2*x+2), x), x) - x^2/(2*x+2)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string()],
        "scaled linear improper rational nested verification should preserve the source denominator domain"
    );

    let input = "integrate(x^2/(-2*x-2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result, "1/2 * x - 1/2 * ln(|x + 1|) - 1/4 * x^2",
        "negative scaled linear improper rational should fold nested rational factors"
    );
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string()],
        "unexpected negative scaled linear improper rational required_conditions: {required:?}"
    );

    let residual = "diff(integrate(x^2/(-2*x-2), x), x) - x^2/(-2*x-2)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string()],
        "negative scaled linear improper rational nested verification should preserve the source denominator domain"
    );

    let input = "integrate((x^3+3*x+5)/(x^3-x^2-x+1), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert!(
        !result.starts_with("integrate("),
        "expected a proper antiderivative, got {result}"
    );
    assert!(
        result.contains("ln(|x + 1|)") && result.contains("ln(|x - 1|)"),
        "expected logarithmic partial-fraction remainder terms, got {result}"
    );
    assert!(
        result.contains("9 / (2 * (x - 1))")
            && !result.contains("9/2 / (x - 1)")
            && !result.contains("+ -"),
        "expected a clean repeated-pole rational remainder term, got {result}"
    );
    assert!(
        result.contains("+ x") && !result.contains("1 * x"),
        "expected the polynomial quotient term to omit the unit factor, got {result}"
    );
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "unexpected required_conditions: {required:?}"
    );

    let residual = "diff(integrate((x^3+3*x+5)/(x^3-x^2-x+1), x), x) - (x^3+3*x+5)/(x^3-x^2-x+1)";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "nested verification should preserve the source denominator domain"
    );
    let (equivalent, mut equiv_required) = evaluated_equiv_with_required_conditions(
        "diff(integrate((x^3+3*x+5)/(x^3-x^2-x+1), x), x)",
        "(x^3+3*x+5)/(x^3-x^2-x+1)",
    );
    equiv_required.sort();
    assert!(
        equivalent,
        "public equivalence should verify the improper rational antiderivative by differentiation"
    );
    assert_eq!(
        equiv_required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "public equivalence should preserve the source denominator domain"
    );

    let input = "integrate((x^3+3*x+5)/(-x^3+x^2+x-1), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert!(
        !result.starts_with("integrate("),
        "expected a proper antiderivative for negative orientation, got {result}"
    );
    assert!(
        result.contains("ln(|x + 1|)") && result.contains("ln(|x - 1|)"),
        "expected logarithmic partial-fraction remainder terms for negative orientation, got {result}"
    );
    assert!(
        result.contains("9 / (2 * (x - 1))")
            && !result.contains("9/2 / (x - 1)")
            && result.contains("- x")
            && !result.contains("+ -"),
        "expected clean oriented polynomial and repeated-pole terms, got {result}"
    );
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "unexpected negative-orientation required_conditions: {required:?}"
    );

    let residual = "diff(integrate((x^3+3*x+5)/(-x^3+x^2+x-1), x), x) - (x^3+3*x+5)/(-x^3+x^2+x-1)";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "negative-orientation nested verification should preserve the source denominator domain"
    );

    let input = "integrate((x^5+3*x+5)/(x^3-x^2-x+1), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert!(
        (result.contains("x^3 / 3") || result.contains("1/3 * x^3"))
            && !result.contains("x^(1 + 2) / (1 + 2)"),
        "higher-degree quotient should render folded polynomial power terms, got {result}"
    );
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "unexpected higher-degree quotient required_conditions: {required:?}"
    );

    let residual = "diff(integrate((x^5+3*x+5)/(x^3-x^2-x+1), x), x) - (x^5+3*x+5)/(x^3-x^2-x+1)";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "higher-degree quotient nested verification should preserve the source denominator domain"
    );
}
#[test]
fn integrate_contract_improper_positive_quadratic_uses_polynomial_division() {
    let input = "integrate((x^3+x+1)/(x^2+1), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "arctan(x) + x^2 / 2");
    assert!(
        required.is_empty(),
        "positive quadratic denominator should not add synthetic required conditions: {required:?}"
    );

    let residual = "diff(integrate((x^3+x+1)/(x^2+1), x), x) - (x^3+x+1)/(x^2+1)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "nested verification should not invent denominator conditions for a positive quadratic"
    );
}
#[test]
fn integrate_contract_divergent_improper_area_function_is_undefined() {
    // int_0^x ln(t)/t dt DIVERGES at t = 0 (antiderivative (ln t)^2/2 -> +inf):
    // it must be `undefined`, not a form carrying a silent infinity^k term that
    // a later diff drops into a false finite derivative.
    assert_eq!(
        simplified_integral("integrate(ln(t)/t, t, 0, x)"),
        "undefined"
    );
    assert_eq!(
        simplified_integral("integrate(ln(t)^2/t, t, 0, x)"),
        "undefined"
    );
    // The hyperbolic/exp ~1/t family (antiderivative ln|...| -> -inf at 0) too:
    // the boundary ln(|sinh(0)|) / ln(|(e^0-1)/...|) folds numerically to ln(0).
    assert_eq!(
        simplified_integral("integrate(coth(t), t, 0, x)"),
        "undefined"
    );
    assert_eq!(
        simplified_integral("integrate(1/(e^t-1), t, 0, x)"),
        "undefined"
    );
    // and the derivative of the divergent area function is undefined, not ln(x)/x.
    assert_eq!(
        simplified_integral("diff(integrate(ln(t)/t, t, 0, x), x)"),
        "undefined"
    );

    // CONVERGENT improper and ordinary area functions are unaffected.
    assert_eq!(
        simplified_integral("integrate(ln(t), t, 0, x)"),
        "x * ln(x) - x"
    );
    assert_eq!(simplified_integral("integrate(t^2, t, 0, x)"), "1/3 * x^3");
    assert_eq!(
        simplified_integral("integrate(1/(1+t^2), t, 0, x)"),
        "arctan(x)"
    );
}
/// `N/(a² − x²)` definite-integrated over an interval strictly OUTSIDE (−a, a):
/// the `atanh` antiderivative is real only inside (−a, a), so the FTC path used
/// to decline; the equal `−N/(x² − a²)` has a real log antiderivative off the
/// poles. Numeric ground truths (Simpson, verified): `∫₂³ 1/(1−x²) = ½ln(2/3)`,
/// `∫₃⁵ 1/(4−x²) = ¼ln(7/15)`, `∫₄⁵ 1/(9−x²) = ⅙ln(4/7)`.
#[test]
fn definite_rational_reciprocal_difference_of_squares_outside_atanh_domain() {
    // Outside (−a, a), pole-free: now evaluates via the log form.
    assert_eq!(
        simplified_integral("integrate(1/(1-x^2), x, 2, 3)"),
        "1/2 * ln(2/3)"
    );
    // Negative interval reflects (even integrand) then evaluates identically.
    assert_eq!(
        simplified_integral("integrate(1/(1-x^2), x, -3, -2)"),
        "1/2 * ln(2/3)"
    );
    assert_eq!(
        simplified_integral("integrate(1/(4-x^2), x, 3, 5)"),
        "1/4 * ln(7/15)"
    );
    assert_eq!(
        simplified_integral("integrate(1/(9-x^2), x, 4, 5)"),
        "1/6 * ln(4/7)"
    );
    // Scaled leading coefficient: 1/(2−2x²) = (1/2)/(1−x²).
    assert_eq!(
        simplified_integral("integrate(1/(2-2*x^2), x, 2, 3)"),
        "1/4 * ln(2/3)"
    );
    // Reversed bounds negate the oriented integral.
    assert_eq!(
        simplified_integral("integrate(1/(1-x^2), x, 3, 2)"),
        "1/2 * ln(3/2)"
    );
}
/// The gate is strict: an interval INSIDE (−a, a) keeps the elegant `atanh`
/// form, a pole crossing stays `undefined`, and non-`a²−x²` shapes are left to
/// their own owners (arctan for `a²+x²`, residual for a linear term).
#[test]
fn definite_rational_atanh_domain_gate_is_strict() {
    // Inside (−a, a): the atanh antiderivative is real, keep it.
    assert_eq!(
        simplified_integral("integrate(1/(1-x^2), x, 0, 1/2)"),
        "atanh(1/2)"
    );
    assert_eq!(
        simplified_integral("integrate(1/(4-x^2), x, 0, 1)"),
        "1/2 * atanh(1/2)"
    );
    // A pole strictly inside the interval: divergent, undefined.
    assert_eq!(
        simplified_integral("integrate(1/(1-x^2), x, 0, 2)"),
        "undefined"
    );
    assert_eq!(
        simplified_integral("integrate(1/(1/4-x^2), x, -2, 1/4)"),
        "undefined"
    );
    // Not the a²−x² family: `a²+x²` is arctan, a linear term declines — the
    // gate must not hijack either.
    assert_eq!(
        simplified_integral("integrate(1/(-1-x^2), x, 2, 3)"),
        "arctan(2) - arctan(3)"
    );
    assert_eq!(
        simplified_integral("integrate(1/(1-x-x^2), x, 2, 3)"),
        "integrate(1 / (1 - x^2 - x), x, 2, 3)"
    );
}
/// The FTC wrapper narrated the SHELL and never the method: "find the
/// antiderivative" states WHAT was obtained and never HOW. The rest of the chain
/// already knows how to narrate the indefinite integral, so it gets handed a
/// synthetic 2-arg step and its narration is spliced in between.
#[test]
fn integrate_contract_definite_integral_shows_how_the_antiderivative_was_found() {
    let (wire, _) =
        cli_eval_json_with_stderr_args("integrate(1/(x^4-1), x, 2, oo)", &["--steps", "on"]);
    let titles: Vec<String> = wire["steps"]
        .as_array()
        .expect("steps with --steps on")
        .iter()
        .filter_map(|step| step["substeps"].as_array())
        .flatten()
        .filter_map(|s| s["title"].as_str().map(str::to_string))
        .collect();
    let find = titles
        .iter()
        .position(|t| t.contains("Hallar la antiderivada"));
    let method = titles.iter().position(|t| t.contains("Descomponer"));
    let evaluate = titles.iter().position(|t| t.contains("los límites"));
    assert!(
        matches!((find, method, evaluate), (Some(f), Some(m), Some(e)) if f < m && m < e),
        "the method must sit BETWEEN finding and evaluating: {titles:?}"
    );
}
