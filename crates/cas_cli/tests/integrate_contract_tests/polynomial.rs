use super::*;

/// G1 sub-cycle Cap. C (2026-07-15): an irreducible-over-ℚ quartic whose
/// resolvent cubic has a positive rational NON-square root (`Φ₅` in `x^5-1`,
/// `Φ₁₀` in `x^5+1`, with `t₀ = 5/4`) is kept WHOLE so the residue solve stays
/// over ℚ; the CONJUGATE quadratic pair over ℚ(√t₀) — `B = φ = 1/2 + √(5/4)` —
/// and the NESTED arctan radii `√(5/2 ∓ √(5/4))` appear only in the render.
/// Emission is gated behind the differentiate-back verifier, whose C-ii
/// nested-radical relation tower confirms the identity (every radius appears at
/// even degree in the residual). All five emissions below were additionally
/// verified symbolically with an independent oracle (sympy: residual 0 away
/// from the declared pole). See docs/G1_RATIONAL_INTEGRATION_SCOPING.md.
#[test]
fn integrate_contract_general_quartic_conjugate_pair_integrates_phi5_family() {
    for input in [
        "integrate(1/(x^5-1), x)",
        "integrate(1/(x^5+1), x)",
        "integrate(x/(x^5-1), x)",
        "integrate((x^3+1)/(x^5-1), x)",
        "integrate(1/(x^4+x^3+x^2+x+1), x)",
    ] {
        let (result, _required) = evaluated_integral_with_required_conditions(input);
        assert!(
            !result.contains("integrate("),
            "should integrate via the conjugate quadratic pair: {input} -> {result}"
        );
        assert!(
            result.contains("sqrt(5/4)") || result.contains("sqrt(5)"),
            "expected the golden-ratio surd in the render: {input} -> {result}"
        );
    }
    // The named G1 exit probe carries the nested arctan radii.
    let (phi5, _) = evaluated_integral_with_required_conditions("integrate(1/(x^5-1), x)");
    assert!(
        phi5.contains("5/2 - sqrt(5/4)") || phi5.contains("sqrt(5/4) + 5/2"),
        "expected the nested arctan radius sqrt(5/2 ∓ sqrt(5/4)): {phi5}"
    );
}
/// G1 verification-budget lift (2026-07-14): a GENERAL (non-constant) numerator
/// over an even-quartic factor produces a correct antiderivative whose combined
/// surd render is larger than the constant-numerator case. The exact algebraic
/// zero-test (`algebraic_rational_zero_test`) previously exceeded its node/term
/// budget on that residual and declined; the raised budget lets it verify (still
/// an EXACT decision procedure — a larger budget only decides bigger inputs, never
/// a false positive), so these now emit. Numerically confirmed correct.
#[test]
fn integrate_contract_general_numerator_even_quartic_now_verifies() {
    for input in [
        "integrate((x^3+5)/(x^6+1), x)",
        "integrate((x^3-x)/(x^6+1), x)",
        "integrate((3*x^3-2*x+5)/(x^6+1), x)",
        "integrate((x^2+1)/(x^4-x^2+1), x)",
        "integrate((2*x^2+3)/(x^8-1), x)",
    ] {
        let (result, _required) = evaluated_integral_with_required_conditions(input);
        assert!(
            !result.contains("integrate("),
            "raised algebraic zero-test budget should verify and emit: {input} -> {result}"
        );
    }
}
/// G1 residual R2 (2026-07-15): an even quartic `x^4 + p·x^2 + r` whose
/// resolvent `u^2 + pu + r` has IRRATIONAL REAL roots (Δ = p² − 4r > 0
/// non-square) is now kept WHOLE over ℚ (`EvenQuarticRealResolvent`) and
/// rendered through the closed conjugate split `(x² − u₁)(x² − u₂)` with
/// `u_i ∈ ℚ(√Δ)`: the inner partial fraction is CLOSED FORM in QuadSurd (no
/// field division — `1/√Δ = √Δ/Δ`, `1/u = conj(u)/r`), a positive `u` yields
/// the real-log ratio around the NESTED radius `√u` (two real poles → NonZero
/// conditions), a negative `u` the arctan around `√(−u)`. Coefficients carry
/// exactly one factor of their radius, so the differentiate-back residual is
/// even in each radius atom and the C-ii nested tower verifies it. All
/// emissions below were also confirmed numerically against an independent
/// oracle (sympy, 30 digits).
#[test]
fn integrate_contract_even_quartic_real_resolvent_integrates_x4_minus_5_family() {
    for input in [
        "integrate(1/(x^4-5), x)",
        "integrate(1/(x^4-2), x)",
        "integrate((x^2+1)/(x^4-5), x)",
        "integrate((x^3+x)/(x^4-5), x)",
        // Both resolvent roots negative: pure arctan pair, no real poles.
        "integrate(1/(x^4+6*x^2+4), x)",
        // Composite squarefree part: rational quadratic × real-resolvent quartic.
        "integrate(1/(x^6-2*x^4-5*x^2+10), x)",
    ] {
        let (result, _required) = evaluated_integral_with_required_conditions(input);
        assert!(
            !result.contains("integrate("),
            "should integrate via the real-resolvent conjugate split: {input} -> {result}"
        );
    }

    // The iconic member: both nested radii present (log ratio + arctan).
    let (result, required) = evaluated_integral_with_required_conditions("integrate(1/(x^4-5), x)");
    assert!(
        result.contains("arctan") && result.contains("ln(|"),
        "expected arctan + real-log ratio: {result}"
    );
    assert!(
        !required.is_empty(),
        "the real poles ±5^(1/4) must surface as conditions: {required:?}"
    );

    // Δ < 0 (complex resolvent roots outside the even/general quartic owners)
    // is out of THIS render's scope — the E-iv universal RootSum closure now
    // claims those shapes instead (verified by its exact-identity proof).
    for input in ["integrate(1/(x^4+5), x)", "integrate(1/(x^4+2*x^2+3), x)"] {
        let (result, _required) = evaluated_integral_with_required_conditions(input);
        assert!(
            result.contains("root_sum("),
            "delta<0 quartics now emit via the RootSum closure: {input} -> {result}"
        );
    }
}
/// G1 residual R3 (2026-07-15): the doubly-even octic `c/(x^8 + P·x^4 + R)`
/// with `S = √R`, `s = √S` rational and `A = √(2S − P)` irrational splits over
/// ℝ as two conjugate quartics in ℚ(A), each splitting into a symmetric surd
/// pair with NESTED radii `√(2s ∓ A)` — the closed two-level render whose
/// level-1 partial fraction is exact in ℚ(A) (γ = c/(2SA), δ = c/(2S)) and
/// whose four quadratics are positive-definite (no abs, no conditions). The
/// differentiate-back residual is even in each radius atom, so the nested
/// relation tower (`t² = 2S−P`, `u² = 2s−t`, `v² = 2s+t`) confirms it under
/// the raised reduction budget. All emissions verified numerically against
/// mpmath/sympy at 30 digits — notably, sympy 1.14's own `integrate` returns
/// literally `0` for `1/(x^8+1)` and `1/(x^8+16)` (a live wrong answer), so
/// this render strictly beats it on the family.
#[test]
fn integrate_contract_doubly_even_octic_integrates_x8_plus_1_family() {
    for input in [
        "integrate(1/(x^8+1), x)",
        "integrate(2/(x^8+1), x)",
        "integrate(1/(x^8+16), x)",
        "integrate(1/(x^8-x^4+1), x)",
        "integrate(1/(x^8+5*x^4+16), x)",
    ] {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert!(
            !result.contains("integrate("),
            "should integrate via the doubly-even octic split: {input} -> {result}"
        );
        assert!(
            result.contains("arctan") && result.contains("ln("),
            "expected the arctan+log closed form: {input} -> {result}"
        );
        assert!(
            required.is_empty(),
            "positive-definite quadratics need no conditions for {input}: {required:?}"
        );
    }

    // A² ≤ 0 (x^8+3x^4+1), s irrational (x^8+4) and non-constant numerators
    // are out of THIS render's scope — the E-iv universal RootSum closure now
    // claims most of them (verified by its exact-identity proof); x/(x^8+1)
    // stays an honest residual (no clean linear log argument in its PRS).
    for input in [
        "integrate(1/(x^8+3*x^4+1), x)",
        "integrate(1/(x^8+4), x)",
        "integrate(x^2/(x^8+1), x)",
    ] {
        let (result, _required) = evaluated_integral_with_required_conditions(input);
        assert!(
            result.contains("root_sum("),
            "out of the R3 gate but claimed by the RootSum closure: {input} -> {result}"
        );
    }
    {
        let input = "integrate(x/(x^8+1), x)";
        let (result, _required) = evaluated_integral_with_required_conditions(input);
        assert!(
            result.contains("integrate("),
            "no clean linear log argument, must stay residual: {input} -> {result}"
        );
    }
}
#[test]
fn integrate_contract_affine_sum_linearity() {
    assert_eq!(simplified_integral("integrate(2*x + 3, x)"), "x^2 + 3 * x");
}
#[test]
fn integrate_contract_positive_quadratic_cube_uses_recurrence() {
    let input = "integrate(1/(x^2+1)^3, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "3/8 * arctan(x) + (3 * x^3 + 5 * x) / (8 * (x^2 + 1)^2)"
    );
    assert!(
        required.is_empty(),
        "positive quadratic cube should not add synthetic required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);
    assert!(
        stderr.is_empty(),
        "positive quadratic cube trace should stay quiet\nstderr:\n{stderr}"
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
    let reduction_substep = substeps
        .iter()
        .find(|substep| substep["title"] == "Reducir el cuadrático positivo al cubo")
        .expect("expected positive quadratic cube reduction substep");
    let reduction_latex = reduction_substep["after_latex"]
        .as_str()
        .expect("reduction substep should expose concrete after_latex");
    assert!(
        reduction_latex.contains("\\frac{3}{8\\cdot ({x}^{2} + 1)}")
            && reduction_latex.contains("{({x}^{2} + 1)}^{3}"),
        "reduction should expose the arctan integrand and rational derivative part, got {reduction_latex}"
    );
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Integrar la parte arctan y la parte racional"),
        "expected final integration substep, got {substeps:?}"
    );

    let residual = "diff(integrate(1/(x^2+1)^3, x), x) - 1/(x^2+1)^3";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "nested verification should not invent denominator conditions for a positive quadratic cube"
    );

    let input = "integrate(1/((x+1)^2+1)^3, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "3/8 * arctan(x + 1) + (3 * x^3 + 9 * x^2 + 14 * x + 8) / (8 * (x^2 + 2 * x + 2)^2)"
    );
    assert!(
        required.is_empty(),
        "shifted positive quadratic cube should not add synthetic required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);
    assert!(
        stderr.is_empty(),
        "shifted positive quadratic cube trace should stay quiet\nstderr:\n{stderr}"
    );
    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    let integration_step = steps
        .iter()
        .find(|step| step["rule"] == "Calcular la integral")
        .expect("expected shifted positive quadratic cube integration step");
    let substeps = integration_step["substeps"]
        .as_array()
        .expect("shifted positive quadratic cube should expose didactic substeps");
    let reduction_latex = substeps
        .iter()
        .find(|substep| substep["title"] == "Reducir el cuadrático positivo al cubo")
        .and_then(|substep| substep["after_latex"].as_str())
        .expect("expected shifted positive quadratic cube reduction substep");
    assert!(
        reduction_latex.contains(" - \\frac{3\\cdot {x}^{4}")
            && !reduction_latex.contains("\\frac{-"),
        "shifted cube reduction should carry the negative rational sign outside the fraction, got {reduction_latex}"
    );

    let residual = "diff(integrate(1/((x+1)^2+1)^3, x), x) - 1/((x+1)^2+1)^3";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "nested verification should not invent denominator conditions for a shifted positive quadratic cube"
    );
    let direct_diff = "diff(integrate(1/((x+1)^2+1)^3, x), x)";
    let (direct_wire, direct_stderr) = cli_eval_json_with_stderr(direct_diff);
    assert!(
        direct_stderr.is_empty(),
        "direct shifted positive-quadratic cube diff/integrate should avoid depth overflow: {direct_stderr}"
    );
    let direct_result = direct_wire["result"].as_str().unwrap_or_default();
    assert!(
        direct_result.contains("^3") && !direct_result.contains("x^6"),
        "direct shifted positive-quadratic cube diff/integrate should preserve compact denominator, got {direct_result}"
    );
    assert_eq!(direct_wire["required_display"], serde_json::json!([]));

    let input = "integrate(1/(4*x^2+1)^3, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "3/16 * arctan(2 * x) + (12 * x^3 + 5 * x) / (8 * (4 * x^2 + 1)^2)"
    );
    assert!(
        required.is_empty(),
        "scaled positive quadratic cube should not add synthetic required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);

    let residual = "diff(integrate(1/(4*x^2+1)^3, x), x) - 1/(4*x^2+1)^3";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "nested verification should not invent denominator conditions for a scaled positive quadratic cube"
    );

    let input = "integrate(x^2/(x^2+1)^3, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "1/8 * arctan(x) + (x^3 - x) / (8 * (x^2 + 1)^2)");
    assert!(
        required.is_empty(),
        "quadratic numerator over positive quadratic cube should not add synthetic required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);

    let residual = "diff(integrate(x^2/(x^2+1)^3, x), x) - x^2/(x^2+1)^3";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "nested verification should not invent denominator conditions for a quadratic numerator over a positive quadratic cube"
    );
    let direct_diff = "diff(integrate(x^2/(x^2+2*x+2)^3, x), x)";
    let (direct_wire, direct_stderr) = cli_eval_json_with_stderr(direct_diff);
    assert!(
        direct_stderr.is_empty(),
        "direct quadratic-numerator positive-quadratic cube diff/integrate should avoid depth overflow: {direct_stderr}"
    );
    let direct_result = direct_wire["result"].as_str().unwrap_or_default();
    assert!(
        direct_result.contains("x^2")
            && direct_result.contains("(x^2 + 2·x + 2)^3")
            && !direct_result.contains("x^6"),
        "direct quadratic-numerator positive-quadratic cube diff/integrate should preserve compact denominator, got {direct_result}"
    );
    assert_eq!(direct_wire["required_display"], serde_json::json!([]));

    let input = "integrate((2*x+1)^2/((2*x+1)^2+1)^3, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "1/16 * arctan(2 * x + 1) + (2 * x^3 + 3 * x^2 + x) / (4 * (4 * x^2 + 4 * x + 2)^2)"
    );
    assert!(
        required.is_empty(),
        "affine quadratic numerator over positive quadratic cube should not add synthetic required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);

    let residual = "diff(integrate((2*x+1)^2/((2*x+1)^2+1)^3, x), x) - (2*x+1)^2/((2*x+1)^2+1)^3";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "nested verification should not invent denominator conditions for an affine quadratic numerator over a positive quadratic cube"
    );

    let input = "integrate(x^3/(x^2+1)^3, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "1 / (4 * (x^2 + 1)^2) - 1 / (2 * (x^2 + 1))");
    assert!(
        required.is_empty(),
        "cubic numerator over positive quadratic cube should not add synthetic required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);

    let residual = "diff(integrate(x^3/(x^2+1)^3, x), x) - x^3/(x^2+1)^3";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "nested verification should not invent denominator conditions for a cubic numerator over a positive quadratic cube"
    );

    let input = "integrate(x^4/(x^2+1)^3, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "3/8 * arctan(x) + (3 * x^3 + 5 * x) / (8 * (x^2 + 1)^2) - x / (x^2 + 1)"
    );
    assert!(
        required.is_empty(),
        "quartic numerator over positive quadratic cube should not add synthetic required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);

    let residual = "diff(integrate(x^4/(x^2+1)^3, x), x) - x^4/(x^2+1)^3";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "nested verification should not invent denominator conditions for a quartic numerator over a positive quadratic cube"
    );

    let input = "integrate((2*x+1)^4/(((2*x+1)^2+1)^3), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "3/16 * arctan(2 * x + 1) + (6 * x^3 + 9 * x^2 + 7 * x + 2) / (4 * (4 * x^2 + 4 * x + 2)^2) - (2 * x + 1) / (2 * (4 * x^2 + 4 * x + 2))"
    );
    assert!(
        required.is_empty(),
        "scaled affine quartic numerator over positive quadratic cube should not add synthetic required conditions: {required:?}"
    );
    assert_eq!(
        assert_antiderivative_verifies(input),
        AntiderivativeVerificationRoute::PublicResidual
    );
    assert_rendered_antiderivative_verifies(input, &result);

    let residual =
        "diff(integrate((2*x+1)^4/(((2*x+1)^2+1)^3), x), x) - (2*x+1)^4/(((2*x+1)^2+1)^3)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "nested verification should not invent denominator conditions for a scaled affine quartic numerator over a positive quadratic cube"
    );
}
