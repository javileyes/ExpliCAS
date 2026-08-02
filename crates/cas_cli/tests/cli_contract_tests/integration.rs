use super::*;

#[test]
fn test_eval_trig_log_integral_residual_compacts_cleanup_noise_json() {
    let output = cli()
        .args([
            "eval",
            "integrate(1/((tan(x)-2)*ln(tan(x))), x)",
            "--format",
            "json",
            "--steps",
            "on",
        ])
        .output()
        .expect("Failed to run CLI");

    assert!(
        output.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).unwrap();
    let wire: Value = serde_json::from_str(&stdout).expect("Invalid wire output");
    let steps = wire["steps"].as_array().expect("steps array");
    let rules: Vec<&str> = steps
        .iter()
        .map(|step| step["rule"].as_str().expect("step rule"))
        .collect();

    assert_eq!(wire["ok"], true);
    assert_eq!(
        wire["result"],
        "integrate(cos(x) / (ln(sin(x) / cos(x))·(sin(x) - 2·cos(x))), x)"
    );
    assert_eq!(
        wire["required_display"],
        serde_json::json!([
            "cos(x) ≠ 0",
            "tan(x) - 1 ≠ 0",
            "tan(x) - 2 ≠ 0",
            "tan(x) > 0"
        ])
    );
    assert_eq!(
        rules,
        vec![
            "Expandir tangente como seno entre coseno",
            "Conservar integral residual"
        ]
    );
}
#[test]
fn test_eval_even_integrand_negative_interval_reflects() {
    // An EVEN integrand over a strictly-negative interval reflects to the positive
    // branch (`∫_{-3}^{-2} f = ∫_2^3 f`). `√(x²-1)`'s antiderivative uses `acosh` (real
    // only for arg >= 1), so the negative branch used to decline; it now evaluates to
    // the SAME closed form as the reflected positive interval.
    for (neg, pos) in [
        (
            "integrate(sqrt(x^2-1), x, -3, -2)",
            "integrate(sqrt(x^2-1), x, 2, 3)",
        ),
        (
            "integrate(sqrt(x^2-4), x, -5, -3)",
            "integrate(sqrt(x^2-4), x, 3, 5)",
        ),
        (
            "integrate(1/sqrt(x^2-1), x, -3, -2)",
            "integrate(1/sqrt(x^2-1), x, 2, 3)",
        ),
    ] {
        let run = |e: &str| -> String {
            let out = cli()
                .args(["eval", e, "--format", "json"])
                .output()
                .expect("run");
            let w: Value = serde_json::from_slice(&out.stdout).expect("json");
            w["result"].as_str().unwrap_or("").to_string()
        };
        let r = run(neg);
        assert_eq!(r, run(pos), "{neg}");
        assert!(!r.contains("integrate("), "{neg} should evaluate, got {r}");
    }
}
#[test]
fn test_eval_apart_partial_fractions() {
    // `apart(p/q)` (alias `partfrac`) gives the partial-fraction decomposition, exact over Q. The
    // result is `Hold`-protected so the fraction-combining rules don't pull it back over a common
    // denominator. Single-variable is inferred; `apart(p/q, x)` names it. An IMPROPER fraction
    // (deg p ≥ deg q) is polynomial-divided first: `p/q = quotient + remainder/q`.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("apart(1/(x^2-1))"), "1/2 / (x - 1) - 1/2 / (x + 1)");
    // Improper fractions: the polynomial quotient is prepended to the proper decomposition.
    assert_eq!(r("apart(x^3/(x^2-1))"), "1/2 / (x - 1) + 1/2 / (x + 1) + x");
    assert_eq!(r("apart(x^2/(x^2-1))"), "1/2 / (x - 1) + 1 - 1/2 / (x + 1)");
    assert_eq!(
        r("apart(x^4/(x^2-1))"),
        "1/2 / (x - 1) + x^2 + 1 - 1/2 / (x + 1)"
    );
    assert_eq!(
        r("apart(1/(x^3-x))"),
        "1/2 / (x - 1) + 1/2 / (x + 1) - 1 / x"
    );
    assert_eq!(
        r("apart(1/((x-1)*(x-2)*(x-3)))"),
        "1/2 / (x - 3) + 1/2 / (x - 1) - 1 / (x - 2)"
    );
    assert_eq!(r("apart((x+3)/(x^2-x-2))"), "5/3 / (x - 2) - 2/3 / (x + 1)");
    assert_eq!(r("apart(1/(x^2+x))"), "1 / x - 1 / (x + 1)");
    // Not a rational fraction, or an irreducible high-degree denominator ⇒ honest residual.
    assert_eq!(r("apart(x^2+1)"), "apart(x^2 + 1)");
    assert_eq!(r("apart(1/(x^3-x-1))"), "apart(1 / (x^3 - x - 1))");
    // Repeated roots get the full multiplicity ladder A_k/(x-r)^k, NOT the
    // Ostrogradsky/Hermite integral form (which dropped the 1/(2(x-1)^2) term and
    // returned a non-equivalent answer). Soundness regression guard for B2.
    assert_eq!(
        r("apart(1/((x-1)^2*(x+1)))"),
        "1/4 / (x + 1) + 1/2 / (x - 1)^2 - 1/4 / (x - 1)"
    );
    assert_eq!(r("apart(1/((x-1)^2))"), "1 / (x - 1)^2");
    assert_eq!(r("apart((x+1)/((x-1)^2))"), "1 / (x - 1) + 2 / (x - 1)^2");
    assert_eq!(r("apart(1/((x-1)^3))"), "1 / (x - 1)^3");
    assert_eq!(
        r("apart(1/(x*(x-1)^2))"),
        "1 / x + 1 / (x - 1)^2 - 1 / (x - 1)"
    );
    // A degree-1 denominator IS already a partial fraction. `apart` used to
    // decline it (min denominator degree 2) and echo an unevaluated
    // `apart(1/(x-2))` residual; it now returns the fraction. An IMPROPER
    // degree-1 fraction gets the polynomial part split off.
    assert_eq!(r("apart(1/(x-2))"), "1 / (x - 2)");
    assert_eq!(r("apart(5/(x-3))"), "5 / (x - 3)");
    assert_eq!(r("apart(1/(2*x-4))"), "1/2 / (x - 2)");
    assert_eq!(r("apart((x+1)/(x-2))"), "3 / (x - 2) + 1");
    assert_eq!(r("apart((3*x+1)/(x+2))"), "3 - 5 / (x + 2)");
    // A shared numerator/denominator factor cancels to a degree-1 pole, which
    // now decomposes (returns to itself) instead of echoing the residual.
    assert_eq!(r("apart((x+1)/(x^2-x-2))"), "1 / (x - 2)");
    // A SCALED monomial numerator `c*x^k` (c != 1) simplifies to `Mul(c, Div(x^k,
    // D))` — the constant pulls OUT of the division — so the old `Expr::Div`-only
    // match echoed an unevaluated `apart(2x/…)` residual while the unit `x/…`
    // decomposed fine. Any fraction-like shape (nested Div / reciprocal factor)
    // now normalizes to `num/den` first. Cross-checked vs sympy.
    assert_eq!(
        r("apart((2*x)/((x-1)^2*(x+1)))"),
        "1/2 / (x - 1) + 1 / (x - 1)^2 - 1/2 / (x + 1)"
    );
    assert_eq!(
        r("apart((3*x)/((x-1)*(x+1)))"),
        "3/2 / (x - 1) + 3/2 / (x + 1)"
    );
    assert_eq!(
        r("apart((2*x^2)/((x-1)^2*(x+1)))"),
        "1/2 / (x + 1) + 1 / (x - 1)^2 + 3/2 / (x - 1)"
    );
    assert_eq!(r("apart((5*x)/((x-2)*(x+3)))"), "2 / (x - 2) + 3 / (x + 3)");
    // Scaled improper fraction: the polynomial quotient is split off.
    assert_eq!(
        r("apart((2*x^3)/((x-1)*(x+2)))"),
        "2/3 / (x - 1) + 16/3 / (x + 2) + 2·x - 2"
    );
}
#[test]
fn test_eval_lineintegral_verb_f4() {
    // Fase 3 · F4: ensamblador puro sobre composición viva — parametrizar,
    // derivar, ensamblar el integrando y delegar en la integral definida.
    let eval_result = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Vector (circulación), escalar (∫f·ds vía ‖r'‖) y hélice 3D.
    assert_eq!(
        eval_result("lineintegral([-y,x],[x,y],[cos(t),sin(t)],t,0,2*pi)"),
        "2·pi"
    );
    assert_eq!(
        eval_result("lineintegral(x^2,[x,y],[cos(t),sin(t)],t,0,pi)"),
        "1/2·pi"
    );
    assert_eq!(
        eval_result("lineintegral([y,-x,1],[x,y,z],[cos(t),sin(t),t],t,0,2*pi)"),
        "0"
    );
    // Fixture de equivalencia verbo ≡ composición (guardrail #5: resultado
    // como contrato): la circulación ensamblada A MANO da lo mismo.
    assert_eq!(
        eval_result(
            "integrate(subs(subs(-y,x,cos(t)),y,sin(t))*diff(cos(t),t) + subs(subs(x,x,cos(t)),y,sin(t))*diff(sin(t),t), t, 0, 2*pi)"
        ),
        "2·pi"
    );
    // Declines honestos: parametrización que menciona una variable del campo,
    // shapes incompatibles (#comps ≠ #vars), t dentro de la lista de vars.
    for probe in [
        "lineintegral(x,[x,y],[t,x],t,0,1)",
        "lineintegral([-y,x],[x,y],[cos(t)],t,0,1)",
        "lineintegral(x^2,[x,t],[cos(t),sin(t)],t,0,1)",
    ] {
        let r = eval_result(probe);
        assert!(
            r.starts_with("lineintegral("),
            "{probe} debe declinar a eco residual, got: {r}"
        );
    }
}
#[test]
fn test_eval_improper_rational_integral_real_root_quadratic_denominator() {
    // An improper `∫_a^∞ p/q` with a `½·ln|p/q|` antiderivative and a quadratic denominator with
    // REAL roots OUTSIDE [a, ∞) used to decline: `nonzero_on_unbounded_interval` returned `Unknown`
    // for a degree-2 factor with non-negative discriminant (it only certified the no-real-root case).
    // Now it decides EXACTLY from the vertex `−b/2a` and the sign of `q` at the bound (no surds). The
    // boundary limit (already supported) supplies the value; tail divergence shows up as `±∞`.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Convergent: roots ±1 / 0,1 / ±2 / −1,−2 all lie below the lower bound -> value computed.
    assert_eq!(r("integrate(1/(x^2-1), x, 2, oo)"), "-1/2·ln(1/3)"); // = ½ln3
    assert_eq!(r("integrate(2/(x^2-1), x, 2, oo)"), "-ln(1/3)"); // = ln3
    assert_eq!(r("integrate(1/(x^2-x), x, 2, oo)"), "-ln(1/2)"); // = ln2
    assert_eq!(r("integrate(1/(x^2+3*x+2), x, 1, oo)"), "-ln(2/3)"); // = ln(3/2)
    assert_eq!(r("integrate(1/(x^2-9), x, 4, oo)"), "-1/6·ln(1/7)"); // = (1/6)ln7
                                                                     // SOUNDNESS: tail-divergent and pole-in-range must NOT fabricate a finite value.
    assert_eq!(r("integrate(x/(x^2-1), x, 2, oo)"), "infinity"); // diverges (~1/x tail)
    assert_eq!(r("integrate(1/(x^2-1), x, 0, oo)"), "undefined"); // pole at x=1 ∈ [0,∞)
    assert_eq!(r("integrate(1/(x^2-1), x, 1/2, oo)"), "undefined"); // pole at x=1 ∈ [1/2,∞)
                                                                    // Unchanged: finite definite (already worked) and the no-real-root quadratic.
    assert_eq!(r("integrate(1/(x^2-1), x, 2, 5)"), "1/2·ln(2)");
    assert_eq!(r("integrate(1/(x^2+1), x, 0, oo)"), "1/2·pi");
}
#[test]
fn test_eval_definite_integral_removable_pole_is_not_undefined() {
    // A rationalization step turns `1/(√x·(1+x))` into `(√x³−√x)/(x³−x)`, inventing a SPURIOUS
    // denominator root at x=1 where the numerator also vanishes (removable). The FTC pole scan used
    // to reject it as an in-interval pole and return a false `undefined` on a convergent / regular
    // proper integral. The (continuous) antiderivative `2·arctan(√x)` is finite at x=1, certifying
    // the singularity removable, so the integral evaluates.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Regular proper interval [1/2, 4] (NO singularity in it): 2·(arctan(2) − arctan(√½)) ≈ 0.9833.
    assert_eq!(
        r("integrate(1/(sqrt(x)*(1+x)),x,1/2,4)"),
        "2·(arctan(2) - arctan(sqrt(1/2)))"
    );
    // Convergent improper integral [1, ∞) = π/2.
    assert_eq!(r("integrate(1/(sqrt(x)*(1+x)),x,1,inf)"), "1/2·pi");
    // The interval clear of the spurious root is unaffected.
    assert_eq!(
        r("integrate(1/(sqrt(x)*(1+x)),x,4,9)"),
        "2·(arctan(3) - arctan(2))"
    );
    // Pure-rational removable singularities also evaluate: (x−1)/(x²−1) = 1/(x+1) on [0,3] = ln(4).
    assert_eq!(r("integrate((x-1)/(x^2-1),x,0,3)"), "ln(4)");
    // SOUNDNESS: a GENUINE interior pole (numerator nonzero) still diverges → undefined.
    assert_eq!(r("integrate(1/(x-1),x,0,2)"), "undefined");
    assert_eq!(r("integrate(1/((x-1)*(x-3)),x,0,4)"), "undefined");
    assert_eq!(r("integrate(1/(x-2)^2,x,1,3)"), "undefined");
}
#[test]
fn test_eval_definite_integral_provably_positive_transcendental_denominator() {
    // `∫ 1/(e^x+1)` computes the antiderivative `ln(e^x/(e^x+1))`, but the DEFINITE
    // form leaked: the pole certificate could not `Polynomial::from_expr` the
    // transcendental denominator `e^x+1`, returned Unknown, and declined. Since
    // `e^x+1 > 0` everywhere (the real-domain sign prover decides `e^x > 0`), it has
    // no pole, so the FTC evaluation is safe. Cross-checked vs sympy.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // `∫₀¹ 1/(e^x+1) = ln(2) + 1 - ln(1+e)` = `ln(2e/(1+e))` ≈ 0.37989.
    assert_eq!(r("integrate(1/(e^x+1), x, 0, 1)"), "ln((e·2)/(1 + e))");
    assert_eq!(r("integrate(1/(e^x+3), x, 0, 1)"), "1/3·ln((e·4)/(3 + e))");
    // Numerator = e^x (antiderivative ln(e^x+1)).
    assert_eq!(r("integrate(e^x/(e^x+1), x, 0, 1)"), "ln(1/2·(1 + e))");
    // SOUNDNESS: a denominator with a REAL root is NOT provably positive everywhere,
    // so it is NOT falsely certified — `e^x-1` vanishes at x=0 (a genuine pole inside
    // [-1,1]) and stays an honest residual, and polynomial poles are still caught.
    assert_eq!(
        r("integrate(1/(e^x-1), x, -1, 1)"),
        "integrate(1 / (e^x - 1), x, -1, 1)"
    );
    assert_eq!(r("integrate(1/(x-1), x, 0, 2)"), "undefined");
    // No regression on the already-working rational and log cases.
    assert_eq!(r("integrate(1/(x^2+1), x, 0, 1)"), "1/4·pi");
    assert_eq!(r("integrate(1/x, x, 1, e)"), "1");
}
#[test]
fn test_eval_nth_root_reciprocal_integral_uses_correct_conjugate() {
    // `1/x^(1/n)` rationalized its denominator by multiplying by the BARE root `x^(1/n)`, which only
    // clears a SQUARE root: `x^(1/4)·x^(1/4) = x^(1/2) ≠ x`. So `1/x^(1/4)` became `x^(1/4)/x = x^(-3/4)`
    // and integrated to a WRONG `4·x^(1/4)` (whose derivative `x^(-3/4)` ≠ the integrand `x^(-1/4)`).
    // The conjugate `x^((n-1)/n)` now clears it correctly: `1/x^(1/4) → x^(3/4)/x → x^(-1/4)`.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Indefinite even-root antiderivatives (true `(n/(n-1))·x^((n-1)/n)`), verified by diff-back.
    assert_eq!(r("integrate(1/x^(1/4),x)"), "4/3·x^(3/4)");
    assert_eq!(r("integrate(1/x^(1/6),x)"), "6/5·x^(5/6)");
    assert_eq!(r("diff(4/3*x^(3/4),x)"), "x^(-1/4)");
    // Definite even-root integrals on [0,1] (true `n/(n-1)`).
    assert_eq!(r("integrate(1/x^(1/4),x,0,1)"), "4/3");
    assert_eq!(r("integrate(1/x^(1/6),x,0,1)"), "6/5");
    assert_eq!(r("integrate(1/x^(1/8),x,0,1)"), "8/7");
    // Square-root rationalization (n=2) is unchanged.
    assert_eq!(r("integrate(1/sqrt(x),x)"), "2·sqrt(x)");
    assert_eq!(r("integrate(1/(x*sqrt(x)),x)"), "-2 / sqrt(x)");
    // ODD-root and general fractional reciprocal powers: the simplifier rationalizes
    // `1/x^(1/3)` to `x^(2/3)/x` (and leaves `1/x^(2/5)` as-is), which the power-rule
    // matcher missed — only the even-root `1/x^(1/(2k))` cases above worked. Folding
    // `(c·)x^a/x^b → c·x^(a-b)` for a FRACTIONAL exponent recovers the power rule, in
    // both the indefinite and definite paths. Verified by diff-back and sympy.
    assert_eq!(r("integrate(1/x^(1/3),x)"), "3/2·x^(2/3)");
    assert_eq!(r("integrate(1/x^(1/3),x,1,8)"), "9/2");
    assert_eq!(r("integrate(1/x^(2/5),x)"), "5/3·x^(3/5)");
    assert_eq!(r("integrate(1/x^(2/3),x)"), "3·x^(1/3)");
    assert_eq!(r("integrate(3/x^(1/2),x)"), "6·sqrt(x)");
    assert_eq!(r("diff(3/2*x^(2/3),x)"), "x^(-1/3)");
    // Integer-exponent quotients keep their existing (unfolded) path.
    assert_eq!(r("integrate(1/x,x)"), "ln(|x|)");
    assert_eq!(r("integrate(x^3/x,x)"), "1/3·x^3");
}
#[test]
fn test_eval_symmetric_surd_even_quartic_integral_verifies() {
    // `c / (x^4 + p·x^2 + r)` whose even quartic factors over ℝ into the symmetric SURD pair
    // `(x²+a·x+s)(x²−a·x+s)` with `s=√r ∈ ℚ` but `a=√(2s−p)` irrational was an unevaluated residual
    // (the rational-coefficient factor path could not carry the √). It now integrates to a verified
    // arctan+log closed form. Numerically checked: F'(x) = integrand (err ~1e-11).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Phi_12 = x^4-x^2+1 (factor √3): no longer a bare `integrate(...)` residual.
    let phi12 = r("integrate(1/(x^4-x^2+1), x)");
    assert!(
        !phi12.starts_with("integrate("),
        "x^4-x^2+1 must integrate to a closed form, got residual: {phi12}"
    );
    assert!(
        phi12.contains("arctan") && phi12.contains("ln") && phi12.contains("sqrt(3)"),
        "expected arctan+log closed form over √3, got: {phi12}"
    );
    // x^4-3x^2+4 uses √7; the scaled numerator stays a closed form too.
    assert!(!r("integrate(1/(x^4-3*x^2+4), x)").starts_with("integrate("));
    assert!(!r("integrate(2/(x^4-x^2+1), x)").starts_with("integrate("));
    // Controls: routes owned elsewhere stay byte-identical.
    assert_eq!(r("integrate(1/(x^2+1), x)"), "arctan(x)");
    // `1/(x^6+1)` = 1/((x^2+1)(x^4-x^2+1)): the even quartic now integrates as a
    // FACTOR via G1 Cap. B (previously out of the symmetric-surd cycle's scope).
    assert!(!r("integrate(1/(x^6+1), x)").starts_with("integrate("));
    // `x^4+3x^2+1` factors into two irreducible quadratics with IRRATIONAL
    // constants (u-roots (-3±√5)/2), not the symmetric-surd form. It graduated
    // via G1 R2 (`EvenQuarticRealResolvent`, 2026-07-15): the conjugate split
    // over ℚ(√5) renders the arctan pair (both u-roots negative, no real
    // poles). Numerically confirmed against sympy at 30 digits.
    let real_resolvent = r("integrate(1/(x^4+3*x^2+1), x)");
    assert!(
        !real_resolvent.starts_with("integrate("),
        "x^4+3x^2+1 must integrate via the real-resolvent split: {real_resolvent}"
    );
    assert!(
        real_resolvent.contains("arctan") && real_resolvent.contains("sqrt(5)"),
        "expected the arctan pair over Q(sqrt(5)): {real_resolvent}"
    );
}

#[test]
fn test_eval_integrate_complex_axis_principal_branch_logs() {
    // D4 (eje de dominio, 2026-08-02): under `--value-domain complex` the
    // logarithmic antiderivatives use the PRINCIPAL BRANCH (`ln(u)`) — `ln|u|`
    // is not a complex antiderivative (|·| is not analytic), and the rest of
    // the engine already treats symbols as complex-valued under this axis
    // (`solve(x²=−1) → {i,−i}`, `sqrt(x²)` stays). Real axis stays `ln|u|`.
    // The single decision point is `cas_math::integration_value_domain`.
    let r = |input: &str, vd: &str| -> String {
        let out = cli()
            .args(["eval", input, "--value-domain", vd, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    for (input, real, complex) in [
        ("integrate(1/x, x)", "ln(|x|)", "ln(x)"),
        ("integrate(cos(x)/sin(x), x)", "ln(|sin(x)|)", "ln(sin(x))"),
        ("integrate(1/(x-1), x)", "ln(|x - 1|)", "ln(x - 1)"),
        ("integrate(tan(x), x)", "-ln(|cos(x)|)", "-ln(cos(x))"),
        (
            "integrate(sec(x), x)",
            "ln(|tan(x) + sec(x)|)",
            "ln(tan(x) + sec(x))",
        ),
        (
            "integrate(1/(x^2-1), x)",
            "1/2·ln(|(x - 1) / (x + 1)|)",
            "1/2·ln((x - 1) / (x + 1))",
        ),
        ("integrate(coth(x), x)", "ln(|sinh(x)|)", "ln(sinh(x))"),
    ] {
        assert_eq!(r(input, "real"), real, "real axis drifted: {input}");
        assert_eq!(r(input, "complex"), complex, "complex axis: {input}");
    }
    // Provably-positive arguments never carried the abs — identical on both
    // axes; polynomial antiderivatives are axis-independent; the definite
    // FTC value matches on both axes (the imaginary parts of the principal
    // branch cancel: ln(−1)−ln(−2) = −ln 2).
    for (input, expected) in [
        ("integrate(x/(x^2+1), x)", "1/2·ln(x^2 + 1)"),
        ("integrate(2*x*(x^2+1)^3, x)", "1/4·(x^2 + 1)^4"),
        ("integrate(1/x, x, -2, -1)", "-ln(2)"),
    ] {
        assert_eq!(r(input, "real"), expected, "real: {input}");
        assert_eq!(r(input, "complex"), expected, "complex: {input}");
    }
}
