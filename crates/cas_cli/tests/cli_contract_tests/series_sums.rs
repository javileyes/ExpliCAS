use super::*;

#[test]
fn test_eval_sum_of_absolute_values_inequality_solves_piecewise() {
    // A SUM of absolute values is piecewise-linear, so the inequality has a real
    // interval solution. The old "isolate one abs and split cases" strategy lost
    // the other terms and wrongly returned "No solution" (or a malformed
    // residual for `>`/`>=`). The piecewise/breakpoint solver returns the exact
    // union of intervals. Ground truth cross-checked against sympy.
    for (input, expected) in [
        ("abs(x) + abs(x-1) < 5", "(-2, 3)"),
        ("abs(x) + abs(x-1) <= 3", "[-1, 2]"),
        // |x|+|x-1| = 1 on all of [0,1], so `<= 1` is the whole closed interval,
        // not just its endpoints (a discrete-vs-interval merge bug guard).
        ("abs(x) + abs(x-1) <= 1", "[0, 1]"),
        ("abs(x-2) + abs(x+2) < 6", "(-3, 3)"),
        ("abs(x) + abs(x+1) < 4", "(-5/2, 3/2)"),
        ("2*abs(x) + abs(x-3) < 8", "(-5/3, 11/3)"),
        // Rational breakpoints (slope 2): bps at -1/2, 1/2.
        ("abs(2*x-1) + abs(2*x+1) < 4", "(-1, 1)"),
        // `>` was previously malformed; now a union of two open rays.
        ("abs(x) + abs(x-1) > 5", "(-infinity, -2) U (3, infinity)"),
        ("abs(x) + abs(x-1) >= 1", "All real numbers"),
        // Three terms.
        ("abs(x) + abs(x-1) + abs(x-2) < 4", "(-1/3, 7/3)"),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
    }

    // Genuinely-empty sums (min of |x|+|x-1| is 1) must still report No solution,
    // and single-abs / non-abs inequalities must be unchanged by the new path.
    for (input, expected) in [
        ("abs(x) + abs(x-1) < 1", "No solution"),
        ("abs(x-2) + abs(x+2) < 3", "No solution"),
        ("abs(x) < 5", "(-5, 5)"),
        ("abs(x-3) >= 2", "(-infinity, 1] U [5, infinity)"),
        ("x^2 - 4 < 0", "(-2, 2)"),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
    }
}
#[test]
fn test_eval_sum_of_absolute_values_equation_solves_piecewise() {
    // The same piecewise/breakpoint solver handles EQUATIONS. The old
    // isolate-one-abs strategy leaked a malformed nested-`solve` residual (and
    // for the flat-minimum case wrongly returned a half-line). On each segment a
    // strictly-linear piece contributes its single crossing; a constant piece
    // equal to the target contributes the whole segment. Cross-checked against an
    // independent exact (fractions) oracle over 400 random sums (0 mismatches).
    for (input, expected) in [
        // Above the minimum (1): two isolated crossings.
        ("abs(x) + abs(x-1) = 3", "{ -1, 2 }"),
        ("abs(x) + abs(x-1) = 2", "{ -1/2, 3/2 }"),
        ("abs(x-2) + abs(x+2) = 8", "{ -4, 4 }"),
        ("abs(x) + abs(x-1) + abs(x-2) = 4", "{ -1/3, 7/3 }"),
        ("2*abs(x) + abs(x-3) = 6", "{ -1, 3 }"),
        // At the flat minimum: the whole closed segment is the solution set.
        ("abs(x) + abs(x-1) = 1", "[0, 1]"),
        ("abs(x+1) + abs(x-1) = 2", "[-1, 1]"),
        // Below the minimum: empty.
        ("abs(x) + abs(x-1) = 1/2", "No solution"),
        // Non-convex signed coefficients: a flat piece yields a ray, and a single
        // crossing yields a point (convexity is NOT assumed by the solver).
        ("abs(x) - abs(x-1) = 0", "{ 1/2 }"),
        ("abs(x) - abs(x-1) = -1", "(-infinity, 0]"),
        ("abs(x) - abs(x-1) = 1", "[1, infinity)"),
        // Affine remainder term folded into the per-segment line.
        ("abs(x) + abs(x-1) + x = 3", "{ -2, 4/3 }"),
        // Single-abs equations are untouched (still the existing path).
        ("abs(x) = 3", "{ 3, -3 }"),
        ("abs(2*x-1) = 5", "{ 3, -2 }"),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
    }
}
#[test]
fn test_eval_sum_of_two_radicals_equation_solves_and_verifies() {
    // `√f + √g = c` used to leak `Solve: solve(x - (c - √g)^(1/(1/2)) = 0, x) = 0`
    // and drop the root. It is now reduced by squaring to the single radical
    // `√(f·g) = (c² - f - g)/2`, solved, and each candidate VERIFIED exactly
    // against the original (both radicands perfect rational squares summing to c),
    // dropping extraneous roots. Cross-checked against an independent oracle over
    // 300 random cases (0 mismatches).
    for (input, expected) in [
        ("sqrt(x+3) + sqrt(x) = 3", "{ 1 }"),
        ("sqrt(x+1) + sqrt(x) = 1", "{ 0 }"),
        ("sqrt(x+5) + sqrt(x) = 5", "{ 4 }"),
        ("sqrt(x-1) + sqrt(x+4) = 5", "{ 5 }"),
        // Symmetric radicands (difference of squares under the reduction).
        ("sqrt(x+1) + sqrt(x-1) = 3", "{ 85/36 }"),
        ("sqrt(x-2) + sqrt(x+2) = 5", "{ 641/100 }"),
        // No real solution: the single candidate is extraneous (or the minimum of
        // the LHS exceeds c) — verification drops it.
        ("sqrt(x) + sqrt(x+8) = 2", "No solution"),
        ("sqrt(x+1) + sqrt(x) = 0", "No solution"),
        // DIFFERENCE of two radicals `√f − √g = c`: the reduction flips the RHS sign and the
        // verification checks `√f − √g == c`, so the sign carries through.
        ("sqrt(x+5) - sqrt(x) = 1", "{ 4 }"),
        ("sqrt(3*x+1) - sqrt(x+4) = 1", "{ 5 }"),
        ("sqrt(x) - sqrt(x-3) = 1", "{ 4 }"),
        // A difference exceeding its bound, and a negatively-signed one, are dropped by verification.
        ("sqrt(x+5) - sqrt(x) = 10", "No solution"),
        ("sqrt(x) - sqrt(x+5) = 1", "No solution"),
        // MONOMIAL reduced RHS (`√(fg) = c·x`, no constant term): these returned a
        // wrong "No solution" (or dropped a root) because the single-radical solver
        // mishandles `√(quad) = c·x`. The reduction now squares to the POLYNOMIAL
        // `fg − reduced_rhs² = 0` and verifies, bypassing that solver. Cross-checked
        // vs sympy solveset.
        ("sqrt(5*x-1) - sqrt(x+2) = 1", "{ 2 }"),
        ("sqrt(4*x+1) - sqrt(x) = 1", "{ 0, 4/9 }"),
        ("sqrt(3*x+1) - sqrt(x) = 1", "{ 0, 1 }"),
        ("sqrt(2*x+7) - sqrt(x+3) = 1", "{ -3, 1 }"),
        ("sqrt(3*x+4) - sqrt(x) = 2", "{ 0, 4 }"),
        // EQUAL radicands with `c = 0` (the both-sides equality `√A = √B`): the candidate makes
        // both radicands equal but IRRATIONAL (√7 at x=2), so the verification must accept the
        // canceling surds rather than demanding each radicand be a perfect square.
        ("sqrt(2*x+3) = sqrt(x+5)", "{ 2 }"),
        ("sqrt(x+1) = sqrt(2*x-3)", "{ 4 }"),
        ("sqrt(2*x+8) - sqrt(x+5) = 0", "{ -3 }"),
        // Equal-slope radicands never meet: genuine no-solution stays no-solution.
        ("sqrt(x+3) - sqrt(x+5) = 0", "No solution"),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
    }
}
#[test]
fn test_eval_complex_angle_sum() {
    // Tanda-3 ciclo 1: argumento complejo MIXTO re+iθ — la suma de ángulos entera
    // (válida ∀ re,θ ∈ ℂ, sin guard, como el puente). Puro-imaginario sigue siendo
    // del puente; real mode intacto. ONE-DIRECTION (la contracción trig casa
    // cos/sin, jamás cosh/sinh — no existe lado de ping-pong).
    let rc = |input: &str| -> String {
        let out = cli()
            .args([
                "eval",
                input,
                "--value-domain",
                "complex",
                "--format",
                "json",
            ])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(rc("sin(1+i)"), "sin(1)·cosh(1) + i·cos(1)·sinh(1)");
    assert_eq!(rc("cos(1+i)"), "cos(1)·cosh(1) - i·sin(1)·sinh(1)");
    assert_eq!(rc("sin(x+i)"), "sin(x)·cosh(1) + i·cos(x)·sinh(1)");
    assert_eq!(rc("sin(2+3*i)"), "sin(2)·cosh(3) + i·cos(2)·sinh(3)");
    assert_eq!(rc("sinh(1+i)"), "cos(1)·sinh(1) + i·sin(1)·cosh(1)");
    assert_eq!(rc("cosh(1+i)"), "cos(1)·cosh(1) + i·sin(1)·sinh(1)");
    // tan compone vía Tan→Sin/Cos + esta regla: cociente honesto expandido.
    assert_eq!(
        rc("tan(1+i)"),
        "(sin(1)·cosh(1) + i·cos(1)·sinh(1)) / (cos(1)·cosh(1) - i·sin(1)·sinh(1))"
    );
    // Ownership intacto: puro-imaginario del puente, real sin i, real mode gated.
    assert_eq!(rc("sin(i)"), "i·sinh(1)");
    assert_eq!(rc("sin(2)"), "sin(2)");
    assert_eq!(r("sin(1+i)"), "sin(1 + i)");
    // Verificación cruzada con el walker (independiente de la regla).
    assert_eq!(rc("approx(sin(1+i))"), "1.29845758142 + 0.634963914785·i");
}
#[test]
fn test_eval_convergent_p_series_even_zeta() {
    // `sum(c/k^p, k, 1, inf)` with EVEN p has Euler's closed form c·ζ(2m) = c·(rational)·π^(2m).
    // Odd p (ζ(3), ζ(5), …, no known closed form in π), the divergent harmonic series (p=1), and
    // any lower bound ≠ 1 MUST stay honest residuals — solving them would be unsound.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("sum(1/k^2, k, 1, inf)"), "1/6·pi^2"); // ζ(2) = π²/6
    assert_eq!(r("sum(1/k^4, k, 1, inf)"), "1/90·pi^4"); // ζ(4) = π⁴/90
    assert_eq!(r("sum(1/k^6, k, 1, inf)"), "1/945·pi^6"); // ζ(6) = π⁶/945
    assert_eq!(r("sum(1/k^8, k, 1, inf)"), "1/9450·pi^8"); // ζ(8) = π⁸/9450
    assert_eq!(r("sum(2/k^2, k, 1, inf)"), "1/3·pi^2"); // 2·ζ(2) = π²/3
    assert_eq!(r("sum(k^(-2), k, 1, inf)"), "1/6·pi^2"); // negative-power form
                                                         // Honest residuals: no elementary closed form, or out of scope.
    assert_eq!(r("sum(1/k^3, k, 1, inf)"), "sum(1 / k^3, k, 1, infinity)"); // Apéry, odd
    assert_eq!(r("sum(1/k^5, k, 1, inf)"), "sum(1 / k^5, k, 1, infinity)"); // odd
    assert_eq!(r("sum(1/k^2, k, 2, inf)"), "sum(1 / k^2, k, 2, infinity)"); // lower bound ≠ 1
}
#[test]
fn test_eval_summation_pole_in_range_is_undefined() {
    // A finite or infinite sum whose summand has a POLE (a `1/0` term) at an integer in the range is
    // UNDEFINED — the telescoping/closed-form builders otherwise compute THROUGH it. The pole
    // detector folds `n^k` exactly (`as_rational_const` declines `Pow`, so a quadratic denominator's
    // root went undetected); ALL roots are checked, incl. the NEGATIVE one and the start itself.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Poles inside the range -> undefined (finite and infinite upper bound).
    assert_eq!(r("sum(1/(n^2-1), n, -2, 5)"), "undefined"); // poles at n=±1
    assert_eq!(r("sum(1/(n^2-4), n, -2, 5)"), "undefined"); // poles at n=±2 (n=-2 is the start)
    assert_eq!(r("sum(1/(n^2-1), n, -1, 5)"), "undefined"); // n=-1 start is itself a pole
    assert_eq!(r("sum(1/(n^2-1), n, 1, 5)"), "undefined");
    assert_eq!(r("sum(1/(n^2-1), n, -2, inf)"), "undefined"); // telescoped to -5/12 before
    assert_eq!(r("sum(1/(n^2-4), n, -2, inf)"), "undefined");
    // No pole in the range -> the exact value is unchanged.
    assert_eq!(r("sum(1/(n^2-1), n, 2, 5)"), "17/30");
    assert_eq!(r("sum(1/(n^2-1), n, -3, -2)"), "11/24");
    assert_eq!(r("sum(1/(n^2-1), n, 2, inf)"), "3/4");
    assert_eq!(r("sum(1/2^k, k, 0, inf)"), "2");
    assert_eq!(r("sum(k, k, -2, 5)"), "12");
}
#[test]
fn test_eval_sign_form_sum_partitions_at_poles() {
    // A SUM of ≥2 sign forms `Σ cᵢ·sign(gᵢ) {op} k` is a step function (the simplifier combines it over a
    // common denominator and the isolation path then returns "No solution" / a garbage residual). It now
    // partitions ℝ at the `gᵢ = 0` poles, evaluates the constant sum on each open region, and keeps the
    // satisfying ones — the poles excluded.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // `sign(x+1) + sign(x-1) > 0` is +2 only on `(1, ∞)`.
    assert_eq!(
        r("solve((x+1)/abs(x+1) + (x-1)/abs(x-1) > 0, x)"),
        "(1, infinity)"
    );
    assert_eq!(
        r("solve(x/abs(x) + (x-2)/abs(x-2) > 0, x)"),
        "(2, infinity)"
    );
    // A difference of signs (`sign(x) − sign(x-2)`) is +2 only between the poles.
    assert_eq!(r("solve(x/abs(x) - (x-2)/abs(x-2) > 0, x)"), "(0, 2)");
    // `= 0` keeps the middle region where the signs cancel.
    assert_eq!(
        r("solve((x+1)/abs(x+1) + (x-1)/abs(x-1) = 0, x)"),
        "(-1, 1)"
    );
    // Three terms, and a constant RHS on the sum.
    assert_eq!(
        r("solve(x/abs(x) + (x-1)/abs(x-1) + (x-2)/abs(x-2) > 1, x)"),
        "(2, infinity)"
    );
    assert_eq!(
        r("solve((x+1)/abs(x+1) + (x-1)/abs(x-1) = 2, x)"),
        "(1, infinity)"
    );
    // A weighted sum; `>= 0` holds on two regions, the pole between them excluded.
    assert_eq!(
        r("solve(2*x/abs(x) + (x-1)/abs(x-1) >= 0, x)"),
        "(0, 1) U (1, infinity)"
    );
    assert_eq!(
        r("solve(x/abs(x) + (x-2)/abs(x-2) >= 0, x)"),
        "(0, 2) U (2, infinity)"
    );
    assert_eq!(
        r("solve(x/abs(x) + (x-2)/abs(x-2) < 0, x)"),
        "(-infinity, 0)"
    );
    // Controls: a SINGLE sign form (n = 1) stays with the dedicated handler.
    assert_eq!(r("solve(x/abs(x) = 1, x)"), "(0, infinity)");
    assert_eq!(r("solve(x/abs(x) + 1 > 0, x)"), "(0, infinity)");
}
#[test]
fn test_eval_rational_sum_inequality_routing() {
    // SOUNDNESS regression: a rational-SUM inequality `x + c/x {op} k` (LHS an Add containing a
    // rational term) used to skip the reliable rational path and have its operator dropped, returning
    // the empty set (strict) or a degenerate point (non-strict). Now the LHS is combined into a single
    // fraction N/D and routed through the verified rational path. Truth cross-checked vs sympy.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("solve(x + 1/x > 2, x)"), "(0, 1) U (1, infinity)");
    assert_eq!(r("solve(x + 1/x < 2, x)"), "(-infinity, 0)");
    assert_eq!(r("solve(x + 1/x >= 2, x)"), "(0, infinity)");
    assert_eq!(r("solve(x + 2/x > 3, x)"), "(0, 1) U (2, infinity)");
    assert_eq!(r("solve(2*x + 1/x > 1, x)"), "(0, infinity)");
    assert_eq!(r("solve(x + 4/x >= 4, x)"), "(0, infinity)");
    assert_eq!(r("solve(x - 2 + 1/x > 0, x)"), "(0, 1) U (1, infinity)");
    assert_eq!(r("solve(x + 3 + 2/x > 0, x)"), "(-2, -1) U (0, infinity)");
    assert_eq!(r("solve(3*x + 12/x > 12, x)"), "(0, 2) U (2, infinity)");
    assert_eq!(r("solve(x + 9/x > 6, x)"), "(0, 3) U (3, infinity)");
    assert_eq!(r("solve(x + 1/(x-1) > 2, x)"), "(1, infinity)");
    assert_eq!(r("solve(2*x + 3/(x-1) > 5, x)"), "(1, infinity)");
    // Surd bounds: x + 1/x >= 3  ⟹  (0, (3-√5)/2] ∪ [(3+√5)/2, ∞).
    assert_eq!(
        r("solve(x + 1/x >= 3, x)"),
        "(0, 1/2·(3 - sqrt(5))] U [1/2·(sqrt(5) + 3), infinity)"
    );
    // Non-strict touch-point cases: the solution is a half-line PLUS the isolated touch point, which
    // requires unioning a Discrete point with a Continuous interval (previously the interval was
    // silently dropped, collapsing the answer to the lone point `[p, p]`).
    assert_eq!(r("solve(x + 1/x <= 2, x)"), "(-infinity, 0) U [1, 1]");
    assert_eq!(r("solve(x + 4/x <= 4, x)"), "(-infinity, 0) U [2, 2]");
    assert_eq!(r("solve(x + 9/x <= 6, x)"), "(-infinity, 0) U [3, 3]");
    // Controls: the single-fraction form and ordinary inequalities are unchanged.
    assert_eq!(r("solve((x^2+1)/x > 2, x)"), "(0, 1) U (1, infinity)");
    assert_eq!(r("solve(1/x < 1, x)"), "(-infinity, 0) U (1, infinity)");
    assert_eq!(r("solve(x^2 > 4, x)"), "(-infinity, -2) U (2, infinity)");
}
#[test]
fn test_eval_finite_geometric_sum_with_symbolic_ratio() {
    // A finite geometric sum with a SYMBOLIC ratio used to decline and echo
    // `sum(r^k, k, 0, n)`; the numeric-ratio builders only handle a rational
    // base. It now emits the closed form `(r^(n+1) - r^a)/(r - 1)` (removable
    // singularity at r=1, matching how the engine simplifies through such holes).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("sum(r^k, k, 0, n)"), "(r^(n + 1) - 1) / (r - 1)");
    assert_eq!(r("sum(r^k, k, 1, n)"), "(r^(n + 1) - r) / (r - 1)");
    assert_eq!(r("sum(x^k, k, 0, n)"), "(x^(n + 1) - 1) / (x - 1)");
    assert_eq!(r("sum((x+1)^k, k, 0, n)"), "((x + 1)^(n + 1) - 1) / x");
    // A numeric ratio keeps the cleaner integer-base form; a numeric upper bound
    // still expands directly. (The arithmetic-geometric `k·r^k` is closed by its
    // own sibling builder — see the arithmetic-geometric contract test.)
    assert_eq!(r("sum(2^k, k, 0, n)"), "2^(n + 1) - 1");
    assert_eq!(r("sum(x^k, k, 0, 3)"), "x^3 + x^2 + x + 1");
    // A leading coefficient (numeric or symbolic, index-free) is carried
    // through: `sum(c·r^k) = c·(r^(n+1) - r^a)/(r - 1)`.
    assert_eq!(r("sum(3*r^k, k, 0, n)"), "(3·r^(n + 1) - 3) / (r - 1)");
    assert_eq!(r("sum(5*r^k, k, 1, n)"), "(5·r^(n + 1) - 5·r) / (r - 1)");
    assert_eq!(r("sum(c*x^k, k, 0, n)"), "(c·x^(n + 1) - c) / (x - 1)");
    // The bare index `k` is NOT a coefficient — `k·r^k` stays with the
    // arithmetic-geometric builder, not hijacked into `k·(...)`.
    assert_eq!(
        r("sum(k*r^k, k, 1, n)"),
        "r·(n·r^(n + 1) + r^n·(-n - 1) + 1) / (1 - r)^2"
    );
}
#[test]
fn test_eval_finite_arithmetic_geometric_sum_with_symbolic_ratio() {
    // A finite arithmetic-geometric sum `sum(k·r^k)` with a SYMBOLIC ratio used
    // to decline: the numeric builder decomposes the ratio as a rational. It now
    // emits the closed form `r(1 - (n+1)r^n + n·r^(n+1))/(1-r)^2` (verified
    // numerically: at r=2, n=3 the value is 34).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(
        r("sum(k*r^k, k, 1, n)"),
        "r·(n·r^(n + 1) + r^n·(-n - 1) + 1) / (1 - r)^2"
    );
    // Lower bound 0 shares the formula (the k=0 term is zero).
    assert_eq!(
        r("sum(k*r^k, k, 0, n)"),
        "r·(n·r^(n + 1) + r^n·(-n - 1) + 1) / (1 - r)^2"
    );
    assert_eq!(
        r("sum(k*x^k, k, 1, n)"),
        "x·(n·x^(n + 1) + x^n·(-n - 1) + 1) / (1 - x)^2"
    );
    // The degree-2 cofactor `k^2*r^k` is now closed by its own sibling builder
    // (see the quadratic-geometric contract test); a lower bound >= 2 still
    // stays a residual (needs a symbolic head correction).
    assert_eq!(r("sum(k*r^k, k, 2, n)"), "sum(k·r^k, k, 2, n)");
    // The pure geometric sum (cycle sibling) is unaffected.
    assert_eq!(r("sum(r^k, k, 0, n)"), "(r^(n + 1) - 1) / (r - 1)");
}
#[test]
fn test_eval_finite_quadratic_geometric_sum_with_symbolic_ratio() {
    // `sum(k^2*r^k)` with a symbolic ratio: the numeric arithmetic-geometric
    // builder handles a rational ratio, but the symbolic case declined. It now
    // emits the `(1-r)^3` closed form (verified numerically: at r=2, n=3 -> 90).
    // The formula is large, so the SumRule must be budget-exempt or it is dropped.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(
        r("sum(k^2*r^k, k, 1, n)"),
        "r·(r^(n + 1)·(2·n^2 + 2·n - 1) + r + 1 - n^2·r^(n + 2) - r^n·(n + 1)^2) / (1 - r)^3"
    );
    assert_eq!(
        r("sum(k^2*x^k, k, 0, n)"),
        "x·(x^(n + 1)·(2·n^2 + 2·n - 1) + x + 1 - n^2·x^(n + 2) - x^n·(n + 1)^2) / (1 - x)^3"
    );
    // Siblings unchanged: degree-1 arith-geo, pure geometric, numeric ratio,
    // Faulhaber, and numeric-bound sums.
    assert_eq!(
        r("sum(k*r^k, k, 1, n)"),
        "r·(n·r^(n + 1) + r^n·(-n - 1) + 1) / (1 - r)^2"
    );
    assert_eq!(r("sum(r^k, k, 0, n)"), "(r^(n + 1) - 1) / (r - 1)");
    assert_eq!(r("sum(k^2, k, 1, n)"), "1/6·n·(n + 1)·(2·n + 1)");
    assert_eq!(r("sum(k, k, 1, 10)"), "55");
    // A degree-3 cofactor stays a residual.
    assert_eq!(r("sum(k^3*r^k, k, 1, n)"), "sum(k^3·r^k, k, 1, n)");
}
