use super::*;

#[test]
fn test_eval_abs_of_quadratic_equals_variable_splits_and_verifies() {
    // `|f(x)| = g(x)` with a degree-≥2 polynomial `f` and a variable RHS leaked an `arcsin`/`sqrt`
    // residual (the isolation path). Split into `f = ±g` and verify each root against the ORIGINAL
    // `|f(r)| = g(r)` (which enforces `g(r) ≥ 0`). Linear `|f|` (piecewise handler) and constant-RHS
    // (isolation, keeps surds) forms are untouched.
    for (input, expected) in [
        // `|x²−1| = √2` splits into `x² = 1±√2`; the `1−√2 < 0` branch is now DISPROVEN (an even root of
        // a provably-negative surd is non-real) and dropped, instead of leaking `±√(1−√2)` (imaginary).
        (
            "abs(x^2 - 1) = sqrt(2)",
            "{ sqrt(sqrt(2) + 1), -sqrt(sqrt(2) + 1) }",
        ),
        // `|E| = 0 ⟺ E = 0`: the FULL zero-set of a factored product (the abs isolation dropped all but
        // the first factor, `|x·(x−2)| = 0 → {0}`).
        ("abs(x*(x-2)) = 0", "{ 0, 2 }"),
        ("abs((x-1)*(x-3)*(x+2)) = 0", "{ -2, 1, 3 }"),
        ("abs(sin(x)) = 0", "{ k·pi : k ∈ ℤ }"),
        // `|x²−1| = x+1`: f=g ⟹ {2,−1}; f=−g ⟹ {0,−1}; all have g ≥ 0.
        ("abs(x^2 - 1) = x + 1", "{ -1, 2, 0 }"),
        ("abs(x^2 - 4) = x + 2", "{ -2, 3, 1 }"),
        // `|f| = |h|` needs no sign condition (both branches kept).
        ("abs(x^2 - 1) = abs(x + 1)", "{ -1, 2, 0 }"),
        // Verification DROPS roots where the RHS is negative.
        ("abs(x^2 - 2) = x", "{ 2, 1 }"),
        ("abs(x^2 - 1) = -x - 5", "No solution"),
        // Controls: linear `|f|` and constant-RHS quadratic keep their existing handlers.
        ("abs(x - 3) = 2*x", "{ 1 }"),
        ("abs(x^2 - 4) = 3", "{ sqrt(7), -sqrt(7), 1, -1 }"),
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
fn test_eval_abs_of_log_threshold_inequality_solves_both_branches() {
    // `|ln(x)| {op} c`: the two-sided reduction was ALREADY correct, but the interval
    // algebra downstream could not ORDER the transcendental endpoints (`e²` vs `1/e²`),
    // so the intersection collapsed (`< 2` → "No solution", `≤ 2` → `[e², e²]`) and the
    // union filled the gap (`> 2` → `(0, ∞)`). `compare_values` now decides constant
    // transcendental endpoints by the exact value-bounds oracle.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("solve(abs(ln(x)) < 2, x)"), "(1 / e^2, e^2)");
    assert_eq!(r("solve(abs(ln(x)) <= 2, x)"), "[1 / e^2, e^2]");
    assert_eq!(
        r("solve(abs(ln(x)) > 2, x)"),
        "(0, 1 / e^2) U (e^2, infinity)"
    );
    assert_eq!(
        r("solve(abs(ln(x)) >= 2, x)"),
        "(0, 1 / e^2] U [e^2, infinity)"
    );
    // The equation sibling stays as it was (already correct).
    assert_eq!(r("solve(abs(ln(x)) = 2, x)"), "{ e^2, 1 / e^2 }");
    // An exponential inside the abs: one side is vacuous (e^x − 1 > −2 always).
    assert_eq!(r("solve(abs(e^x - 1) < 2, x)"), "(-infinity, ln(3))");
    // Polynomial controls keep their surd-endpoint rendering.
    assert_eq!(r("solve(abs(x) < 2, x)"), "(-2, 2)");
}
#[test]
fn test_eval_rational_exponent_constants_are_sign_decidable() {
    // A constant `base^(p/q)` (`e^(1/3)`, `2^(1/3)`, ...) is now sign-decidable via
    // exact n-th-root value bounds (`const_sign::interval_pow`), closing the P0-F-log
    // family (an out-of-domain negative root `e^(1/3)/(1-e^(1/3))` was kept) and its
    // guard siblings (even-root threshold, abs-split, quadratic discriminant) that
    // previously only decided rationals/linear surds.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Log-equation domain filter: the candidate root is provably negative (x > 0
    // required) and the conditional's constant conditions are provably false, so the
    // whole thing prunes to a clean "No solution".
    assert_eq!(r("solve(ln(x)-ln(x+1)=1/3, x)"), "No solution");
    // Control: a positive in-domain root is KEPT (e^(-1/3)/(1-e^(-1/3)) ~ 2.53 > 0),
    // unconditionally (the coefficient 1 - e^(-1/3) is provably nonzero).
    assert_eq!(
        r("solve(ln(x)-ln(x+1)=-1/3, x)"),
        "{ e^(-1/3) / (1 - e^(-1/3)) }"
    );
    // Even-root RANGE correction with a transcendental-power threshold (`√ >= 0`).
    assert_eq!(r("solve(sqrt(x) < -e^(1/3), x)"), "No solution");
    assert_eq!(r("solve(sqrt(x) >= -2^(1/3), x)"), "[0, infinity)");
    // abs-split: the `x² = 1 - e^(1/3)` branch radicand is provably negative — the
    // spurious complex pair is dropped, keeping only the real pair.
    assert_eq!(
        r("solve(abs(x^2-1) = e^(1/3), x)"),
        "{ (e^(1/3) + 1)^(1/2), -((e^(1/3) + 1)^(1/2)) }"
    );
    // Quadratic with a provably-negative transcendental-power constant.
    assert_eq!(r("solve(x^2 = 1-e^(1/3), x)"), "No solution");
    // Control: positive threshold still squares (the sound branch).
    assert_eq!(r("solve(sqrt(x) > e^(1/3), x)"), "(cbrt(e^2), infinity)");
}
#[test]
fn test_eval_ln_of_even_numerator_power_uses_abs() {
    // `ln(x^(p/q))` with q ODD and p EVEN is real for EVERY x != 0 (under the
    // engine's real power semantics `(-8)^(2/3) = 4`), so it expands to
    // `(p/q)·ln|x|` over the domain x != 0. The engine used to emit `(p/q)·ln(x)`,
    // which wrongly NARROWS the domain to x > 0 (dropping the x < 0 branch).
    for (input, expected) in [
        ("ln(x^(2/3))", "2/3·ln(|x|)"),
        ("ln(x^(4/3))", "4/3·ln(|x|)"),
        ("ln(x^(2/5))", "2/5·ln(|x|)"),
        ("ln(x^(-2/3))", "-2/3·ln(|x|)"),
        ("ln(x^(6/3))", "2·ln(|x|)"), // reduces to the even integer 2
        // Even INTEGER already used |x|; unchanged.
        ("ln(x^2)", "2·ln(|x|)"),
        // ODD numerator keeps the sign of x -> domain x > 0, bare ln(x).
        ("ln(x^(1/3))", "1/3·ln(x)"),
        ("ln(x^(5/3))", "5/3·ln(x)"),
        ("ln(x^3)", "3·ln(x)"),
        // q EVEN forces x >= 0 already -> bare ln(x).
        ("ln(x^(1/2))", "1/2·ln(x)"),
        ("ln(x^(3/2))", "3/2·ln(x)"),
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
fn test_eval_sqrt_of_perfect_square_inequality_is_abs() {
    // `√(perfect square) {op} affine` is `|·| {op} affine`: `√(x²-6x+9) = |x-3|`. The
    // solve path used to keep the raw radical and emit a wrong conditional
    // (`√(x²-6x+9) > x-3 → "All real numbers if x-3 >= 0"`). Simplifying the sides before
    // the abs hook collapses `√(square) → |·|` so the exact segment method applies.
    for (input, expected) in [
        ("sqrt(x^2-6*x+9) > x-3", "(-infinity, 3)"),
        ("sqrt(x^2-6*x+9) <= x-3", "[3, infinity)"),
        ("sqrt((x-3)^2) > x-3", "(-infinity, 3)"),
        ("sqrt(x^2) > x", "(-infinity, 0)"),
        ("sqrt(x^2) >= x", "All real numbers"),
        ("sqrt(x^2) < x", "No solution"),
        ("sqrt((x-1)^2) <= x", "[1/2, infinity)"),
        ("sqrt((x-2)^2) < x", "(1, infinity)"),
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
fn test_eval_abs_threshold_and_ln_square_inequalities() {
    // `|g(x)| {op} c` (constant `c`) and `ln(x)^2 {op} c` are NON-MONOTONIC: the isolation/
    // split path dropped the operator and returned the boundary equation (`|x^2-2x| < 1` ->
    // "No solution"; `ln(x)^2 > 1` -> "All real numbers if x>0"). Both now reduce to the
    // two sub-inequalities the engine already solves exactly, intersected (`<`) or unioned
    // (`>`). The non-strict cases additionally exercise the closed-endpoint root filter, which
    // keeps the `e^{±√t}` band intact (`compare_values` cannot order `E`-bearing bounds).
    for (input, expected) in [
        // abs of a quadratic WITH a linear term (the symmetric `|x^2-k|` already worked).
        ("abs(x^2-2x) < 1", "(1 - sqrt(2), 1) U (1, sqrt(2) + 1)"),
        (
            "abs(x^2-2x) > 1",
            "(-infinity, 1 - sqrt(2)) U (sqrt(2) + 1, infinity)",
        ),
        ("abs(x^2-2x) <= 1", "[1 - sqrt(2), sqrt(2) + 1]"),
        ("abs(x^2-5x+6) < 2", "(1, 4)"),
        ("abs(x^2-5x+6) <= 2", "[1, 4]"),
        // c <= 0 edges.
        ("abs(x^2-2x) > 0", "(-infinity, 0) U (0, 2) U (2, infinity)"),
        // ln(x)^2 {op} c: strict and non-strict, integer and surd thresholds.
        ("ln(x)^2 > 1", "(0, 1 / e) U (e, infinity)"),
        ("ln(x)^2 < 1", "(1 / e, e)"),
        ("ln(x)^2 >= 1", "(0, 1 / e] U [e, infinity)"),
        ("ln(x)^2 <= 1", "[1 / e, e]"),
        ("ln(x)^2 <= 4", "[1 / e^2, e^2]"),
        ("ln(x)^2 < 4", "(1 / e^2, e^2)"),
        // ln(x)^2 c-edge cases: domain-aware, never a fabricated "All reals".
        ("ln(x)^2 > 0", "(0, 1) U (1, infinity)"),
        ("ln(x)^2 <= 0", "[1, 1]"),
        ("ln(x)^2 < -1", "No solution"),
        // Regression: genuinely-dropped isolated roots of non-strict inequalities survive.
        ("(x-2)^2*(x+1) <= 0", "(-infinity, -1] U [2, 2]"),
        ("x+1/x <= 2", "(-infinity, 0) U [1, 1]"),
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
fn test_eval_single_abs_inequality_uses_segment_method() {
    // A SINGLE `|f| {op} g` with an affine (non-constant) RHS used to fall to the
    // isolate-one-abs path, which solves the boundary EQUATION and returns the root
    // (`|x| > x+1 → {-1/2}`) or "No solution" instead of the interval. Route single-abs
    // INEQUALITIES through the exact piecewise/segment method (single-abs equations and
    // sum-of-abs are unchanged). Verified by a membership oracle over 300 cases.
    for (input, expected) in [
        ("abs(x) > x", "(-infinity, 0)"),
        ("abs(x) >= x", "All real numbers"),
        ("abs(x) < x", "No solution"),
        ("abs(x-3) > x-3", "(-infinity, 3)"),
        ("abs(x-1) <= x-1", "[1, infinity)"),
        ("abs(x) > x+1", "(-infinity, -1/2)"),
        ("abs(x) < x+1", "(-1/2, infinity)"),
        ("abs(2*x) <= x+3", "[-1, 3]"),
        ("abs(x-2) > 2*x", "(-infinity, 2/3)"),
        // Unchanged: abs vs constant, sum-of-abs, single-abs equation.
        ("abs(x) > 2", "(-infinity, -2) U (2, infinity)"),
        ("abs(x)+abs(x-1) < 3", "(-1, 2)"),
        ("abs(x) = x+1", "{ -1/2 }"),
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
fn test_eval_const_over_abs_denominator_vs_zero_reduces_to_sign() {
    // SOUNDNESS: `c/g {op} 0` with an abs INSIDE the denominator (`1/(|x|-1) < 0`)
    // fell to the generic rational path, which cannot find g's zeros through the
    // abs and returned garbage (`< 0 → ℝ`; `> 0 → (-∞,-∞)∪(∞,∞)`). Since `c/g` is
    // never 0 and shares g's sign, `c/g {op} 0 ⟺ g {op'} 0` with a STRICT op' (the
    // pole g=0 is undefined, not 0, so `≤/≥` collapse to `</>`). Delegate to the
    // abs solver.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // `< 0 ⟺ |x|-1 < 0 ⟺ |x| < 1`.
    assert_eq!(r("solve(1/(abs(x) - 1) < 0, x)"), "(-1, 1)");
    // `> 0 ⟺ |x|-1 > 0 ⟺ |x| > 1`.
    assert_eq!(
        r("solve(1/(abs(x) - 1) > 0, x)"),
        "(-infinity, -1) U (1, infinity)"
    );
    assert_eq!(
        r("solve(1/(abs(x) - 2) > 0, x)"),
        "(-infinity, -2) U (2, infinity)"
    );
    // Always-positive denominator: the reduction gives an always-true `|x|+1 > 0`.
    assert_eq!(r("solve(1/(abs(x) + 1) > 0, x)"), "All real numbers");
    // Shifted abs argument and a non-unit numerator constant.
    assert_eq!(
        r("solve(5/(abs(x-3) - 1) > 0, x)"),
        "(-infinity, 2) U (4, infinity)"
    );
    // Coefficiented abs argument.
    assert_eq!(
        r("solve(1/(abs(2*x) - 1) > 0, x)"),
        "(-infinity, -1/2) U (1/2, infinity)"
    );
    // Non-strict operators keep the pole OPEN (the value is undefined at g=0, not 0).
    assert_eq!(r("solve(1/(abs(x) - 1) <= 0, x)"), "(-1, 1)");
    assert_eq!(
        r("solve(1/(abs(x) - 1) >= 0, x)"),
        "(-infinity, -1) U (1, infinity)"
    );
    // Negative numerator flips the reduced sign: `-1/(|x|-1) < 0 ⟺ |x|-1 > 0`.
    assert_eq!(
        r("solve(-1/(abs(x) - 1) < 0, x)"),
        "(-infinity, -1) U (1, infinity)"
    );
    assert_eq!(r("solve(3/(abs(x) - 2) < 0, x)"), "(-2, 2)");

    // The reduction needs only the numerator's SIGN, decided exactly via the shared
    // const-sign chokepoint — so a surd (`√2`) or transcendental (`e−3`, `π`)
    // numerator works too, not just a rational.
    assert_eq!(
        r("solve(sqrt(2)/(abs(x) - 1) > 0, x)"),
        "(-infinity, -1) U (1, infinity)"
    );
    assert_eq!(r("solve(-sqrt(2)/(abs(x) - 1) > 0, x)"), "(-1, 1)");
    // `e − 3 < 0` flips the reduced sign; `π > 0` keeps it.
    assert_eq!(r("solve((e-3)/(abs(x) - 1) > 0, x)"), "(-1, 1)");
    assert_eq!(r("solve(pi/(abs(x) - 2) < 0, x)"), "(-2, 2)");

    // NO REGRESSION: non-abs reciprocal denominators, the bare `A/|g| {op} c` forms
    // (c ≠ 0), and equations keep their existing owners.
    assert_eq!(r("solve(1/(x - 1) < 0, x)"), "(-infinity, 1)");
    assert_eq!(r("solve(5/(x - 3) > 0, x)"), "(3, infinity)");
    assert_eq!(r("solve(1/abs(x) > 2, x)"), "(-1/2, 0) U (0, 1/2)");
}
#[test]
fn test_eval_reciprocal_power_inequality_keeps_pole_sign_split() {
    // `c/xⁿ {op} k` with an ODD `n ≥ 3` (or a surd-border even `n`) used to drop the sign-flip across
    // the x=0 pole, returning a complement / phantom ray / a union with the pole filled in. The
    // sign-split candidate is now ordered exactly (cube/4th/5th-root bounds) and verified, so each
    // case is the correct punctured union. Verified numerically against the ground-truth predicate.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(
        r("solve(2/x^3 > -1, x)"),
        "(-infinity, -cbrt(2)) U (0, infinity)"
    );
    assert_eq!(r("solve(1/x^3 > 2, x)"), "(0, cbrt(1/2))");
    assert_eq!(
        r("solve(1/x^3 < 2, x)"),
        "(-infinity, 0) U (cbrt(1/2), infinity)"
    );
    assert_eq!(r("solve(1/x^5 > 2, x)"), "(0, root(1/2, 5))");
    assert_eq!(r("solve(3/x^3 > 1, x)"), "(0, cbrt(3))");
    assert_eq!(r("solve(2/x^3 < -1, x)"), "(-cbrt(2), 0)");
    // Surd-border even power: the pole at 0 must be EXCLUDED (punctured union, not a single interval).
    assert_eq!(
        r("solve(1/x^4 > 1/4, x)"),
        "(-root(4, 4), 0) U (0, root(4, 4))"
    );
    // Controls that must stay correct: rational-border even power and linear denominator.
    assert_eq!(r("solve(1/x^2 > 1, x)"), "(-1, 0) U (0, 1)");
    assert_eq!(r("solve(1/x > 2, x)"), "(0, 1/2)");
}
#[test]
fn test_eval_sign_via_abs_excludes_pole() {
    // `g/|g| {op} c` is `sign(g) {op} c`, sign in {-1, +1} and undefined at g=0. The generic path
    // returned a CLOSED ray including the 0/0 point (`x/|x| = 1 -> [0, infinity)`) or "No solution" for
    // the inequality forms. It now reduces to a strict sign condition on g, with OPEN pole exclusion.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("solve(x/abs(x) = 1, x)"), "(0, infinity)");
    assert_eq!(r("solve(x/abs(x) = -1, x)"), "(-infinity, 0)");
    assert_eq!(r("solve(abs(x)/x = 1, x)"), "(0, infinity)");
    assert_eq!(r("solve((x-2)/abs(x-2) = 1, x)"), "(2, infinity)");
    assert_eq!(r("solve((x-2)/abs(x-2) >= 1, x)"), "(2, infinity)");
    assert_eq!(
        r("solve((x-2)/abs(x-2) <= 1, x)"),
        "(-infinity, 2) U (2, infinity)"
    );
    assert_eq!(r("solve((x-2)/abs(x-2) < 1, x)"), "(-infinity, 2)");
    // sign(g) is never 0 or out of {-1,+1}.
    assert_eq!(r("solve(x/abs(x) = 2, x)"), "No solution");
    assert_eq!(r("solve(x/abs(x) = 0, x)"), "No solution");
    // Controls: genuine abs equations/inequalities (denominator is not |numerator|) are unchanged.
    assert_eq!(r("solve(abs(x) = 3, x)"), "{ 3, -3 }");
    assert_eq!(r("solve(abs(x)/2 = 1, x)"), "{ -2, 2 }");
    assert_eq!(
        r("solve(abs(x-1) > 2, x)"),
        "(-infinity, -1) U (3, infinity)"
    );
}
#[test]
fn test_eval_sign_via_abs_with_coefficient_excludes_pole() {
    // The sign form carries a COEFFICIENT: `c·g/|g| = c·sign(g)`. The bare detector required the
    // numerator to equal the abs-argument exactly, so any coefficient (`-x/|x|`, `3x/|x|`) fell to the
    // generic path that returned a CLOSED ray including the `0/0` pole — or "No solution" for the
    // inequalities. Peeling `c` reduces to `sign(g) {op} k/c` (flipping a strict op when `c < 0`), with
    // OPEN pole exclusion.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Negative unit coefficient (`-sign(g)`): the ray flips and the pole is OPEN.
    assert_eq!(r("solve(-x/abs(x) = 1, x)"), "(-infinity, 0)");
    assert_eq!(r("solve(-x/abs(x) = -1, x)"), "(0, infinity)");
    assert_eq!(r("solve(-(x-2)/abs(x-2) = 1, x)"), "(-infinity, 2)");
    assert_eq!(r("solve(-3*x/abs(3*x) = 1, x)"), "(-infinity, 0)");
    // Negated inequalities were "No solution"; now the correct half-line.
    assert_eq!(r("solve(-x/abs(x) >= 1, x)"), "(-infinity, 0)");
    assert_eq!(r("solve(-x/abs(x) < 1, x)"), "(0, infinity)");
    // Positive coefficient ≠ 1 (the abs-argument is bare `x`, not `c·x`): also excludes the pole now.
    assert_eq!(r("solve(3*x/abs(x) = 3, x)"), "(0, infinity)");
    assert_eq!(r("solve(2*x/abs(x) = 2, x)"), "(0, infinity)");
    // `|g|/g` with a negated denominator: `|x|/(-x) = -sign(x)` (was a garbage conditional).
    assert_eq!(r("solve(abs(x)/(-x) = 1, x)"), "(-infinity, 0)");
    // ABS in the NUMERATOR with a coefficient/negation: `c·|g|/g = c·sign(g)`. `2*abs(x)/x` and
    // `-abs(x)/x` simplify to `Div(Mul(c, |x|), x)`, whose raw numerator is not a bare abs, so the
    // coefficient sibling of `|x|/x` returned a garbage `All real numbers if [linear] >= 0`
    // conditional (a wrong answer). Peeling `c` from BOTH sides of the division fixes it.
    assert_eq!(r("solve(-abs(x)/x = 1, x)"), "(-infinity, 0)");
    assert_eq!(r("solve(-abs(x)/x = -1, x)"), "(0, infinity)");
    assert_eq!(r("solve(2*abs(x)/x = 2, x)"), "(0, infinity)");
    assert_eq!(r("solve(-2*abs(x)/x = 2, x)"), "(-infinity, 0)");
    assert_eq!(r("solve(abs(x)/(2*x) = 1/2, x)"), "(0, infinity)");
    // Controls: the bare and matched-coefficient forms are unchanged; a rescaled RHS that no sign
    // value can hit is empty.
    assert_eq!(r("solve(x/abs(x) = 1, x)"), "(0, infinity)");
    assert_eq!(r("solve(2*x/abs(2*x) = 1, x)"), "(0, infinity)");
    assert_eq!(r("solve(-x/abs(x) = 2, x)"), "No solution");
    assert_eq!(r("solve(3*x/abs(x) = 2, x)"), "No solution");
}
#[test]
fn test_eval_sign_via_abs_with_additive_constant_excludes_pole() {
    // An ADDITIVE constant on the sign form (`sign(g) + d {op} k`) was not peeled, so the detector
    // declined and the generic path returned "No solution" (or a closed ray with the `0/0` pole). The
    // constant now folds into the reduced RHS: `coeff·sign(g) + offset {op} k ⟺ sign(g) {op} (k-offset)/coeff`.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // `sign(x) + 1 > 0` ⟺ `sign(x) > -1` ⟺ `x > 0` (pole open).
    assert_eq!(r("solve(x/abs(x) + 1 > 0, x)"), "(0, infinity)");
    assert_eq!(r("solve(x/abs(x) + 1 = 2, x)"), "(0, infinity)");
    assert_eq!(r("solve(x/abs(x) - 1 < 0, x)"), "(-infinity, 0)");
    assert_eq!(r("solve(x/abs(x) - 1 = 0, x)"), "(0, infinity)");
    // `sign(x) > -2` holds for both sign values, so the whole domain minus the pole.
    assert_eq!(
        r("solve(2 + x/abs(x) > 0, x)"),
        "(-infinity, 0) U (0, infinity)"
    );
    assert_eq!(
        r("solve(x/abs(x) + 2 > 0, x)"),
        "(-infinity, 0) U (0, infinity)"
    );
    // Negated sign with an offset: `-sign(x) + 1 > 0` ⟺ `sign(x) < 1` ⟺ `x < 0`. The `3 - sign(x)`
    // constant exceeds the sign range, so again everything but the pole.
    assert_eq!(r("solve(-x/abs(x) + 1 > 0, x)"), "(-infinity, 0)");
    assert_eq!(
        r("solve(3 - x/abs(x) > 0, x)"),
        "(-infinity, 0) U (0, infinity)"
    );
    // Control: no offset (Family 3) and an unreachable reduced RHS stay correct.
    assert_eq!(r("solve(x/abs(x) = 1, x)"), "(0, infinity)");
    assert_eq!(r("solve(x/abs(x) + 1 > 3, x)"), "No solution");
}
#[test]
fn test_eval_sign_form_equals_variable_rhs_splits_on_sign() {
    // `coeff·sign(g) + offset = h(x)` with a VARIABLE RHS (`x/|x| = x`) leaked a
    // malformed residual (the isolation cleared the denominator to `x = x·|x|`).
    // The sign form is a step function, so it splits on `sign(g) = ±1`: solve
    // `h = coeff+offset` on `g > 0` and `h = -coeff+offset` on `g < 0`, unioning
    // (the pole `g = 0` excluded by the STRICT branch). Verified by substitution.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("solve(x/abs(x) = x, x)"), "{ 1, -1 }");
    assert_eq!(r("solve(abs(x)/x = x, x)"), "{ 1, -1 }");
    // Coefficiented sign form: `2·sign(x) = x`.
    assert_eq!(r("solve(2*x/abs(x) = x, x)"), "{ 2, -2 }");
    assert_eq!(r("solve(x/abs(x) = 2*x, x)"), "{ 1/2, -1/2 }");
    // A branch's root can fall OUTSIDE its sign-domain and be dropped: `sign(x) = x²`
    // keeps only x=1 (x=-1 has sign -1 ≠ 1).
    assert_eq!(r("solve(x/abs(x) = x^2, x)"), "{ 1 }");
    assert_eq!(r("solve(x/abs(x) = x - 2, x)"), "{ 3 }");
    // `sign(x) = -x` and `-sign(x) = x` have NO solution (neither ±1 lands in its
    // own half-line) — the audit's stated "{-1,1}" was itself wrong.
    assert_eq!(r("solve(x/abs(x) = -x, x)"), "No solution");
    assert_eq!(r("solve(-x/abs(x) = x, x)"), "No solution");

    // NO REGRESSION: constant-RHS equations and inequalities keep their handler.
    assert_eq!(r("solve(x/abs(x) = 1, x)"), "(0, infinity)");
    assert_eq!(r("solve(-abs(x)/x = 1, x)"), "(-infinity, 0)");
    assert_eq!(r("solve(x/abs(x) > 0, x)"), "(0, infinity)");
}
#[test]
fn test_eval_abs_equation_quadratic_arg_split() {
    // `|arg(x)| = c` (constant `c ≥ 0`) with a quadratic argument carrying a linear term leaked a
    // circular residual `solve(x − (2x+3)^(1/2)=0)` from the recursive isolation, even though
    // `solve(x²-2x = 3)` returns `{-1, 3}`. The `|arg|=c → arg=±c` split now solves it.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("solve(abs(x^2-2*x)=3, x)"), "{ -1, 3 }");
    assert_eq!(r("solve(abs(x^2-x)=2, x)"), "{ -1, 2 }");
    assert_eq!(r("solve(abs(x^2-2*x)=0, x)"), "{ 0, 2 }"); // c = 0: single branch, no duplicate
    assert_eq!(r("solve(abs(x^2-2*x)=-1, x)"), "No solution"); // c < 0
                                                               // Both branches contribute: |x²-5x|=6 has four roots {-1, 2, 3, 6}.
    let four = r("solve(abs(x^2-5*x)=6, x)");
    assert!(
        four.contains("-1")
            && four.contains("6")
            && four.matches(", ").count() == 3
            && !four.contains("Solve"),
        "|x^2-5x|=6 -> {four}"
    );
    // Cases the normal path already solved are unchanged.
    assert_eq!(r("solve(abs(x^2+x)=2, x)"), "{ -2, 1 }");
    assert_eq!(r("solve(abs(x-3)=2, x)"), "{ 5, 1 }");
}
#[test]
fn test_eval_irreducible_polynomial_inequality_sign_analysis() {
    // An irreducible polynomial inequality was rewritten to `Equal(p, 0)`, dropping the operator and
    // returning the equation's root SET — so `> 0` and `< 0` gave byte-identical output. Sign analysis
    // over the (closed-form) real roots now yields the correct interval union, respecting the operator
    // and using open endpoints for strict ops, closed for non-strict.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Δ>0 cubic (one real root): half-line, operator-sensitive, no longer a root set.
    let gt = r("x^3+x+1>0");
    let lt = r("x^3+x+1<0");
    assert_ne!(gt, lt, "operator must matter (was the P0 defect)");
    assert!(
        gt.contains("infinity") && !gt.contains('{') && !gt.contains("Solve"),
        "x^3+x+1>0 -> {gt}"
    );
    assert!(gt.starts_with('(') && gt.ends_with("infinity)"), "{gt}"); // (r, infinity)
    assert!(lt.starts_with("(-infinity"), "{lt}"); // (-infinity, r)
                                                   // Non-strict closes the endpoint at the root.
    let geq = r("x^3+x+1>=0");
    assert!(geq.starts_with('[') && geq.ends_with("infinity)"), "{geq}");
    // Casus irreducibilis (three real roots): a two-piece interval union.
    let casus = r("x^3-3*x+1>0");
    assert!(
        casus.contains(" U ") && casus.contains("infinity") && !casus.contains('{'),
        "x^3-3x+1>0 -> {casus}"
    );
    assert_ne!(casus, r("x^3-3*x+1<0"), "operator must matter");
    // Controls: factorable inequalities and the underlying equation are unchanged.
    assert_eq!(r("x^2-1>0"), "(-infinity, -1) U (1, infinity)");
    assert_eq!(r("x^2-1<0"), "(-1, 1)");
    assert_eq!(r("x^3-2=0"), "{ cbrt(2) }");
}
#[test]
fn test_eval_rational_constant_inequality_sign_split() {
    // `N/D {op} c` with a polynomial denominator. With `P = N − c·D`, solve `P {op} 0`
    // where `D > 0` and `P {flip op} 0` where `D < 0` (poles excluded), then verify the
    // candidate numerically before returning. The general division path otherwise
    // reciprocates without flipping (`1/(x²+1) < 1/2 → (-1,1)`, `1/x³ < 8 → (-∞,1/2)`,
    // both wrong). Only verified candidates are emitted; an unorderable higher-surd
    // answer (`1/x⁴ > 1/4 → ±4^(1/4)`) declines to its prior behaviour rather than risk
    // a fresh wrong answer (next step: surd-aware interval ordering).
    for (input, expected) in [
        // Positive-definite quadratic denominators (D > 0 everywhere).
        ("1/(x^2+1) < 1/2", "(-infinity, -1) U (1, infinity)"),
        ("2/(x^2+1) < 1", "(-infinity, -1) U (1, infinity)"),
        ("1/(x^2+1) > 2", "No solution"),
        ("5/(x^2+4) <= 1", "(-infinity, -1] U [1, infinity)"),
        ("1/(x^2+1) < 0", "No solution"), // constant target, never holds
        ("1/(x^2+1) >= 0", "All real numbers"),
        // Even-power poles at 0 (D ≥ 0, vanishing at 0): the pole is excluded.
        ("1/x^2 < 4", "(-infinity, -1/2) U (1/2, infinity)"),
        ("1/x^2 > 4", "(-1/2, 0) U (0, 1/2)"),
        ("1/x^2 > 0", "(-infinity, 0) U (0, infinity)"),
        ("1/(x-1)^2 < 4", "(-infinity, 1/2) U (3/2, infinity)"),
        // Sign-varying denominators (linear, odd powers): the sign split flips on D < 0.
        ("1/(x+3) < 1/2", "(-infinity, -3) U (-1, infinity)"),
        ("1/x < 4", "(-infinity, 0) U (1/4, infinity)"),
        ("1/x^3 < 8", "(-infinity, 0) U (1/2, infinity)"),
        ("1/x^4 < 16", "(-infinity, -1/2) U (1/2, infinity)"),
        ("1/x^3 >= -1", "(-infinity, -1] U (0, infinity)"),
        ("2/x^4 >= 2", "[-1, 0) U (0, 1]"),
        // Quadratic-surd / golden-ratio endpoints, compared exactly during verification.
        ("5/x^2 > 1/4", "(-2·sqrt(5), 0) U (0, 2·sqrt(5))"),
        (
            "(1+x)/x^2 <= 1",
            "(-infinity, 1/2·(1 - sqrt(5))] U [phi, infinity)",
        ),
        // Numerator and denominator share a factor: the removable pole at 0 stays
        // excluded (NOT cancelled — `x/(x³−x) ≤ 0` is `(-1,0)∪(0,1)`, not `(-1,1)`).
        ("x/(x^3-x) <= 0", "(-1, 0) U (0, 1)"),
        // Reciprocal-power form `x^(-n) {op} c`: the splitter folds it to `c/x^n`, so it
        // routes through the same verified path (was a flipped/inverted wrong answer).
        ("x^(-2) > 4", "(-1/2, 0) U (0, 1/2)"),
        ("x^(-2) < 4", "(-infinity, -1/2) U (1/2, infinity)"),
        ("x^(-3) < 8", "(-infinity, 0) U (1/2, infinity)"),
        ("x^(-4) < 16", "(-infinity, -1/2) U (1/2, infinity)"),
        ("2*x^(-3) >= 2", "(0, 1]"),
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
fn test_eval_single_abs_affine_equation_recovers_instead_of_leaking() {
    // A single-abs equation that reorients to `var = α·|arg| + β` (the variable
    // ends up on the abs side, e.g. an effective negative slope) used to leak a
    // malformed nested-`solve` residual. It is piecewise-linear with one
    // breakpoint, so the shared exact segment core solves it. The decompose step
    // distributes a constant factor over the sum, so divided/scaled forms
    // (`2x + |x-1| = 3`, `(|x|+|x-1|)/2 = 1`) are handled too. Cross-checked
    // against an independent exact (fractions) oracle (0 mismatches).
    for (input, expected) in [
        // Reoriented `var = c - |arg|` (the previously-leaking shape).
        ("x + abs(x-1) = 3", "{ 2 }"),
        ("abs(x-1) = 3 - x", "{ 2 }"),
        ("x + abs(x-1) = 5", "{ 3 }"),
        ("abs(x-2) = 4 - x", "{ 3 }"),
        // Nested absolute value: the outer split feeds the single-abs solver.
        ("abs(x + abs(x-1)) = 3", "{ 2 }"),
        // Coefficient ≠ 1 on the variable / on the abs: the reorientation divides
        // by the leading coefficient, which the decompose step now distributes.
        ("2*x - abs(x) = 1", "{ 1 }"),
        ("2*x + 2*abs(x-2) + 1 = 6", "{ 9/4 }"),
        // Divided sum (top-level, ≥2 abs terms under a constant denominator).
        ("(abs(x) + abs(x-1))/2 = 1", "{ -1/2, 3/2 }"),
        // Degenerate-slope branch yields a ray, not a point.
        ("x = abs(x)", "[0, infinity)"),
        ("x - abs(x-2) = 0", "{ 1 }"),
        // Working single-abs cases are unchanged (positive-slope RHS path).
        ("abs(x-1) = x + 1", "{ 0 }"),
        ("abs(2*x-1) = x", "{ 1, 1/3 }"),
        ("abs(x) = 2*x - 3", "{ 3 }"),
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
fn test_eval_polynomial_in_absolute_value_substitutes_and_splits_branches() {
    // `|x|² − 3|x| + 2 = 0` reaches solve as `x² − 3|x| + 2 = 0` (the simplifier
    // folds `|x|² → x²`). Because `x² = |x|²` it is a quadratic in `u = |x|`:
    // `u² − 3u + 2 = 0 ⟹ u ∈ {1,2} ⟹ x ∈ {±1, ±2}`. It used to leak a malformed
    // `solve(x − √(3|x| − 2) = 0, …)` residual, dropping the negative branch and
    // every root.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(
        r("solve(abs(x)^2 - 3*abs(x) + 2 = 0, x)"),
        "{ 1, -1, 2, -2 }"
    );
    assert_eq!(r("solve(x^2 - 3*abs(x) + 2 = 0, x)"), "{ 1, -1, 2, -2 }");
    assert_eq!(
        r("solve(2*abs(x)^2 - 3*abs(x) + 1 = 0, x)"),
        "{ 1/2, -1/2, 1, -1 }"
    );
    // c = 0 border: `u(u−3) = 0 ⟹ u ∈ {0,3}`, and `u = 0 ⟹ x = 0` is a single root.
    assert_eq!(r("solve(abs(x)^2 - 3*abs(x) = 0, x)"), "{ 0, 3, -3 }");
    assert_eq!(r("solve(abs(x)^3 - abs(x) = 0, x)"), "{ 0, 1, -1 }");
    // Higher even degree: `u⁴ − 5u² + 4 = 0 ⟹ u ∈ {1,2}`.
    assert_eq!(r("solve(x^4 - 5*abs(x)^2 + 4 = 0, x)"), "{ -2, -1, 1, 2 }");
    // A negative `u`-root has no real pre-image and is dropped: `u = (1±√5)/2`,
    // keep only `φ`.
    assert_eq!(r("solve(abs(x)^2 - abs(x) - 1 = 0, x)"), "{ phi, -phi }");
    // Every `u`-root negative ⇒ no real solution.
    assert_eq!(r("solve(abs(x)^2 + 3*abs(x) + 2 = 0, x)"), "No solution");

    // GATES: a term that breaks evenness in x (`x + |x|`) is not a polynomial in
    // |x| — it declines here and the piecewise handler solves it.
    assert_eq!(r("solve(x + abs(x) - 4 = 0, x)"), "{ 2 }");
    assert_eq!(r("solve(x^2 + abs(x) - x = 0, x)"), "{ 0 }");
    // Plain polynomials and the degree-1 `|x|` isolation are untouched.
    assert_eq!(r("solve(x^2 - 3*x + 2 = 0, x)"), "{ 1, 2 }");
    assert_eq!(r("solve(abs(x) = 2, x)"), "{ 2, -2 }");
}
#[test]
fn test_eval_single_abs_equals_polynomial_solves_both_branches_with_domain() {
    // A single `|f|` term with a non-constant degree-≥2 remainder is `|f| = g(x)`.
    // Isolating the abs and recursing is unsound: the generic path solved only the
    // `f = g` branch and skipped `g ≥ 0`, so `x² + |x−1| − 3 = 0` returned the
    // spurious `{−2.56, 1.56}` (missing the real `−1`), and `x² − 3|x−1| + 2 = 0`
    // leaked a malformed residual. Both branches are now solved and each root kept
    // only when `g(r) ≥ 0` (decided exactly, so surd roots verify).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Was `{ -2.56, 1.56 }` (spurious root, missing `-1`).
    assert_eq!(
        r("solve(x^2 + abs(x-1) - 3 = 0, x)"),
        "{ 1/2·(sqrt(17) - 1), -1 }"
    );
    // Orientation-independent (abs on the RHS).
    assert_eq!(
        r("solve(3 - x^2 = abs(x-1), x)"),
        "{ 1/2·(sqrt(17) - 1), -1 }"
    );
    // Was a malformed `solve(x − √(3|x−1| − 2))` residual.
    assert_eq!(
        r("solve(x^2 - 3*abs(x-1) + 2 = 0, x)"),
        "{ 1/2·(-sqrt(13) - 3), 1/2·(sqrt(13) - 3) }"
    );
    // A coefficient on the abs term.
    assert_eq!(
        r("solve(x^2 + 2*abs(x-1) - 5 = 0, x)"),
        "{ 2·sqrt(2) - 1, -1 }"
    );
    // `g(r) ≥ 0` verification keeps the on-domain root and drops the off-domain
    // one: `|x−1| = x²−1` keeps `{1, −2}` (both have `x²−1 ≥ 0`).
    assert_eq!(r("solve(abs(x-1) = x^2 - 1, x)"), "{ 1, -2 }");
    // Every candidate has `g < 0` ⇒ no real solution.
    assert_eq!(r("solve(x^2 + abs(x-1) + 3 = 0, x)"), "No solution");
    assert_eq!(r("solve(abs(x-5) = -x^2 - 1, x)"), "No solution");

    // NO REGRESSION: linear `g` stays with the isolation path, constant `g` and
    // bare `|x|` polynomials with their own handlers, multi-abs with the
    // piecewise handler.
    assert_eq!(r("solve(abs(x-2) = x, x)"), "{ 1 }");
    assert_eq!(r("solve(abs(2*x-1) = x + 1, x)"), "{ 2, 0 }");
    assert_eq!(r("solve(abs(x^2-2*x) = 3, x)"), "{ -1, 3 }");
    assert_eq!(r("solve(x^2 - 3*abs(x) + 2 = 0, x)"), "{ 1, -1, 2, -2 }");
    assert_eq!(r("solve(abs(x-1) + abs(x+1) = 4, x)"), "{ -2, 2 }");
}
#[test]
fn test_eval_single_abs_polynomial_inequality_sign_splits_at_the_abs_zero() {
    // A polynomial inequality with a single `|f|` term was solved by an opaque
    // path that returned a WRONG "No solution" (`x² − 3|x| + 2 < 0` is
    // `(−2,−1) ∪ (1,2)`, not ∅). It now splits at `f = 0` into the `|f| = ±f`
    // branches, solves each, intersects with the branch domain, and unions.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(
        r("solve(abs(x)^2 - 3*abs(x) + 2 < 0, x)"),
        "(-2, -1) U (1, 2)"
    );
    assert_eq!(r("solve(x^2 - 3*abs(x) + 2 < 0, x)"), "(-2, -1) U (1, 2)");
    // The `>=` complement, with closed boundaries.
    assert_eq!(
        r("solve(x^2 - 3*abs(x) + 2 >= 0, x)"),
        "(-infinity, -2] U [-1, 1] U [2, infinity)"
    );
    // `<=` includes the boundary points.
    assert_eq!(r("solve(x^2 - 3*abs(x) + 2 <= 0, x)"), "[-2, -1] U [1, 2]");
    assert_eq!(
        r("solve(x^2 - abs(x) - 2 > 0, x)"),
        "(-infinity, -2) U (2, infinity)"
    );
    // Shifted abs argument (the split is at x = 1, not symmetric).
    assert_eq!(
        r("solve(x^2 - 3*abs(x-1) + 2 < 0, x)"),
        "(1/2·(-sqrt(13) - 3), 1/2·(sqrt(13) - 3))"
    );
    assert_eq!(
        r("solve(2*abs(x-1) + x^2 - 5 < 0, x)"),
        "(-1, 2·sqrt(2) - 1)"
    );
    // Always-signed remainders: empty / full without a spurious split.
    assert_eq!(r("solve(x^2 + abs(x) + 1 < 0, x)"), "No solution");
    assert_eq!(r("solve(x^2 + abs(x) + 1 > 0, x)"), "All real numbers");

    // NO REGRESSION: bare `|f| {op} c`, two-abs, sign-form, and top-level
    // `|quadratic| {op} c` keep their own handlers.
    assert_eq!(r("solve(abs(x-1) <= 2, x)"), "[-1, 3]");
    assert_eq!(r("solve(abs(x-1) < abs(x-3), x)"), "(-infinity, 2)");
    assert_eq!(r("solve(x/abs(x) = 1, x)"), "(0, infinity)");
}
#[test]
fn test_eval_single_abs_polynomial_equation_sign_splits_at_the_abs_zero() {
    // An EQUATION with a single `|f|` term entangled MULTIPLICATIVELY with a
    // polynomial (`x·|x| = 4`) is not `|f| = g` (isolated) nor a pure
    // polynomial-in-|x| (the odd `x` factor is not a function of `|x|`). The
    // isolation path reoriented to `x = 4/|x|` and leaked a malformed
    // `solve(x − 4/|x| = 0)` residual. The sign split at `f = 0` (same handler as
    // the inequality form) now solves each `|f| = ±f` polynomial branch and keeps
    // the roots on that branch's half-line.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // `x·|x|` is a strictly increasing bijection: exactly one real root, its sign
    // matching the RHS. Positive RHS lands in the `x ≥ 0` branch (`x² = c`).
    assert_eq!(r("solve(x*abs(x) = 4, x)"), "{ 2 }");
    assert_eq!(r("solve(x*abs(x) = 2, x)"), "{ sqrt(2) }");
    // Negative RHS lands in the `x < 0` branch (`−x² = c`).
    assert_eq!(r("solve(x*abs(x) = -4, x)"), "{ -2 }");
    assert_eq!(r("solve(x*abs(x) + 1 = 0, x)"), "{ -1 }");
    // Rational leading coefficient is cleared before the split.
    assert_eq!(r("solve(2*x*abs(x) = 8, x)"), "{ 2 }");
    // A quadratic branch keeps ALL in-domain roots and drops the out-of-domain
    // one: `x·|x| − x = 0` is `x(|x|−1) = 0` → `{−1, 0, 1}`.
    assert_eq!(r("solve(x*abs(x) - x = 0, x)"), "{ 0, 1, -1 }");
    assert_eq!(r("solve(x*abs(x) + 2*x = 3, x)"), "{ 1 }");
    // Shifted abs argument: the split is at x = 1, and `u·|u|` (u = x−1) is a
    // bijection, so a single root.
    assert_eq!(r("solve((x-1)*abs(x-1) = 4, x)"), "{ 3 }");
    assert_eq!(r("solve(x*abs(x-1) = 6, x)"), "{ 3 }");

    // NO REGRESSION: isolated-abs (`|f| = g`), poly-in-|x|, bare `|f| = c`, and
    // the sign form keep their own, already-correct equation handlers.
    assert_eq!(r("solve(abs(x) = 4, x)"), "{ 4, -4 }");
    assert_eq!(r("solve(x^2 - 3*abs(x) + 2 = 0, x)"), "{ 1, -1, 2, -2 }");
    assert_eq!(
        r("solve(abs(x-1) = 3 - x^2, x)"),
        "{ 1/2·(sqrt(17) - 1), -1 }"
    );
    assert_eq!(r("solve(x/abs(x) = 1, x)"), "(0, infinity)");
}
#[test]
fn test_eval_abs_as_a_factor_inequality_sign_splits() {
    // When the abs is a FACTOR rather than an added term (`|x|³ − |x| = |x|(x²−1)`),
    // removing it leaves a constant remainder, so the earlier "non-constant
    // remainder" gate wrongly declined and the generic path returned "No
    // solution". The gate now also fires on a degree-≥2 branch, so the sign
    // split still applies. `|x|³ − |x| < 0` is `(−1,0) ∪ (0,1)` — 0 excluded
    // (the value there is exactly 0).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("solve(abs(x)^3 - abs(x) < 0, x)"), "(-1, 0) U (0, 1)");
    assert_eq!(
        r("solve(abs(x)^3 - 4*abs(x) > 0, x)"),
        "(-infinity, -2) U (2, infinity)"
    );
    // `>=` includes the isolated zero at x = 0.
    assert_eq!(
        r("solve(abs(x)^3 - abs(x) >= 0, x)"),
        "(-infinity, -1] U [0, 0] U [1, infinity)"
    );
    // No regression on the added-term forms or bare `|f| {op} c`.
    assert_eq!(r("solve(x^2 - 3*abs(x) + 2 < 0, x)"), "(-2, -1) U (1, 2)");
    assert_eq!(r("solve(abs(x-1) <= 2, x)"), "[-1, 3]");
}
#[test]
fn test_eval_multi_abs_polynomial_relation_partitions_at_breakpoints() {
    // Two-or-more affine `|f|` terms PLUS a degree-≥2 remainder — the linear
    // sum-of-abs handler carries only a linear remainder, so `x² + |x−1| + |x+1|
    // < 5` used to return a wrong "No solution" (the true set is (1−√6, √6−1)).
    // Partition at the breakpoints and solve the polynomial per segment.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(
        r("solve(x^2 + abs(x-1) + abs(x+1) < 5, x)"),
        "(1 - sqrt(6), sqrt(6) - 1)"
    );
    // The equation form gives the two boundary points.
    assert_eq!(
        r("solve(x^2 + abs(x-1) + abs(x+1) = 5, x)"),
        "{ 1 - sqrt(6), sqrt(6) - 1 }"
    );
    // Minimum is 2 at x=0, so `<= 2` is the isolated point {0}.
    assert_eq!(r("solve(x^2 + abs(x-1) + abs(x+1) <= 2, x)"), "[0, 0]");
    assert_eq!(
        r("solve(x^2 - abs(x-1) - abs(x+1) > 0, x)"),
        "(-infinity, -2) U (2, infinity)"
    );
    // Three abs terms.
    assert_eq!(
        r("solve(x^2 + abs(x) + abs(x-1) + abs(x+1) < 6, x)"),
        "(1/2·(3 - sqrt(33)), 1/2·(sqrt(33) - 3))"
    );
    // Shifted breakpoints (min 4 at 0, so `< 5` is (-1, 1)).
    assert_eq!(r("solve(x^2 + abs(x-2) + abs(x+2) < 5, x)"), "(-1, 1)");
    // Empty result stays empty.
    assert_eq!(r("solve(x^2 - abs(x-1) - abs(x+1) < -3, x)"), "No solution");

    // NO REGRESSION: a LINEAR remainder keeps the existing sum-of-abs handler.
    assert_eq!(r("solve(abs(x-1) + abs(x+1) < 3, x)"), "(-3/2, 3/2)");
    assert_eq!(
        r("solve(abs(x) + abs(x-2) >= 4, x)"),
        "(-infinity, -1] U [3, infinity)"
    );
    // Single abs stays with the sign-split handler.
    assert_eq!(r("solve(x^2 - 3*abs(x) + 2 < 0, x)"), "(-2, -1) U (1, 2)");
}
