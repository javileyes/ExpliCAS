use super::*;

#[test]
fn test_eval_irrational_fractional_base_exponential_inequality_flips() {
    // `base^x {op} c` where the CONSTANT base is provably in (0, 1) but IRRATIONAL
    // (`sin(1)`, `cos(1)`): the `log(base, ·)` isolation must flip the direction
    // (decreasing exponential), decided by the exact value-bounds oracle. It used to
    // flip only for exact rationals, returning the reversed ray. Bases > 1 and the
    // equation form stay unflipped.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // (The `if sin(1) > 0` guard is PRUNED — the sign is provable — and the isolation
    // renders the equivalent `ln` ratio; the direction is what this contract fixes.)
    assert_eq!(
        r("solve(sin(1)^x > 2, x)"),
        "(-infinity, ln(2) / ln(sin(1)))"
    );
    assert_eq!(
        r("solve(cos(1)^x >= 3, x)"),
        "(-infinity, ln(3) / ln(cos(1))]"
    );
    assert_eq!(
        r("solve(sin(1)^x < 2, x)"),
        "(ln(2) / ln(sin(1)), infinity)"
    );
    // Controls: base > 1 keeps direction; equations never flip; symbolic base untouched.
    assert_eq!(r("solve(pi^x > 5, x)"), "(ln(5) / ln(pi), infinity)");
    assert_eq!(r("solve(sin(1)^x = 2, x)"), "{ ln(2) / ln(sin(1)) }");
    assert_eq!(r("solve(a^x > 2, x)"), "(log(a, 2), infinity) if a > 0");
}
#[test]
fn test_eval_const_over_surd_affine_denominator_keeps_true_pole_only() {
    // `c/(a·x + b) {op} 0` with a NON-RATIONAL constant intercept `b`: the simplifier
    // rationalizes the denominator through its conjugate (`1/(x+√2) → (√2−x)/(2−x²)`),
    // fabricating a spurious REMOVABLE pole at the conjugate that the rational-inequality
    // path punched out of the answer (`(−√2,√2)∪(√2,∞)`), collapsed odd-root denominators
    // to a false "No solution", and returned the conjugate as a root of `c/g = 0`. The
    // raw-tree reduction `c/g {op} 0 ⟺ g {op'} 0` keeps only the TRUE pole.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("solve(1/(x+sqrt(2))>0, x)"), "(-sqrt(2), infinity)");
    assert_eq!(r("solve(-1/(x-sqrt(2))>0, x)"), "(-infinity, sqrt(2))");
    assert_eq!(r("solve(2/(x+sqrt(3))>=0, x)"), "(-sqrt(3), infinity)");
    assert_eq!(r("solve(1/(2*x+sqrt(2))>0, x)"), "(-(2^(-1/2)), infinity)");
    assert_eq!(r("solve(1/(x+2^(1/3))>0, x)"), "(-(2^(1/3)), infinity)");
    assert_eq!(r("solve(1/(1+sqrt(2)-x)>0, x)"), "(-infinity, sqrt(2) + 1)");
    // Non-strict + negative constant: the `≥` split's equation branch must NOT
    // resurrect the conjugate as a boundary singleton.
    assert_eq!(
        r("solve(-2/3/(2*x+sqrt(2))>=0, x)"),
        "(-infinity, -(2^(-1/2)))"
    );
    // A nonzero constant over ANYTHING is never zero (raw check, before the
    // rationalizer plants a conjugate numerator root).
    assert_eq!(r("solve(-2/3/(2*x+sqrt(2))=0, x)"), "No solution");
    // Nonzero thresholds solve in `u = g(x)` space (all-rational breakpoints) and map
    // back through the affine — these previously returned a false "No solution" or a
    // malformed residual.
    // El extremo negativo imprimía `-2·2^(-1/2)`, la forma sin plegar que el
    // hermano positivo sí plegaba: la corrección del signo en la combinación de
    // potencias (2026-07-28) lo deja en `-sqrt(2)`, el MISMO número y la
    // misma forma que usa el resto de la expresión. El extremo `1 - sqrt(2)`
    // no cambia, que es la señal de que solo se normalizó la presentación.
    assert_eq!(r("solve(1/(x+sqrt(2))>1, x)"), "(-sqrt(2), 1 - sqrt(2))");
    assert_eq!(
        r("solve(1/(x+sqrt(2))<1, x)"),
        "(-infinity, -sqrt(2)) U (1 - sqrt(2), infinity)"
    );
    assert_eq!(r("solve(1/(x+sqrt(2))=1, x)"), "{ 1 - sqrt(2) }");
    assert_eq!(r("solve(2/(x-sqrt(3))<=-1, x)"), "[sqrt(3) - 2, sqrt(3))");
    // Orientation flips for a negative slope.
    assert_eq!(r("solve(1/(-x+sqrt(2))>2, x)"), "(sqrt(2) - 1/2, sqrt(2))");
    // Controls: rational pole, symbolic intercept, bare 1/x, and the equation
    // forms with a variable numerator keep their owners.
    assert_eq!(r("solve(1/(x-2)>0, x)"), "(2, infinity)");
    assert_eq!(r("solve(1/(x+a)>0, x)"), "(-a, infinity)");
    assert_eq!(r("solve(1/x>0, x)"), "(0, infinity)");
    assert_eq!(r("solve(x/(x-2)=0, x)"), "{ 0 }");
    assert_eq!(r("solve(1/(x-2)=3, x)"), "{ 7/3 }");
    assert_eq!(r("solve(1/(x-2)>1, x)"), "(2, 3)");
}
#[test]
fn test_eval_rational_power_polynomial_equation_solves_by_substitution() {
    // Equations that are a polynomial of degree >= 2 in x^(1/q) (a
    // quadratic-in-disguise) used to leak a malformed internal `Solve: solve(...)`
    // residual under ok=true and drop every root. They are now solved by the
    // u = x^(1/q) substitution, with the correct real-root domain on
    // back-substitution: even q drops negative u-roots, odd q keeps them.
    // Cross-checked against an independent exact (fractions) oracle over 300
    // random cases (0 mismatches).
    for (input, expected) in [
        // Quadratic in sqrt(x): even root, both u-roots non-negative.
        ("x - 3*sqrt(x) + 2 = 0", "{ 1, 4 }"),
        ("x - 5*sqrt(x) + 6 = 0", "{ 4, 9 }"),
        // A negative u-root is dropped by the even-root domain (sqrt(x) = -3 has no
        // real solution), leaving only the valid root.
        ("x + sqrt(x) - 6 = 0", "{ 4 }"),
        // Quadratic in x^(1/3): the ODD root keeps the negative u-root (x^(1/3) = -1).
        ("x^(2/3) - x^(1/3) - 2 = 0", "{ -1, 8 }"),
        ("x^(2/3) + x^(1/3) - 6 = 0", "{ -27, 8 }"),
        // sqrt(x)^2 normalizes to x; still a quadratic in sqrt(x).
        ("sqrt(x)^2 - 3*sqrt(x) + 2 = 0", "{ 1, 4 }"),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
    }

    // Must NOT disturb the existing paths: plain polynomials, exponential
    // substitution, single-power equations, and surd-root quadratics are unchanged.
    for (input, expected) in [
        ("x^4 - 5*x^2 + 4 = 0", "{ -2, -1, 1, 2 }"),
        ("e^(2*x) - 3*e^x + 2 = 0", "{ ln(2), 0 }"),
        ("sqrt(x) = 2", "{ 4 }"),
        ("x^2 - 5*x + 6 = 0", "{ 2, 3 }"),
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
fn test_eval_log_polynomial_equation_solves_by_substitution() {
    // Equations that are a polynomial of degree >= 2 in ln(x) used to leak a
    // malformed `Solve: solve(x - e^(...))` residual and drop every root. They are
    // now solved by the u = ln(x) substitution, back-substituting ln(x) = u_root
    // (= e^(u_root), the existing path with the ln domain). Cross-checked against
    // an independent oracle over 250 random ln-polynomials (0 mismatches).
    for (input, expected) in [
        ("ln(x)^2 - ln(x) - 2 = 0", "{ 1 / e, e^2 }"),
        ("ln(x)^2 - 3*ln(x) + 2 = 0", "{ e, e^2 }"),
        ("ln(x)^2 = ln(x)", "{ 1, e }"),
        ("ln(x)^2 - 1 = 0", "{ 1 / e, e }"),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
    }

    // Single-log equations are unchanged (degree-1, handled directly).
    for (input, expected) in [("ln(x) = 2", "{ e^2 }"), ("ln(x) - 1 = 0", "{ e }")] {
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
fn test_eval_single_radical_equals_polynomial_squares_and_verifies() {
    // `√(quadratic) = polynomial` (`√(5x²+9x−2) = 3x`): the isolation core
    // mis-filtered after squaring — a wrong "No solution" (true `{1/4, 2}`) or a
    // dropped root (`√(5x²+9x) = 3x → {0}`, missing `9/4`). Square exactly to
    // `f − g² = 0`, solve, and keep roots with `g(r) ≥ 0`. Cross-checked vs sympy.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // The confirmed wrong-answers, now fixed.
    assert_eq!(r("solve(sqrt(5*x^2+9*x-2) = 3*x, x)"), "{ 1/4, 2 }");
    assert_eq!(r("solve(sqrt(5*x^2+9*x) = 3*x, x)"), "{ 0, 9/4 }");
    assert_eq!(r("solve(sqrt(x^2-4) = x-1, x)"), "{ 5/2 }");
    // `g(r) < 0` extraneous roots dropped: `√(6x²+x−1) = 2x` has candidates {1/2, −1};
    // only 1/2 has `2x ≥ 0`.
    assert_eq!(r("solve(sqrt(6*x^2+x-1) = 2*x, x)"), "{ 1/2 }");
    // Squared quadratic with complex reduced roots, and a constant `f − g²`, stay
    // "No solution".
    assert_eq!(r("solve(sqrt(9*x^2+1) = 3*x, x)"), "No solution");
    assert_eq!(r("solve(sqrt(x^2+5*x+6) = x+1, x)"), "No solution");

    // NO REGRESSION: previously-correct degree-2 radicands (rational and surd
    // roots), degree-1 radicands (isolation path), and the perfect-square identity
    // are unchanged.
    assert_eq!(r("solve(sqrt(2*x^2+x) = 2*x, x)"), "{ 0, 1/2 }");
    assert_eq!(r("solve(sqrt(x^2+7*x) = x+3, x)"), "{ 9 }");
    assert_eq!(
        r("solve(sqrt(3*x^2+5*x-2) = 2*x, x)"),
        "{ 1/2·(5 - sqrt(17)), 1/2·(sqrt(17) + 5) }"
    );
    assert_eq!(r("solve(sqrt(x+1) = 2, x)"), "{ 3 }");
}
#[test]
fn test_eval_radical_inequality_keeps_argument_domain() {
    // `sqrt(g(x)) {<,<=} c` requires g(x) >= 0, but for a COMPOUND argument the
    // engine dropped that domain, returning e.g. `sqrt(x-1) < 3 → (-inf, 10)`
    // (which wrongly includes points where the radicand is negative) instead of
    // `[1, 10)`. The fix intersects with the solved argument domain `g(x) >= 0`
    // (`g(x) > 0` for ln). Ground truth cross-checked against sympy.
    for (input, expected) in [
        ("sqrt(x-1) < 3", "[1, 10)"),
        ("sqrt(2*x-1) <= 3", "[1/2, 5]"),
        // Bare-variable argument unchanged.
        ("sqrt(x) < 2", "[0, 4)"),
        ("sqrt(x) >= 2", "[4, infinity)"),
        // `>` / `>=` already implied the domain via the bound; still correct.
        ("sqrt(x-1) > 2", "(5, infinity)"),
        ("sqrt(x+2) > 1", "(-1, infinity)"),
        // Range correction (sqrt ≥ 0): a negative upper threshold is impossible.
        ("sqrt(x-1) < -1", "No solution"),
        // A negative SURD threshold: decided exactly (`√x < −√2` impossible; `√x > −√2` holds on the
        // whole domain) — it used to fall through to the unsound squaring branch (`√x < −√2 → [0,2)`).
        ("sqrt(x) < -sqrt(2)", "No solution"),
        ("sqrt(x) > -sqrt(2)", "[0, infinity)"),
        ("sqrt(x-1) >= -sqrt(3)", "[1, infinity)"),
        // sqrt(g) <= 0 forces g = 0: a single point in the domain (a degenerate interval `[p, p]`).
        ("sqrt(x+3) <= 0", "[-3, -3]"),
        // ln argument domain is g(x) > 0 (open).
        ("ln(x-1) < 0", "(1, 2)"),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
    }

    // Quadratic radicand: the domain `x²-4 >= 0` splits the solution into two
    // intervals (the lone interval before the fix dropped the |x|>=2 domain).
    // The `-√13` lower bound renders via the existing surd-bound style; assert
    // the structural domain split rather than the exact surd spelling.
    let output = cli()
        .args(["eval", "sqrt(x^2-4) < 3", "--format", "json"])
        .output()
        .expect("Failed to run CLI");
    let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
    let result = wire["result"].as_str().expect("result string");
    assert!(
        result.contains("-2]") && result.contains("[2,") && result.contains(" U "),
        "sqrt(x^2-4) < 3 must split on the x²-4>=0 domain, got {result:?}"
    );
}
#[test]
fn test_eval_radical_inequality_case_splits_on_rhs_sign() {
    // A radical inequality `√f {op} g` with a NON-constant RHS must case-split on
    // the sign of g, not square blindly. Squaring loses the RHS-sign branches:
    // `√x < x-2` is `(4, ∞)` (the `[0,1)` the naive square keeps fails `g > 0`),
    // and `√(x-2) > 4-x` is `(3, ∞)` (`4-x < 0` already satisfies `>` for x > 4).
    // For a LINEAR radicand the domain `f >= 0` is rational-bounded, so the
    // case-split intersections compare rational-vs-surd endpoints exactly. Verified
    // against an independent membership oracle over 350 random cases (0 mismatches).
    for (input, expected) in [
        ("sqrt(x) < x-2", "(4, infinity)"),
        ("sqrt(x) < x+1", "[0, infinity)"),
        ("sqrt(x+1) > x-1", "[-1, 3)"),
        ("sqrt(x-2) > 4-x", "(3, infinity)"),
        // Non-strict touch point `√f = g = 0` is an isolated solution the squared
        // intersection drops as a degenerate overlap; recovered via `solve(√f = g)` (rendered `[p, p]`).
        ("sqrt(x+3) <= -x-3", "[-3, -3]"),
        // Detached point unioned with an interval: `√0 = 0 = -2+2` AND [0, ∞).
        ("sqrt(2*x+4) <= x+2", "[-2, -2] U [0, infinity)"),
        // Boundary at the open endpoint of a non-empty branch stays closed.
        ("sqrt(2*x+4) <= 2*x-2", "[5/2, infinity)"),
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
fn test_eval_radical_inequality_fractional_constant_and_degenerate() {
    // Hardening cases an adversarial workflow surfaced (all previously wrong):
    for (input, expected) in [
        // FRACTIONAL RHS slope: g² must be built EXPANDED, not Pow(g,2) (the factored
        // form dropped the squared outer rational factor → wrong "No solution").
        ("sqrt(x) < x/2 - 3", "(2·(sqrt(7) + 4), infinity)"),
        ("sqrt(4*x+2) >= (1/2)*x - 6", "[-1/2, 2·(sqrt(66) + 10)]"),
        // Fractional RHS in a NON-STRICT branch: the boundary `√f = g` is now solved as
        // the polynomial `f = g² ∧ g >= 0` (the radical-equation solver leaks on
        // fractional g). The `[2, ...]` endpoint stays CLOSED.
        (
            "sqrt(x^2-4) <= (1/2)*x+5",
            "[2/3·(5 - 4·sqrt(7)), -2] U [2, 2/3·(4·sqrt(7) + 5)]",
        ),
        // Boundary touch with fractional g: `√(9-x²) = (1/3)x-1` at x=3 (`√0=0`). The
        // single-radical equation solver now resolves this boundary (it previously
        // leaked, so the non-strict root re-union was skipped), so x=3 re-unions as the
        // degenerate `[3, 3]` — the engine's standard form for a point-only non-strict
        // solution (`x² <= 0 → [0, 0]`), not the bug-dependent `{ 3 }`.
        ("sqrt(-x^2+9) <= (1/3)*x-1", "[3, 3]"),
        // CONSTANT g: `solve(const, x)` errors, so the sign is taken from the constant.
        ("sqrt(4-x^2) < 5", "[-2, 2]"),
        ("sqrt(x-2) >= 0*x - 4", "[2, infinity)"),
        // DEGENERATE radicand: `-x²` has domain {0}; the single-point `f >= 0` must
        // survive the case-split intersections (a bare Discrete operand collapsed to ∅).
        ("sqrt(-x^2) < x+1", "{ 0 }"),
        ("sqrt(-(x-1)^2) < x", "{ 1 }"),
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
fn test_eval_radical_equation_drops_extraneous_root_via_rhs_sign() {
    // A single-radical equation `√f = g` carries the range constraint `g ≥ 0` (√ is nonnegative);
    // squaring loses it, so the solver returned BOTH quadratic roots. `√(x+1) = -x` yielded
    // `{φ, ½(1-√5)}`, but `φ > 0` makes the RHS `-x < 0` — extraneous. Recording `NonNegative(g)` lets
    // the EXACT surd-sign prover drop it. The golden-ratio root is the named constant `phi`, whose sign
    // the surd parser cannot read; the `const_value_bounds` fallback (arbitrary-precision interval
    // arithmetic) decides `-phi < 0` exactly. A valid root has `g = √f ≥ 0`, so this never overdrops.
    for (input, expected) in [
        ("sqrt(x+1) = -x", "{ 1/2·(1 - sqrt(5)) }"),
        ("sqrt(x+1) = -1*x", "{ 1/2·(1 - sqrt(5)) }"),
        // φ is VALID here (`√(φ+1) = φ`), so it must be KEPT — the condition `x ≥ 0` holds at φ.
        ("sqrt(x+1) = x", "{ phi }"),
        // RATIONAL squared-roots already filtered, but the condition is consistent.
        ("sqrt(x+6) = -x", "{ -2 }"),
        ("sqrt(x) = x - 2", "{ 4 }"),
        ("sqrt(x-1) = x - 3", "{ 5 }"),
        ("sqrt(x+1) = x - 1", "{ 3 }"),
        // Surd squared-roots with a non-unit RHS slope stay correct.
        ("sqrt(x+1) = -2*x", "{ 1/8·(1 - sqrt(17)) }"),
        // No real root survives the RHS-sign constraint.
        ("sqrt(x-1) = -x", "No solution"),
        // Pure isolation (constant RHS) unaffected.
        ("sqrt(x) = 2", "{ 4 }"),
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
fn test_eval_irreducible_cubic_single_real_root_by_cardano() {
    // An irreducible cubic (no rational root) with a SINGLE real root (Cardano Δ > 0) is solved
    // exactly by radicals instead of leaking a residual. The root is `∛(-q/2+√Δ) + ∛(-q/2-√Δ) - B/3`
    // (real cube roots). These are numerically verified to satisfy the cubic in the dev probes
    // (e.g. `x³+x²+3` → −1.8637, `x³-x-1` → the plastic number 1.3247).
    for input in [
        "solve(x^3+x^2+3=0, x)",
        "solve(x^3-2*x^2-4*x-2=0, x)",
        "solve(x^3+x-1=0, x)",
        "solve(x^3-x-1=0, x)",
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        let result = wire["result"].as_str().unwrap_or("");
        // A single discrete real root expressed by radicals: `{ … }`, not a residual `Solve:`/`if`.
        assert!(
            result.starts_with("{ ")
                && result.contains("cbrt(")
                && !result.contains("Solve")
                && !result.contains(" if "),
            "{input} -> {result}"
        );
    }
    // FACTOR case: a higher-degree polynomial `(rational factors)·(irreducible Δ>0 cubic)` peels its
    // rational roots, then solves the leftover cubic by Cardano and unions — previously the cubic
    // factor's real root was silently dropped (`x⁴+x³+3x → {0}` lost the root of `x³+x²+3`). The
    // rational roots are reported as a DISTINCT set (the `x²` factor's double `0` collapses to one).
    for input in [
        "solve(x^4+x^3+3*x=0, x)",         // x·(x³+x²+3)
        "solve(x^4-2*x^3-4*x^2-2*x=0, x)", // x·(x³-2x²-4x-2)
        "solve(x^5+x^4+3*x^2=0, x)",       // x²·(x³+x²+3), double 0 deduped
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        let result = wire["result"].as_str().unwrap_or("");
        // `{ 0, <cubic radical root> }`: rational root 0 plus the cubic's single real root, no residual,
        // no duplicate `0`.
        assert!(
            result.starts_with("{ 0, ")
                && result.contains("cbrt(")
                && !result.contains("0, 0")
                && !result.contains("Solve")
                && !result.contains(" if "),
            "{input} -> {result}"
        );
    }
    // Rational-root and clean cubics are unaffected (NOT routed to Cardano).
    for (input, expected) in [
        ("solve(x^3-1=0, x)", "{ 1 }"),
        ("solve(x^3-6*x^2+11*x-6=0, x)", "{ 1, 2, 3 }"),
        ("solve(x^3-2=0, x)", "{ cbrt(2) }"),
        ("solve(x^3+3*x^2+3*x+1=0, x)", "{ -1 }"),
        ("solve(x^3-3*x+2=0, x)", "{ -2, 1 }"),
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
fn test_eval_casus_irreducibilis_cubic_three_real_roots() {
    // The casus irreducibilis: an irreducible cubic with Δ < 0 has THREE distinct real roots that
    // cannot be written with real radicals, so they are emitted in trigonometric form
    // `2√(-p/3)·cos(φ/3 - 2πk/3) - B/3` (the engine collapses special arccos values to sin/cos
    // ratios). Each root is numerically verified to satisfy its cubic in the dev probes
    // (e.g. `x³-3x+1` → {1.532, 0.347, -1.879}, `x³-7x+7` → {1.692, 1.357, -3.049}).
    let three_root_cases = [
        "solve(x^3-3*x+1=0, x)",
        "solve(x^3-7*x+7=0, x)",
        "solve(x^3-3*x^2+1=0, x)",
    ];
    for input in three_root_cases {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        let result = wire["result"].as_str().unwrap_or("");
        // Three real roots in a trig closed form: `{ a, b, c }`, no residual.
        assert!(
            result.starts_with("{ ")
                && (result.contains("cos(") || result.contains("sin("))
                && result.matches(", ").count() == 2
                && !result.contains("Solve")
                && !result.contains(" if "),
            "{input} -> {result}"
        );
    }
    // As a FACTOR of a higher-degree polynomial, the casus-irreducibilis cubic is now also solved:
    // `x⁴-3x²+x = x·(x³-3x+1)` yields the rational root 0 plus the three trig roots (4 total).
    let factor = cli()
        .args(["eval", "solve(x^4-3*x^2+x=0, x)", "--format", "json"])
        .output()
        .expect("Failed to run CLI");
    assert!(factor.status.success());
    let fwire: Value = serde_json::from_slice(&factor.stdout).expect("Invalid wire output");
    let fresult = fwire["result"].as_str().unwrap_or("");
    assert!(
        fresult.starts_with("{ 0, ")
            && (fresult.contains("cos(") || fresult.contains("sin("))
            && fresult.matches(", ").count() == 3,
        "factor casus -> {fresult}"
    );
}
#[test]
fn test_eval_fraction_base_power_is_parenthesized() {
    // A non-integer rational base under a power must keep its parentheses: `(3/2)^(1/3)`, NOT
    // `3/2^(1/3)` — the latter re-parses (under standard precedence, `^` binds tighter than `/`) as
    // `3/(2^(1/3))`, a DIFFERENT, wrong value. This is most visible in Cardano radicals like
    // `solve(10x³-4x²+18x-27=0)` whose real root is `1/15·((17161/2)^(1/3) + 2 - 262^(1/3))`.
    for input in ["(3/2)^(1/3)", "(17161/2)^(1/3)", "(7/3)^(1/5)", "(2/3)^x"] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        let result = wire["result"].as_str().unwrap_or("");
        // The result must be IDEMPOTENT: re-evaluating the printed form yields the same string. A
        // dropped-paren form would re-parse differently, so this catches the precedence bug directly.
        let reparse = cli()
            .args(["eval", result, "--format", "json"])
            .output()
            .expect("Failed to re-run CLI");
        let rwire: Value = serde_json::from_slice(&reparse.stdout).expect("Invalid wire output");
        assert_eq!(
            rwire["result"].as_str(),
            Some(result),
            "{input} -> {result} did not round-trip"
        );
        // And it must literally carry the parenthesized base (not the bare `n/m^...`).
        assert!(
            result.contains(")^") && !result.contains("/2^(") && !result.contains("/3^("),
            "{input} -> {result}"
        );
    }
    // Integer bases are NOT over-parenthesized.
    for (input, expected) in [("2^(1/3)", "2^(1/3)"), ("262^(1/3)", "262^(1/3)")] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
    }
    // The Cardano radical that exposed the bug now renders unambiguously.
    let cardano = cli()
        .args([
            "eval",
            "solve(10*x^3-4*x^2+18*x-27=0, x)",
            "--format",
            "json",
        ])
        .output()
        .expect("Failed to run CLI");
    let cwire: Value = serde_json::from_slice(&cardano.stdout).expect("Invalid wire output");
    let cresult = cwire["result"].as_str().unwrap_or("");
    assert!(
        cresult.contains("cbrt(17161/2)"),
        "cardano fraction radicand -> {cresult}"
    );
}
#[test]
fn test_eval_polynomial_in_log_inequality_back_substitutes_through_exp() {
    // `P(ln(x)) {op} 0` (degree ≥ 2 in `ln(x)`) used to collapse to "No solution": the polynomial-in-u
    // path solved the EQUATION but the inequality dropped the band. It now solves for `u = ln(x)` and
    // maps each u-interval directly through the increasing `x = e^u`: `a < ln(x) < b  ⟺  e^a < x < e^b`,
    // with `-∞ → 0` (the `x > 0` domain edge, OPEN) and `+∞ → +∞`. Building `e^bound` directly avoids the
    // bound comparator (which could not order `1/e²` against `e²` and previously emptied the band).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Distinct rational roots: the band between the two exponentials.
    assert_eq!(r("solve(ln(x)^2 - 3*ln(x) + 2 < 0, x)"), "(e, e^2)");
    assert_eq!(r("solve(ln(x)^2 - 3*ln(x) + 2 <= 0, x)"), "[e, e^2]");
    // Complement: the two outer rays, `(0, e)` keeping the `x > 0` domain edge.
    assert_eq!(
        r("solve(ln(x)^2 - 3*ln(x) + 2 > 0, x)"),
        "(0, e) U (e^2, infinity)"
    );
    // A root at 0 (`ln(x)(ln(x) - 2)`) maps to `x = 1`.
    assert_eq!(r("solve(ln(x)^2 - 2*ln(x) < 0, x)"), "(1, e^2)");
    assert_eq!(r("solve(ln(x)^2 - 5*ln(x) + 6 < 0, x)"), "(e^2, e^3)");
    // Symmetric `ln² - 4`: the band is `(e^-2, e^2)`, rendered with the reciprocal lower bound.
    assert_eq!(r("solve(ln(x)^2 - 4 < 0, x)"), "(1 / e^2, e^2)");
    assert_eq!(
        r("solve(ln(x)^2 - 4 >= 0, x)"),
        "(0, 1 / e^2] U [e^2, infinity)"
    );
    // Controls: a single `ln` (degree 1) stays the ordinary monotonic isolation, and the equation form
    // is unchanged.
    assert_eq!(r("solve(ln(x) > 1, x)"), "(e, infinity)");
    assert_eq!(r("solve(ln(x)^2 - 3*ln(x) + 2 = 0, x)"), "{ e, e^2 }");
}
#[test]
fn test_eval_affine_argument_polynomial_in_log_inequality() {
    // A polynomial-in-`ln(g)` inequality with an AFFINE argument `g = a·x + b` (`ln(2x)`, `ln(x-1)`)
    // used to return "No solution" (the handler was restricted to the bare `ln(x)`). The u-band now maps
    // back through the affine inverse `x = (e^u − b)/a`: `u ∈ (p, q) ⟺ x ∈ ((e^p − b)/a, (e^q − b)/a)`,
    // with the bounds swapping when a < 0 and the `−∞` end giving the domain edge `−b/a`.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Scaled argument `ln(2x)`: band `e^-2 < 2x < e^2`.
    assert_eq!(r("solve(ln(2*x)^2 - 4 < 0, x)"), "(1 / (2·e^2), 1/2·e^2)");
    // Shifted argument `ln(x-1)`: band `e < x-1 < e^2`.
    assert_eq!(
        r("solve(ln(x-1)^2 - 3*ln(x-1) + 2 < 0, x)"),
        "(1 + e, e^2 + 1)"
    );
    // Complement with the domain edge `x > 1/3` kept open.
    assert_eq!(
        r("solve(ln(3*x-1)^2 - 4 >= 0, x)"),
        "(1/3, (e^2 + 1) / (3·e^2)] U [1/3·(e^2 + 1), infinity)"
    );
    // Negative slope `ln(1-x)` (a = -1): the bounds swap, giving `1 - e^2 < x < 1 - e^-2`.
    assert_eq!(
        r("solve(ln(1-x)^2 - 4 < 0, x)"),
        "(1 - e^2, -(1 - e^2) / e^2)"
    );
    // AllReals in u ⇒ the affine DOMAIN `g > 0` (`2x > 0 ⟺ x > 0`), NOT a blanket `x > 0` coincidence.
    assert_eq!(r("solve(ln(2*x)^2 + 1 > 0, x)"), "(0, infinity)");
    assert_eq!(r("solve(ln(x-1)^2 + 1 < 0, x)"), "No solution");
    // Controls: the bare `ln(x)` case and the single-`ln` affine isolation are unchanged.
    assert_eq!(r("solve(ln(x)^2 - 4 < 0, x)"), "(1 / e^2, e^2)");
    assert_eq!(r("solve(ln(2*x) > 1, x)"), "(1/2·e, infinity)");
}
#[test]
fn test_eval_rational_power_polynomial_inequality() {
    // A polynomial-in-`x^(1/q)` inequality (`x − 3√x + 2 < 0`, a quadratic in `√x`) used to emit an
    // honest-but-incomplete residual. It now substitutes `u = x^(1/q)`, solves the u-inequality, and
    // maps the u-band back through `x = u^q`, honouring the `u ≥ 0` (and `x ≥ 0`) domain for even q.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Quadratic in `√x` (q = 2 even): `1 < √x < 2 ⟺ 1 < x < 4`.
    assert_eq!(r("solve(x - 3*sqrt(x) + 2 < 0, x)"), "(1, 4)");
    assert_eq!(r("solve(x - 3*sqrt(x) + 2 <= 0, x)"), "[1, 4]");
    // Complement keeps the domain edge `x = 0` (`√x < 1 ⟺ 0 ≤ x < 1`).
    assert_eq!(
        r("solve(x - 3*sqrt(x) + 2 > 0, x)"),
        "[0, 1) U (4, infinity)"
    );
    assert_eq!(
        r("solve(x - 3*sqrt(x) + 2 >= 0, x)"),
        "[0, 1] U [4, infinity)"
    );
    assert_eq!(r("solve(x - 5*sqrt(x) + 6 < 0, x)"), "(4, 9)");
    // Quadratic in `x^(1/3)` (q = 3 odd): the whole real line is the u-domain, so the band is signed.
    assert_eq!(r("solve(x^(2/3) - x^(1/3) - 2 < 0, x)"), "(-1, 8)");
    assert_eq!(
        r("solve(x^(2/3) - x^(1/3) - 2 > 0, x)"),
        "(-infinity, -1) U (8, infinity)"
    );
    // No constant term (`u² - 3u = u(u-3)`): `0 < √x < 3 ⟺ 0 < x < 9`, the pole at u=0 open.
    assert_eq!(r("solve(x - 3*sqrt(x) < 0, x)"), "(0, 9)");
    // Controls: a degree-1 `√x` stays the ordinary monotonic isolation, a plain polynomial is unchanged,
    // and the equation form is untouched.
    assert_eq!(r("solve(sqrt(x) - 2 < 0, x)"), "[0, 4)");
    assert_eq!(r("solve(x^2 - 3*x + 2 < 0, x)"), "(1, 2)");
    assert_eq!(r("solve(x - 5*sqrt(x) + 6 = 0, x)"), "{ 4, 9 }");
}
#[test]
fn test_eval_high_degree_polynomial_inequality_with_rational_root() {
    // `xⁿ - c > 0` for odd n with a RATIONAL root (`x⁵-1 = (x-1)(x⁴+x³+x²+x+1)`) used to return
    // "No solution": the inequality path declined because it could not certify the positive-definite
    // residual quartic, while the EQUATION path finds the real root {1}. Running the sign analysis
    // over the equation's roots (its alternation + end-behaviour guards keep it sound) recovers the
    // interval. This also unblocks the reciprocal form `1/xⁿ > c` for n up to 12.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Direct polynomial inequalities (were "No solution").
    assert_eq!(r("solve(x^5 > 1, x)"), "(1, infinity)");
    assert_eq!(r("solve(x^9 > 1, x)"), "(1, infinity)");
    assert_eq!(r("solve(x^7 < 1, x)"), "(-infinity, 1)");
    assert_eq!(r("solve(x^5 > 32, x)"), "(2, infinity)");
    // Reciprocal forms for odd n >= 5 with a rational boundary (were inventing the negative ray).
    assert_eq!(r("solve(1/x^5 > 1, x)"), "(0, 1)");
    assert_eq!(r("solve(1/x^7 > 1, x)"), "(0, 1)");
    assert_eq!(r("solve(1/x^9 > 1, x)"), "(0, 1)");
    assert_eq!(r("solve(1/x^7 < 1, x)"), "(-infinity, 0) U (1, infinity)");
    // Surd-boundary and lower-degree controls remain correct.
    assert_eq!(r("solve(x^5 > 2, x)"), "(root(2, 5), infinity)");
    assert_eq!(r("solve(1/x^3 > 2, x)"), "(0, cbrt(1/2))");
    assert_eq!(r("solve(x^3 - x > 0, x)"), "(-1, 0) U (1, infinity)");
}
#[test]
fn test_eval_squared_irrational_quadratic_factor_keeps_its_roots() {
    // A polynomial with a SQUARED (or equal-constant) irreducible quadratic factor dropped that
    // factor's irrational roots: `(x²-3)²·(x-1) = 0` returned `{1}`, losing ±√3. The quartic-factor
    // solver factors the deflated monic quartic into `(x²+px+q)(x²+rx+s)`, but when the two factors
    // share a constant term (`q = s`, the perfect-square case) the `p = (d-qb)/(s-q)` formula divided
    // by zero, so that case was skipped — the roots of the repeated quadratic vanished. The `q = s`
    // branch now solves `p,r` from `t²-bt+(c-2q)=0` directly.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // The in-core biquadratic residual solver now owns these (2026-07-14): same set,
    // cleaner `-sqrt(3)` render (was the quartic-factor owner's `-3·3^(-1/2)`).
    assert_eq!(
        r("solve((x^2-3)^2*(x-1) = 0, x)"),
        "{ 1, -sqrt(3), sqrt(3) }"
    );
    assert_eq!(
        r("solve((x^2-7)^2*(x-3) = 0, x)"),
        "{ 3, -sqrt(7), sqrt(7) }"
    );
    // A general (non-symmetric) irreducible quadratic, squared: roots (3±√5)/2.
    assert_eq!(
        r("solve((x^2-3*x+1)^2*(x-1) = 0, x)"),
        "{ 1, 1/2·(sqrt(5) + 3), 1/2·(3 - sqrt(5)) }"
    );
    // The bug survives full expansion (same quintic, factored back internally).
    assert_eq!(
        r("solve(x^5 - x^4 - 6*x^3 + 6*x^2 + 9*x - 9 = 0, x)"),
        "{ 1, -sqrt(3), sqrt(3) }"
    );
    // Degree-6 with two rational cofactor roots; the squared factor still contributes ±√3.
    assert_eq!(
        r("solve((x^2-3)^2*(x^2-4) = 0, x)"),
        "{ -2, 2, -sqrt(3), sqrt(3) }"
    );
    // Controls: the DISTINCT-quadratic-factor case and a plain quadratic are unchanged.
    // The negative root used to print `-5·5^(-1/2)` — the very form the comment
    // at the top of this test calls the OLD owner's ugly render. It stayed here
    // because the fold that produces `-sqrt(5)` only fired for a POSITIVE
    // coefficient: `5·5^(-1/2)` combined, `(-5)·5^(-1/2)` did not, so the two
    // roots of one equation printed in two different shapes. Fixed 2026-07-28
    // by peeling the sign before comparing bases.
    assert_eq!(
        r("solve(x^5-5*x^3+x^2-5 = 0, x)"),
        "{ -1, sqrt(5), -sqrt(5) }"
    );
    assert_eq!(r("solve(x^2-5*x+6 = 0, x)"), "{ 2, 3 }");
}
#[test]
fn test_eval_content_scaled_squared_quadratic_factor_keeps_roots() {
    // A CONTENT / scalar-multiple wrapper on the squared-quadratic case dropped the irrational roots:
    // `2·(x²-3)²·(x-1) = 0` returned `{1}`. After peeling the rational root, the deflated quotient is
    // `2·(x²-3)²` — a NON-monic quartic, which the factorizer rejected. Normalizing the quotient to
    // monic (dividing by the leading coefficient preserves the roots) recovers ±√3. The remaining
    // higher-multiplicity cases (`(x²-3)³`, two distinct irrational-root factors) deflate past degree 4
    // and stay residual — they need general ℚ-factorization.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Outer scalar content.
    assert_eq!(
        r("solve(2*(x^2-3)^2*(x-1) = 0, x)"),
        "{ 1, -sqrt(3), sqrt(3) }"
    );
    // Content folded INTO the squared factor (`(2x²-6)² = 4·(x²-3)²`).
    assert_eq!(
        r("solve((2*x^2-6)^2*(x-1) = 0, x)"),
        "{ 1, -sqrt(3), sqrt(3) }"
    );
    // A different scalar and root.
    assert_eq!(
        r("solve(3*(x^2-5)^2*(x-2) = 0, x)"),
        "{ 2, -sqrt(5), sqrt(5) }"
    );
    // NEGATIVE content (leading coefficient < 0) normalizes the same way.
    assert_eq!(
        r("solve(-2*(x^2-3)^2*(x-1) = 0, x)"),
        "{ 1, -sqrt(3), sqrt(3) }"
    );
    // Content on a non-repeated quartic (distinct factors) stays correct.
    assert_eq!(
        r("solve(2*x^4 - 10*x^2 + 12 = 0, x)"),
        "{ -sqrt(2), -sqrt(3), sqrt(2), sqrt(3) }"
    );
    // Control: the monic case is unchanged.
    assert_eq!(
        r("solve((x^2-3)^2*(x-1) = 0, x)"),
        "{ 1, -sqrt(3), sqrt(3) }"
    );
}
#[test]
fn test_eval_unsound_power_monomial_inequality_declines_to_residual() {
    // A power-monomial inequality `c·x^e {op} k` is solved by the engine's MONOTONIC isolation, which
    // emits a single ray — correct ONLY when `x^e` is strictly monotonic (`e > 0`, odd numerator).
    // An even-numerator VALLEY (`x^(2/3) = |x|^(2/3)`) is now SOLVED exactly by the `|x| {op} k^(q/p)`
    // reduction (its truth is two rays / a bounded interval). A NEGATIVE non-integer exponent
    // (`1/x^(1/3)`, `1/√x`) — a reciprocal fractional power with a pole — is still declined to an honest
    // residual (correct solving of the reciprocals is the next rung).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Even-numerator valleys are now SOLVED exactly (two rays for `>`, a bounded interval for `<`).
    assert_eq!(
        r("solve(x^(2/3) > 2, x)"),
        "(-infinity, -(2^(3/2))) U (2^(3/2), infinity)"
    );
    assert_eq!(r("solve(x^(2/3) < 2, x)"), "(-(2^(3/2)), 2^(3/2))");
    assert_eq!(
        r("solve(x^(2/5) > 2, x)"),
        "(-infinity, -(2^(5/2))) U (2^(5/2), infinity)"
    );
    // Negative non-integer exponents / reciprocal fractional powers (were complement / pole) — declined.
    assert_eq!(r("solve(1/x^(1/3) > 2, x)"), "(0, 1/8)");
    assert_eq!(r("solve(1/x^(1/2) > 2, x)"), "(0, 1/4)");
    assert_eq!(r("solve(x^(-1/3) > 2, x)"), "solve(x^(-1 / 3) > 2, x)");
    // KEEP: strictly-monotonic powers (e > 0, odd numerator) stay solved EXACTLY.
    assert_eq!(r("solve(x^(1/3) > 2, x)"), "(8, infinity)");
    assert_eq!(r("solve(x^(1/2) < 2, x)"), "[0, 4)");
    assert_eq!(r("solve(x^(3/2) > 2, x)"), "(2^(2/3), infinity)");
    assert_eq!(r("solve(x^(5/3) > 2, x)"), "(2^(3/5), infinity)");
    // KEEP: integer-exponent reciprocals are owned by the rational-constant path (Class B).
    assert_eq!(r("solve(1/x^3 > 2, x)"), "(0, cbrt(1/2))");
    assert_eq!(r("solve(1/x > 2, x)"), "(0, 1/2)");
    // KEEP: the EQUATION form is untouched (op gate) — both valley roots are found.
    assert_eq!(r("solve(x^(2/3) = 8, x)"), "{ -16·sqrt(2), 16·sqrt(2) }");
}
#[test]
fn test_eval_wrapped_non_monotonic_power_inequality_declines_to_residual() {
    // An even-numerator VALLEY through its WRAPPERS — a shifted/scaled affine base `(x-1)^(2/3)`, an
    // additive constant `x^(2/3) + 1` — is now SOLVED exactly by the `|a·x+b| {op} k^(q/p)` reduction.
    // The `sqrt` FUNCTION reciprocal `1/sqrt(x)` is SOLVED since U2 (w-space).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Shifted / scaled affine base (even-numerator valley) — SOLVED.
    assert_eq!(
        r("solve((x-1)^(2/3) > 4, x)"),
        "(-infinity, -7) U (9, infinity)"
    );
    assert_eq!(
        r("solve((2*x-3)^(2/3) > 4, x)"),
        "(-infinity, -5/2) U (11/2, infinity)"
    );
    // Additive constant on the power — SOLVED.
    assert_eq!(
        r("solve(x^(2/3) + 1 > 5, x)"),
        "(-infinity, -8) U (8, infinity)"
    );
    assert_eq!(r("solve(5 - x^(2/3) > 1, x)"), "(-8, 8)");
    // sqrt FUNCTION reciprocal — SOLVED since U2 via the w-space substitution.
    assert_eq!(r("solve(1/sqrt(x) > 2, x)"), "(0, 1/4)");
    assert_eq!(r("solve(1/sqrt(x-1) > 2, x)"), "(1, 5/4)");
    // KEEP: a shifted/scaled STRICTLY-MONOTONIC power (e > 0, odd numerator) stays solved exactly.
    assert_eq!(r("solve((x-1)^(1/3) > 2, x)"), "(9, infinity)");
    assert_eq!(r("solve(sqrt(x-1) > 2, x)"), "(5, infinity)");
    assert_eq!(r("solve(sqrt(x) < 2, x)"), "[0, 4)");
    // KEEP: an integer power of an affine base is a polynomial inequality, solved exactly.
    assert_eq!(
        r("solve((x-1)^2 > 4, x)"),
        "(-infinity, -1) U (3, infinity)"
    );
    assert_eq!(r("solve(1/(x-1) > 2, x)"), "(1, 3/2)");
}
#[test]
fn test_eval_uncombined_like_power_terms_valley_inequality() {
    // The solve path extracts the power term from the RAW LHS, where `x^(2/3) + x^(2/3)` (the variable
    // on BOTH sides of the `Add`) hit the `(_, _) => None` arm and bypassed the valley reduction — the
    // monotonic fall-through then dropped the `x < 0` ray (`> 8` gave `(8, ∞)`) or emitted garbage
    // (`>= 8` gave `[-8, -8] ∪ [8, ∞)`). The extractor now COMBINES like power terms (same affine base
    // and exponent), matching the standalone simplifier's `→ 2·x^(2/3)` fold.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(
        r("solve(x^(2/3) + x^(2/3) > 8, x)"),
        "(-infinity, -8) U (8, infinity)"
    );
    assert_eq!(
        r("solve(x^(2/3) + x^(2/3) >= 8, x)"),
        "(-infinity, -8] U [8, infinity)"
    );
    assert_eq!(r("solve(x^(2/3) + x^(2/3) < 8, x)"), "(-8, 8)");
    // Mixed coefficients combine to 3·x^(2/3); three terms also fold.
    assert_eq!(
        r("solve(x^(2/3) + 2*x^(2/3) > 9, x)"),
        "(-infinity, -(3^(3/2))) U (3^(3/2), infinity)"
    );
    assert_eq!(
        r("solve(x^(2/3) + x^(2/3) + x^(2/3) > 12, x)"),
        "(-infinity, -8) U (8, infinity)"
    );
    // Shifted base combines too.
    assert_eq!(
        r("solve((x-1)^(2/3) + (x-1)^(2/3) > 8, x)"),
        "(-infinity, -7) U (9, infinity)"
    );
    // UNLIKE exponents sharing the SAME base (`x^(2/3) + x^(4/3)`, a quartic in `x^(1/3)`) are not a
    // single valley, but the rational-power-polynomial handler now solves them (`u⁴ + u² - 8 > 0`).
    assert_eq!(
        r("solve(x^(2/3) + x^(4/3) > 8, x)"),
        "(-infinity, -((1/2·(sqrt(33) - 1))^(3/2))) U ((1/2·(sqrt(33) - 1))^(3/2), infinity)"
    );
    // A DIFFERENT base (`(x-1)^(2/3)`) is not an `x`-power polynomial, so it stays residual.
    assert_eq!(
        r("solve(x^(2/3) + (x-1)^(2/3) > 8, x)"),
        "solve(x - (8 - (x - 1)^(2/3))^(1 / 2/3) = 0, x)"
    );
    // Exact cancellation is empty; the odd-power and integer-power forms stay correct.
    assert_eq!(r("solve(x^(2/3) - x^(2/3) > 0, x)"), "No solution");
    assert_eq!(r("solve(x^(1/3) + x^(1/3) > 8, x)"), "(64, infinity)");
    assert_eq!(
        r("solve(x^2 + x^2 > 8, x)"),
        "(-infinity, -2) U (2, infinity)"
    );
}
#[test]
fn test_eval_variable_base_log_inequality_declines() {
    // SOUNDNESS: `log(x, c) ≷ k` reads x as the BASE, so logₓ(c)=ln(c)/ln(x) is NON-monotonic
    // (decreasing on x>1, sign change at x=1). The engine's monotonic isolation emitted a wrong ray
    // (and an `undefined` endpoint for k=0). With no exact split representation it now declines to an
    // honest residual (ok=true). Constant-base log and equations are unaffected.
    let run = |input: &str| -> (bool, String) {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        (
            wire["ok"].as_bool().unwrap_or(false),
            wire["result"].as_str().unwrap_or("").to_string(),
        )
    };
    // Variable-base log inequalities decline to a residual (ok=true), never a wrong ray / "undefined".
    for input in [
        "log(x,2)>3",
        "log(x,4)>2",
        "log(x,3)<1",
        "log(x,1/2)>1",
        "log(x,2)>0",
    ] {
        let (ok, result) = run(input);
        assert!(
            ok,
            "{input} should be ok=true (honest residual), got {result:?}"
        );
        assert!(
            result.contains("solve(") && !result.contains("undefined"),
            "{input} should be a clean residual, got {result:?}"
        );
    }
    let plain = |input: &str| run(input).1;
    // EQ-safety: equations still solve.
    assert_eq!(plain("log(x,2)=3"), "{ cbrt(2) }");
    // Constant-base log (monotonic) is unaffected.
    assert_eq!(plain("log(2,x)>3"), "(8, infinity)");
    assert_eq!(plain("log(2,x)<3"), "(0, 8)");
    assert_eq!(plain("ln(x)<0"), "(0, 1)");
}
#[test]
fn test_eval_exponential_positivity_inequality() {
    // SOUNDNESS: `b^x {>,>=} c` with a positive base and c <= 0 is identically TRUE (b^x > 0 always),
    // so the solution is ℝ — not the empty set the op-agnostic EmptySet classification produced. The
    // product/sum cascade self-heals via AllReals ∩ s = s.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("e^x>0"), "All real numbers");
    assert_eq!(r("e^x>=0"), "All real numbers");
    assert_eq!(r("e^x>-1"), "All real numbers");
    assert_eq!(r("2^x>0"), "All real numbers");
    assert_eq!(r("(1/2)^x>0"), "All real numbers");
    assert_eq!(r("x*e^x>0"), "(0, infinity)"); // sign(x·e^x) = sign(x)
    assert_eq!(r("x^2*e^x>0"), "(-infinity, 0) U (0, infinity)"); // ℝ∖{0}
                                                                  // Controls: `<`/`<=`/`=` against c <= 0 stay empty (b^x is never <0, ≤0, or =0); rhs > 0 solves normally.
    assert_eq!(r("e^x<0"), "No solution");
    assert_eq!(r("e^x<=0"), "No solution");
    assert_eq!(r("e^x=0"), "No solution");
    assert_eq!(r("e^x>5"), "(ln(5), infinity)");
    assert_eq!(r("e^x-1>0"), "(0, infinity)");
    // SOUNDNESS: a bare additive single-exponential `a*base^x + c {op} k` was declined by the
    // substitution gate (only `base^x` with no higher power) and fell to the boundary equation,
    // returning "No solution" (or a malformed conditional) when the isolated threshold is negative —
    // truth is all reals since base^x > 0. Now it substitutes u=base^x and the u>0 clamp answers it.
    assert_eq!(r("e^x+1>0"), "All real numbers");
    assert_eq!(r("e^x+5>2"), "All real numbers");
    assert_eq!(r("3^x+1>0"), "All real numbers");
    assert_eq!(r("e^x+1>=0"), "All real numbers");
    assert_eq!(r("2*e^x+3>0"), "All real numbers");
    assert_eq!(r("e^x+1<0"), "No solution"); // base^x = -1 has no solution, so < never holds
                                             // Equation narration is unchanged (the bare gate is inequality-only).
    assert_eq!(r("e^x=2"), "{ ln(2) }");
    assert_eq!(r("e^x+1=0"), "No solution");
}
#[test]
fn test_eval_exponential_reciprocal_polynomial_clears_the_reciprocal() {
    // Equations that mix an exponential with its RECIPROCAL (`e^x + e^(−x)`, the hyperbolic form) used
    // to bail — `función [cosh] no definida` for base `e`, `Cannot isolate 'x'` for general bases —
    // because `simplify` folds `e^x + e^(−x)` into `2·cosh(x)`. The Laurent map `u = base^x` (built on
    // the raw tree, so `simplify` never runs) clears the `1/u` and solves the polynomial in `u`.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // `u = base^x`: `u² − 2u + 1 = 0 ⟹ u = 1 ⟹ x = 0`.
    assert_eq!(r("solve(e^x + e^(-x) = 2, x)"), "{ 0 }");
    assert_eq!(r("solve(3^x + 3^(-x) = 2, x)"), "{ 0 }");
    // `u² − 1 = 0 ⟹ u = 1` (the `u = −1` root is dropped: `base^x > 0`).
    assert_eq!(r("solve(e^x - e^(-x) = 0, x)"), "{ 0 }");
    // Distinct positive roots: `u² − 3u + 2 = 0 ⟹ u ∈ {1, 2}`.
    assert_eq!(r("solve(e^x + 2*e^(-x) = 3, x)"), "{ 0, ln(2) }");
    // An affine exponent (`2^(1−x) = 2·2^(−x)`) folds the `2` into the coefficient.
    assert_eq!(r("solve(2^x - 3 + 2^(1-x) = 0, x)"), "{ 0, 1 }");
    // `2^x + 2^(−x) = 5/2 ⟹ u ∈ {1/2, 2} ⟹ x ∈ {−1, 1}` (`ln(1/2)/ln(2)` folds to −1).
    assert_eq!(r("solve(2^x + 2^(-x) = 5/2, x)"), "{ -1, 1 }");
    // `cosh(x) ≥ 1` always, so `= 1/2·2 = 1` (i.e. sum `= 1`) has NO real solution.
    assert_eq!(r("solve(e^x + e^(-x) = 1, x)"), "No solution");
    // SURD-discriminant roots: BOTH `u = 2 ± √3` are provably positive, so both back-substitute. The
    // exact-surd-sign upgrade to the positivity prover keeps the second root (it used to drop it behind
    // a spurious `2 − √3 > 0` guard).
    assert_eq!(
        r("solve(e^x + e^(-x) = 4, x)"),
        "{ ln(2 - sqrt(3)), ln(sqrt(3) + 2) }"
    );
    // `u² − 2u − 1 = 0 ⟹ u = 1 ± √2`; the negative `1 − √2` is now DISPROVEN positive ⟹ dropped.
    assert_eq!(r("solve(e^x - e^(-x) = 2, x)"), "{ ln(sqrt(2) + 1) }");
    assert_eq!(
        r("solve(e^x + e^(-x) = 3, x)"),
        "{ ln(1/2·(3 - sqrt(5))), ln(1/2·(sqrt(5) + 3)) }"
    );
    // Controls: the pure positive-power forms are owned by the existing path and must be UNCHANGED.
    assert_eq!(r("solve(e^(2*x) - 3*e^x + 2 = 0, x)"), "{ ln(2), 0 }");
    assert_eq!(r("solve(4^x - 3*2^x + 2 = 0, x)"), "{ 0, 1 }");
}
#[test]
fn test_eval_fractional_base_exponential_inequality_direction() {
    // SOUNDNESS: `a^x ≷ k` with 0 < a < 1 (decreasing) must FLIP the inequality direction when
    // isolating x through the logarithm. Previously the engine kept the direction, returning the
    // reversed (wrong) ray. The bound is exact; only the direction was wrong. Truth vs sympy.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // (1/2)^x > 4 ⟺ x < -2. The change-of-base boundary `ln(c)/ln(b)` folds to
    // the exact rational for these fractional-base/argument pairs.
    assert_eq!(r("(1/2)^x>4"), "(-infinity, -2)");
    assert_eq!(r("(1/2)^x<1/4"), "(2, infinity)"); // x > 2
    assert_eq!(r("(1/2)^x>=2"), "(-infinity, -1]"); // x <= -1
    assert_eq!(r("0.3^x<0.09"), "(2, infinity)"); // x > 2 (9/100 = (3/10)²)
    assert_eq!(r("(1/3)^x>1/9"), "(-infinity, 2)"); // x < 2
                                                    // Controls: base > 1 keeps direction; equations are never flipped.
    assert_eq!(r("2^x>4"), "(2, infinity)");
    assert_eq!(r("2^x<4"), "(-infinity, 2)");
    assert_eq!(r("2^x>=8"), "[3, infinity)");
    assert_eq!(r("(1/2)^x=4"), "{ -2 }");
    assert_eq!(r("2^x=4"), "{ 2 }");
    // SOUNDNESS: an ADDITIVE/scaled single exponential `a*base^x + c {op} k` is isolated to the pure
    // `base^x {op'} (k-c)/a` and solved by the terminal for EVERY base — including a fractional base
    // with a positive threshold (`(1/2)^x - 4 > 0 -> (1/2)^x > 4`) or a negative threshold
    // (`(1/2)^x + 1 > 0 -> (1/2)^x > -1 -> all reals`). The substitution path would decline a
    // fractional base to a residual, so the isolation runs first.
    assert_eq!(r("(1/2)^x-4>0"), "(-infinity, -2)"); // x < -2
    assert_eq!(r("(1/2)^x-1>0"), "(-infinity, 0)");
    assert_eq!(r("(1/2)^x+1>0"), "All real numbers");
    assert_eq!(r("(1/2)^x+1<0"), "No solution");
    assert_eq!(r("(1/3)^x-1>0"), "(-infinity, 0)");
}
#[test]
fn test_eval_exponential_polynomial_inequality_back_substitution() {
    // SOUNDNESS (B3): a polynomial-in-`u = e^x` INEQUALITY was solved in u-space and the interval was
    // returned WITHOUT back-substituting `x = ln(u)` (the equation path back-substituted, the
    // inequality path forgot). `e^(2x)-3e^x+2<0` leaked the u-interval `(1, 2)` instead of `(0, ln 2)`.
    // The fix clamps the u-solution to `u > 0` (range of e^x) and maps each endpoint through ln.
    // Truth verified vs sympy.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // u = e^x in (1, 2) ⟺ x in (0, ln 2). All four operators, base e.
    assert_eq!(r("e^(2*x)-3*e^x+2<0"), "(0, ln(2))");
    assert_eq!(r("e^(2*x)-3*e^x+2>0"), "(-infinity, 0) U (ln(2), infinity)");
    assert_eq!(r("e^(2*x)-3*e^x+2<=0"), "[0, ln(2)]");
    assert_eq!(
        r("e^(2*x)-3*e^x+2>=0"),
        "(-infinity, 0] U [ln(2), infinity)"
    );
    // A base > 1 other than e maps through log_base: 2^x in [1, 2] ⟺ x in [0, 1].
    assert_eq!(r("2^(2*x)-3*2^x+2<=0"), "[0, 1]");
    // u must be > 0: a root <= 0 is clamped away. u in (-2, 1) ⟺ (0, 1) ⟺ x < 0; u in (-2, -1) ⟺ empty.
    assert_eq!(r("e^(2*x)+e^x-2<0"), "(-infinity, 0)");
    assert_eq!(r("e^(2*x)+3*e^x+2<0"), "No solution");
    // U3: the IRRATIONAL roots (e^x = (1±√5)/2) now back-substitute exactly —
    // the surd sign oracles classify the endpoints ((1−√5)/2 clamps away as
    // provably negative; (1+√5)/2 = φ maps through the boundary equation to
    // ln(φ)). Previously an honest decline.
    assert_eq!(r("e^(2*x)-e^x-1<0"), "(-infinity, ln(phi))");
    assert_eq!(r("e^(2*x)-e^x-1>0"), "(ln(phi), infinity)");
    // A FRACTIONAL base (0 < a < 1) likewise declines to the residual (decreasing inverse + ln-ratio
    // bounds the downstream interval comparison cannot order) rather than leak the u-interval.
    assert_eq!(
        r("(1/2)^(2*x)-3*(1/2)^x+2<0"),
        "solve((1/2)^(2·x) + 2 - 3·(1/2)^x < 0, x)"
    );
    // Controls: the equation path still back-substitutes; e^(2x) = -5 has no real solution.
    assert_eq!(r("e^(2*x)-3*e^x+2=0"), "{ ln(2), 0 }");
    assert_eq!(r("e^(2*x)=-5"), "No solution");
}
#[test]
fn test_eval_exponential_coefficient_equals_base_inequality() {
    // SOUNDNESS: when the linear coefficient equals the base, the simplifier merges
    // `c·base^x = base^(x+1)`, and the exponential substitution could not match the `Add`-in-exponent
    // `base^(x+1)`. The strategy declined and the fallback returned the EQUATION root, dropping the
    // operator: `2^(2x)-2·2^x<0` -> `{1}` instead of `(-inf, 1)`. Now `substitute_expr_pattern` maps the
    // affine exponent `base^(x+1) -> base^1·u` (numeric base, integer constant), so the inequality solves
    // and back-substitutes correctly. Truth vs sympy.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // u = base^x in (0, base) <=> base^x < base <=> x < 1. All four operators, bases 2/3/10.
    assert_eq!(r("2^(2*x)-2*2^x<0"), "(-infinity, 1)");
    assert_eq!(r("2^(2*x)-2*2^x>0"), "(1, infinity)");
    assert_eq!(r("2^(2*x)-2*2^x<=0"), "(-infinity, 1]");
    assert_eq!(r("2^(2*x)-2*2^x>=0"), "[1, infinity)");
    assert_eq!(r("3^(2*x)-3*3^x<0"), "(-infinity, 1)");
    assert_eq!(r("10^(2*x)-10*10^x<0"), "(-infinity, 1)");
    // The affine-exponent substitution also drives the equation form: 2^(x+1) = 8 <=> 2·u = 8 <=> x = 2.
    assert_eq!(r("2^(2*x)-2*2^x=0"), "{ 1 }");
    assert_eq!(r("2^(x+1)=8"), "{ 2 }");
    // Controls: a coefficient that is NOT the base does not merge, so the inner base^x substitutes
    // as before (2^(2x)-4·2^x stays a clean u^2-4u): boundary x=2, not 1.
    assert_eq!(r("2^(2*x)-4*2^x<0"), "(-infinity, 2)");
}
#[test]
fn test_eval_factorable_exponential_inequality() {
    // SOUNDNESS (peldaño 1): a degree-2 exponential inequality collapsed to one side with NO constant
    // term, `A·base^(2x) + B·base^x {op} 0`, factors out base^x > 0 to the single exponential
    // `base^x {op} -B/A`. For base e the coefficient merges (`e·e^x = e^(x+1)`) so the substitution
    // was blocked and the fallback leaked the equation root `{1}`; for a SYMBOLIC coefficient the
    // polynomial-in-u inequality solver errored (ok=false). Both now reduce and solve. Truth vs sympy.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // base e, coefficient == base: e^x in (0, e) <=> x < 1.
    assert_eq!(r("e^(2*x)-e*e^x<0"), "(-infinity, 1)");
    assert_eq!(r("e^(2*x)-e*e^x>0"), "(1, infinity)");
    assert_eq!(r("e^(2*x)-e*e^x<=0"), "(-infinity, 1]");
    assert_eq!(r("e^(2*x)-e*e^x>=0"), "[1, infinity)");
    // SYMBOLIC coefficient pi (was a loud ok=false "symbolic coefficient" error): e^x < pi <=> x < ln(pi).
    assert_eq!(r("e^(2*x)-pi*e^x<0"), "(-infinity, ln(pi))");
    assert_eq!(r("e^(2*x)-pi*e^x>0"), "(ln(pi), infinity)");
    assert_eq!(r("e^(2*x)-2*pi*e^x<0"), "(-infinity, ln(2·pi))");
    // Controls: a constant term keeps the substitution path (B3), and the equation is unchanged.
    assert_eq!(r("e^(2*x)-3*e^x+2<0"), "(0, ln(2))");
    assert_eq!(r("e^(2*x)-e*e^x=0"), "{ 1 }");
}
#[test]
fn test_eval_nonunit_exponent_exponential_inequality() {
    // SOUNDNESS: a single exponential with a NON-UNIT integer exponent, `base^(k*x) {op} c`, could not
    // be isolated by the unit-exponent terminal (`e^(2x)<2` -> residual, `e^(2x)<e` -> ok=false). Since
    // `base^(k*x)` (base>1) is strictly increasing, recover the ray from the boundary EQUATION
    // `base^(k*x)=c` (which solves) + monotonicity. This also closes the degree-3+ inequality: the
    // factor-out cofactor `e^(2x)-e` of `e^(3x)-e*e^x<0` resolves here. Truth vs sympy.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Non-unit single exponential, base e: e^(2x) {op} c <=> 2x {op} ln(c) <=> x {op} ln(c)/2.
    assert_eq!(r("e^(2*x)<e"), "(-infinity, 1/2)");
    assert_eq!(r("e^(2*x)>e"), "(1/2, infinity)");
    assert_eq!(r("e^(2*x)<=e"), "(-infinity, 1/2]");
    assert_eq!(r("e^(2*x)<2"), "(-infinity, 1/2·ln(2))");
    // Non-positive threshold resolves by sign (base^(k*x) > 0 always); positivity stays correct.
    assert_eq!(r("e^(2*x)<0"), "No solution");
    assert_eq!(r("e^(2*x)>0"), "All real numbers");
    assert_eq!(r("e^(2*x)>1"), "(0, infinity)");
    // Degree-3 (and degree-4) collapsed: factor out e^x>0 to the non-unit cofactor.
    assert_eq!(r("e^(3*x)-e*e^x<0"), "(-infinity, 1/2)"); // was the WRONG point {1/2}
    assert_eq!(r("e^(3*x)-e*e^x>0"), "(1/2, infinity)");
    assert_eq!(r("e^(3*x)-e*e^x<=0"), "(-infinity, 1/2]");
    assert_eq!(r("e^(3*x)-pi*e^x<0"), "(-infinity, 1/2·ln(pi))");
    assert_eq!(r("e^(4*x)-e*e^x<0"), "(-infinity, 1/3)");
    // SYMBOLIC-CONSTANT thresholds/coefficients beyond bare e/pi: the threshold sign is delegated to
    // the boundary equation (Discrete root -> ray, Empty -> sign), so e^2, sqrt(2), 2*e all solve; a
    // provably non-positive threshold (-e) resolves by sign.
    assert_eq!(r("e^(2*x)<e^2"), "(-infinity, 1)");
    assert_eq!(r("e^(2*x)>e^2"), "(1, infinity)");
    assert_eq!(r("e^(2*x)<e^3"), "(-infinity, 3/2)");
    assert_eq!(r("e^(3*x)-e^2*e^x>0"), "(1, infinity)"); // was the WRONG point {1}
    assert_eq!(r("e^(3*x)-e^2*e^x<0"), "(-infinity, 1)");
    assert_eq!(r("e^(3*x)+e*e^x<0"), "No solution"); // e^x(e^(2x)+e) > 0 always
    assert_eq!(r("e^(3*x)+e*e^x>0"), "All real numbers");
    assert_eq!(r("e^(3*x)+pi*e^x<0"), "No solution");
    // Controls: a degree-3 with RATIONAL roots stays on the substitution path; equations unchanged.
    assert_eq!(r("e^(3*x)-e^x<0"), "(-infinity, 0)");
    assert_eq!(r("e^(3*x)-e*e^x=0"), "{ 1/2 }");
}
#[test]
fn test_eval_reducible_quartic_factor_roots() {
    // A polynomial whose deflated quartic factor splits into two rational quadratics dropped the
    // quadratic factor's roots: `x⁵-5x³+x²-5 = (x+1)(x²-5)(x²-x+1)` returned only `{-1}`, losing the
    // `±√5` roots of `x²-5`. The quartic is now factored into quadratics and each is solved.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // The quintic recovers -1 plus the ±√5 surd roots (3 real roots, no residual).
    let quintic = r("solve(x^5-5*x^3+x^2-5=0, x)");
    assert!(
        quintic.contains("-1")
            && quintic.contains("sqrt(5)")
            && quintic.matches(", ").count() == 2
            && !quintic.contains("Solve"),
        "x^5-5x^3+x^2-5 -> {quintic}"
    );
    // Standalone reducible quartics with only-rational or mixed real roots.
    assert_eq!(r("solve(x^4+x^3-x-1=0, x)"), "{ -1, 1 }"); // (x²-1)(x²+x+1)
    assert_eq!(r("solve(x^4-3*x^2-4=0, x)"), "{ -2, 2 }"); // (x²-4)(x²+1)
                                                           // An IRREDUCIBLE quartic correctly declines (Ferrari deferred) — stays an honest residual.
    assert!(r("solve(x^4-x-1=0, x)").contains("solve("));
    // The reducible-quartic INEQUALITY now works through the sign-analysis chain.
    assert_eq!(r("x^4-3*x^2-4>0"), "(-infinity, -2) U (2, infinity)");
    // Controls: biquadratics and lower-degree solves are unchanged.
    assert_eq!(r("solve(x^4-5*x^2+4=0, x)"), "{ -2, -1, 1, 2 }");
    assert_eq!(r("solve(x^3-2=0, x)"), "{ cbrt(2) }");
}
#[test]
fn test_eval_biquadratic_surd_roots() {
    // A biquadratic `a·x⁴ + b·x² + c` whose x-roots are surds leaked a circular residual
    // (`solve(x − (8x²−15)^(1/4)=0)`); the `z = x²` substitution now solves it. Roots verified
    // numerically in the dev probes (|p(root)| < 1e-13).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Four surd roots {±√3, ±√5}.
    let four = r("x^4-8*x^2+15=0");
    assert!(
        four.contains("sqrt(5)")
            && four.contains("sqrt(3)")
            && four.matches(", ").count() == 3
            && !four.contains("Solve"),
        "x^4-8x^2+15=0 -> {four}"
    );
    // Only the non-negative z root survives: {±√3} (z = -1 dropped).
    let two = r("x^4-2*x^2-3=0");
    assert!(
        two.contains("sqrt(3)") && two.matches(", ").count() == 1 && !two.contains("Solve"),
        "x^4-2x^2-3=0 -> {two}"
    );
    // No real roots when both z roots are negative or complex.
    assert_eq!(r("x^4+x^2+1=0"), "No solution");
    assert_eq!(r("x^4+3*x^2+2=0"), "No solution");
    // Rational-root biquadratics and general (non-biquadratic) quartics are unchanged.
    assert_eq!(r("x^4-5*x^2+4=0"), "{ -2, -1, 1, 2 }");
    assert!(r("x^4-x-1=0").contains("solve(")); // general quartic stays a residual (Ferrari deferred)
                                                // The biquadratic INEQUALITY is now operator-sensitive (biquad solver → Discrete → sign analysis).
    let gt = r("x^4-8*x^2+15>0");
    let lt = r("x^4-8*x^2+15<0");
    assert_ne!(gt, lt, "operator must matter");
    assert!(
        gt.contains(" U ") && !gt.contains("Solve") && !gt.contains('{'),
        "x^4-8x^2+15>0 -> {gt}"
    );
}
#[test]
fn test_eval_symbolic_power_of_power_guards_base_sign() {
    // `(x^a)^b = x^(a·b)` holds for ALL real x only when both exponents are integers; with a
    // non-integer exponent it needs `x ≥ 0` (for x<0, `x^a` is not real and the fold drops the sign,
    // so `((-2)^a)^b ≠ (-2)^(a·b)`). The old unconditional fold was a wrong value. Now: integer and
    // provably-non-negative bases still fold; a non-provably-non-negative or negative base declines
    // in the default (generic) domain (honest unevaluated form), and `--domain assume` opts in.
    for (input, expected) in [
        ("(x^2)^3", "x^6"), // integer exponents: unconditional, valid for all x
        ("(x^3)^2", "x^6"),
        ("((-2)^3)^2", "64"), // integer exponents over a negative base: still exact
        ("(2^a)^b", "2^(a·b)"), // provably-positive base: unconditional fold
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
    }
    // Symbolic exponents over an unknown- or negative-sign base no longer fold to a wrong value in
    // the default domain — they stay an honest unevaluated form.
    for input in ["(x^a)^b", "((-2)^a)^b"] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(
            wire["result"].as_str(),
            Some(input),
            "{input} should stay unevaluated"
        );
    }
    // `--domain assume` opts into the analytic fold (the user accepts x ≥ 0).
    let assumed = cli()
        .args(["eval", "(x^a)^b", "--domain", "assume", "--format", "json"])
        .output()
        .expect("Failed to run CLI");
    let wire: Value = serde_json::from_slice(&assumed.stdout).expect("Invalid wire output");
    assert_eq!(wire["result"].as_str(), Some("x^(a·b)"));
}
#[test]
fn test_eval_even_power_exponential_keeps_positive_root() {
    // `a^(2x) = k` is solved as `(a^x)^2 = k -> a^x = ±√k`. The POSITIVE root gives the real solution
    // `x = log_a(√k)`; the NEGATIVE root `a^x = -√k` is unsatisfiable (a^x > 0). The back-substitution
    // aggregator used to let the negative root's guarded (false) conditional OVERWRITE the real
    // solution, returning the empty `{…} if -√k > 0`. Discrete solutions now survive a sibling
    // conditional branch.
    // Cases with a clean closed form:
    for (input, expected) in [
        ("solve(2^(2*x)=2, x)", "{ 1/2 }"),
        ("solve(e^(2*x)=5, x)", "{ 1/2·ln(5) }"),
        // Unchanged controls (clean even powers / direct log / negative or zero RHS):
        ("solve(3^(2*x)=9, x)", "{ 1 }"),
        ("solve(3^(2*x)=81, x)", "{ 2 }"),
        ("solve(3^(2*x)=16, x)", "{ ln(4) / ln(3) }"),
        ("solve(e^(2*x)=-5, x)", "No solution"),
        ("solve(3^(2*x)=0, x)", "No solution"),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
    }
    // Cases whose value is correct but not fully simplified (e.g. `3^(2x)=27` is `3/2`): assert they
    // return a single real solution rather than the old empty `{…} if -√k > 0` conditional.
    for input in [
        "solve(3^(2*x)=27, x)",
        "solve(2^(2*x)=8, x)",
        "solve(5^(2*x)=125, x)",
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        let result = wire["result"].as_str().unwrap_or("");
        assert!(
            result.starts_with("{ ") && !result.contains("if") && !result.contains("No solution"),
            "{input} -> {result}"
        );
    }
}
#[test]
fn test_eval_nonstrict_inequality_includes_isolated_roots() {
    // For a NON-STRICT inequality `f ≤ 0` / `f ≥ 0`, every real in-domain root of `f` is a solution
    // (the value `0` satisfies the relation), but the interval sign-analysis emits only the
    // sign-CHANGE regions and drops the isolated roots of even-multiplicity factors that fall
    // outside them. Those roots are now unioned back in (as degenerate `[p, p]` intervals or a
    // `{p}` discrete set), with poles excluded by construction.
    for (input, expected) in [
        ("solve((x-2)^2*(x+1)<=0, x)", "(-infinity, -1] U [2, 2]"),
        ("solve((x+1)^2*(x-3)^3>=0, x)", "[-1, -1] U [3, infinity)"),
        ("solve(x^2/(x-1)>=0, x)", "[0, 0] U (1, infinity)"),
        (
            "solve(x^2*(x^2-4)>=0, x)",
            "(-infinity, -2] U [0, 0] U [2, infinity)",
        ),
        ("solve(x^3*(x-2)^2<=0, x)", "(-infinity, 0] U [2, 2]"),
        (
            "solve((x-1)*(x-2)^2*(x-3)>=0, x)",
            "(-infinity, 1] U [2, 2] U [3, infinity)",
        ),
        ("solve((x-1)^4*(x+1)<=0, x)", "(-infinity, -1] U [1, 1]"),
        ("solve(x^2/((x-1)*(x-2))<=0, x)", "[0, 0] U (1, 2)"),
        ("solve((x-3)^2/(x-1)<=0, x)", "(-infinity, 1) U [3, 3]"),
        ("solve((x+3)^2*(x-1)*(x-5)<=0, x)", "[-3, -3] U [1, 5]"),
        // Pure touch point -> the single solution, rendered as a degenerate interval `[p, p]` (the
        // root flows through the interval-union machinery once Discrete∪interval unions keep both sides).
        ("solve((x-2)^2<=0, x)", "[2, 2]"),
        ("solve(-(x-2)^2>=0, x)", "[2, 2]"),
        // STRICT controls: `0` does NOT satisfy `<`/`>`, so NO isolated root is added.
        ("solve((x-2)^2*(x+1)<0, x)", "(-infinity, -1)"),
        ("solve(x^2/(x-1)>0, x)", "(1, infinity)"),
        // Squares are everywhere-nonnegative; a pole is never a solution.
        ("solve((x-2)^4>=0, x)", "All real numbers"),
        ("solve(1/(x-2)^2<=0, x)", "No solution"),
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
fn test_eval_nested_power_text_is_parenthesized_and_round_trips() {
    // `^` is right-associative, so a surviving nested power must be parenthesized
    // in the TEXT output. `(4*x^2)^(1/2)` simplifies to `2·(x^2)^(1/2)` but was
    // rendered `2·x^2^(1/2)`, which re-parses as `2·x^(2^(1/2)) = 2·x^√2` — a
    // different, wrong expression. The fix wraps the power base in parentheses so
    // the text round-trips to the same value.
    let output = cli()
        .args(["eval", "(4*x^2)^(1/2)", "--format", "json"])
        .output()
        .expect("Failed to run CLI");
    let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
    let result = wire["result"].as_str().expect("result string");
    assert!(
        result.contains("(x^2)"),
        "nested power base must be parenthesized, got {result:?}"
    );

    // Round-trip: feed the rendered text back in; it must evaluate to the true
    // value `2·|x|`, not the mis-parsed `2·x^√2`.
    let reparse = cli()
        .args(["eval", &result.replace('·', "*"), "--format", "json"])
        .output()
        .expect("Failed to run CLI");
    let wire2: Value = serde_json::from_slice(&reparse.stdout).expect("Invalid wire output");
    assert_eq!(
        wire2["result"].as_str(),
        Some("2·|x|"),
        "rendered nested-power text must round-trip to 2·|x|, got {:?}",
        wire2["result"]
    );

    // Other clean power renderings are unchanged.
    for (input, expected) in [
        ("x^2", "x^2"),
        ("(x+1)^2", "(x + 1)^2"),
        ("x^2*y^3", "x^2·y^3"),
        ("(x^2)^(1/2)", "|x|"),
        ("x^2^3", "x^8"),
    ] {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let w: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        assert_eq!(w["result"].as_str(), Some(expected), "{input}");
    }
}
#[test]
fn test_eval_ln_quotient_change_of_base_folds_fractional_bases() {
    // `ln(c)/ln(b) = log_b(c)` now folds for reciprocal/fractional rationals
    // (a negative rational), not just integer-power pairs. It used to leak
    // `ln(8)/ln(1/2)` into a solve boundary as `(ln(8)/ln(1/2), inf)` instead of
    // the folded `(-3, inf)`.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("ln(8)/ln(1/2)"), "-3");
    assert_eq!(r("ln(1/8)/ln(2)"), "-3");
    assert_eq!(r("ln(1/4)/ln(2)"), "-2");
    assert_eq!(r("ln(16)/ln(1/2)"), "-4");
    // Integer-power pairs still fold; irrational ratios still decline.
    assert_eq!(r("ln(8)/ln(2)"), "3");
    assert_eq!(r("ln(8)/ln(4)"), "3/2");
    assert_eq!(r("ln(7)/ln(2)"), "ln(7) / ln(2)");
    // The exponential-inequality boundary now folds to the exact rational.
    assert_eq!(r("solve((1/2)^x < 8, x)"), "(-3, infinity)");
    assert_eq!(r("solve((1/3)^x > 9, x)"), "(-infinity, -2)");
}
#[test]
fn test_eval_reciprocal_root_laurent_equation_solves() {
    // A Laurent polynomial in `sqrt(x)` — a root mixed with its reciprocal — used
    // to leak a malformed `solve(x - (x^(-1/2)+1)^(1/(1/2)))` residual. It now
    // substitutes `u = x^(1/q)`, clears the `1/u^k`, and solves. `√x - 1/√x = 1`
    // is `u^2 - u - 1 = 0`, so `u = φ` (the negative surd root is dropped since
    // √x >= 0), giving `x = φ^2 = 1 + φ`.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("solve(sqrt(x) - 1/sqrt(x) = 1, x)"), "{ 1 + phi }");
    assert_eq!(r("solve(sqrt(x) + 1/sqrt(x) = 5/2, x)"), "{ 1/4, 4 }");
    assert_eq!(r("solve(2*sqrt(x) - 3/sqrt(x) = 1, x)"), "{ 9/4 }");
    assert_eq!(r("solve(sqrt(x) - 2/sqrt(x) = 0, x)"), "{ 2 }");
    // A double root and a genuinely empty case.
    assert_eq!(r("solve(sqrt(x) + 4/sqrt(x) = 4, x)"), "{ 4 }");
    assert_eq!(r("solve(sqrt(x) + 1/sqrt(x) = 1, x)"), "No solution");
    // No regression: pure-positive-power forms keep the sibling handler, plain
    // polynomials and Laurent-in-x are untouched.
    assert_eq!(r("solve(x - 3*sqrt(x) + 2 = 0, x)"), "{ 1, 4 }");
    assert_eq!(r("solve(x^(2/3) - x^(1/3) - 2 = 0, x)"), "{ -1, 8 }");
    assert_eq!(r("solve(1/x + x = 5/2, x)"), "{ 1/2, 2 }");
}
#[test]
fn test_eval_reciprocal_root_laurent_combined_fraction_and_higher_roots() {
    // `simplify` combines the reciprocal-root Laurent over a common denominator
    // (`x^(1/3) − 1/x^(1/3) → (x^(4/3) − x^(2/3))/x`) or renders a term as
    // `x^(2/3)/x`. Handling the top-level `Div(N, x^m)` and the term-level
    // `x^a/x^b` closes the cube/fourth-root reciprocal family (odd roots keep the
    // negative solution).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("solve(x^(1/3) - 1/x^(1/3) = 0, x)"), "{ -1, 1 }");
    assert_eq!(r("solve(x^(1/3) - 2/x^(1/3) = 1, x)"), "{ -1, 8 }");
    assert_eq!(r("solve(x^(1/3) + 1/x^(1/3) = 5/2, x)"), "{ 1/8, 8 }");
    // An even root drops the negative branch (x^(1/4) >= 0).
    assert_eq!(r("solve(x^(1/4) - 1/x^(1/4) = 0, x)"), "{ 1 }");
    // The cycle-sibling sqrt Pow-sum forms remain correct.
    assert_eq!(r("solve(sqrt(x) + 1/sqrt(x) = 5/2, x)"), "{ 1/4, 4 }");
    // No regression: an ordinary rational `(x^2-1)/x = 0` is untouched.
    assert_eq!(r("solve((x^2-1)/x = 0, x)"), "{ -1, 1 }");
}
/// `cbrt(x)` es la raíz cúbica y su LaTeX salía como `\text{cbrt}(x)`: el mismo
/// agujero de la tabla de funciones que tenía `root(a, n)` en la cabecera.
///
/// El álgebra sigue SIN unificarse: `cbrt(x)^3` no pliega a `x` y
/// `cbrt(x)/root(x,3)` —que es 1— no se cancela, porque `cbrt` no se canonicaliza
/// a `x^(1/3)` como sí hace su sinónimo. Canonicalizarla se probó y se MIDIÓ en el
/// mismo ciclo: el gate de sombra de claims pasa de 68 s a más de 240 s (el motor
/// NO se vuelve lento para el usuario — `integrate(1/(x^3-2), x)` sigue en 0,4 s —,
/// lo que explota es el número de pasos crudos que ese gate verifica). Queda como
/// peldaño con su número, no como olvido.
#[test]
fn test_eval_cbrt_renders_as_a_radical_on_both_surfaces() {
    let wire = |input: &str| -> Value {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        serde_json::from_slice(&out.stdout).expect("Invalid wire output")
    };

    assert_eq!(wire("cbrt(x + 1)")["result"], "cbrt(x + 1)");
    assert_eq!(wire("cbrt(x + 1)")["result_latex"], "\\sqrt[3]{x + 1}");
    assert_eq!(wire("cbrt(x + 1)")["input_latex"], "\\sqrt[3]{x + 1}");
    // Y el numérico se pliega igual que su sinónimo.
    assert_eq!(wire("cbrt(8)")["result"], "2");
    assert_eq!(wire("root(8,3)")["result"], "2");
}
