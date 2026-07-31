use super::*;

#[test]
fn test_eval_inverse_trig_canonical_reciprocal_root_exact_values_json() {
    let cases = [
        ("arcsin(sqrt(2)/2)", "1/4\u{00b7}pi"),
        ("arccos(sqrt(2)/2)", "1/4\u{00b7}pi"),
        ("arcsin(sqrt(3)/2)", "1/3\u{00b7}pi"),
        ("arccos(sqrt(3)/2)", "1/6\u{00b7}pi"),
        ("arctan(3^(-1/2))", "1/6\u{00b7}pi"),
    ];

    for (input, expected) in cases {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");

        assert!(output.status.success(), "input: {input}");

        let stdout = String::from_utf8(output.stdout).unwrap();
        let wire: Value = serde_json::from_str(&stdout).expect("Invalid wire output");

        assert_eq!(wire["ok"], true, "input: {input}");
        assert_eq!(wire["result"], expected, "input: {input}");
        assert_eq!(wire["warnings"], serde_json::json!([]), "input: {input}");
        assert_eq!(
            wire["required_display"],
            serde_json::json!([]),
            "input: {input}"
        );
    }
}
#[test]
fn test_eval_const_over_trig_equation_reduces_to_full_family() {
    // SOUNDNESS: `c/trig(x) = k` (`2/sin(x)=4`) isolated to the boundary and
    // returned only the PRINCIPAL value `{π/6}`, dropping the second branch and
    // all periodicity; the coefficient-1 form (`1/sin(x)=2`) folded `1/sin → csc`
    // mid-isolation and leaked `solve(csc(x)=2)`. Reduce `c/trig(g)=k` to
    // `trig(g)=c/k` and route to the bare-trig solver for the full periodic family.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    let sin_half =
        "{ 1/6\u{b7}pi + k\u{b7}2\u{b7}pi, 5/6\u{b7}pi + k\u{b7}2\u{b7}pi : k \u{2208} \u{2124} }";
    let cos_half =
        "{ 1/3\u{b7}pi + k\u{b7}2\u{b7}pi, 5/3\u{b7}pi + k\u{b7}2\u{b7}pi : k \u{2208} \u{2124} }";
    let tan_one = "{ 1/4\u{b7}pi + k\u{b7}pi : k \u{2208} \u{2124} }";
    // Numerator ≠ 1: was principal-value-only.
    assert_eq!(r("solve(2/sin(x)=4, x)"), sin_half);
    assert_eq!(r("solve(5/cos(x)=10, x)"), cos_half);
    // Numerator = 1: was the `solve(csc(x)=...)` leak.
    assert_eq!(r("solve(1/sin(x)=2, x)"), sin_half);
    assert_eq!(r("solve(1/cos(x)=2, x)"), cos_half);
    // Tangent (reduces to `tan(g)=c/k`, not the cot homogeneous path).
    assert_eq!(r("solve(3/tan(x)=3, x)"), tan_one);
    assert_eq!(r("solve(1/tan(x)=1, x)"), tan_one);
    // The reduced target is scale-invariant: `4/sin=8` and `2/sin=4` both give sin=1/2.
    assert_eq!(r("solve(4/sin(x)=8, x)"), sin_half);
    // Negative numerator flips the sign: sin(x) = -1/2.
    assert_eq!(
        r("solve(-2/sin(x)=4, x)"),
        "{ -1/6\u{b7}pi + k\u{b7}2\u{b7}pi, 7/6\u{b7}pi + k\u{b7}2\u{b7}pi : k \u{2208} \u{2124} }"
    );
    // Shifted/scaled argument routes through the full-family solver too.
    assert_eq!(
        r("solve(2/sin(2*x)=4, x)"),
        "{ 1/12\u{b7}pi + k\u{b7}pi, 5/12\u{b7}pi + k\u{b7}pi : k \u{2208} \u{2124} }"
    );
    // Range honesty: `|c/k| > 1` for sin/cos has no solution.
    assert_eq!(r("solve(1/sin(x)=1/2, x)"), "No solution");
    assert_eq!(r("solve(2/cos(x)=1, x)"), "No solution");

    // NO REGRESSION: bare csc/sec/cot and `trig(x)/c` (constant DENOMINATOR) keep
    // their own handling.
    assert_eq!(r("solve(csc(x)=2, x)"), sin_half);
    assert_eq!(r("solve(sin(x)/2=1, x)"), "No solution");
}
#[test]
fn test_eval_periodic_trig_inequality_declines() {
    // SOUNDNESS: a periodic `sin`/`cos`/`tan` inequality has an infinite periodic-union solution
    // that the monotonic inversion used to emit as a single wrong ray. Since cycle P2 the sin/cos
    // interior cases SOLVE exactly via PeriodicIntervalUnion; tan still declines honestly (P3).
    // The bare out-of-range cases (ℝ/∅) and equations are unaffected.
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
    for (input, expected) in [
        ("sin(x)>0", "{ (k·2·pi, pi + k·2·pi) : k ∈ ℤ }"),
        ("cos(x)<0", "{ (1/2·pi + k·2·pi, 3/2·pi + k·2·pi) : k ∈ ℤ }"),
        (
            "sin(x)>1/2",
            "{ (1/6·pi + k·2·pi, 5/6·pi + k·2·pi) : k ∈ ℤ }",
        ),
        ("tan(x)>1", "{ (1/4·pi + k·pi, 1/2·pi + k·pi) : k ∈ ℤ }"),
        ("sin(2*x)>0", "{ (k·pi, 1/2·pi + k·pi) : k ∈ ℤ }"),
        (
            "cos(x)>=1/2",
            "{ [-1/3·pi + k·2·pi, 1/3·pi + k·2·pi] : k ∈ ℤ }",
        ),
    ] {
        let (ok, result) = run(input);
        assert!(ok, "{input} should be ok=true, got {result:?}");
        assert_eq!(result, expected, "{input}");
    }
    let plain = |input: &str| run(input).1;
    // Out-of-range bare sin/cos are still answered exactly (not pre-empted by the residual decline).
    assert_eq!(plain("cos(x)<=1"), "All real numbers");
    assert_eq!(plain("sin(x)>2"), "No solution");
    // Equations and constant-trig (variable is linear) still solve (two-family periodic set).
    assert_eq!(
        plain("sin(x)=1/2"),
        "{ 1/6·pi + k·2·pi, 5/6·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(plain("sin(2)*x>0"), "(0, infinity)");
}
#[test]
fn test_eval_periodic_trig_equation_emits_family() {
    // A bare `sin/cos/tan(x)=c` equation has an INFINITE periodic family; the unary-inverse path
    // returned only the principal root (`solve(tan(x)=1)→{π/4}`, dropping `+kπ`). Emit the whole
    // family via the `Periodic` SolutionSet. tan is period π for every c; sin/cos collapse to a
    // single family only for c ∈ {0,±1} (other c are two families → decline, unchanged).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Period π families.
    assert_eq!(r("solve(sin(x)=0, x)"), "{ k·pi : k ∈ ℤ }");
    assert_eq!(r("solve(tan(x)=0, x)"), "{ k·pi : k ∈ ℤ }");
    assert_eq!(r("solve(cos(x)=0, x)"), "{ 1/2·pi + k·pi : k ∈ ℤ }");
    assert_eq!(r("solve(tan(x)=1, x)"), "{ 1/4·pi + k·pi : k ∈ ℤ }");
    assert_eq!(r("solve(tan(x)=sqrt(3), x)"), "{ 1/3·pi + k·pi : k ∈ ℤ }");
    // tan is complete even for a symbolic threshold.
    assert_eq!(r("solve(tan(x)=2, x)"), "{ arctan(2) + k·pi : k ∈ ℤ }");
    // Scaled argument `trig(a·x)=c`: divide base and period by `a` (a>1 shrinks the period).
    assert_eq!(r("solve(cos(2*x)=1, x)"), "{ k·pi : k ∈ ℤ }");
    assert_eq!(r("solve(sin(2*x)=0, x)"), "{ k·1/2·pi : k ∈ ℤ }");
    assert_eq!(r("solve(tan(2*x)=1, x)"), "{ 1/8·pi + k·1/2·pi : k ∈ ℤ }");
    assert_eq!(r("solve(sin(x/2)=0, x)"), "{ k·2·pi : k ∈ ℤ }");
    // Squared trig via the double-angle reduction `sin(arg)^2=c <=> cos(2·arg)=1-2c`.
    assert_eq!(r("solve(sin(x)^2=1, x)"), "{ 1/2·pi + k·pi : k ∈ ℤ }");
    assert_eq!(r("solve(cos(x)^2=1, x)"), "{ k·pi : k ∈ ℤ }");
    assert_eq!(r("solve(sin(x)^2=1/2, x)"), "{ 1/4·pi + k·1/2·pi : k ∈ ℤ }");
    assert_eq!(r("solve(sin(2*x)^2=1, x)"), "{ 1/4·pi + k·1/2·pi : k ∈ ℤ }");
    // sin(x)^2=1/4 -> cos(2x)=1/2 -> the TWO families {π/6+kπ, 5π/6+kπ}.
    assert_eq!(
        r("solve(sin(x)^2=1/4, x)"),
        "{ 1/6·pi + k·pi, 5/6·pi + k·pi : k ∈ ℤ }"
    );
    // Period 2π families (c = ±1, the two roots coincide -> one family).
    assert_eq!(r("solve(sin(x)=1, x)"), "{ 1/2·pi + k·2·pi : k ∈ ℤ }");
    assert_eq!(r("solve(cos(x)=1, x)"), "{ k·2·pi : k ∈ ℤ }");
    assert_eq!(r("solve(cos(x)=-1, x)"), "{ pi + k·2·pi : k ∈ ℤ }");
    // Two-family `sin/cos=c` (0 < |c| < 1): BOTH principal roots, shared period 2π.
    assert_eq!(
        r("solve(sin(x)=1/2, x)"),
        "{ 1/6·pi + k·2·pi, 5/6·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(cos(x)=1/2, x)"),
        "{ 1/3·pi + k·2·pi, 5/3·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(sin(x)=-1/2, x)"),
        "{ -1/6·pi + k·2·pi, 7/6·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(sin(x)=1/3, x)"),
        "{ arcsin(1/3) + k·2·pi, pi - arcsin(1/3) + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(r("solve(sin(x)=2, x)"), "No solution"); // |c|>1
                                                        // A SURD constant in the `= 0` form (`2·cos(x) − √3 = 0`) used to collapse to the principal root
                                                        // `{π/6}`, dropping the periodic family AND the second base root — the `A·trig + B` normalization
                                                        // required a RATIONAL offset `B`, so a surd `B` fell through to the principal-inverse isolation. The
                                                        // offset is now kept symbolically, so the `= 0` form matches the trusted direct-RHS form.
    assert_eq!(
        r("solve(2*cos(x)-sqrt(3)=0, x)"),
        "{ 1/6·pi + k·2·pi, 11/6·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(sin(x)-sqrt(3)/2=0, x)"),
        "{ 1/3·pi + k·2·pi, 2/3·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(r("solve(tan(x)-sqrt(3)=0, x)"), "{ 1/3·pi + k·pi : k ∈ ℤ }");
    assert_eq!(
        r("solve(cos(x)+sqrt(2)/2=0, x)"),
        "{ 3/4·pi + k·2·pi, 5/4·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(sin(2*x)-sqrt(3)/2=0, x)"),
        "{ 1/6·pi + k·pi, 1/3·pi + k·pi : k ∈ ℤ }"
    );
}
#[test]
fn test_eval_pi_shifted_argument_trig_keeps_periodic_family() {
    // `trig(a·x + b) = c` with `b` a π-multiple additive shift: the simplifier expands the
    // angle-addition (`sin(x + π/4) → (√2/2)·(sin x + cos x)`), and the isolation then returned only the
    // PRINCIPAL root (`sin(x + π/4) = 1/2 → {−π/12}`, dropping both the `+2kπ` family and the second
    // branch). Now `trig(u) = c` is solved for `u = a·x + b` and mapped back through `x = (u − b)/a`.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Both branches, full 2π period.
    assert_eq!(
        r("solve(sin(x + pi/4) = 1/2, x)"),
        "{ -1/12·pi + k·2·pi, 7/12·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(cos(x + pi/3) = 1/2, x)"),
        "{ k·2·pi, 4/3·pi + k·2·pi : k ∈ ℤ }"
    );
    // A coefficient AND a π-shift: base and period both scale by `1/a`.
    assert_eq!(
        r("solve(sin(2*x + pi/4) = 1/2, x)"),
        "{ -1/24·pi + k·pi, 7/24·pi + k·pi : k ∈ ℤ }"
    );
    // Single-family `c ∈ {0, ±1}` cases, and tan (period π).
    assert_eq!(
        r("solve(cos(x - pi/6) = 0, x)"),
        "{ 2/3·pi + k·pi : k ∈ ℤ }"
    );
    assert_eq!(r("solve(tan(x + pi/4) = 1, x)"), "{ k·pi : k ∈ ℤ }");
    assert_eq!(r("solve(sin(x - pi/2) = 1, x)"), "{ pi + k·2·pi : k ∈ ℤ }");
    // Out of range stays unsatisfiable.
    assert_eq!(r("solve(sin(x + pi/4) = 2, x)"), "No solution");
    // A SYMBOLIC (non-π) shift — `arctan`, surd — is mishandled the same way and now also keeps the
    // full family (the auxiliary-angle dispatch target `sin(x + arctan(b/a)) = c` relies on this).
    assert_eq!(
        r("solve(sin(x + arctan(4/3)) = 1, x)"),
        "{ 1/2·pi - arctan(4/3) + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(sin(x + sqrt(2)) = 1/2, x)"),
        "{ 1/6·pi - sqrt(2) + k·2·pi, 5/6·pi - sqrt(2) + k·2·pi : k ∈ ℤ }"
    );
    // Controls: a PLAIN-rational additive shift and the bare/coefficient forms are handled by the
    // existing periodic path and must be UNCHANGED (this handler declines — it gates on a symbolic shift).
    assert_eq!(
        r("solve(sin(x + 1) = 1/2, x)"),
        "{ 1/6·(pi - 6) + k·2·pi, 5/6·pi - 1 + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(sin(2*x) = 1/2, x)"),
        "{ 1/12·pi + k·pi, 5/12·pi + k·pi : k ∈ ℤ }"
    );
}
#[test]
fn test_eval_periodic_trig_product_equation_unions_families() {
    // A PRODUCT of periodic trig factors (or a `cos(a)±cos(b)` / `sin(a)±sin(b)` that reduces to one
    // via sum-to-product) used to drop periodicity: each factor was solved for its PRINCIPAL root
    // only and the roots unioned into a wrong finite set (`solve(cos(2x)-cos(x))→{0}`). Now every
    // factor yields its full `Periodic` family and the families are unioned over a common period.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Explicit products, equal period: the sin·cos product now reduces through
    // the double angle (`sin·cos = 0 ⇔ sin(2x) = 0`), yielding the SAME set in
    // its compact single-family form: {kπ} ∪ {π/2+kπ} ≡ {kπ/2} (k even ↦ kπ,
    // k odd ↦ π/2+(k−1)π/2·2... i.e. exact set equality).
    assert_eq!(r("solve(sin(x)*cos(x)=0, x)"), "{ k·1/2·pi : k ∈ ℤ }");
    assert_eq!(
        r("solve((2*cos(x)+1)*(cos(x)-1)=0, x)"),
        "{ 2/3·pi + k·2·pi, 4/3·pi + k·2·pi, k·2·pi : k ∈ ℤ }"
    );
    // Mixed periods (π and 2π): expand to the common period 2π, then union.
    assert_eq!(
        r("solve(sin(x)*(2*cos(x)-1)=0, x)"),
        "{ k·2·pi, pi + k·2·pi, 1/3·pi + k·2·pi, 5/3·pi + k·2·pi : k ∈ ℤ }"
    );
    // `cos(2x) − cos(x)` simplifies (in the solve context) to the single-atom polynomial
    // `2·cos(x)² − cos(x) − 1`, so the double-angle poly-in-`cos` path solves it (`cos ∈ {1, −1/2}`);
    // the family order reflects that path (the same complete set as sum-to-product).
    assert_eq!(
        r("solve(cos(2*x)-cos(x), x)"),
        "{ 2/3·pi + k·2·pi, 4/3·pi + k·2·pi, k·2·pi : k ∈ ℤ }"
    );
    // `sin(2x) − sin(x) = sin(x)·(2·cos(x) − 1)` stays on the sum-to-product / product path.
    assert_eq!(
        r("solve(sin(2*x)-sin(x), x)"),
        // Factor-wise family union (post-2026-07-13): same set, `k·2π` base last.
        "{ 1/3·pi + k·2·pi, pi + k·2·pi, 5/3·pi + k·2·pi, k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(cos(2*x)+cos(x), x)"),
        "{ pi + k·2·pi, 1/3·pi + k·2·pi, 5/3·pi + k·2·pi : k ∈ ℤ }"
    );
    // SOUNDNESS: a product mixing a trig factor with a non-periodic factor cannot be one periodic
    // set; it must stay an honest residual rather than emit a half-solved/wrong set.
    assert_eq!(r("solve((x-1)*sin(x)=0, x)"), "Solve: sin(x)·(x - 1) = 0");
    // Non-trig products are unaffected.
    assert_eq!(r("solve((x-1)*(x-2)=0, x)"), "{ 1, 2 }");
}
#[test]
fn test_eval_quadratic_in_trig_equation_unions_periodic_roots() {
    // A polynomial of degree ≥ 2 in a single trig atom (`2·sin(x)² − 3·sin(x) + 1 = 0`, NOT a perfect
    // square, so the squared-trig reduction misses it) leaked an `arcsin(… − cos(2x) …)` residual once
    // the double-angle identity fired. Substitute `u = sin(x)`, solve `P(u) = 0`, back-substitute each
    // root through the periodic solver (range guard drops `|u| > 1`), and union the families over a
    // common period — `union_solution_sets` drops a `Periodic ∪ Periodic`, so the handler combines them.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // `(2sin-1)(sin-1)=0`: BOTH families kept (`sin = 1/2` and `sin = 1`).
    assert_eq!(
        r("solve(2*sin(x)^2 - 3*sin(x) + 1 = 0, x)"),
        "{ 1/6·pi + k·2·pi, 5/6·pi + k·2·pi, 1/2·pi + k·2·pi : k ∈ ℤ }"
    );
    // `(2cos-1)(cos+1)=0`.
    assert_eq!(
        r("solve(2*cos(x)^2 + cos(x) - 1 = 0, x)"),
        "{ pi + k·2·pi, 1/3·pi + k·2·pi, 5/3·pi + k·2·pi : k ∈ ℤ }"
    );
    // Mixed periods: `sin = 0` (period π) and `sin = 1` (period 2π) combine over 2π.
    assert_eq!(
        r("solve(sin(x)^2 - sin(x) = 0, x)"),
        "{ k·2·pi, pi + k·2·pi, 1/2·pi + k·2·pi : k ∈ ℤ }"
    );
    // SOUNDNESS: a root outside `[-1, 1]` is dropped (`cos = 2` has no angle).
    assert_eq!(
        r("solve(cos(x)^2 - cos(x) - 2 = 0, x)"),
        "{ pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(2*sin(x)^2 + 5*sin(x) + 2 = 0, x)"),
        "{ -1/6·pi + k·2·pi, 7/6·pi + k·2·pi : k ∈ ℤ }"
    );
    // Controls: a pure square stays with the squared-trig reduction (compact form); a single trig and a
    // Pythagorean mix (two distinct atoms) are unchanged.
    assert_eq!(
        r("solve(2*sin(x)^2 - 1 = 0, x)"),
        "{ 1/4·pi + k·1/2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(sin(x) = 1/2, x)"),
        "{ 1/6·pi + k·2·pi, 5/6·pi + k·2·pi : k ∈ ℤ }"
    );
    // A MIXED `sin²(x) + cos(x)` now reduces via the Pythagorean identity (`sin² = 1 − cos²`) to a
    // polynomial in `cos(x)` and solves (`cos(x)·(1 − cos(x)) = 0 ⟹ cos(x) ∈ {0, 1}`).
    assert_eq!(
        r("solve(sin(x)^2 + cos(x) = 1, x)"),
        "{ 1/2·pi + k·2·pi, 3/2·pi + k·2·pi, k·2·pi : k ∈ ℤ }"
    );
}
#[test]
fn test_eval_double_angle_and_mixed_trig_reduce_to_single_atom() {
    // A double-angle `cos(2x)` folds (via the simplifier) to `2·cos(x)² − 1`; when the rest is a
    // polynomial in `cos(x)` the equation becomes a single-atom quadratic. When it mixes `sin` and
    // `cos` (e.g. `cos(2x) − sin(x) → 2·cos(x)² − sin(x) − 1`) the Pythagorean identity eliminates the
    // all-even atom. Both were `arccos(…)` / `arcsin(…)` residuals before.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // `cos(2x) + 3cos(x) + 2 = 0 ⟹ 2cos² + 3cos + 1 = 0 ⟹ cos ∈ {−1, −1/2}`.
    assert_eq!(
        r("solve(cos(2*x) + 3*cos(x) + 2 = 0, x)"),
        "{ pi + k·2·pi, 2/3·pi + k·2·pi, 4/3·pi + k·2·pi : k ∈ ℤ }"
    );
    // Mixed via Pythagorean: `cos(2x) = sin(x) ⟹ 2cos² − sin − 1 ⟹ −2sin² − sin + 1 = 0 ⟹
    // sin ∈ {1/2, −1}`. The `A = B` form and the pre-expanded form agree.
    assert_eq!(
        r("solve(cos(2*x) = sin(x), x)"),
        "{ -1/2·pi + k·2·pi, 1/6·pi + k·2·pi, 5/6·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(2*cos(x)^2 - sin(x) - 1 = 0, x)"),
        "{ -1/2·pi + k·2·pi, 1/6·pi + k·2·pi, 5/6·pi + k·2·pi : k ∈ ℤ }"
    );
    // `2sin²(x) + 3cos(x) − 3 = 0 ⟹ 2cos² − 3cos + 1 = 0 ⟹ cos ∈ {1, 1/2}`.
    assert_eq!(
        r("solve(2*sin(x)^2 + 3*cos(x) - 3 = 0, x)"),
        "{ 1/3·pi + k·2·pi, 5/3·pi + k·2·pi, k·2·pi : k ∈ ℤ }"
    );
    // Controls: a pure single-atom quadratic and a two-term `cos(2x) + cos(x)` (solved as a PRODUCT via
    // sum-to-product) are unchanged.
    assert_eq!(
        r("solve(2*sin(x)^2 - 3*sin(x) + 1 = 0, x)"),
        "{ 1/6·pi + k·2·pi, 5/6·pi + k·2·pi, 1/2·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(cos(2*x) + cos(x) = 0, x)"),
        "{ pi + k·2·pi, 1/3·pi + k·2·pi, 5/3·pi + k·2·pi : k ∈ ℤ }"
    );
}
#[test]
fn test_eval_homogeneous_linear_trig_equation_reduces_to_tangent() {
    // A HOMOGENEOUS linear trig equation `a·sin(g) + b·cos(g) = 0` (same argument `g`, `a ≠ 0`) reduces
    // to `tan(g) = −b/a` — dividing by `cos(g)` loses nothing since `cos(g) = 0` is never a solution when
    // `a ≠ 0`. The isolation path otherwise leaks an `arcsin(cos(x)·…)` residual.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // `sin = cos ⟹ tan = 1`, `sin + cos = 0 ⟹ tan = −1` (period π, one family).
    assert_eq!(r("solve(sin(x) = cos(x), x)"), "{ 1/4·pi + k·pi : k ∈ ℤ }");
    assert_eq!(
        r("solve(sin(x) + cos(x) = 0, x)"),
        "{ -1/4·pi + k·pi : k ∈ ℤ }"
    );
    // Irrational coefficient: `√3·sin − cos = 0 ⟹ tan = 1/√3 ⟹ π/6`.
    assert_eq!(
        r("solve(sqrt(3)*sin(x) - cos(x) = 0, x)"),
        "{ 1/6·pi + k·pi : k ∈ ℤ }"
    );
    // A non-notable ratio keeps the exact `arctan`.
    assert_eq!(
        r("solve(2*sin(x) - 3*cos(x) = 0, x)"),
        "{ arctan(3/2) + k·pi : k ∈ ℤ }"
    );
    // Affine argument: `sin(2x) = cos(2x) ⟹ tan(2x) = 1 ⟹ π/8 + kπ/2`.
    assert_eq!(
        r("solve(sin(2*x) = cos(2*x), x)"),
        "{ 1/8·pi + k·1/2·pi : k ∈ ℤ }"
    );
    // Controls: bare `sin/cos = 0` (owned by the periodic handler) and a product (not a sum) are
    // unchanged. (The inhomogeneous `… = c` is now solved by the auxiliary-angle handler — see below.)
    assert_eq!(r("solve(sin(x) = 0, x)"), "{ k·pi : k ∈ ℤ }");
    // (Same compact-form recontract as the product-union test above:
    // sin·cos = 0 reduces via sin(2x) = 0 to the equivalent {kπ/2}.)
    assert_eq!(r("solve(sin(x)*cos(x) = 0, x)"), "{ k·1/2·pi : k ∈ ℤ }");
}
#[test]
fn test_eval_inhomogeneous_linear_trig_uses_auxiliary_angle() {
    // `a·sin(g) + b·cos(g) = c` (`c ≠ 0`) reduces by the auxiliary angle to
    // `sin(g + arctan(b/a)) = c/√(a²+b²)` (normalizing `a > 0`), dispatched to the shifted-argument
    // solver. It was an `arcsin(… − cos(x) …)` residual before.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // `c/R = 1` (tangent): a single family. `3·sin + 4·cos = 5 ⟹ sin(x + arctan(4/3)) = 1`.
    assert_eq!(
        r("solve(3*sin(x) + 4*cos(x) = 5, x)"),
        "{ 1/2·pi - arctan(4/3) + k·2·pi : k ∈ ℤ }"
    );
    // `c/R < 1` (notable): `sin + cos = 1 ⟹ sin(x + π/4) = 1/√2 ⟹ {2kπ, π/2 + 2kπ}`.
    assert_eq!(
        r("solve(sin(x) + cos(x) = 1, x)"),
        "{ k·2·pi, 1/2·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(sin(x) - cos(x) = 1, x)"),
        "{ 1/2·pi + k·2·pi, pi + k·2·pi : k ∈ ℤ }"
    );
    // SOUNDNESS: `|c| > R ⟹ |c/R| > 1 ⟹` No solution (the surd range guard).
    assert_eq!(r("solve(3*sin(x) + 4*cos(x) = 6, x)"), "No solution");
    assert_eq!(r("solve(3*sin(x) + 4*cos(x) = 10, x)"), "No solution");
    // Irrational (provable-sign surd) coefficients: `sin + √3·cos = 1 ⟹ R = 2, φ = arctan(√3) = π/3`.
    assert_eq!(
        r("solve(sin(x) + sqrt(3)*cos(x) = 1, x)"),
        "{ -1/6·pi + k·2·pi, 1/2·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(sqrt(3)*sin(x) + cos(x) = 1, x)"),
        "{ k·2·pi, 2/3·pi + k·2·pi : k ∈ ℤ }"
    );
    // A COMPOUND coefficient `2·√2` (rational × surd): `classify_linear_trig_leaf` now multiplies the
    // outer factor by the inner coefficient (it used to discard the `√2`). `R = √(1+8) = 3`.
    assert_eq!(
        r("solve(sin(x) + 2*sqrt(2)*cos(x) = 3, x)"),
        "{ 1/2·pi - arctan(2·sqrt(2)) + k·2·pi : k ∈ ℤ }"
    );
    // Controls: the homogeneous `c = 0` is the tangent reduction (and its compound-coefficient case is
    // now correct too, thanks to the same `classify_linear_trig_leaf` fix).
    assert_eq!(r("solve(sin(x) = cos(x), x)"), "{ 1/4·pi + k·pi : k ∈ ℤ }");
    assert_eq!(
        r("solve(2*sqrt(2)*sin(x) - cos(x) = 0, x)"),
        "{ arctan(2^(-1/2) / 2) + k·pi : k ∈ ℤ }"
    );
}
#[test]
fn test_eval_abs_of_trig_equation_keeps_periodicity() {
    // `|A| = c` with a trig-bearing argument was solved to PRINCIPAL roots by the generic abs isolation
    // (`|2·sin(x)−1| = 1 → {π/2, 0}`). It now splits into `A = c ∨ A = −c`, solving each branch fully so
    // trig stays periodic, then unions the families (over a common period when they differ).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // `sin = 1` (period 2π) ∪ `sin = 0` (period π) — combined over 2π.
    assert_eq!(
        r("solve(abs(2*sin(x) - 1) = 1, x)"),
        "{ 1/2·pi + k·2·pi, k·2·pi, pi + k·2·pi : k ∈ ℤ }"
    );
    // `tan = ±1`, period π.
    assert_eq!(
        r("solve(abs(tan(x)) = 1, x)"),
        "{ 1/4·pi + k·pi, -1/4·pi + k·pi : k ∈ ℤ }"
    );
    // Both branches' non-principal `π − arcsin` roots are kept.
    assert_eq!(
        r("solve(abs(sin(x) - 1/2) = 1/4, x)"),
        "{ arcsin(3/4) + k·2·pi, pi - arcsin(3/4) + k·2·pi, arcsin(1/4) + k·2·pi, pi - arcsin(1/4) + k·2·pi : k ∈ ℤ }"
    );
    // One branch is out of range (`cos = 2`) and contributes nothing.
    assert_eq!(
        r("solve(abs(cos(x) - 1) = 1, x)"),
        "{ 1/2·pi + k·pi : k ∈ ℤ }"
    );
    // `c = 0` is a single branch.
    assert_eq!(
        r("solve(abs(2*sin(x) - 1) = 0, x)"),
        "{ 1/6·pi + k·2·pi, 5/6·pi + k·2·pi : k ∈ ℤ }"
    );
    // Both branches out of range ⇒ empty.
    assert_eq!(r("solve(abs(2*sin(x) - 1) = 5, x)"), "No solution");
    // Controls: bare `|trig| = c` keeps the periodic-trig reduction's form; non-trig `|A|` and a
    // negative RHS are unchanged.
    assert_eq!(
        r("solve(abs(sin(x)) = 1/2, x)"),
        "{ 1/6·pi + k·2·pi, 5/6·pi + k·2·pi, -1/6·pi + k·2·pi, 7/6·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(r("solve(abs(x - 1) = 2, x)"), "{ 3, -1 }");
    assert_eq!(r("solve(abs(sin(x)) = -1, x)"), "No solution");
}
#[test]
fn test_eval_trig_power_equation_keeps_periodicity() {
    // A trig EXPRESSION that simplifies to a perfect square / odd power of a single trig
    // (`cos(x)^2-1 -> -sin(x)^2`, `sin(x)*tan(x) -> sin^2/cos`, `(cos+1)(cos-1)sin -> -sin^3`)
    // collapsed to a single (often duplicated) finite root because the squared-trig reduction only
    // saw a bare `trig^2 = c` with the constant on the OTHER side and `n = 2`. Peeling a leading
    // coefficient/`Neg` and reducing `trig(arg)^n = 0` to `trig(arg) = 0` (with a complementary-
    // denominator guard for the quotient form) restores the full periodic family.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Squared / Neg / conjugate-factor forms.
    assert_eq!(r("solve(cos(x)^2-1, x)"), "{ k·pi : k ∈ ℤ }");
    assert_eq!(r("solve(sin(x)^2-1, x)"), "{ 1/2·pi + k·pi : k ∈ ℤ }");
    assert_eq!(r("solve((cos(x)-1)*(cos(x)+1), x)"), "{ k·pi : k ∈ ℤ }");
    assert_eq!(r("solve(-cos(x)^2, x)"), "{ 1/2·pi + k·pi : k ∈ ℤ }");
    // Odd-power forms (sin^3 used to decline; -sin^3 collapsed to {0}).
    assert_eq!(r("solve(sin(x)^3, x)"), "{ k·pi : k ∈ ℤ }");
    assert_eq!(r("solve(cos(x)^3, x)"), "{ 1/2·pi + k·pi : k ∈ ℤ }");
    assert_eq!(
        r("solve((cos(x)+1)*(cos(x)-1)*sin(x), x)"),
        "{ k·pi : k ∈ ℤ }"
    );
    // Quotient form with a complementary denominator (sin*tan = sin^2/cos).
    assert_eq!(r("solve(sin(x)*tan(x), x)"), "{ k·pi : k ∈ ℤ }");
    // Controls: the `= c` squared forms and non-trig equations are unchanged.
    assert_eq!(r("solve(sin(x)^2 = 1, x)"), "{ 1/2·pi + k·pi : k ∈ ℤ }");
    assert_eq!(
        r("solve(4*cos(x)^2 = 1, x)"),
        "{ 1/3·pi + k·pi, 2/3·pi + k·pi : k ∈ ℤ }"
    );
    assert_eq!(r("solve(x^2 - 1, x)"), "{ -1, 1 }");
}
#[test]
fn test_eval_trig_equation_with_surd_rhs_keeps_full_periodic_family() {
    // `sin(x) = √2/2` (and the other special-angle SURD right-hand sides) returned only the principal
    // value `{π/4}`: the periodic solver classified the RHS magnitude with `as_rational_const`, which
    // bails on an irrational, so the whole periodic path declined and the generic inverse leaked one
    // root. The classification is now exact over a quadratic surd (`linear_surd_sign`), so the full
    // two-branch periodic family is emitted — and `arcsin(√2/2)` simplifies to `π/4`.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(
        r("solve(sin(x) = sqrt(2)/2, x)"),
        "{ 1/4·pi + k·2·pi, 3/4·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(sin(x) = sqrt(3)/2, x)"),
        "{ 1/3·pi + k·2·pi, 2/3·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(cos(x) = sqrt(3)/2, x)"),
        "{ 1/6·pi + k·2·pi, 11/6·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(cos(x) = -sqrt(2)/2, x)"),
        "{ 3/4·pi + k·2·pi, 5/4·pi + k·2·pi : k ∈ ℤ }"
    );
    // Controls: rational RHS (special angle and general), the ±1 / 0 boundaries, and out-of-range
    // are all unchanged.
    assert_eq!(
        r("solve(sin(x) = 1/2, x)"),
        "{ 1/6·pi + k·2·pi, 5/6·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(sin(x) = 1/3, x)"),
        "{ arcsin(1/3) + k·2·pi, pi - arcsin(1/3) + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(r("solve(sin(x) = 1, x)"), "{ 1/2·pi + k·2·pi : k ∈ ℤ }");
    assert_eq!(r("solve(sin(x) = 0, x)"), "{ k·pi : k ∈ ℤ }");
    assert_eq!(r("solve(sin(x) = 2, x)"), "No solution");
}
#[test]
fn test_eval_trig_equation_affine_argument_and_odd_power_keep_family() {
    // Two more periodic-family-drop forms. (b) an AFFINE argument `sin(x - 1) = 0` returned only the
    // principal `{1}` — the arg detector handled `a·x` but not `a·x + b`; it now peels the offset and
    // shifts the family (`x = (u - b)/a`). (c) an ODD power `cos(x)^3 = 1` returned `{0}` — it now
    // reduces `trig^n = c` (n odd) to `trig = c^(1/n)` (a bijection on ℝ) and recurses.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // (b) affine argument: shifted, scaled+shifted, and the c=1 single-family form.
    assert_eq!(r("solve(sin(x-1)=0, x)"), "{ 1 + k·pi : k ∈ ℤ }");
    assert_eq!(r("solve(cos(x+1)=0, x)"), "{ 1/2·(pi - 2) + k·pi : k ∈ ℤ }");
    assert_eq!(r("solve(sin(3*x-1)=0, x)"), "{ 1/3 + k·1/3·pi : k ∈ ℤ }");
    assert_eq!(r("solve(cos(x-1)=1, x)"), "{ 1 + k·2·pi : k ∈ ℤ }");
    // (c) odd power = constant: real n-th root, then the full family.
    assert_eq!(r("solve(cos(x)^3=1, x)"), "{ k·2·pi : k ∈ ℤ }");
    assert_eq!(r("solve(sin(x)^3=1, x)"), "{ 1/2·pi + k·2·pi : k ∈ ℤ }");
    assert_eq!(r("solve(cos(x)^3=-1, x)"), "{ pi + k·2·pi : k ∈ ℤ }");
    // sin(x)^5 = 1/32 -> sin(x) = 1/2 -> both branches.
    assert_eq!(
        r("solve(sin(x)^5=1/32, x)"),
        "{ 1/6·pi + k·2·pi, 5/6·pi + k·2·pi : k ∈ ℤ }"
    );
    // SOUNDNESS: sin(x)^n ∈ [-1, 1], so an out-of-range RHS has NO real solution (must not leak the
    // spurious non-real arcsin(2^(1/3)) the cube-root reduction would otherwise produce).
    assert_eq!(r("solve(sin(x)^3=2, x)"), "No solution");
    assert_eq!(r("solve(cos(x)^3=8, x)"), "No solution");
    // Controls: the n=2 square reduction and the bare/scaled forms are unchanged.
    assert_eq!(r("solve(cos(x)^2=1, x)"), "{ k·pi : k ∈ ℤ }");
    assert_eq!(
        r("solve(sin(2*x)=1/2, x)"),
        "{ 1/12·pi + k·pi, 5/12·pi + k·pi : k ∈ ℤ }"
    );
}
#[test]
fn test_eval_even_power_and_abs_trig_equation_keeps_family() {
    // `trig(x)^n = c` for EVEN n >= 4 (and `|trig(x)| = c`) collapsed the infinite periodic root set
    // to a finite pair, or leaked a spurious arcsin(>1) for an out-of-range RHS. Now reduced to
    // `trig = ±c^(1/n)` (resp. `trig = ±c`) with a range guard.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Even power: the full two-branch periodic family (sin=+/-1, +/-c^(1/n), ...).
    assert_eq!(
        r("solve(sin(x)^4=1, x)"),
        "{ 1/2·pi + k·2·pi, -1/2·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(r("solve(cos(x)^4=1, x)"), "{ k·2·pi, pi + k·2·pi : k ∈ ℤ }");
    assert_eq!(
        r("solve(sin(x)^4=1/16, x)"),
        "{ 1/6·pi + k·2·pi, 5/6·pi + k·2·pi, -1/6·pi + k·2·pi, 7/6·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(r("solve(sin(x)^4=0, x)"), "{ k·pi : k ∈ ℤ }");
    // An n-th-root RHS (not a quadratic surd) now also emits the full family.
    assert_eq!(
        r("solve(sin(x)^4=1/4, x)"),
        "{ arcsin(root(1/4, 4)) + k·2·pi, pi - arcsin(root(1/4, 4)) + k·2·pi, -arcsin(root(1/4, 4)) + k·2·pi, arcsin(root(1/4, 4)) + pi + k·2·pi : k ∈ ℤ }"
    );
    // SOUNDNESS: an out-of-range RHS has NO real solution (no spurious arcsin(>1)).
    assert_eq!(r("solve(sin(x)^4=4, x)"), "No solution");
    assert_eq!(r("solve(sin(x)^6=2, x)"), "No solution");
    assert_eq!(r("solve(sin(x)^4=-1, x)"), "No solution");
    // |trig(x)| = c reduces the same way.
    assert_eq!(
        r("solve(abs(sin(x))=1, x)"),
        "{ 1/2·pi + k·2·pi, -1/2·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(r("solve(abs(cos(x))=0, x)"), "{ 1/2·pi + k·pi : k ∈ ℤ }");
    assert_eq!(r("solve(abs(sin(x))=2, x)"), "No solution");
    // Controls: n=2, odd power, and the bare form are unchanged.
    assert_eq!(r("solve(sin(x)^2=1, x)"), "{ 1/2·pi + k·pi : k ∈ ℤ }");
    assert_eq!(r("solve(cos(x)^3=1, x)"), "{ k·2·pi : k ∈ ℤ }");
    assert_eq!(
        r("solve(sin(x)=1/2, x)"),
        "{ 1/6·pi + k·2·pi, 5/6·pi + k·2·pi : k ∈ ℤ }"
    );
}
#[test]
fn test_eval_boundary_trig_inequality_is_periodic_point_set_or_residual() {
    // A bare sin/cos inequality at the EXACT range boundary +-1 returned a wrong ray
    // (`sin(x) >= 1 -> [pi/2, infinity)`). The TOUCH side holds only where the trig equals the extreme,
    // so it is the periodic point set; the COMPLEMENT side is R minus those points (not representable)
    // and declines to an honest residual instead of the wrong ray.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Touch side -> periodic point set (Periodic variant).
    assert_eq!(r("solve(sin(x) >= 1, x)"), "{ 1/2·pi + k·2·pi : k ∈ ℤ }");
    assert_eq!(r("solve(sin(x) <= -1, x)"), "{ -1/2·pi + k·2·pi : k ∈ ℤ }");
    assert_eq!(r("solve(cos(x) >= 1, x)"), "{ k·2·pi : k ∈ ℤ }");
    assert_eq!(r("solve(cos(x) <= -1, x)"), "{ pi + k·2·pi : k ∈ ℤ }");
    // Complement side -> honest residual (no more wrong ray).
    assert_eq!(
        r("solve(cos(x) < 1, x)"),
        "{ (k·2·pi, 2·pi + k·2·pi) : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(sin(x) > -1, x)"),
        "{ (-1/2·pi + k·2·pi, 3/2·pi + k·2·pi) : k ∈ ℤ }"
    );
    // Range-guard combinations stay exact R / empty.
    assert_eq!(r("solve(sin(x) <= 1, x)"), "All real numbers");
    assert_eq!(r("solve(sin(x) > 1, x)"), "No solution");
    assert_eq!(r("solve(sin(x) >= -1, x)"), "All real numbers");
    assert_eq!(r("solve(cos(x) < -1, x)"), "No solution");
}
#[test]
fn test_eval_periodic_trig_equation_with_outside_coefficient_emits_full_family() {
    // SOUNDNESS: an OUTSIDE coefficient/offset (`2·sin x = 1`, `2·cos x + 1 = 0`) left the trig side a
    // `Mul`/`Add` that the bare-trig detector could not see, so the equation fell through to the
    // unary-inverse path and returned only the PRINCIPAL value (`{π/6}`) — an incomplete solution set
    // presented as complete, with ok=true and no warning. Normalising `A·trig(a·x)+B=C` to
    // `trig(a·x)=(C−B)/A` before detection now routes it through the full `Periodic` generator.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Outside coefficient -> the SAME family the bare `trig=c'` form yields.
    assert_eq!(
        r("solve(2*sin(x)=1, x)"),
        "{ 1/6·pi + k·2·pi, 5/6·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(2*cos(x)+1=0, x)"),
        "{ 2/3·pi + k·2·pi, 4/3·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(r("solve(3*tan(x)=3, x)"), "{ 1/4·pi + k·pi : k ∈ ℤ }");
    assert_eq!(r("solve(5*sin(x)=5, x)"), "{ 1/2·pi + k·2·pi : k ∈ ℤ }"); // c=1 single family
    assert_eq!(
        r("solve(3*sin(x)=1, x)"),
        "{ arcsin(1/3) + k·2·pi, pi - arcsin(1/3) + k·2·pi : k ∈ ℤ }"
    );
    // Negative coefficient (sign folds into c), additive offset, and scaled argument all work.
    assert_eq!(
        r("solve(-2*sin(x)=1, x)"),
        "{ -1/6·pi + k·2·pi, 7/6·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(2*sin(x)+1=2, x)"),
        "{ 1/6·pi + k·2·pi, 5/6·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(2*sin(2*x)=1, x)"),
        "{ 1/12·pi + k·pi, 5/12·pi + k·pi : k ∈ ℤ }"
    );
    // SOUNDNESS edges: out-of-range stays empty; c=±1 single family.
    assert_eq!(r("solve(2*sin(x)=3, x)"), "No solution");
    assert_eq!(r("solve(2*sin(x)=2, x)"), "{ 1/2·pi + k·2·pi : k ∈ ℤ }");
    // SQUARED trig with an outside coefficient: `A·trig(arg)^2=c` folds to `trig(arg)^2=c/A` so the
    // double-angle reduction runs (previously `4·cos²x=1` dropped the `+kπ` and returned `{π/3, 2π/3}`).
    assert_eq!(
        r("solve(4*cos(x)^2=1, x)"),
        "{ 1/3·pi + k·pi, 2/3·pi + k·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(4*sin(x)^2=1, x)"),
        "{ 1/6·pi + k·pi, 5/6·pi + k·pi : k ∈ ℤ }"
    );
    assert_eq!(r("solve(2*cos(x)^2=1, x)"), "{ 1/4·pi + k·1/2·pi : k ∈ ℤ }");
    assert_eq!(r("solve(3*sin(x)^2=3, x)"), "{ 1/2·pi + k·pi : k ∈ ℤ }"); // sin²=1 single family
    assert_eq!(r("solve(4*cos(x)^2=5, x)"), "No solution"); // cos²=5/4 > 1
}
#[test]
fn test_eval_trig_inequality_out_of_range() {
    // SOUNDNESS: `sin(x)`/`cos(x)` ≷ c with c PROVABLY outside [-1, 1] is ℝ or ∅, not the finite ray
    // (sometimes with a non-real `arcsin(c)` endpoint) the generic monotonic inversion produced.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("cos(x)<=1"), "All real numbers"); // cos ≤ 1 always
    assert_eq!(r("cos(x)>1"), "No solution"); // cos > 1 never
    assert_eq!(r("sin(x)>2"), "No solution"); // out of range, no non-real arcsin(2) endpoint
    assert_eq!(r("cos(x)<-1"), "No solution");
    assert_eq!(r("sin(x)<2"), "All real numbers");
    assert_eq!(r("cos(x)>=-1"), "All real numbers"); // cos ≥ -1 always
    assert_eq!(r("sin(x)>=2"), "No solution");
    // Controls: an in-range threshold now SOLVES exactly (cycle P2 PeriodicIntervalUnion; the old
    // `(1/6·pi, infinity)` ray was unsound: sin(x)>1/2 is false at x=pi, which lies in that ray).
    // Equations are unchanged.
    assert_eq!(
        r("sin(x)>1/2"),
        "{ (1/6·pi + k·2·pi, 5/6·pi + k·2·pi) : k ∈ ℤ }"
    );
    assert_eq!(r("cos(x)=2"), "No solution");
    assert_eq!(
        r("sin(x)=1/3"),
        "{ arcsin(1/3) + k·2·pi, pi - arcsin(1/3) + k·2·pi : k ∈ ℤ }"
    );
}
#[test]
fn test_eval_complementary_inverse_trig_respects_domain() {
    // `arcsin(x) + arccos(x) = π/2` (and the `arcsec + arccsc` form that reduces
    // to it) holds only where both terms are real, i.e. on `[-1, 1]` for
    // arcsin/arccos. For a concrete argument provably OUTSIDE that interval both
    // terms are undefined, so the identity must NOT collapse the sum to π/2.
    // Previously `arccos(2) + arcsin(2)` and `arcsec(1/2) + arccsc(1/2)` returned
    // π/2 — a wrong answer.
    for input in [
        "arccos(2) + arcsin(2)",
        "arcsin(2) + arccos(2)",
        "arccos(3) + arcsin(3)",
        "arcsec(1/2) + arccsc(1/2)",
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_ne!(
            wire["result"].as_str(),
            Some("1/2·pi"),
            "{input}: out-of-domain inverse-trig sum must not collapse to π/2"
        );
    }

    // Valid arguments still apply the identity: symbolic (with the domain
    // condition), in-interval constants, and the `|x| >= 1` arcsec/arccsc form.
    for input in [
        "arccos(x) + arcsin(x)",
        "arccos(1/2) + arcsin(1/2)",
        "arccos(1) + arcsin(1)",
        "arccos(-1) + arcsin(-1)",
        "arcsec(2) + arccsc(2)",
        "arcsec(x) + arccsc(x)",
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(
            wire["result"].as_str(),
            Some("1/2·pi"),
            "{input}: valid complementary inverse-trig sum must give π/2"
        );
    }

    // The symbolic arcsin/arccos form carries its `-1 ≤ x ≤ 1` domain condition.
    let output = cli()
        .args(["eval", "arccos(x) + arcsin(x)", "--format", "json"])
        .output()
        .expect("Failed to run CLI");
    let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
    let displays = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert!(
        displays.iter().any(|v| v.as_str() == Some("-1 ≤ x ≤ 1")),
        "arccos(x)+arcsin(x) must carry the -1 ≤ x ≤ 1 condition, got {displays:?}"
    );

    // The symbolic arcsec/arccsc form collapses to π/2 too, but `arcsec`/`arccsc`
    // are real only for `|arg| ≥ 1`, so the sum MUST carry the exterior-interval
    // condition (it is `x ≤ -1 or x ≥ 1` for the bare variable, and scales with an
    // affine argument). Previously the condition was dropped: the collapse to π/2
    // removed the `arccos(1/x)` witness before the per-function domain was attached.
    for (input, expected_condition) in [
        ("arcsec(x) + arccsc(x)", "x ≤ -1 or x ≥ 1"),
        ("arccsc(x) + arcsec(x)", "x ≤ -1 or x ≥ 1"),
        ("arcsec(2*x) + arccsc(2*x)", "x ≤ -1/2 or x ≥ 1/2"),
        ("arcsec(x + 1) + arccsc(x + 1)", "x ≤ -2 or x ≥ 0"),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(
            wire["result"].as_str(),
            Some("1/2·pi"),
            "{input}: in-domain arcsec/arccsc sum must give π/2"
        );
        let displays = wire["required_display"]
            .as_array()
            .expect("required_display");
        assert!(
            displays
                .iter()
                .any(|v| v.as_str() == Some(expected_condition)),
            "{input} must carry the {expected_condition:?} domain condition, got {displays:?}"
        );
    }
}
