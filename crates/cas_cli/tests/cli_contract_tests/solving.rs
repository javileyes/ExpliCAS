use super::*;

#[test]
fn test_eval_symbolic_quadratic_with_negative_constant_discriminant_is_empty() {
    // `x² = c` with a PROVABLY-NEGATIVE constant `c` (surd OR transcendental) has no real root, but the
    // symbolic-coefficient quadratic path emitted `±√(negative)/(2a)` as if real (a mixed surd /
    // transcendental radicand doesn't syntactically expose its sign). The discriminant now gates on
    // `provable_const_sign` — a proven-negative constant delta ⇒ No solution.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("solve(x^2 = 1-sqrt(2), x)"), "No solution");
    assert_eq!(r("solve(abs(x^2+2) = sqrt(2), x)"), "No solution");
    assert_eq!(r("solve(x^2 = -pi, x)"), "No solution");
    assert_eq!(r("solve(x^2 = e-3, x)"), "No solution");
    // Controls: a POSITIVE constant discriminant keeps both real roots; a free-variable (symbolic)
    // quadratic is untouched (sign undecidable ⇒ kept).
    assert_eq!(
        r("solve(x^2 = sqrt(2)-1, x)"),
        "{ -sqrt(sqrt(2) - 1), sqrt(sqrt(2) - 1) }"
    );
    assert_eq!(
        r("solve(a*x^2+b*x+c=0, x)"),
        "{ (-sqrt(b^2 - 4·a·c) - b) / (2·a), (sqrt(b^2 - 4·a·c) - b) / (2·a) }"
    );
}
#[test]
fn test_eval_solver_function_aliases_solve_via_canonical_forms() {
    // `log2`/`log10`/`cbrt` used to error `función [...] no definida` in solve():
    // they now rewrite to their canonical invertible forms (`log(2,·)`, `log(10,·)`,
    // `u^(1/3)`) at the solve entry. The reciprocal trig aliases (`csc`/`sec`/`cot`)
    // are handled at the EQUATION level (a subtree `1/sin` rewrite gets re-folded to
    // `csc` by the simplifier): `csc ⟺ sin = 1/c`, `sec ⟺ cos = 1/c`,
    // `cot(g) = c ⟺ cos − c·sin = 0` — the cos/sin form keeps `cot = 0 → π/2 + kπ`,
    // which a `1/tan` rewrite would lose.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("solve(log2(x)=3, x)"), "{ 8 }");
    assert_eq!(r("solve(log10(x)=2, x)"), "{ 100 }");
    assert_eq!(r("solve(abs(log2(x))<3, x)"), "(1/8, 8)");
    assert_eq!(r("solve(cbrt(x)=-2, x)"), "{ -8 }");
    assert_eq!(
        r("solve(csc(x)=2, x)"),
        "{ 1/6\u{b7}pi + k\u{b7}2\u{b7}pi, 5/6\u{b7}pi + k\u{b7}2\u{b7}pi : k \u{2208} \u{2124} }"
    );
    assert_eq!(
        r("solve(sec(x)=2, x)"),
        "{ 1/3\u{b7}pi + k\u{b7}2\u{b7}pi, 5/3\u{b7}pi + k\u{b7}2\u{b7}pi : k \u{2208} \u{2124} }"
    );
    assert_eq!(
        r("solve(cot(x)=0, x)"),
        "{ 1/2\u{b7}pi + k\u{b7}pi : k \u{2208} \u{2124} }"
    );
    // Range honesty comes free from the owning solver: |1/c| > 1 has no solution.
    assert_eq!(r("solve(csc(x)=1/2, x)"), "No solution");
    assert_eq!(r("solve(csc(x)=0, x)"), "No solution");
}
#[test]
fn test_eval_solve_calculus_binder_solution_survives() {
    // Chip del barrido F0 (2026-07-19): `solve(limit(1/x,x,infinity)=y, y)`
    // afirmaba "No solution" — los walkers non-finite recursaban en los ARGS de
    // la call y el `infinity` de la COTA (notación, no valor: el límite ES 0)
    // marcaba la solución como no-finita. Los binders de cálculo son opacos.
    let eval_result = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // F9 GRADUÓ el limit anidado: ahora el limit RESUELVE dentro de solve
    // (limit(1/x,x,∞)→0 ⇒ y=0) — estrictamente mejor que la solución residual
    // que este pin fijaba mientras el limit no evaluaba (contrato actualizado
    // por intención). El fix de los walkers non-finite sigue pineado por los
    // "No solution" legítimos de abajo.
    assert_eq!(
        eval_result("solve(limit(1/x, x, infinity) = y, y)"),
        "{ 0 }"
    );
    assert_eq!(
        eval_result("solve(y = limit(1/x, x, infinity), y)"),
        "{ 0 }"
    );
    // Pins: los "No solution" legítimos y los guards de no-realidad intactos.
    assert_eq!(eval_result("solve(x = x+1, x)"), "No solution");
    assert_eq!(eval_result("solve(abs(x) = -1, x)"), "No solution");
    assert_eq!(eval_result("solve(cos(x) = 2, x)"), "No solution");
    assert_eq!(eval_result("solve(sin(x) = sqrt(2), x)"), "No solution");
}
#[test]
fn test_eval_solve_critical_points() {
    // Cierre vectorial · V7d (decisión del usuario): los diff inline se pre-evalúan
    // en el path de solve_system (con fold numérico de los artefactos x^(2-1)), así
    // que el flujo de puntos críticos con gradiente LINEAL resuelve one-shot; el
    // no-lineal sigue declinando honesto (scope-out: Gröbner = mate-nueva).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(
        r("solve([diff(x^2+y^2-2*x-4*y, x)=0, diff(x^2+y^2-2*x-4*y, y)=0], [x,y])"),
        "{ x = 1, y = 2 }"
    );
    // Sistema acoplado (parciales cruzadas): ∇(x²+xy+y²−3x) = 0 → (2, −1).
    assert_eq!(
        r("solve([diff(x^2+x*y+y^2-3*x, x)=0, diff(x^2+x*y+y^2-3*x, y)=0], [x,y])"),
        "{ x = 2, y = -1 }"
    );
    // El flujo curricular COMPLETO compone: clasificación en el crítico hallado.
    assert_eq!(
        r("subs(subs(det(hessian(x^2+y^2-2*x-4*y,[x,y])), x, 1), y, 2)"),
        "4"
    );
    // Pins: lineal puro intacto; y desde S2 (frente sistemas) el gradiente
    // no-lineal AISLABLE resuelve por sustitución verificada — el contrato
    // V7d «declina honesto» graduó a capacidad SIN Gröbner: ∇(x³+y³−3xy)=0
    // → y=x² sustituida da x⁴=x → (0,0) y (1,1), completo sobre ℝ (el
    // factor x²+x+1 no tiene raíces reales).
    assert_eq!(r("solve([x+y=3, x-y=1], [x,y])"), "{ x = 2, y = 1 }");
    assert_eq!(
        r("solve([diff(x^3+y^3-3*x*y, x)=0, diff(x^3+y^3-3*x*y, y)=0], [x,y])"),
        "{ x = 0, y = 0 } or { x = 1, y = 1 }"
    );
    // El scope-out REAL sigue declinando honesto: sin ecuación aislable
    // (ninguna es lineal-con-coeficiente-constante en una incógnita),
    // Gröbner sigue siendo mate-nueva fuera del alcance.
    let out = cli()
        .args(["eval", "solve([x^2+y^3=5, x^3-y^2=1], [x,y])"])
        .output()
        .expect("Failed to run CLI");
    let text =
        String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr);
    assert!(
        text.contains("polynomial conversion") || text.contains("non-linear"),
        "el sistema no-aislable debe declinar honesto, got: {text}"
    );
}
#[test]
fn test_eval_reciprocal_positive_function_inequality_flips() {
    // SOUNDNESS: `c/f(x) OP k` with a provably-positive function denominator (abs, …) and k > 0 must
    // FLIP when isolating the denominator: `c/f > k ⟺ f < c/k`. Previously the engine kept the
    // direction, returning the COMPLEMENT (`1/abs(x)>2 → (-∞,-1/2)∪(1/2,∞)`). The denominator pole is
    // conveyed via the `x ≠ ...` required condition (so the interval ∩ condition is the true set).
    let run = |input: &str| -> (String, Vec<String>) {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        let conds = wire["required_display"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        (wire["result"].as_str().unwrap_or("").to_string(), conds)
    };
    // 1/abs(x) > 2 ⟺ 0 < abs(x) < 1/2 — the SET itself now punctures the pole
    // (scout family C: "(-1/2, 1/2)" asserted x=0 in the set and relied on the
    // reader combining it with the side condition).
    assert_eq!(
        run("solve(1/abs(x)>2, x)"),
        ("(-1/2, 0) U (0, 1/2)".into(), vec!["x ≠ 0".into()])
    );
    assert_eq!(
        run("solve(2/abs(x)>1, x)"),
        ("(-2, 0) U (0, 2)".into(), vec!["x ≠ 0".into()])
    );
    assert_eq!(
        run("solve(1/abs(x-1)>2, x)"),
        ("(1/2, 1) U (1, 3/2)".into(), vec!["x ≠ 1".into()])
    );
    // The `<` direction is unchanged (it was already the larger side): abs(x) > 1/2.
    assert_eq!(
        run("solve(1/abs(x)<2, x)").0,
        "(-infinity, -1/2) U (1/2, infinity)"
    );
    // Controls: bare-variable reciprocal (sign-split path) and equality are unchanged.
    assert_eq!(run("solve(1/x>2, x)").0, "(0, 1/2)");
    assert_eq!(run("solve(1/x<2, x)").0, "(-infinity, 0) U (1/2, infinity)");
}
#[test]
fn test_eval_two_sided_rational_inequality_moves_to_one_side() {
    // `A(x) {op} B(x)` with the variable on BOTH sides and a rational difference (`1/(x-1) > 1/(x+1)`)
    // reached a path that emitted a garbage `inf^(1/2)` bound when the difference numerator is a nonzero
    // constant — or `{2}` / "No solution" for other shapes — even though the explicit-difference form
    // solved correctly. It is now moved to one side (`(A - B) {op} 0`) and routed through the verified
    // `N/D {op} 0` path.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Constant-numerator difference (the `inf^(1/2)` garbage case).
    assert_eq!(
        r("solve(1/(x-1) > 1/(x+1), x)"),
        "(-infinity, -1) U (1, infinity)"
    );
    assert_eq!(r("solve(1/(x+2) > 1/(x-2), x)"), "(-2, 2)");
    assert_eq!(
        r("solve(3/(x-1) > 3/(x+1), x)"),
        "(-infinity, -1) U (1, infinity)"
    );
    assert_eq!(r("solve(1/(x-1) < 1/(x+1), x)"), "(-1, 1)");
    // A linear-numerator difference (was returning the boundary point `{2}`).
    assert_eq!(r("solve(1/(x-1) > 3/(x+1), x)"), "(-infinity, -1) U (1, 2)");
    // Fraction vs a polynomial side (was "No solution"); irrational golden-ratio bounds.
    assert_eq!(
        r("solve(1/(x-1) > x, x)"),
        "(-infinity, 1/2·(1 - sqrt(5))) U (1, phi)"
    );
    // Non-strict keeps the numerator zero as a CLOSED endpoint, poles excluded.
    assert_eq!(
        r("solve(2/(x-1) >= 3/(x-2), x)"),
        "(-infinity, -1] U (1, 2)"
    );
    // Controls: an already-correct two-sided form, a radical two-sided (NOT preempted), and a
    // polynomial two-sided (declines the rational path, solved by its own).
    assert_eq!(r("solve(1/(x-1) > 2/(x+1), x)"), "(-infinity, -1) U (1, 3)");
    assert_eq!(r("solve(sqrt(x) > x - 2, x)"), "[0, 4)");
    assert_eq!(r("solve(x^2 > x, x)"), "(-infinity, 0) U (1, infinity)");

    // A nonzero constant numerator over a SINGLE-POLE linear-SURD/π denominator reduces exactly to the
    // boundary `g {op'} 0` (`Polynomial::from_expr` declines the irrational intercept `x − √2`, which
    // used to leave a garbage `(√2+∞, ∞)` interval on the legacy path).
    assert_eq!(r("solve(1/(x-sqrt(2)) > 0, x)"), "(sqrt(2), infinity)");
    assert_eq!(r("solve(1/(x-pi) > 0, x)"), "(pi, infinity)");
    assert_eq!(r("solve(1/(x-pi) < 0, x)"), "(-infinity, pi)");
    assert_eq!(r("solve(2/(x-sqrt(3)) < 0, x)"), "(-infinity, sqrt(3))");
}
#[test]
fn test_eval_parametric_linear_degenerate_branch() {
    // A parametric linear equation whose coefficient cancels (`a·x = a`) dropped the `a ≠ 0` guard
    // and the `a = 0 ⇒ ℝ` branch, returning a bare `{1}`. It now emits the full conditional, matching
    // the structurally identical compound `(a-1)·x = a-1`.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(
        r("solve(a*x=a, x)"),
        "{ 1 } if a != 0; All real numbers if a = 0"
    );
    assert_eq!(
        r("solve(a*x=2*a, x)"),
        "{ 2 } if a != 0; All real numbers if a = 0"
    );
    assert_eq!(
        r("solve(b*x=b, x)"),
        "{ 1 } if b != 0; All real numbers if b = 0"
    );
    assert_eq!(
        r("solve(a*(x-1)=0, x)"),
        "{ 1 } if a != 0; All real numbers if a = 0"
    );
    // Controls: a numeric-coefficient equation, a non-degenerate parametric solve (root still
    // contains the parameter), the compound form, and a non-linear equation are all UNCHANGED.
    assert_eq!(r("solve(2*x=4, x)"), "{ 2 }");
    assert_eq!(r("solve(a*x=b, x)"), "{ b / a }");
    assert_eq!(
        r("solve((a-1)*x=a-1, x)"),
        "{ 1 } if a - 1 != 0; All real numbers if a - 1 = 0"
    );
    assert_eq!(r("solve(x^2=4, x)"), "{ -2, 2 }");
}

#[test]
fn test_eval_parametric_content_factor_branch() {
    // The higher-degree sibling of the linear recovery: a var-free PARAMETRIC
    // content factor of a polynomial product was divided away with both the
    // guard and the `= 0 ⇒ ℝ` branch (`y·(x−1)·(x+2) = 0 → {−2, 1}` hid that
    // y = 0 makes the equation `0 = 0`, i.e. ALL reals). Detected on the RAW
    // tree — the simplifier expands the product into a sum, destroying the
    // very structure being matched.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(
        r("solve(y*(x-1)*(x+2)=0, x)"),
        "{ -2, 1 } if y != 0; All real numbers if y = 0"
    );
    // An EMPTY incumbent is the same class: for y = 0 the true set is ℝ.
    assert_eq!(
        r("solve(y*(x^2+1)=0, x)"),
        "No solution if y != 0; All real numbers if y = 0"
    );
    // Compound and multi-parameter contents ride the same split.
    assert_eq!(
        r("solve((y+1)*(x-1)*(x+2)=0, x)"),
        "{ -2, 1 } if y + 1 != 0; All real numbers if y + 1 = 0"
    );
    assert_eq!(
        r("solve(a*b*(x-1)=0, x)"),
        "{ 1 } if a·b != 0; All real numbers if a·b = 0"
    );
    // Controls: numeric and transcendental-constant contents can never be
    // zero-or-not (no branch), an all-var product keeps its owner, and a
    // RADICAL factor declines (its zero branch needs the expression's domain,
    // not ℝ — named stepping stone).
    assert_eq!(r("solve(2*(x-1)*(x+2)=0, x)"), "{ -2, 1 }");
    assert_eq!(r("solve(pi*(x-1)*(x+2)=0, x)"), "{ -2, 1 }");
    assert_eq!(r("solve(x*(x-1)*(x+2)=0, x)"), "{ -2, 0, 1 }");
    assert_eq!(r("solve(y*sqrt(x+1)*(x-2)=0, x)"), "{ -1, 2 }");
}

#[test]
fn test_eval_flipped_zero_product_orientation() {
    // P0 (2026-08-01): the FLIPPED spelling `0 = A·B` fell to the generic mul
    // isolation, which divided by the moved factor even when it carried the
    // solve variable — dropping that factor's roots (`0 = (x−1)·(x+2)` gave
    // `{ -2 }`). Division by a variable-carrying factor is sound exactly when
    // the other side is nonzero; the `= 0` shape now splits per factor at the
    // isolation chokepoint (zero-product), matching the normal orientation.
    // Sibling `0 != A·B` closed 2026-08-02 (see
    // test_eval_flipped_zero_product_neq below); the TRIPLE product under
    // `!=` remains broken in BOTH orientations (normal spelling errors with
    // «Cycle detected», flipped falls to the lossy division) — pre-existing
    // n-ary Neq-product family, named stepping stone with exact repro.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("solve(0=(x-1)*(x+2), x)"), "{ -2, 1 }");
    assert_eq!(r("solve(0=x*(x-1), x)"), "{ 0, 1 }");
    assert_eq!(r("solve(0=(x-1)*(x+2)*(x-3), x)"), "{ -2, 1, 3 }");
    assert_eq!(r("solve(0=-(x-1)*(x+2), x)"), "{ 1, -2 }");
    // The parametric content-factor hook now wraps a HEALTHY incumbent.
    assert_eq!(
        r("solve(0=y*(x-1)*(x+2), x)"),
        "{ 1, -2 } if y != 0; All real numbers if y = 0"
    );
    // Controls: the normal orientation, constant-factor division, nonzero RHS
    // (division stays sound there), a rootless factor (`e^x` → Empty branch),
    // and the trig-product owner are all UNCHANGED.
    assert_eq!(r("solve((x-1)*(x+2)=0, x)"), "{ -2, 1 }");
    assert_eq!(r("solve(0=2*(x-1), x)"), "{ 1 }");
    assert_eq!(r("solve(4=(x-1)*(x+2), x)"), "{ -3, 2 }");
    assert_eq!(r("solve(0=e^x*(x-1), x)"), "{ 1 }");
}

#[test]
fn test_eval_flipped_zero_product_neq() {
    // The `!=` sibling of the flipped zero-product P0 (2026-08-02): dividing
    // `A·B ≠ 0` by a variable-carrying factor dropped that factor's
    // EXCLUSIONS (`0 ≠ (x−1)·(x+2)` kept only x ≠ −2). Eq and Neq are
    // orientation-symmetric, so the isolation guard re-poses the equation in
    // the normal orientation and delegates to the standard Neq-product owner
    // (ℝ minus ALL roots), with a reentry latch instead of a loop when that
    // owner declines. The n-ary product is owned since the follow-up cycle —
    // see test_eval_neq_product_nary_complement.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(
        r("solve(0 != (x-1)*(x+2), x)"),
        "(-infinity, -2) U (-2, 1) U (1, infinity)"
    );
    assert_eq!(
        r("solve(0 != x*(x-1), x)"),
        "(-infinity, 0) U (0, 1) U (1, infinity)"
    );
    // Controls: normal orientation (its own owner, untouched), constant
    // factor, rootless factor, and the Eq guard are all UNCHANGED.
    assert_eq!(
        r("solve((x-1)*(x+2) != 0, x)"),
        "(-infinity, -2) U (-2, 1) U (1, infinity)"
    );
    assert_eq!(
        r("solve(0 != 2*(x-1), x)"),
        "(-infinity, 1) U (1, infinity)"
    );
    assert_eq!(
        r("solve(0 != e^x*(x-1), x)"),
        "(-infinity, 1) U (1, infinity)"
    );
    assert_eq!(r("solve(0 = (x-1)*(x+2), x)"), "{ -2, 1 }");
}

#[test]
fn test_eval_neq_product_nary_complement() {
    // The n-ary `product != 0` had NO owner: the zero-product strategy only
    // accepted `=`, so the normal orientation fell through to isolation,
    // whose Neq reorientation guard delegated to the IDENTICAL equation —
    // its fingerprint was still active in the solve stack, so every explicit
    // product the quadratic owner didn't cover crashed with «Cycle detected»
    // (`(x−1)(x+2)(x−3) ≠ 0`, and even the non-polynomial pair
    // `e^x·(x−1) ≠ 0`), while the flipped orientation leaked a lossy
    // division (`ℝ∖{−2}`). The zero-product owner now owns BOTH relational
    // shapes: under `!=` it solves the same `factor = 0` branches and
    // answers the complement (n+1 open intervals, exactly value-ordered),
    // declining honestly on non-discrete aggregates (trig factors) and
    // parametric factors, which keep their pre-existing paths.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // The former «Cycle detected» crashes, both orientations.
    assert_eq!(
        r("solve((x-1)*(x+2)*(x-3) != 0, x)"),
        "(-infinity, -2) U (-2, 1) U (1, 3) U (3, infinity)"
    );
    assert_eq!(
        r("solve(0 != (x-1)*(x+2)*(x-3), x)"),
        "(-infinity, -2) U (-2, 1) U (1, 3) U (3, infinity)"
    );
    assert_eq!(
        r("solve((x-1)*(x+2)*(x-3)*(x+4) != 0, x)"),
        "(-infinity, -4) U (-4, -2) U (-2, 1) U (1, 3) U (3, infinity)"
    );
    assert_eq!(
        r("solve(e^x*(x-1) != 0, x)"),
        "(-infinity, 1) U (1, infinity)"
    );
    // Exact surd ordering rides the same complement builder: this input
    // previously errored («symbolic coefficients not supported»).
    assert_eq!(
        r("solve((x-sqrt(2))*(x+1) != 0, x)"),
        "(-infinity, -1) U (-1, sqrt(2)) U (sqrt(2), infinity)"
    );
    // Honest declines keep their current owners/paths byte-identical:
    // trig factor (non-discrete aggregate; the missing kπ exclusions are a
    // PRE-EXISTING division-fallback loss, named in the ledger) and the
    // parametric factor (missing y=0 branch, named family).
    assert_eq!(
        r("solve(sin(x)*(x-1) != 0, x)"),
        "(-infinity, 1) U (1, infinity)"
    );
    assert_eq!(
        r("solve(y*(x-1) != 0, x)"),
        "(-infinity, 1) U (1, infinity)"
    );
    // The parametric gate checks the WHOLE product, not the top-level split
    // factors: the nested shape `(y·(x−1))·(x+2)` must keep its pre-existing
    // soft error, NOT publish `ℝ∖{−2,1}` (over-claims on the y=0 branch,
    // where the product is identically zero). Caught by adversarial sweep.
    assert!(r("solve((y*(x-1))*(x+2) != 0, x)").is_empty());
    // The `=` owner is untouched.
    assert_eq!(r("solve((x-1)*(x+2)*(x-3) = 0, x)"), "{ -2, 1, 3 }");
}

#[test]
fn test_eval_neq_polynomial_complement() {
    // Expanded polynomials of degree ≥ 3 under `!=` had no owner: they fell
    // through to isolation, whose even-root/abs-split terminal solved the
    // ASSOCIATED `= 0` and published its ROOTS as the `!=` answer —
    // `x⁴−5x²+4 ≠ 0 → {±1, ±2}` (the four points that are NOT solutions),
    // and «No solution» for the rootless `x⁴+x²+1 ≠ 0` (the exact
    // negation). The polynomial `!=` owner (quadratic_strategy) now solves
    // the associated equation through the full recursive solver — the
    // cycle-guard fingerprint includes the relational op, so the
    // delegation with identical (lhs, rhs) is not a false cycle — and
    // answers the complement; a provably sign-definite polynomial settles
    // as AllReals without solving. Non-Discrete associated results decline
    // honestly (casus irreducibilis / Cardano stay with their current
    // paths: the isolation-terminal op-loss there is a named next rung).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Former wrong answers (roots published as the != solution).
    assert_eq!(
        r("solve(x^4 - 5*x^2 + 4 != 0, x)"),
        "(-infinity, -2) U (-2, -1) U (-1, 1) U (1, 2) U (2, infinity)"
    );
    assert_eq!(
        r("solve(x^6 - 9*x^3 + 8 != 0, x)"),
        "(-infinity, 1) U (1, 2) U (2, infinity)"
    );
    assert_eq!(
        r("solve((x^2-1)^2 - 4 != 0, x)"),
        "(-infinity, -sqrt(3)) U (-sqrt(3), sqrt(3)) U (sqrt(3), infinity)"
    );
    // Former exact negation: rootless quartic said «No solution».
    assert_eq!(r("solve(x^4 + x^2 + 1 != 0, x)"), "All real numbers");
    assert_eq!(r("solve(-x^4 - 1 != 0, x)"), "All real numbers");
    // Former cbrt-echo residuals (the original stepping stone), both
    // orientations, plus the expanded shapes of last cycle's leftovers.
    assert_eq!(
        r("solve(x^3 - 2*x^2 - 5*x + 6 != 0, x)"),
        "(-infinity, -2) U (-2, 1) U (1, 3) U (3, infinity)"
    );
    assert_eq!(
        r("solve(0 != x^3 - 2*x^2 - 5*x + 6, x)"),
        "(-infinity, -2) U (-2, 1) U (1, 3) U (3, infinity)"
    );
    assert_eq!(
        r("solve((x-1)^2*(x+2) != 0, x)"),
        "(-infinity, -2) U (-2, 1) U (1, infinity)"
    );
    assert_eq!(
        r("solve(x^5 - x != 0, x)"),
        "(-infinity, -1) U (-1, 0) U (0, 1) U (1, infinity)"
    );
    // Controls: degrees ≤ 2 keep their sound owners byte-identical, the
    // parametric polynomial declines to its current path, and `=` owners
    // are untouched.
    assert_eq!(
        r("solve(3*x+2 != 0, x)"),
        "(-infinity, -2/3) U (-2/3, infinity)"
    );
    assert_eq!(
        r("solve(x^2+x-2 != 0, x)"),
        "(-infinity, -2) U (-2, 1) U (1, infinity)"
    );
    assert_eq!(r("solve(x^4-5*x^2+4 = 0, x)"), "{ -2, -1, 1, 2 }");
}

#[test]
fn test_eval_neq_root_recovery_complement() {
    // The post-pipeline root recoveries (irreducible-cubic Cardano /
    // trigonometric casus irreducibilis / quartic-factor deflation) solve
    // the ASSOCIATED `= 0` and used to REPLACE a residual incumbent with its
    // root set REGARDLESS of the op — publishing, under `!=`, exactly the
    // points that are NOT solutions (`x³+x+1 ≠ 0 → {cardano root}`,
    // `x³−3x+1 ≠ 0 → {three trig roots}`, `x⁵−5x³+x²−5 ≠ 0 → {−1, ±√5}`).
    // The recoveries now adapt to the op: under `!=` a discrete recovery
    // flips to its exactly-ordered complement (const_value_bounds orders the
    // trig-root triple), and `=`/order-inequality behavior is untouched.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Cardano single-real-root cubic: complement of one exact radical root.
    assert_eq!(
        r("solve(x^3 + x + 1 != 0, x)"),
        "(-infinity, cbrt(1/6\u{b7}(-sqrt(31/3) - 3)) + cbrt(1/6\u{b7}(sqrt(31/3) - 3))) U (cbrt(1/6\u{b7}(-sqrt(31/3) - 3)) + cbrt(1/6\u{b7}(sqrt(31/3) - 3)), infinity)"
    );
    // Casus irreducibilis: three trig roots, exactly value-ordered
    // (≈ −1.879 < 0.347 < 1.532).
    assert_eq!(
        r("solve(x^3 - 3*x + 1 != 0, x)"),
        "(-infinity, -sin(2/9\u{b7}pi) / sin(pi / 9)) U (-sin(2/9\u{b7}pi) / sin(pi / 9), sin(pi / 9) / sin(4/9\u{b7}pi)) U (sin(pi / 9) / sin(4/9\u{b7}pi), sin(4/9\u{b7}pi) / sin(2/9\u{b7}pi)) U (sin(4/9\u{b7}pi) / sin(2/9\u{b7}pi), infinity)"
    );
    // Quartic-factor deflation (quintic = (x+1)(x²−5)(x²−x+1)).
    assert_eq!(
        r("solve(x^5 - 5*x^3 + x^2 - 5 != 0, x)"),
        "(-infinity, -sqrt(5)) U (-sqrt(5), -1) U (-1, sqrt(5)) U (sqrt(5), infinity)"
    );
    // The `=` recoveries and the order-inequality path stay untouched.
    assert_eq!(
        r("solve(x^5 - 5*x^3 + x^2 - 5 = 0, x)"),
        "{ -1, sqrt(5), -sqrt(5) }"
    );
    assert_eq!(
        r("solve(x^3+x^2+3 > 0, x)"),
        "(cbrt(1/6\u{b7}(-sqrt(85) - 83/9)) + cbrt(1/6\u{b7}(sqrt(85) - 83/9)) - 1/3, infinity)"
    );
}
#[test]
fn test_eval_solve_all_reals_inlines_domain_condition() {
    // An identity equation whose solution is all reals RESTRICTED by a domain condition must show
    // that condition in the default text surface (`All real numbers if x > 0`), matching the in-set
    // conditional convention (`1/x=1/x → "… if x != 0"`), not a dishonest bare `All real numbers`.
    for (input, expected) in [
        ("solve(ln(x^2)=2*ln(x), x)", "All real numbers if x > 0"),
        ("solve(2*ln(x)=ln(x^2), x)", "All real numbers if x > 0"),
        ("solve(e^(ln(x))=x, x)", "All real numbers if x > 0"),
        ("solve(sqrt(x)^2=x, x)", "All real numbers if x ≥ 0"),
        (
            "solve(ln(x^2)=2*ln(abs(x)), x)",
            "All real numbers if x ≠ 0",
        ),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
        // The LaTeX surface uses the matching `\begin{cases}` conditional form.
        assert!(
            wire["result_latex"]
                .as_str()
                .unwrap_or("")
                .contains("\\begin{cases}"),
            "{input} latex"
        );
    }
    // Controls: an in-set conditional is NOT double-rendered; an unconditional identity stays bare;
    // a `simplify` result whose `required_display` is intentionally JSON-only keeps its bare text.
    for (input, expected) in [
        ("solve(1/x=1/x, x)", "All real numbers if x != 0"),
        ("solve(0*x=0, x)", "All real numbers"),
        ("sqrt(x)*sqrt(x)", "x"),
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
fn test_eval_inequality_intersects_factor_function_domain() {
    // A domain-restricted function (`ln`, `√`) appearing as a FACTOR (not the bare LHS) must still
    // exclude its undefined region: `ln(x)·(x−2)² ≤ 0` is `(0,1]∪{2}`, NOT `(−∞,1]∪{2}` (`ln` is
    // undefined for x ≤ 0). The inequality result is now intersected with the LHS's implicit domain.
    for (input, expected) in [
        ("solve(ln(x)*(x-2)^2<=0, x)", "(0, 1] U [2, 2]"),
        ("solve(ln(x)*(x-2)>=0, x)", "(0, 1] U [2, infinity)"),
        ("solve(ln(x)*(x-3)<=0, x)", "[1, 3]"),
        // Bare-function controls (already correct) stay correct.
        ("solve(ln(x)<=0, x)", "(0, 1]"),
        ("solve(ln(x)>=0, x)", "[1, infinity)"),
        ("solve(sqrt(x)>=2, x)", "[4, infinity)"),
        // No domain restriction -> unchanged.
        ("solve(x^2-1>0, x)", "(-infinity, -1) U (1, infinity)"),
        ("solve((x-1)*(x-3)<=0, x)", "[1, 3]"),
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
fn test_eval_equation_with_undefined_side_has_no_solution() {
    // A relation with an `undefined` side has NO real solution — nothing equals/compares to
    // `undefined`. In RealOnly, `ln(-2)` / `ln(-1)` / `1/0` simplify to `undefined`, so these are
    // unsatisfiable. The isolation path used to emit a degenerate `All real numbers if undefined = 0`
    // conditional (its guard is never true).
    for input in [
        "solve(ln(x)=ln(-2), x)",
        "solve(x=ln(-1), x)",
        "solve(x=ln(-2), x)",
        "solve(ln(x)=undefined, x)",
        "solve(x+1=undefined, x)",
        "solve(x=1/0, x)",
        // Matrix-equation members: the scalar-broadcast of `A*X` (2x2) minus the
        // 2x1 column RHS folds to `undefined` AFTER the subtraction, so the
        // var-eliminated residual — not a bare side — is non-finite. Under the
        // engine's scalar-X semantics a 2x2 can never equal a 2x1 column, so the
        // sound answer is "No solution", not "All real numbers if undefined = 0".
        "solve([[1,2],[3,4]]*X=[[5],[6]], X)",
        "solve([[1,0],[0,1]]*X=[[2],[3]], X)",
        "solve(X*[[1,2],[3,4]]=[[5],[6]], X)",
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some("No solution"), "{input}");
    }
    // Controls: a defined (possibly non-real-rejected) RHS is unaffected.
    for (input, expected) in [
        ("solve(ln(x)=ln(2), x)", "{ 2 }"),
        ("solve(x=sqrt(-4), x)", "No solution"),
        ("solve(ln(x)=2, x)", "{ e^2 }"),
        ("solve(x^2=4, x)", "{ -2, 2 }"),
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
fn solve_system_parametric_s1_contract() {
    // Frente S · S1: la lista de incógnitas manda en la clasificación — todo
    // símbolo fuera de ella es PARÁMETRO; Cramer simbólico exacto 2×2 con
    // `det ≠ 0` por el canal canónico de condiciones; y un sistema bien
    // formado JAMÁS sale del wire como error interno (decline honesto ok).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input])
            .output()
            .expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    };
    let json_of = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stdout).to_string()
    };

    // Paramétrico insignia: coeficiente simbólico, condición det ≠ 0 visible
    // en el resultado Y normalizada en required_display.
    let flagship = r("solve([a*x+y=1, x-y=0], [x, y])");
    assert!(
        flagship.starts_with("{ x = 1 / (a + 1), y = 1 / (a + 1) }"),
        "{flagship}"
    );
    assert!(flagship.contains("requires: a + 1 != 0"), "{flagship}");
    let flagship_json = json_of("solve([a*x+y=1, x-y=0], [x, y])");
    assert!(flagship_json.contains("\"ok\": true"), "{flagship_json}");
    assert!(flagship_json.contains("a ≠ -1"), "{flagship_json}");

    // Parámetros en el RHS con det constante: cociente polinómico EXACTO,
    // sin condición (det = -2 racional).
    assert_eq!(r("solve([x+y=u+v, x-y=u-v], [x, y])"), "{ x = u, y = v }");

    // Los declines honestos NUNCA son errores de wire (antes: E_INTERNAL).
    for decline in [
        "solve([x^2+y=1, x-y=0], [x, y])",
        "solve([a*x+y=1, a*x+y=2], [x, y])",
        "solve([sqrt(2)*x+y=1, x-y=0], [x, y])",
    ] {
        let j = json_of(decline);
        assert!(
            j.contains("\"ok\": true"),
            "decline must be honest ok: {decline}: {j}"
        );
        assert!(
            !j.contains("E_INTERNAL"),
            "decline must not be E_INTERNAL: {decline}"
        );
    }

    // No-robo: la ruta racional byte-idéntica, en ambas sintaxis.
    assert_eq!(r("solve([x+y=3, x-y=1], [x, y])"), "{ x = 2, y = 1 }");
    assert_eq!(r("solve_system(x+y=3; x-y=1; x; y)"), "{ x = 2, y = 1 }");
    // La forma semicolon también resuelve el paramétrico (mismo pipeline).
    let semi = r("solve_system(a*x+y=1; x-y=0; x; y)");
    assert!(semi.contains("x = 1 / (a + 1)"), "{semi}");
}
#[test]
fn solve_system_nonlinear_s2_contract() {
    // Frente S · S2: no-lineales 2×2 por composición aislar → sustituir →
    // solve univariable → back-substitute, con gate de verificación exacta
    // POR PAR contra AMBAS ecuaciones originales (D5 transferido).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input])
            .output()
            .expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    };

    // Parábola-recta racional (curso elemental).
    assert_eq!(
        r("solve([y=x^2, y=x+2], [x, y])"),
        "{ x = -1, y = 1 } or { x = 2, y = 4 }"
    );
    // Hipérbola x·y=6 + recta (sustitución clásica).
    assert_eq!(
        r("solve([x*y=6, x+y=5], [x, y])"),
        "{ x = 2, y = 3 } or { x = 3, y = 2 }"
    );
    // Círculo + recta textbook.
    assert_eq!(
        r("solve([x^2+y^2=25, x+y=7], [x, y])"),
        "{ x = 3, y = 4 } or { x = 4, y = 3 }"
    );
    // Pares SURD verificados exactos (el gate maneja aritmética surd).
    let surd = r("solve([x^2+y=3, x-y=1], [x, y])");
    assert!(surd.contains("sqrt(17)") && surd.contains(" or "), "{surd}");
    // Sin intersección real: vacío PROBADO por el solve univariable.
    assert!(r("solve([x^2+y^2=1, x+y=5], [x, y])").contains("no solution"));
    // Guard paramétrico: no-lineal con parámetro sigue residual honesto.
    let param = r("solve([x^2+y^2=a, x+y=1], [x, y])");
    assert!(param.contains("non-linear"), "{param}");
    // Dos cuadráticas sin ecuación aislable: desde S5 resuelve por la
    // resultante de Sylvester (el peldaño llegó — 4 pares verificados).
    assert_eq!(
        r("solve([x*y=6, x^2+y^2=13], [x, y])"),
        "{ x = -3, y = -2 } or { x = -2, y = -3 } or { x = 2, y = 3 } or { x = 3, y = 2 }"
    );
    // Composición bonus: coeficiente surd en sistema LINEAL resuelto vía la
    // misma ruta (el techo multipoly-sobre-Q de S1 graduó por composición).
    assert_eq!(
        r("solve([sqrt(2)*x+y=1, x-y=0], [x, y])"),
        "{ x = sqrt(2) - 1, y = sqrt(2) - 1 }"
    );
    // No-robo: lineal racional y paramétrico S1 byte-idénticos.
    assert_eq!(r("solve([x+y=3, x-y=1], [x, y])"), "{ x = 2, y = 1 }");
    let s1 = r("solve([a*x+y=1, x-y=0], [x, y])");
    assert!(s1.starts_with("{ x = 1 / (a + 1)"), "{s1}");
}
#[test]
fn solve_system_educational_s3_contract() {
    // Frente S · S3: la mitad educativa — narración es/en por familia, cero
    // fugas de micro-pasos internos, y eco fiel de la forma que el usuario
    // tecleó (lista ↔ lista, semicolon ↔ solve_system).
    let json_of = |input: &str, lang: Option<&str>| -> String {
        let mut args = vec!["eval", input, "--steps", "on", "--format", "json"];
        if let Some(l) = lang {
            args.extend(["--lang", l]);
        }
        let out = cli().args(&args).output().expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stdout).to_string()
    };

    // Narración no-lineal con contenido real (aislamiento + univariable +
    // gate) en español…
    let es = json_of("solve([x*y=6, x+y=5], [x, y])", None);
    assert!(es.contains("Aislar y de la ecuación 2"), "{es}");
    assert!(es.contains("resolver la univariable en x"), "{es}");
    assert!(es.contains("2 pares verificados"), "{es}");
    // …y en inglés vía la tabla es/en.
    let en = json_of("solve([x*y=6, x+y=5], [x, y])", Some("en"));
    assert!(en.contains("Isolate y from equation 2"), "{en}");
    assert!(en.contains("verified pairs emitted"), "{en}");

    // Paramétrico narra la condición del determinante.
    let par = json_of("solve([a*x+y=1, x-y=0], [x, y])", None);
    assert!(
        par.contains("determinante debe ser distinto de cero"),
        "{par}"
    );
    // Lineal racional narra método; degenerado narra el porqué.
    // El pin decía «Cramer/Gauss» — un hedge sobre un snapshot de relleno: los
    // tres pasos arrastraban `exprs.first()`, así que el método se anunciaba
    // sobre una ecuación que no estaba resolviendo y la ecuación 2 no salía en
    // la traza. Desde 2026-07-28 se nombra el método que REALMENTE corre
    // (`solve_nxn_gauss`: escalonar la matriz ampliada sobre ℚ y
    // back-substituir) y cada paso lleva su propia ecuación.
    let lin = json_of("solve([x+y=3, x-y=1], [x, y])", None);
    assert!(lin.contains("Eliminación gaussiana exacta"), "{lin}");
    assert!(lin.contains("Ecuación 2 de 2 del sistema"), "{lin}");
    assert!(lin.contains("Verificación exacta"), "{lin}");
    let inc = json_of("solve([x+y=1, x+y=2], [x, y])", Some("en"));
    assert!(
        inc.contains("inconsistent (no assignment satisfies all)"),
        "{inc}"
    );

    // Cero fugas: los micro-pasos internos de S2 no llegan al canal steps.
    let leak = json_of("solve([x*y=6, x+y=5], [x, y])", None);
    assert!(leak.contains("\"steps_count\": 0"), "{leak}");

    // Eco fiel: la forma lista se ecoa como lista (JSON escapa los
    // backslash del LaTeX: \\ en bytes).
    assert!(
        leak.contains("\\\\operatorname{solve}\\\\left(\\\\left["),
        "{leak}"
    );
    // …y la forma semicolon conserva su eco solve_system.
    let semi = json_of("solve_system(x+y=3; x-y=1; x; y)", None);
    assert!(
        semi.contains("\\\\operatorname{solve\\\\_system}"),
        "{semi}"
    );
}
#[test]
fn solve_system_surface_s4_contract() {
    // Frente S · S4: superficie — una familia de sintaxis (lista) en ambos
    // comandos, help publicando los declines honestos como contrato, y
    // completado. Cierra la brecha señalada por el usuario: sistemas en
    // solve al mismo nivel de integración que dsolve.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input])
            .output()
            .expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    };

    // Paridad de sintaxis lista en solve_system (wire).
    assert_eq!(
        r("solve_system([x+y=3, x-y=1], [x, y])"),
        "{ x = 2, y = 1 }"
    );
    assert_eq!(
        r("solve_system([x*y=6, x+y=5], [x, y])"),
        "{ x = 2, y = 3 } or { x = 3, y = 2 }"
    );
    // El paramétrico fluye con su condición por la forma lista.
    let par = r("solve_system([a*x+y=1, x-y=0], [x, y])");
    assert!(par.contains("requires: a + 1 != 0"), "{par}");
    // La semicolon clásica sigue byte-idéntica (no-robo).
    assert_eq!(r("solve_system(x+y=3; x-y=1; x; y)"), "{ x = 2, y = 1 }");

    // Help: solve documenta la forma lista; solve_system publica familias Y
    // declines honestos como contrato (help es superficie REPL — molde O7).
    let help_of = |topic: &str| -> String {
        let out = cli()
            .arg("repl")
            .write_stdin(format!("help {topic}\nexit\n"))
            .output()
            .expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stdout).to_string()
    };
    let solve_help = help_of("solve");
    assert!(solve_help.contains("solve([eq1, eq2, ...]"), "{solve_help}");
    let sys_help = help_of("solve_system");
    assert!(sys_help.contains("Parametric 2x2"), "{sys_help}");
    assert!(sys_help.contains("Honest declines"), "{sys_help}");
    assert!(sys_help.contains("isolate-substitute-"), "{sys_help}");
}
#[test]
fn solve_system_resultant_s5_contract() {
    // Frente S · S5: la resultante de Sylvester — sistemas bivariados SIN
    // ecuación aislable (dos cónicas) por eliminación exacta + solve
    // univariable + gate de verificación por par (D5). La emisión es sana
    // incondicionalmente (gate); la afirmación de vacío exige completitud
    // de candidatos (raíces de la resultante + coeficientes líderes).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input])
            .output()
            .expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    };

    // Hipérbola x·y=6 ∩ círculo — el ejercicio estándar de bachillerato.
    assert_eq!(
        r("solve([x*y=6, x^2+y^2=13], [x, y])"),
        "{ x = -3, y = -2 } or { x = -2, y = -3 } or { x = 2, y = 3 } or { x = 3, y = 2 }"
    );
    // Elipse ∩ hipérbola: NINGUNA incógnita lineal en NINGUNA ecuación.
    assert_eq!(
        r("solve([x^2+4*y^2=25, x^2-y^2=5], [x, y])"),
        "{ x = -3, y = -2 } or { x = -3, y = 2 } or { x = 3, y = -2 } or { x = 3, y = 2 }"
    );
    // Círculo ∩ círculo.
    assert_eq!(
        r("solve([x^2+y^2=25, (x-1)^2+y^2=18], [x, y])"),
        "{ x = 4, y = -3 } or { x = 4, y = 3 }"
    );
    // Círculos concéntricos: resultante constante ≠ 0 → vacío PROBADO.
    assert!(r("solve([x^2+y^2=1, x^2+y^2=4], [x, y])").contains("no solution"));

    // Narración del camino resultante, es/en.
    let steps_of = |input: &str, lang: Option<&str>| -> String {
        let mut args = vec!["eval", input, "--steps", "on", "--format", "json"];
        if let Some(l) = lang {
            args.extend(["--lang", l]);
        }
        let out = cli().args(&args).output().expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stdout).to_string()
    };
    let es = steps_of("solve([x*y=6, x^2+y^2=13], [x, y])", None);
    assert!(es.contains("resultante de Sylvester"), "{es}");
    assert!(es.contains("4 pares verificados"), "{es}");
    let en = steps_of("solve([x*y=6, x^2+y^2=13], [x, y])", Some("en"));
    assert!(en.contains("Sylvester resultant"), "{en}");

    // Pins de no-robo y declines que SIGUEN:
    // — la ruta de sustitución S2 intacta (más barata, va primero);
    assert_eq!(
        r("solve([x*y=6, x+y=5], [x, y])"),
        "{ x = 2, y = 3 } or { x = 3, y = 2 }"
    );
    // — paramétrico no-lineal sigue residual honesto (guard);
    assert!(r("solve([x^2+y^2=a, x+y=1], [x, y])").contains("non-linear"));
    // — grados fuera del techo univariable siguen residual honesto.
    let hard = r("solve([x^2+y^3=5, x^3-y^2=1], [x, y])");
    assert!(hard.starts_with("Error"), "{hard}");
}
#[test]
fn solve_system_declared_constant_shadow_contract() {
    // Frente S (chip e-como-incógnita): «la lista de incógnitas manda»
    // aplicado a los NOMBRES — e/pi/phi DECLARADAS en la lista son variables
    // dentro del canal de sistemas (espejo D14: la excepción vive donde vive
    // el contexto); fuera de la lista conservan su significado global.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input])
            .output()
            .expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    };

    // La bandera del chip: el 5×5 con e declarada resuelve.
    assert_eq!(
        r("solve([a+b+c+d+e=15, b+c+d+e=14, c+d+e=12, d+e=9, e=5], [a, b, c, d, e])"),
        "{ a = 1, b = 2, c = 3, d = 4, e = 5 }"
    );
    assert_eq!(r("solve([e+x=3, x-1=0], [x, e])"), "{ x = 1, e = 2 }");
    assert_eq!(r("solve([pi+x=4, x-1=0], [x, pi])"), "{ x = 1, pi = 3 }");
    // El shadowing compone con el camino no-lineal (phi por sustitución).
    let phi = r("solve([phi^2+y=3, phi-y=1], [phi, y])");
    assert!(phi.contains("phi = 1/2·(sqrt(17) - 1)"), "{phi}");

    // Pins: FUERA de la lista las constantes siguen siendo ellas mismas —
    // Euler y π como coeficientes exactos, byte-idénticos.
    assert_eq!(
        r("solve([x+e*y=1, x-y=0], [x, y])"),
        "{ x = 1 / (1 + e), y = 1 / (1 + e) }"
    );
    assert_eq!(
        r("solve([x+pi*y=1, x-y=0], [x, y])"),
        "{ x = 1 / (1 + pi), y = 1 / (1 + pi) }"
    );
    // `i` NO se sombrea (unidad imaginaria estructural): declina honesto.
    let i_decl = r("solve([i+x=3, x-1=0], [x, i])");
    assert!(
        !i_decl.contains("i = 2"),
        "i declarada no debe sombrearse: {i_decl}"
    );
}
#[test]
fn solve_system_parametric_3x3_s6_contract() {
    // Frente S · S6 (peldaño graduado): Cramer simbólico 3×3 — la partición
    // generalizada a n incógnitas + determinantes polinómicos (el
    // poly_determinant COMPARTIDO con la resultante S5); det ≠ 0 como
    // condición estructurada; n ≥ 4 simbólico sigue decline honesto
    // (presupuesto deliberado del cofactor).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input])
            .output()
            .expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    };

    let flagship = r("solve([a*x+y+z=1, x-y=0, y-z=0], [x, y, z])");
    assert!(
        flagship.starts_with("{ x = 1 / (a + 2), y = 1 / (a + 2), z = 1 / (a + 2) }"),
        "{flagship}"
    );
    assert!(flagship.contains("requires: a + 2 != 0"), "{flagship}");

    // Dos parámetros densos: cocientes polinómicos completos.
    let dense = r("solve([a*x+b*y+z=1, x+y+z=0, x-y=2], [x, y, z])");
    assert!(dense.contains("x = (2·b - 1) / (a + b - 2)"), "{dense}");
    assert!(dense.contains("requires: a + b - 2 != 0"), "{dense}");

    // RHS simbólico con det constante: exacto sin condición.
    assert_eq!(
        r("solve([x+y+z=p, x-y=0, y-z=0], [x, y, z])"),
        "{ x = 1/3·p, y = 1/3·p, z = 1/3·p }"
    );
    // Degenerado simbólico 3×3 con par de filas coef-idénticas y aumentados
    // 1 vs 2: la clasificación ESTRUCTURAL (cross-minors exactos) lo decide
    // inconsistente para TODO valor de a — ya no declina.
    let degen = r("solve([a*x+y+z=1, a*x+y+z=2, x-y=0], [x, y, z])");
    assert!(degen.contains("System has no solution"), "{degen}");
    // El rank INTERMEDIO paramétrico (y la base que puede anularse) siguen
    // siendo el decline honesto: para a=0 este sistema es 0=1 (inconsistente)
    // y para a≠0 es dependiente — la clasificación depende del parámetro.
    let param_edge = r("solve([a*x+a*y=1, 2*a*x+2*a*y=2], [x, y])");
    assert!(
        param_edge.contains("rank classification is a future rung"),
        "{param_edge}"
    );
    // Proporcionalidad estructural completa (aumentado incluido) con base que
    // nunca se anula: dependiente para todo a.
    let dep = r("solve([a*x+y=1, 2*a*x+2*y=2], [x, y])");
    assert!(
        dep.contains("System has infinitely many solutions"),
        "{dep}"
    );
    // Coef-proporcionales con desajuste aumentado constante: inconsistente
    // para todo a.
    let inc = r("solve([x+a*y=1, x+a*y=2], [x, y])");
    assert!(inc.contains("System has no solution"), "{inc}");
    // El par NO-LINEAL proporcional (misma curva) nombra su residual propio —
    // sin la coletilla lineal (contradiría al camino no-lineal) y sin afirmar
    // cardinalidad (la curva compartida puede ser vacía o un punto en ℝ).
    let curve = r("solve([x*y=6, 2*x*y=12], [x, y])");
    assert!(
        curve.contains("proportional non-linear equations")
            && curve.contains("same curve")
            && !curve.contains("only handles linear equations"),
        "{curve}"
    );

    // Pins de no-robo: racional 3×3 y paramétrico 2×2 byte-idénticos.
    assert_eq!(
        r("solve([x+y+z=6, x-y=0, z=3], [x, y, z])"),
        "{ x = 3/2, y = 3/2, z = 3 }"
    );
    let s1 = r("solve([a*x+y=1, x-y=0], [x, y])");
    assert!(s1.starts_with("{ x = 1 / (a + 1)"), "{s1}");
    // El 4×4 paramétrico graduó en S7 (misma tanda — ver su contrato).
    let four = r("solve([a*x+y+z+w=1, x-y=0, y-z=0, z-w=0], [x, y, z, w])");
    assert!(four.starts_with("{ x = 1 / (a + 3)"), "{four}");
}
#[test]
fn solve_system_parametric_nxn_s7_contract() {
    // Frente S · S7: el Cramer simbólico generalizado cableado para n ≥ 4 —
    // el presupuesto del cofactor es el guard deliberado contra el blowup;
    // el determinante degenerado sigue declinando honesto.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input])
            .output()
            .expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    };
    let flagship = r("solve([a*x+y+z+w=1, x-y=0, y-z=0, z-w=0], [x, y, z, w])");
    assert!(
        flagship
            .starts_with("{ x = 1 / (a + 3), y = 1 / (a + 3), z = 1 / (a + 3), w = 1 / (a + 3) }"),
        "{flagship}"
    );
    assert!(flagship.contains("requires: a + 3 != 0"), "{flagship}");
    assert_eq!(
        r("solve([x+y+z+w=p, x-y=0, y-z=0, z-w=0], [x, y, z, w])"),
        "{ x = 1/4·p, y = 1/4·p, z = 1/4·p, w = 1/4·p }"
    );
    // Racional 4×4 byte-idéntico (no-robo); el par coef-idéntico con
    // aumentados 1 vs 2 ahora clasifica estructuralmente: inconsistente para
    // todo a (el rank intermedio paramétrico sigue declinando — ver el pin
    // 3×3 de la suite S6).
    assert_eq!(
        r("solve([x+y+z+w=10, y+z+w=9, z+w=7, w=4], [x, y, z, w])"),
        "{ x = 1, y = 2, z = 3, w = 4 }"
    );
    let degen = r("solve([a*x+y+z+w=1, a*x+y+z+w=2, x-y=0, y-z=0], [x, y, z, w])");
    assert!(degen.contains("System has no solution"), "{degen}");
}
