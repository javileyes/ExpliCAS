use super::*;

#[test]
fn test_eval_matrix_commutator_does_not_collapse_to_zero() {
    // Matrix multiplication is non-commutative, so the commutator A·B − B·A is
    // generally nonzero. The engine's exact-zero / equivalent-pair root shortcuts
    // and the additive cancellation matchers compare products as commutative
    // factor multisets, which previously collapsed A·B − B·A to 0 (a wrong
    // answer). The bug only surfaced in the steps-off fast path (the steps-on
    // path evaluates the products first), so BOTH modes are checked here.
    let cases = [
        (
            "[[1,2],[3,4]]*[[5,6],[7,8]] - [[5,6],[7,8]]*[[1,2],[3,4]]",
            "[[-4, -12], [12, 4]]",
        ),
        (
            // Nilpotent generators: [E12, E21] = E11 − E22.
            "[[0,1],[0,0]]*[[0,0],[1,0]] - [[0,0],[1,0]]*[[0,1],[0,0]]",
            "[[1, 0], [0, -1]]",
        ),
        (
            "[[5,6],[7,8]]*[[1,2],[3,4]] - [[1,2],[3,4]]*[[5,6],[7,8]]",
            "[[4, 12], [-12, -4]]",
        ),
    ];
    for (input, expected) in cases {
        for mode in ["off", "on"] {
            let output = cli()
                .args(["eval", input, "--format", "json", "--steps", mode])
                .output()
                .expect("Failed to run CLI");
            assert!(output.status.success(), "{input} (steps={mode})");
            let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
            assert_eq!(
                wire["result"].as_str(),
                Some(expected),
                "{input} (steps={mode}): matrix commutator must not collapse to 0"
            );
        }
    }

    // A genuinely identical product difference A·B − A·B is still the zero matrix
    // / 0 (order-preserving structural equality is sound), and the scalar
    // commutator x·y − y·x stays 0 (scalar multiplication IS commutative).
    for (input, expected) in [
        (
            "[[1,2],[3,4]]*[[5,6],[7,8]] - [[1,2],[3,4]]*[[5,6],[7,8]]",
            "0",
        ),
        ("x*y - y*x", "0"),
    ] {
        for mode in ["off", "on"] {
            let output = cli()
                .args(["eval", input, "--format", "json", "--steps", mode])
                .output()
                .expect("Failed to run CLI");
            let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
            assert_eq!(
                wire["result"].as_str(),
                Some(expected),
                "{input} (steps={mode}): genuine cancellation must still hold"
            );
        }
    }
}
#[test]
fn test_eval_matrix_power_zero_is_identity_not_scalar_one() {
    // `M^0` is the n×n IDENTITY matrix (the multiplicative identity of the matrix ring), NOT the
    // scalar `1`; a non-square matrix has no `M^0`. The scalar `x^0 -> 1` rule used to collapse a
    // matrix base to `1`, fabricating nonsense (`M^0 + 5 -> 6`, `trace(M^0) -> trace(1)`).
    for (input, expected) in [
        ("[[1,2],[3,4]]^0", "[[1, 0], [0, 1]]"),
        ("[[1,2],[3,4]]^0 + [[1,2],[3,4]]", "[[2, 2], [3, 5]]"),
        ("3*[[1,2],[3,4]]^0", "[[3, 0], [0, 3]]"),
        ("trace([[1,2],[3,4]]^0)", "2"),
        ("[[a,b],[c,d]]^0", "[[1, 0], [0, 1]]"),
        ("[[0,0],[0,0]]^0", "[[1, 0], [0, 1]]"), // ring identity even for the zero matrix
        ("[[1,2,3],[4,5,6]]^0", "undefined"),    // non-square has no M^0
        // Scalar `x^0` is unaffected.
        ("5^0", "1"),
        ("(x+1)^0", "1"),
        ("0^0", "undefined"),
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
fn test_eval_matrix_shape_mismatch_is_undefined() {
    // A shape-incompatible matrix operation has no value, so it must return the `undefined` sentinel
    // (like `1/0` or a singular inverse), never echo the malformed operation back as a valid result.
    // Previously these returned the operation unchanged with `ok:true`.
    for input in [
        "[[1,2],[3,4]] + [[1,2,3],[4,5,6]]", // add, different dims
        "[[1,2],[3,4]] - [[1,2,3]]",         // sub, different dims
        "[[1,2],[3,4]] * [[1,2,3]]",         // mul, inner dims 2 != 1
        "[[1,2,3],[4,5,6]]^2",               // non-square power
        "[[1,2,3]] + 5",                     // matrix + scalar (no broadcast)
        "([[1,2],[3,4]] + [[1,2,3],[4,5,6]]) * [[1,2],[3,4]]", // distributed mismatch propagates
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some("undefined"), "{input}");
    }
    // Compatible matrix operations are unaffected (must still compute exactly).
    for (input, expected) in [
        ("[[1,2],[3,4]] + [[5,6],[7,8]]", "[[6, 8], [10, 12]]"),
        (
            "[[1,2],[3,4]] * [[1,2,3],[4,5,6]]",
            "[[9, 12, 15], [19, 26, 33]]",
        ),
        ("3 * [[1,2],[3,4]]", "[[3, 6], [9, 12]]"),
        ("det([[1,2],[3,4]])", "-2"),
        ("[[1,2,3]] + [[4,5,6]]", "[5, 7, 9]"),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
    }
    // A matrix plus a SYMBOLIC operand is left untouched (the symbol may later bind to a matrix),
    // never prematurely declared `undefined`.
    let out = cli()
        .args(["eval", "[[1,2,3]] + y", "--format", "json"])
        .output()
        .expect("Failed to run CLI");
    let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
    assert_ne!(
        wire["result"].as_str(),
        Some("undefined"),
        "matrix + symbolic must not be undefined"
    );
}
#[test]
fn test_eval_binary_matrix_ops_dot_cross_linsolve() {
    // The 2-argument matrix/vector operations: dot product, cross product, and linear-system
    // solving. dot/cross fold numerically and stay exact symbolically; linsolve returns the UNIQUE
    // solution by exact rational RREF of [A|b], declining (residual) on a singular or inconsistent
    // system. Cross-checked against numpy.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // dot
    assert_eq!(r("dot([1,2,3],[4,5,6])"), "32");
    assert_eq!(r("dot([a,b],[c,d])"), "a·c + b·d");
    assert_eq!(r("dot([1,2],[3,4,5])"), "dot([[1], [2]], [[3], [4], [5]])"); // length mismatch
                                                                             // cross
    assert_eq!(r("cross([1,0,0],[0,1,0])"), "[[0], [0], [1]]");
    assert_eq!(r("cross([2,3,4],[5,6,7])"), "[[-3], [6], [-3]]");
    // linsolve
    assert_eq!(r("linsolve([[1,1],[1,-1]], [3,1])"), "[[2], [1]]");
    assert_eq!(
        r("linsolve([[1,2,3],[0,1,4],[5,6,0]], [6,5,11])"),
        "[[1], [1], [1]]"
    );
    // Singular and inconsistent systems decline to honest residuals.
    assert_eq!(
        r("linsolve([[1,2],[2,4]], [3,6])"),
        "linsolve([[1, 2], [2, 4]], [[3], [6]])"
    );
    assert_eq!(
        r("linsolve([[1,1],[1,1]], [1,2])"),
        "linsolve([[1, 1], [1, 1]], [[1], [2]])"
    );
}
#[test]
fn test_eval_vector_norm() {
    // `norm(v)` is the Euclidean / Frobenius norm √(Σ |entryᵢ|²) and it is VALUE-DEPENDENT
    // (`|entry|² = entry²` only for real entries), so it is domain-threaded (Fase 2 V0):
    // - real mode (default): `i` is an ordinary symbol — the same contract as the gated
    //   Gaussian rules (`abs(3+4i)` stays residual) — so every entry squares RAW and an
    //   `i`-carrying vector stays an honest unevaluated radical.
    // - complex mode: a Gaussian entry folds its MAGNITUDE (`|a+bi|² = a²+b²`) exactly and
    //   a symbolic entry squares its modulus (`|x|²`) — `x²` would be wrong for ℂ-valued x
    //   (x:=i would make `sqrt(x²+1)` collapse to 0 while the norm is sqrt(2)).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
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
    // Real entries: identical in both domains.
    assert_eq!(r("norm([3,4])"), "5");
    assert_eq!(r("norm([1,2,2])"), "3");
    assert_eq!(r("norm([1,1])"), "sqrt(2)");
    assert_eq!(r("norm([3,-4])"), "5");
    assert_eq!(r("norm([[3,4],[0,12]])"), "13"); // Frobenius norm of a matrix
                                                 // El texto del resultado ECOA la notación del input (2026-07-29): estas entradas
                                                 // no escriben ni raíz ni potencia fraccionaria, así que reciben la forma que un
                                                 // alumno reconoce — `sqrt(a^2 + b^2)`, no `(a^2 + b^2)^(1/2)`. Y re-parsea.
    assert_eq!(r("norm([a,b])"), "sqrt(a^2 + b^2)"); // symbolic, real-valued symbols
    assert_eq!(rc("norm([3,4])"), "5");
    assert_eq!(rc("norm([[3,4],[0,12]])"), "13");
    // Real mode + `i`: honest symbolic radical (i is a plain symbol here; the Imaginary
    // Usage Warning nudges to complex mode). Folding these to 5 / sqrt(2) was the V0
    // incoherence: the metric layer treated `i` as imaginary while every gated Gaussian
    // rule (and `abs(3+4i)`) kept it symbolic.
    assert_eq!(r("norm([3,4i])"), "sqrt(9 + 16·i^2)");
    assert_eq!(r("norm([1,i])"), "sqrt(1 + i^2)");
    assert_eq!(r("norm([1+i,1])"), "sqrt(2 + i^2 + 2·i)");
    assert_eq!(r("norm([2i])"), "2·|i|"); // sqrt((2i)²) = |2i| with i symbolic
    assert_eq!(r("norm([3i,4i])"), "5·|i|");
    // Complex mode: the magnitude fold lives HERE (its correct domain).
    assert_eq!(rc("norm([3,4i])"), "5"); // NOT sqrt(9+(4i)^2) = i·sqrt(7)
    assert_eq!(rc("norm([1,i])"), "sqrt(2)"); // NOT sqrt(1+i^2) = 0
    assert_eq!(rc("norm([1+i,1])"), "sqrt(3)"); // |1+i|^2 + 1 = 3
    assert_eq!(rc("norm([2i])"), "2");
    assert_eq!(rc("norm([3i,4i])"), "5");
    // Complex mode + symbols: Hermitian form — the V0 P0 fix. `(x^2+y^2)^(1/2)` here was a
    // latent wrong answer (x:=i, y:=1 → 0 instead of sqrt(2)).
    assert_eq!(rc("norm([x,y])"), "sqrt(|x|^2 + |y|^2)");
    // Conscious contract: `dot` stays BILINEAR (no conjugation — SymPy's default), so over
    // ℂ `norm(v) ≠ sqrt(dot(v,v))` by design: dot([i,1],[i,1]) = i²+1 = 0 while the norm
    // of [i,1] is sqrt(2) (pinned above).
    assert_eq!(rc("dot([i,1],[i,1])"), "0");
}
#[test]
fn test_eval_limit_matrix_componentwise() {
    // P0 2026-07-19 (barrido adversarial de F0): `depends_on` no atravesaba
    // Matrix, así que la regla de constante afirmaba `[[1/x,0],[0,1]]` como su
    // PROPIO límite (un "valor" x-dependiente). Ahora: componentwise
    // all-or-nothing — todas resuelven → matriz de límites; una DNE probada →
    // undefined del conjunto; una declina → residual de la matriz ENTERA.
    let eval_full = |input: &str| -> (String, String) {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        let warnings = wire["warnings"]
            .as_array()
            .map(|w| {
                w.iter()
                    .filter_map(|x| x["assumption"].as_str())
                    .collect::<Vec<_>>()
                    .join(" | ")
            })
            .unwrap_or_default();
        (wire["result"].as_str().unwrap_or("").to_string(), warnings)
    };
    assert_eq!(
        eval_full("limit([[1/x,0],[0,1]], x, infinity)").0,
        "[[0, 0], [0, 1]]"
    );
    assert_eq!(
        eval_full("limit([[x,1],[2,3]], x, 2)").0,
        "[[2, 1], [2, 3]]"
    );
    let (r, w) = eval_full("limit([[1/x,2]], x, 0)");
    assert_eq!(r, "undefined", "una entrada DNE decide el conjunto");
    assert!(w.contains("matrix limit does not exist"), "got: {w}");
    let (r, w) = eval_full("limit([[e^(i*x),1]], x, infinity)");
    assert!(
        r.starts_with("limit("),
        "entrada que declina => residual de la matriz entera, got: {r}"
    );
    assert!(w.contains("matrix entry declines"), "got: {w}");
    // Bajo complex el kill-switch va PRIMERO: residual de la matriz entera.
    let out = cli()
        .args([
            "eval",
            "limit([[1/x,0],[0,1]], x, infinity)",
            "--value-domain",
            "complex",
            "--format",
            "json",
        ])
        .output()
        .expect("Failed to run CLI");
    let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
    assert!(wire["result"].as_str().unwrap_or("").starts_with("limit("));
}
#[test]
fn test_eval_subs_noop_guard_and_definite_matrix_integrate_f3() {
    // Fase 3 · F3: (a) el subs no-op y las cadenas anidadas colapsan en el
    // nodo EXTERIOR en UN rewrite — la resolución hijo-primero re-observaba la
    // huella del hijo y el CycleDetector global-de-fase lo leía como ciclo
    // period-1 (blocked-hint "requires cos(t) (defined)" en toda circulación
    // por-componente / Green). El fixture pinea la AUSENCIA de blocked_hints.
    // (b) integrate DEFINIDO sobre Matrix: componentwise all-or-nothing.
    let eval_wire = |input: &str| -> Value {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        serde_json::from_slice(&out.stdout).expect("Invalid wire output")
    };
    // (a) circulación por-componente: valor limpio y CERO blocked_hints.
    let wire = eval_wire(
        "integrate(subs(subs(-y,x,cos(t)),y,sin(t))*diff(cos(t),t) + subs(subs(x,x,cos(t)),y,sin(t))*diff(sin(t),t), t, 0, 2*pi)",
    );
    assert_eq!(wire["result"].as_str().unwrap_or(""), "2·pi");
    assert!(
        wire["blocked_hints"]
            .as_array()
            .is_none_or(|h| h.is_empty()),
        "la circulación por-componente no debe llevar blocked_hints: {}",
        wire["blocked_hints"]
    );
    // Formas aisladas: no-op, cadena con no-op interior, cadena con no-op
    // exterior, y efectiva — todas limpias.
    for (probe, expected) in [
        ("subs(cos(t), x, 1)", "cos(t)"),
        ("subs(subs(-y,x,cos(t)),y,sin(t))", "-sin(t)"),
        ("subs(subs(x,x,cos(t)),y,sin(t))", "cos(t)"),
        ("subs(subs(x^2+y^2,x,1),y,2)", "5"),
    ] {
        let wire = eval_wire(probe);
        assert_eq!(wire["result"].as_str().unwrap_or(""), expected, "{probe}");
        assert!(
            wire["blocked_hints"]
                .as_array()
                .is_none_or(|h| h.is_empty()),
            "{probe} no debe llevar blocked_hints"
        );
    }
    // (b) integrate definido componentwise; una entrada que declina => residual entero.
    assert_eq!(
        eval_wire("integrate([cos(t),sin(t)], t, 0, pi)")["result"]
            .as_str()
            .unwrap_or(""),
        "[[0], [2]]"
    );
    assert_eq!(
        eval_wire("integrate([2*t,3*t^2], t, 0, 1)")["result"]
            .as_str()
            .unwrap_or(""),
        "[[1], [1]]"
    );
    // Pins: subs con Requires y el orden order-safe intactos; definido escalar.
    assert_eq!(
        eval_wire("subs(x*y/(x^2+y^2),y,k*x)")["result"]
            .as_str()
            .unwrap_or(""),
        "k / (k^2 + 1)"
    );
    assert_eq!(
        eval_wire("subs(diff(x^2*y,x),x,1)")["result"]
            .as_str()
            .unwrap_or(""),
        "2·y"
    );
    assert_eq!(
        eval_wire("integrate(x^2,x,0,1)")["result"]
            .as_str()
            .unwrap_or(""),
        "1/3"
    );
}
#[test]
fn test_eval_componentwise_diff_over_matrix() {
    // Fase 2 V1: `diff` distributes componentwise over a `Matrix` target, ALL-OR-NOTHING
    // (a non-differentiable component keeps the whole call an honest residual), and the
    // higher-order desugar composes for free (`diff(M, x, 2)` = nested componentwise diffs).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("diff([x^2,x^3],x)"), "[[2·x], [3·x^2]]");
    assert_eq!(r("diff([x^2*y, sin(x)],x)"), "[[2·x·y], [cos(x)]]");
    // Higher-order rides the target-agnostic desugar: diff(M,x,2) → diff(diff(M,x),x).
    assert_eq!(r("diff([x^2,x^3],x,2)"), "[[2], [6·x]]");
    // All-or-nothing: sign(x) has no derivative here, so NOTHING is derived (never a
    // half-differentiated matrix).
    assert_eq!(r("diff([x, sign(x)],x)"), "diff([[x], [sign(x)]], x)");
    // Var-list stays a decline (the list-of-vars arity belongs to the vectorial verbs, V3+).
    assert_eq!(r("diff(x^2+y^2,[x,y])"), "diff(x^2 + y^2, [[x], [y]])");
    // Scalar pins: the componentwise arm must not disturb the scalar cascade.
    assert_eq!(r("diff(x^2*y^3,x,y)"), "6·x·y^2");
    assert_eq!(r("wronskian([x^2,x^3],x)"), "x^4");
    // Narration: the componentwise arm emits a visible step (the diff call is the root, so
    // the Matrix-as-leaf wire gap does not swallow it).
    let out = cli()
        .args([
            "eval",
            "diff([x^2,x^3],x)",
            "--steps",
            "on",
            "--format",
            "json",
        ])
        .output()
        .expect("Failed to run CLI");
    let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
    let steps = wire["steps"].as_array().expect("steps array");
    assert!(
        !steps.is_empty(),
        "componentwise diff at the root must narrate"
    );
    assert_eq!(
        steps[0]["rule"].as_str().unwrap_or(""),
        "Calcular la derivada"
    );
}
#[test]
fn test_eval_abs_vector_and_componentwise_integrate() {
    // Fase 2 V7: (a) |v| of a VECTOR is its Euclidean norm, inheriting V0's domain
    // decision wholesale (never re-deciding it); a general matrix stays residual.
    // (b) integrate distributes componentwise, ALL-OR-NOTHING and conditions-
    // conservative: a non-elementary component (or one whose antiderivative carries
    // required conditions) declines the WHOLE call — the north star's protected
    // residuals never end up half-integrated inside a matrix.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
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
    // V7a — abs of a vector is the norm, in BOTH domains (V0 inheritance).
    assert_eq!(r("abs([3,4])"), "5");
    assert_eq!(r("abs([x,y])"), "sqrt(x^2 + y^2)");
    assert_eq!(rc("abs([x,y])"), "sqrt(|x|^2 + |y|^2)");
    assert_eq!(rc("abs([1,i])"), "sqrt(2)");
    // General matrix: honest residual (matrix modulus ≠ Frobenius norm); scalar abs
    // untouched (the abs family is 4-historic-P0 territory).
    assert_eq!(r("abs([[1,2],[3,4]])"), "|[[1, 2], [3, 4]]|");
    assert_eq!(r("abs(-3)"), "3");
    // V7b — componentwise antiderivatives.
    assert_eq!(r("integrate([x, x^2], x)"), "[[1/2·x^2], [1/3·x^3]]");
    assert_eq!(r("integrate([cos(x), e^x], x)"), "[[sin(x)], [e^x]]");
    assert_eq!(r("integrate([1/x, x], x)"), "[[ln(|x|)], [1/2·x^2]]");
    // ALL-OR-NOTHING: e^(-x²) has no elementary antiderivative — the whole call echoes.
    assert_eq!(
        r("integrate([x, e^(-x^2)], x)"),
        "integrate([[x], [1 / e^(x^2)]], x)"
    );
    // Definite integrals over a Matrix: GRADUATED by F3 (Fase 3) — the V7b
    // indefinite-only boundary was the honest residual until the definite
    // componentwise arm landed; the deep pin (all-or-nothing) lives in
    // test_eval_subs_noop_guard_and_definite_matrix_integrate_f3.
    assert_eq!(r("integrate([x,x^2], x, 0, 1)"), "[[1/2], [1/3]]");
}
#[test]
fn test_eval_steps_under_matrix_literal() {
    // Fase 2 V2 (P0-wire): steps that fire UNDER a `Matrix` node used to be silently
    // discarded — `rewrite_at_expr_path_with` treated Matrix as a leaf, so the step's
    // global snapshot came back unchanged and the didactic pipeline dropped it (correct
    // values, EMPTY narration). Matrix now descends like Function (flat cell index).
    let steps_of = |input: &str| -> Vec<Value> {
        let out = cli()
            .args(["eval", input, "--steps", "on", "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["steps"].as_array().cloned().unwrap_or_default()
    };
    // The fixed case: two derivatives inside a matrix literal narrate one step each,
    // with GLOBAL (matrix-shaped) snapshots.
    let steps = steps_of("[diff(x^2,x), diff(x^3,x)]");
    assert!(
        steps.len() >= 2,
        "steps under a Matrix literal must narrate (got {})",
        steps.len()
    );
    let first_before = steps[0]["before"].as_str().unwrap_or("");
    assert!(
        first_before.contains("[["),
        "snapshots must be global (matrix-shaped), got: {first_before}"
    );
    // Differential controls (the two shapes that always worked): unchanged emission.
    assert!(
        steps_of("diff(x^2,x) + diff(x^3,x)").len() >= 2,
        "steps under Add"
    );
    assert!(
        !steps_of("sqrt(diff(x^4,x))").is_empty(),
        "steps under Function arg"
    );
}
#[test]
fn test_eval_matmul_function() {
    // `matmul` sat in the eval gate with NO dispatch arm — the live gate-without-rule
    // gotcha (silent residual while `A*B` evaluated). Now it shares the `*` math; a
    // dimension mismatch declines to an honest residual (not undefined).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(
        r("matmul([[1,2],[3,4]],[[5,6],[7,8]])"),
        "[[19, 22], [43, 50]]"
    );
    assert_eq!(r("matmul([[1,2]],[[3],[4]])"), "[11]");
    assert_eq!(r("matmul([[1,2]],[[3,4]])"), "matmul([1, 2], [3, 4])");
}
#[test]
fn test_eval_matrix_nullspace() {
    // `nullspace(A)` (aliases `null`/`kernel`) returns a basis of {x : A·x = 0} by exact rational
    // RREF, rows = basis vectors. Verified elsewhere by A·v = 0. A trivial kernel is the zero vector;
    // symbolic entries decline.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("nullspace([[1,2],[2,4]])"), "[-2, 1]");
    assert_eq!(r("nullspace([[1,2,3],[4,5,6],[7,8,9]])"), "[1, -2, 1]");
    assert_eq!(r("nullspace([[1,0],[0,1]])"), "[0, 0]"); // trivial kernel
    assert_eq!(r("nullspace([[1,1,1]])"), "[[-1, 1, 0], [-1, 0, 1]]"); // 2-D kernel
    assert_eq!(r("nullspace([[a,b],[c,d]])"), "nullspace([[a, b], [c, d]])");
}
#[test]
fn test_eval_vector_projection_and_angle() {
    // `proj(u,v)` = (⟨u,v⟩/⟨v,v⟩)·v (vector projection of u onto v, in v's shape) and
    // `angle(u,v)` = arccos(⟨u,v⟩/(‖u‖‖v‖)). Both require numeric vectors; the engine folds
    // arccos at the standard cosines, so nice vectors give clean closed forms. A zero direction
    // / zero vector, or symbolic / irrational-entry operands, decline to honest residuals.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Projection (returned as a column in v's shape).
    assert_eq!(r("proj([3,4],[1,0])"), "[[3], [0]]");
    assert_eq!(r("proj([3,4],[1,1])"), "[[7/2], [7/2]]");
    assert_eq!(r("proj([2,3,6],[1,2,2])"), "[[20/9], [40/9], [40/9]]");
    // Zero direction ⇒ honest residual (projection undefined).
    assert_eq!(r("proj([3,4],[0,0])"), "proj([[3], [4]], [[0], [0]])");
    // Angle: standard cosines fold to exact multiples of π.
    assert_eq!(r("angle([1,0],[0,1])"), "1/2·pi"); // perpendicular
    assert_eq!(r("angle([1,0],[1,0])"), "0"); // parallel
    assert_eq!(r("angle([1,0],[-1,0])"), "pi"); // antiparallel
    assert_eq!(r("angle([1,0],[1,1])"), "1/4·pi");
    assert_eq!(r("angle([3,4],[4,3])"), "arccos(24/25)"); // generic ⇒ exact arccos
                                                          // Zero vector ⇒ honest residual.
    assert_eq!(r("angle([0,0],[1,1])"), "angle([[0], [0]], [[1], [1]])");
}
#[test]
fn test_eval_matrix_eigenvectors_rational() {
    // `eigenvectors(A)` (capstone of the linear-algebra core) returns, for each distinct RATIONAL
    // eigenvalue, the null-space basis of A−λI by exact rational RREF — rows are the eigenvectors.
    // Verified elsewhere by A·v = λ·v. A defective matrix yields fewer vectors (geometric
    // multiplicity); surd / complex / symbolic spectra decline to honest residuals.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("eigenvectors([[2,1],[1,2]])"), "[[1, 1], [-1, 1]]");
    assert_eq!(r("eigenvectors([[2,0],[0,3]])"), "[[0, 1], [1, 0]]");
    // Defective matrix (Jordan block): repeated eigenvalue 1 with a SINGLE eigenvector.
    assert_eq!(r("eigenvectors([[1,1],[0,1]])"), "[1, 0]");
    // Repeated eigenvalue with a full 2-D eigenspace plus a simple one.
    assert_eq!(
        r("eigenvectors([[5,4,2],[4,5,2],[2,2,2]])"),
        "[[-1, 1, 0], [-1/2, 0, 1], [2, 2, 1]]"
    );
    // Surd / complex / symbolic spectra → honest residual.
    assert_eq!(
        r("eigenvectors([[2,-1,0],[-1,2,-1],[0,-1,2]])"),
        "eigenvectors([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])"
    );
    assert_eq!(
        r("eigenvectors([[0,-1],[1,0]])"),
        "eigenvectors([[0, -1], [1, 0]])"
    );
    assert_eq!(
        r("eigenvectors([[a,b],[c,d]])"),
        "eigenvectors([[a, b], [c, d]])"
    );
}
#[test]
fn test_eval_matrix_rref() {
    // Reduced row echelon form was unimplemented. It now computes the exact RREF by Gauss-Jordan
    // over BigRational, with an honest residual for symbolic entries.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("rref([[1,2],[3,4]])"), "[[1, 0], [0, 1]]"); // full rank → identity
    assert_eq!(r("rref([[1,2],[2,4]])"), "[[1, 2], [0, 0]]"); // rank 1
    assert_eq!(
        r("rref([[1,2,3],[4,5,6],[7,8,9]])"),
        "[[1, 0, -1], [0, 1, 2], [0, 0, 0]]"
    );
    assert_eq!(r("rref([[2,4,6],[1,2,3]])"), "[[1, 2, 3], [0, 0, 0]]"); // pivot normalized
    assert_eq!(r("rref([[0,1],[1,0]])"), "[[1, 0], [0, 1]]"); // pivot swap
    assert_eq!(r("rref([[a,b],[c,d]])"), "rref([[a, b], [c, d]])"); // symbolic residual
}
#[test]
fn test_eval_matrix_eigenvalues_real() {
    // `eigenvalues(A)` was unimplemented. It now returns the REAL spectrum as the roots of the
    // characteristic polynomial: rational roots peeled exactly, a deflated quadratic closed by the
    // quadratic formula. A complex-conjugate pair (negative discriminant) declines to an honest
    // residual — this is a real-domain engine. Cross-checked against numpy.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Rational spectra.
    assert_eq!(r("eigenvalues([[2,1],[1,2]])"), "[3, 1]");
    assert_eq!(r("eigenvalues([[1,1],[0,1]])"), "[1, 1]"); // repeated eigenvalue
    assert_eq!(r("eigenvalues([[5,4,2],[4,5,2],[2,2,2]])"), "[1, 10, 1]");
    // Rational root peeled, then a surd quadratic factor: 2, 2 ± √2.
    assert_eq!(
        r("eigenvalues([[2,-1,0],[-1,2,-1],[0,-1,2]])"),
        "[2, sqrt(2) + 2, 2 - sqrt(2)]"
    );
    // Complex spectrum (rotation) → honest residual in the real domain.
    assert_eq!(
        r("eigenvalues([[0,-1],[1,0]])"),
        "eigenvalues([[0, -1], [1, 0]])"
    );
    // Symbolic / non-square → honest residual.
    assert_eq!(
        r("eigenvalues([[a,b],[c,d]])"),
        "eigenvalues([[a, b], [c, d]])"
    );
}
#[test]
fn test_eval_matrix_charpoly() {
    // `charpoly(A) = det(λI − A)` was unimplemented. It now returns the monic characteristic
    // polynomial in `lambda`, for numeric and symbolic matrices, 2×2 and 3×3. (A bounded
    // budget exemption lets the cofactor expansion of a small numeric matrix commit instead of
    // being rejected by the anti-worsen node budget.)
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // [[2,1],[1,2]]: λ² − 4λ + 3 (eigenvalues 1, 3).
    assert_eq!(r("charpoly([[2,1],[1,2]])"), "lambda^2 + 3 - 4·lambda");
    // Diagonal 3×3 factors directly to (λ−1)(λ−2)(λ−3).
    assert_eq!(
        r("charpoly([[1,0,0],[0,2,0],[0,0,3]])"),
        "(lambda - 3)·(lambda - 2)·(lambda - 1)"
    );
    // Tridiagonal: λ³ − 6λ² + 10λ − 4 (trace 6, det 4).
    assert_eq!(
        r("charpoly([[2,-1,0],[-1,2,-1],[0,-1,2]])"),
        "lambda^3 + 10·lambda - 6·lambda^2 - 4"
    );
    // Symbolic 2×2: λ² − (a+d)λ + (ad − bc), kept in det form.
    assert_eq!(
        r("charpoly([[a,b],[c,d]])"),
        "(lambda - a)·(lambda - d) - b·c"
    );
    // Non-square stays an honest residual.
    assert_eq!(r("charpoly([[1,2,3]])"), "charpoly([1, 2, 3])");
}
#[test]
fn test_eval_matrix_adjugate() {
    // `adjugate(A)` (alias `adj`) is the transpose of the cofactor matrix — a polynomial in the
    // entries, ALWAYS defined (no det≠0 condition), so it works symbolically too. Satisfies
    // A·adj(A) = det(A)·I (verified separately).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("adjugate([[1,2],[3,4]])"), "[[4, -2], [-3, 1]]");
    assert_eq!(r("adjugate([[a,b],[c,d]])"), "[[d, -b], [-c, a]]"); // symbolic
    assert_eq!(
        r("adjugate([[1,2,3],[0,1,4],[5,6,0]])"),
        "[[-24, 18, 5], [20, -15, -4], [-5, 4, 1]]"
    );
    // A·adj(A) = det(A)·I.
    assert_eq!(
        r("[[1,2],[3,4]] * adjugate([[1,2],[3,4]])"),
        "[[-2, 0], [0, -2]]"
    );
}
#[test]
fn test_eval_matrix_integer_power() {
    // `M^n` for an integer exponent: `n=0 → I`, `n=1 → M`, `|n|≥2` for an all-numeric square matrix
    // is repeated multiplication (negative ⇒ inverse powered), folding exactly. A bounded budget
    // exemption lets the unfolded products commit. Cross-checked against numpy.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("[[1,1],[0,1]]^3"), "[[1, 3], [0, 1]]");
    assert_eq!(r("[[1,1],[1,0]]^5"), "[[8, 5], [5, 3]]"); // Fibonacci
    assert_eq!(r("[[2,0],[0,3]]^2"), "[[4, 0], [0, 9]]");
    assert_eq!(
        r("[[1,2,0],[0,1,1],[0,0,1]]^4"),
        "[[1, 8, 12], [0, 1, 4], [0, 0, 1]]"
    );
    assert_eq!(r("[[1,2],[3,4]]^0"), "[[1, 0], [0, 1]]"); // M^0 = I
    assert_eq!(r("[[2,0],[0,2]]^(-2)"), "[[1/4, 0], [0, 1/4]]"); // negative power via inverse
                                                                 // Controls: a singular base to a negative power is undefined; symbolic power / inverse stay
                                                                 // honest residuals; a non-square base is undefined.
    assert_eq!(r("[[1,2],[2,4]]^(-1)"), "undefined");
    assert_eq!(r("[[a,b],[c,d]]^2"), "[[a, b], [c, d]]^2");
    assert_eq!(r("[[a,b],[c,d]]^(-1)"), "inverse([[a, b], [c, d]])");
    assert_eq!(r("[[1,2,3],[4,5,6]]^2"), "undefined");
}
#[test]
fn test_eval_matrix_rank_exact() {
    // Matrix rank was recognized-but-unimplemented (returned an error). It now computes the
    // exact rank by Gaussian elimination over BigRational, for any shape, with an honest
    // residual for symbolic entries.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("rank([[1,2],[2,4]])"), "1");
    assert_eq!(r("rank([[1,2],[3,4]])"), "2");
    assert_eq!(r("rank([[1,2,3],[4,5,6],[7,8,9]])"), "2");
    assert_eq!(r("rank([[1,0,0],[0,1,0],[0,0,1]])"), "3");
    assert_eq!(r("rank([[0,0],[0,0]])"), "0");
    assert_eq!(r("rank([[1,2,3],[2,4,6]])"), "1"); // 2x3, rank 1
                                                   // Symbolic entry stays an honest residual (no fabricated number).
    assert_eq!(r("rank([[a,2],[3,4]])"), "rank([[a, 2], [3, 4]])");
    // Controls: the sibling matrix functions are unchanged.
    assert_eq!(r("det([[1,2],[3,4]])"), "-2");
    assert_eq!(r("trace([[1,2],[3,4]])"), "5");
}
#[test]
fn test_eval_matrix_inverse_routes_and_no_scalar_broadcast() {
    // `M^(-1)` / `c/M` used to fall to scalar arithmetic and fabricate `1/[[…]]`
    // (a non-square matrix has NO inverse; a symbolic one is not elementwise 1/entry).
    // They now route to the matrix inverse, and `ScalarMatrixRule` no longer broadcasts
    // a matrix-valued operand (e.g. `inverse(M)`) as if it were a scalar.
    for (input, expected) in [
        // Numeric square: the actual inverse.
        ("[[1,2],[3,4]]^(-1)", "[[-2, 1], [3/2, -1/2]]"),
        ("1/[[1,2],[3,4]]", "[[-2, 1], [3/2, -1/2]]"),
        ("2/[[1,2],[3,4]]", "[[-4, 2], [3, -1]]"),
        // Round-trip M·M^(-1) = I.
        ("[[1,2],[3,4]] * [[1,2],[3,4]]^(-1)", "[[1, 0], [0, 1]]"),
        // Symbolic / non-square: honest residual (NOT `1/[[…]]`).
        ("[[a,b],[c,d]]^(-1)", "inverse([[a, b], [c, d]])"),
        ("[[1,2,3],[4,5,6]]^(-1)", "inverse([[1, 2, 3], [4, 5, 6]])"),
        // Singular: undefined (no inverse exists).
        ("[[1,2],[2,4]]^(-1)", "undefined"),
        // Facet 2: a symbolic inverse times a matrix stays a residual, not a broadcast.
        (
            "[[a,b],[c,d]]^(-1) * [[1,0],[0,1]]",
            "inverse([[a, b], [c, d]])·[[1, 0], [0, 1]]",
        ),
        // Ordinary scalar·matrix and matrix·matrix are unaffected.
        ("3 * [[1,2],[3,4]]", "[[3, 6], [9, 12]]"),
        ("[[1,2],[3,4]] * [[5,6],[7,8]]", "[[19, 22], [43, 50]]"),
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
