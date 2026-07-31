use super::*;

#[test]
fn value_domain_complex_reflected() {
    let (output, _code) = run_cli(&[
        "eval",
        "1+1",
        "--format",
        "json",
        "--value-domain",
        "complex",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["semantics"]["value_domain"], "complex");
}
#[test]
fn const_fold_safe_complex_sqrt_negative_reaches_cli_runtime() {
    let (output, _code) = run_cli(&[
        "eval",
        "sqrt(-1)",
        "--format",
        "json",
        "--value-domain",
        "complex",
        "--complex",
        "on",
        "--const-fold",
        "safe",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["options"]["const_fold"], "safe");
    assert_eq!(wire["semantics"]["value_domain"], "complex");
    assert_eq!(wire["result"], "i");
    assert_eq!(
        wire["warnings"].as_array().map(Vec::len),
        Some(0),
        "complex+safe should not emit imaginary-usage warning"
    );
}
#[test]
fn const_fold_safe_complex_i_squared_reaches_cli_runtime() {
    let (output, _code) = run_cli(&[
        "eval",
        "i^2",
        "--format",
        "json",
        "--value-domain",
        "complex",
        "--complex",
        "on",
        "--const-fold",
        "safe",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["options"]["const_fold"], "safe");
    assert_eq!(wire["semantics"]["value_domain"], "complex");
    assert_eq!(wire["result"], "-1");
}
/// Default (rule-based) folding of negative even roots in complex mode. This is the
/// path the rewrite rules drive (`SqrtNegativeRule` / `NegativeBaseHalfPowerRule`),
/// distinct from the `--const-fold safe` path above. Regression guard for the
/// value-domain-aware non-finite backstop: it must NOT revert `(-1)^(1/2) -> i`
/// (an "undefined over ℝ" form that is a defined finite value `i` in ℂ).
#[test]
fn complex_negative_even_root_folds_to_imaginary_by_rules() {
    for (src, expected) in [
        ("(-1)^(1/2)", "i"),
        ("sqrt(-1)", "i"),
        ("sqrt(-4)", "2·i"),
        ("(-9)^(1/2)", "3·i"),
        ("(-3)^(1/2)", "i·sqrt(3)"),
        // Additive cancellation of a now-defined complex value is sound.
        ("(-1)^(1/2) - (-1)^(1/2)", "0"),
        ("sqrt(-4) - sqrt(-4)", "0"),
    ] {
        let (output, _code) =
            run_cli(&["eval", src, "--format", "json", "--value-domain", "complex"]);
        let wire = parse_wire(&output);
        assert_eq!(wire["semantics"]["value_domain"], "complex");
        assert_eq!(
            wire["result"], expected,
            "complex `{src}` should fold to `{expected}`"
        );
    }

    // REAL mode must keep the same forms symbolic (never `i`): the backstop's
    // real-domain semantics are unchanged.
    for (src, expected) in [("(-1)^(1/2)", "(-1)^(1/2)"), ("sqrt(-4)", "2·sqrt(-1)")] {
        let (output, _code) = run_cli(&["eval", src, "--format", "json", "--value-domain", "real"]);
        let wire = parse_wire(&output);
        assert_eq!(wire["result"], expected, "real `{src}` must stay symbolic");
    }

    // Genuine undefined (`c/0`) stays undefined in complex mode: the backstop is only
    // loosened for ℂ-defined forms, never for division by a provably-zero denominator.
    let (output, _code) = run_cli(&[
        "eval",
        "1/(x-x) - 1/(x-x)",
        "--format",
        "json",
        "--value-domain",
        "complex",
    ]);
    let wire = parse_wire(&output);
    assert_eq!(
        wire["result"], "undefined",
        "`1/(x-x) - 1/(x-x)` must stay undefined in ℂ (1/0 is undefined in ℂ too)"
    );
}
#[test]
fn root_ctx_exact_quotient_survives_runtime_before_conjugate_rationalization() {
    let (output, _code) = run_cli(&[
        "eval",
        "((1/sqrt(u))^2 + 2*(1/sqrt(u)) + 1)/((1/sqrt(u)) + 1)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "u^(-1/2) + 1");

    let steps = wire["steps"].as_array().expect("steps array");
    let exact_step = steps
        .iter()
        .find(|step| step["rule"] == "Racionalizar el denominador")
        .expect("exact quotient step");
    assert_eq!(exact_step["after"], "1/sqrt(u) + 1");

    let required = wire["required_display"]
        .as_array()
        .expect("required display");
    assert_eq!(
        required
            .iter()
            .filter_map(|item| item.as_str())
            .collect::<Vec<_>>(),
        vec!["u > 0"],
        "runtime should compress reciprocal-sqrt guards to a single positivity guard"
    );
}
#[test]
fn complex_solve_pow_isolation_declines_honestly() {
    // T2-ciclo1 (pow-isolation domain-aware): over ℂ, `x^n = c` with integer
    // n >= 3 has n roots the real n-th-root isolation cannot produce, and the
    // "even power cannot be negative" verdict is a REAL fact. Both decline to
    // an honest residual instead of a silent subset / wrong Empty.
    for src in ["solve(x^5 = 1, x)", "solve(x^7 = 3, x)"] {
        let (output, _code) =
            run_cli(&["eval", src, "--format", "json", "--value-domain", "complex"]);
        let wire = parse_wire(&output);
        let result = wire["result"].as_str().unwrap_or_default();
        assert!(
            result.contains("solve("),
            "complex `{src}` must stay an honest residual, got `{result}`"
        );
    }

    // GRADUATED 2026-07-30 (ciclo Q3c-005): `x^4 = −1` reaches the exact
    // biquadratic complex branch (disc < 0 decided on BigRational, roots by
    // construction) and now yields all four 4th roots of −1 — the canonical
    // conjugate quartet (±1±i)/√2 — instead of the honest residual this pin
    // used to protect.
    {
        let (output, _code) = run_cli(&[
            "eval",
            "solve(x^4 = -1, x)",
            "--format",
            "json",
            "--value-domain",
            "complex",
        ]);
        let wire = parse_wire(&output);
        let result = wire["result"].as_str().unwrap_or_default();
        for root in ["(1 + i)", "(-1 - i)", "(1 - i)", "(-1 + i)"] {
            assert!(
                result.contains(root),
                "complex x^4 = -1 must carry the conjugate quartet, missing `{root}` in `{result}`"
            );
        }
    }

    // REAL mode keeps the real answers (subset semantics are correct over ℝ).
    for (src, expected) in [
        ("solve(x^5 = 1, x)", "{ 1 }"),
        ("solve(x^4 = -1, x)", "No solution"),
    ] {
        let (output, _code) = run_cli(&["eval", src, "--format", "json", "--value-domain", "real"]);
        let wire = parse_wire(&output);
        assert_eq!(
            wire["result"], expected,
            "real `{src}` should keep its real-domain answer"
        );
    }
}
#[test]
fn complex_solve_backend_twins_emit_conjugate_pairs() {
    // T2-ciclo1: the backend quartic-factor and biquadratic helpers emit the
    // disc<0 / z<0 conjugate pairs under ComplexEnabled instead of dropping
    // them (x^6-1 -> ALL six 6th roots of unity).
    let (output, _code) = run_cli(&[
        "eval",
        "solve(x^6 - 1 = 0, x)",
        "--format",
        "json",
        "--value-domain",
        "complex",
    ]);
    let wire = parse_wire(&output);
    let result = wire["result"].as_str().unwrap_or_default();
    for root in ["1", "-1", "i·sqrt(3)"] {
        assert!(
            result.contains(root),
            "complex x^6-1 should contain `{root}`, got `{result}`"
        );
    }
    assert_eq!(
        result.matches("sqrt(3)").count(),
        4,
        "complex x^6-1 should carry the four complex roots, got `{result}`"
    );

    // GRADUATED 2026-07-30 (ciclo Q3c-005): x^4+1 has complex z-roots and the
    // exact biquadratic complex branch now emits the conjugate quartet
    // (±1±i)/√2 — never Empty, and no longer a residual.
    let (output, _code) = run_cli(&[
        "eval",
        "solve(x^4 + 1 = 0, x)",
        "--format",
        "json",
        "--value-domain",
        "complex",
    ]);
    let wire = parse_wire(&output);
    let result = wire["result"].as_str().unwrap_or_default();
    for root in ["(1 + i)", "(-1 - i)", "(1 - i)", "(-1 + i)"] {
        assert!(
            result.contains(root),
            "complex x^4+1 must carry the conjugate quartet, missing `{root}` in `{result}`"
        );
    }

    // Real mode: both keep their real answers.
    let (output, _code) = run_cli(&[
        "eval",
        "solve(x^6 - 1 = 0, x)",
        "--format",
        "json",
        "--value-domain",
        "real",
    ]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "{ -1, 1 }");
}
#[test]
fn complex_mode_keeps_sqrt_of_square_symbolic() {
    // T2-ciclo2 (transform-layer domain gate): `√(u²) = |u|` is a real-domain
    // identity fabricated by the TRANSFORM preorder pass (transform_pow's
    // early-detection), which runs BEFORE any gated rule. Over ℂ,
    // `√(i²) = i ≠ 1 = |i|`, so the whole family stays symbolic.
    for src in ["(x^2)^(1/2)", "sqrt(x^2)", "sqrt(x^2*y^2)", "sqrt((x+1)^2)"] {
        let (output, _code) =
            run_cli(&["eval", src, "--format", "json", "--value-domain", "complex"]);
        let wire = parse_wire(&output);
        let result = wire["result"].as_str().unwrap_or_default();
        assert!(
            !result.contains('|'),
            "complex `{src}` must NOT fabricate an abs form, got `{result}`"
        );
    }

    // Real mode keeps the classic identity.
    for (src, expected) in [("sqrt(x^2)", "|x|"), ("(x^2)^(1/2)", "|x|")] {
        let (output, _code) = run_cli(&["eval", src, "--format", "json", "--value-domain", "real"]);
        let wire = parse_wire(&output);
        assert_eq!(wire["result"], expected, "real `{src}` should fold to |x|");
    }
}
#[test]
fn complex_mode_applies_euler_formula() {
    // T3-ciclo1 (B2): e^(a+iθ) folds via Euler under ComplexEnabled; the
    // trig of rational π folds downstream to the exact closed forms.
    for (src, expected) in [
        ("e^(i*pi)", "-1"),
        ("e^(pi*i/2)", "i"),
        ("e^(2*pi*i)", "1"),
        ("exp(i*x)", "cos(x) + i·sin(x)"),
    ] {
        let (output, _code) =
            run_cli(&["eval", src, "--format", "json", "--value-domain", "complex"]);
        let wire = parse_wire(&output);
        assert_eq!(
            wire["result"], expected,
            "complex `{src}` should fold via Euler to `{expected}`"
        );
    }

    // REAL mode: Euler is gated off — e^(iπ) stays symbolic.
    let (output, _code) = run_cli(&[
        "eval",
        "e^(i*pi)",
        "--format",
        "json",
        "--value-domain",
        "real",
    ]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "e^(pi·i)");
}
#[test]
fn complex_mode_computes_principal_log_and_arg() {
    // T3-ciclo2 (B3): ln(z) = ln|z| + i·Arg(z) on closed Gaussians, and
    // arg(a+bi) via the exact 9-case atan2 sign table over (-π, π].
    for (src, expected) in [
        ("ln(-1)", "pi·i"),
        ("ln(i)", "1/2·pi·i"),
        ("ln(-2)", "ln(2) + pi·i"),
        ("ln(1+i)", "1/2·ln(2) + 1/4·pi·i"),
        ("arg(i)", "1/2·pi"),
        ("arg(-1)", "pi"),
        ("arg(-1-i)", "-3/4·pi"),
        ("arg(0)", "undefined"),
        ("arg(3)", "0"),
    ] {
        let (output, _code) =
            run_cli(&["eval", src, "--format", "json", "--value-domain", "complex"]);
        let wire = parse_wire(&output);
        assert_eq!(
            wire["result"], expected,
            "complex `{src}` should yield `{expected}`"
        );
    }

    // REAL mode is byte-identical to pre-B3: ln of a negative stays the
    // honest real-domain `undefined`, `arg` stays a symbolic call, and
    // the positive-rational log keeps its owner.
    for (src, expected) in [
        ("ln(-1)", "undefined"),
        ("arg(i)", "arg(i)"),
        ("ln(2)", "ln(2)"),
    ] {
        let (output, _code) = run_cli(&["eval", src, "--format", "json", "--value-domain", "real"]);
        let wire = parse_wire(&output);
        assert_eq!(
            wire["result"], expected,
            "real `{src}` must stay `{expected}`"
        );
    }
}
#[test]
fn complex_principal_branch_condition_survives_to_output() {
    // T4-ciclo1 (B4a): every principal-branch fold carries the structured
    // condition, and it SURVIVES the triviality filter even though the
    // witnesses are closed constants (the `!contains_variable` trap).
    for (src, witness) in [("ln(-1)", "-1"), ("ln(1+i)", "1 + i"), ("arg(i)", "i")] {
        let (output, _code) =
            run_cli(&["eval", src, "--format", "json", "--value-domain", "complex"]);
        let wire = parse_wire(&output);
        let conds = wire["required_conditions"]
            .as_array()
            .expect("required_conditions array");
        assert_eq!(conds.len(), 1, "`{src}` should carry exactly one condition");
        assert_eq!(conds[0]["kind"], "PrincipalBranch", "{src}");
        assert_eq!(conds[0]["expr_display"], witness, "{src}");
        let display = wire["required_display"][0].as_str().expect("display");
        assert!(
            display.contains("principal branch"),
            "`{src}` display: {display}"
        );
    }

    // An undefined verdict is not a branch choice: arg(0) emits none.
    let (output, _code) = run_cli(&[
        "eval",
        "arg(0)",
        "--format",
        "json",
        "--value-domain",
        "complex",
    ]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "undefined");
    assert_eq!(
        wire["required_conditions"].as_array().map(Vec::len),
        Some(0)
    );

    // REAL mode: no complex fold, no branch condition — byte-identical.
    let (output, _code) = run_cli(&[
        "eval",
        "ln(-1)",
        "--format",
        "json",
        "--value-domain",
        "real",
    ]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "undefined");
    assert_eq!(
        wire["required_conditions"].as_array().map(Vec::len),
        Some(0)
    );
}
#[test]
fn complex_mode_computes_general_powers_and_gaussian_sqrt() {
    // T4-ciclo2 (B4b): z^w = e^(w·ln z) on closed Gaussians and the exact
    // principal square root for perfect-square norms. Every fold carries
    // the PrincipalBranch condition.
    for (src, expected) in [
        ("i^i", "e^(-1/2·pi)"),
        ("2^i", "cos(ln(2)) + i·sin(ln(2))"),
        ("sqrt(3+4*i)", "2 + i"),
        ("sqrt(3-4*i)", "2 - i"),
        ("sqrt(i)", "sqrt(1/2)·(1 + i)"),
    ] {
        let (output, _code) =
            run_cli(&["eval", src, "--format", "json", "--value-domain", "complex"]);
        let wire = parse_wire(&output);
        assert_eq!(wire["result"], expected, "complex `{src}`");
        let conds = wire["required_conditions"]
            .as_array()
            .expect("conditions array");
        assert!(
            conds.iter().any(|c| c["kind"] == "PrincipalBranch"),
            "`{src}` must carry the PrincipalBranch condition"
        );
    }

    // Owners keep their exact forms — the polar route must NOT degrade them.
    for (src, expected) in [
        // Migrated by C1 (cartesian display order): real part first.
        ("(-8)^(1/3)", "1 + i·sqrt(3)"),
        ("sqrt(-4)", "2·i"),
        ("(2+i)^2", "3 + 4·i"),
        ("2^(1/2)", "sqrt(2)"),
        ("e^(i*pi)", "-1"),
    ] {
        let (output, _code) =
            run_cli(&["eval", src, "--format", "json", "--value-domain", "complex"]);
        let wire = parse_wire(&output);
        assert_eq!(wire["result"], expected, "complex owner `{src}`");
    }

    // REAL mode: everything stays symbolic or keeps its real-domain verdict.
    for (src, expected) in [
        ("i^i", "i^i"),
        ("2^i", "2^i"),
        ("sqrt(3+4*i)", "sqrt(3 + 4·i)"),
        ("(-8)^(1/3)", "-2"),
    ] {
        let (output, _code) = run_cli(&["eval", src, "--format", "json", "--value-domain", "real"]);
        let wire = parse_wire(&output);
        assert_eq!(
            wire["result"], expected,
            "real `{src}` must stay `{expected}`"
        );
    }
}
#[test]
fn complex_mode_approximates_closed_complex_values() {
    // T4-ciclo4: approx() of closed complex values goes through the B1
    // walker into a cartesian `a + b·i` decimal (presentation surface —
    // approx is f64 by contract).
    for (src, expected) in [
        ("approx(ln(i))", "1.57079632679·i"),
        ("approx(e^(i*pi/3))", "0.5 + 0.866025403784·i"),
        ("approx(2^i)", "0.769238901364 + 0.638961276314·i"),
        ("approx(ln(-2))", "0.69314718056 + 3.14159265359·i"),
        ("approx(ln(-e))", "1 + 3.14159265359·i"),
    ] {
        let (output, _code) =
            run_cli(&["eval", src, "--format", "json", "--value-domain", "complex"]);
        let wire = parse_wire(&output);
        assert_eq!(wire["result"], expected, "complex `{src}`");
    }

    // Real-valued and symbolic arguments keep their owners.
    for (src, expected) in [
        ("approx(i^i)", "0.207879576351"),
        ("approx(sqrt(2))", "1.41421356237"),
        ("approx(x + i)", "approx(x + i)"),
    ] {
        let (output, _code) =
            run_cli(&["eval", src, "--format", "json", "--value-domain", "complex"]);
        let wire = parse_wire(&output);
        assert_eq!(wire["result"], expected, "complex `{src}`");
    }

    // WYSIWYG payload contract (T5-ciclo1): the decimal node wraps the
    // ROUNDED 12-significant-digit rational. Large magnitudes round too —
    // no raw f64 quantization junk — and wear calculator scientific
    // notation (the ×10^19 reading IS the stored rounded rational).
    let (output, _code) = run_cli(&["eval", "approx(10^20/7)", "--format", "json"]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "1.42857142857×10^19");

    // Sticky fold-through (T5-ciclo2): arithmetic sees through decimal(q)
    // exactly and the result keeps the decimal display preference. The
    // headline WYSIWYG identity: subtracting the displayed value back
    // yields EXACTLY zero.
    for (src, expected) in [
        ("approx(3/7) - 0.428571428571", "0"),
        ("approx(1/3) + 1/3", "0.666666666666"),
        ("approx(1/3)*3", "0.999999999999"),
        ("approx(1/3)+approx(1/4)", "0.583333333333"),
        ("2 * approx(3/7)", "0.857142857142"),
        ("approx(0)*x", "0"),
        ("approx(1)*x", "x"),
        ("approx(0) + x", "x"),
        ("1/approx(0)", "undefined"),
    ] {
        let (output, _code) = run_cli(&["eval", src, "--format", "json"]);
        let wire = parse_wire(&output);
        assert_eq!(wire["result"], expected, "`{src}`");
    }

    // Exact-only expressions never acquire the decimal preference.
    for (src, expected) in [
        ("1/3 + 1/3", "2/3"),
        ("3/7 - 0.428571428571", "3/7000000000000"),
    ] {
        let (output, _code) = run_cli(&["eval", src, "--format", "json"]);
        let wire = parse_wire(&output);
        assert_eq!(wire["result"], expected, "exact `{src}` untouched");
    }

    // Closed-subtree mapping (T5-ciclo3): free variables no longer block
    // approx — the maximal closed numeric parts fold to ONE decimal
    // coefficient each; integer-only shapes decline (D5).
    for (src, expected) in [
        // Migrated by the decimal-coefficient-first display fix.
        ("approx(sqrt(2)*pi*e*x)", "12.0770079568·x"),
        ("approx(1/3 + x)", "0.333333333333 + x"),
        ("approx(sin(x) + pi)", "sin(x) + 3.14159265359"),
        ("approx(approx(1/3))", "0.333333333333"),
        ("approx(2*x)", "approx(2·x)"),
    ] {
        let (output, _code) = run_cli(&["eval", src, "--format", "json"]);
        let wire = parse_wire(&output);
        assert_eq!(wire["result"], expected, "`{src}`");
    }

    // Complex closed subtrees compose with the cartesian surface.
    let (output, _code) = run_cli(&[
        "eval",
        "approx(sqrt(2)*i*x)",
        "--format",
        "json",
        "--value-domain",
        "complex",
    ]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "1.41421356237·i·x");

    // Session axis (T5-ciclo4): --numeric-display decimal applies the same
    // walker at the OUTPUT BOUNDARY — internally everything stays exact.
    for (src, expected) in [
        ("1/3 + 1/3", "0.666666666667"),
        ("sqrt(2)*pi*e", "12.0770079568"),
        ("solve(2*x=1, x)", "{ 0.5 }"),
        ("solve(x^2 < 2, x)", "(-1.41421356237, 1.41421356237)"),
        ("sqrt(2)+x", "1.41421356237 + x"),
        ("2+2", "4"),
    ] {
        let (output, _code) = run_cli(&[
            "eval",
            src,
            "--format",
            "json",
            "--numeric-display",
            "decimal",
        ]);
        let wire = parse_wire(&output);
        assert_eq!(wire["result"], expected, "decimal-display `{src}`");
    }

    // Default axis is Exact: byte-identical to the pre-axis contract.
    let (output, _code) = run_cli(&["eval", "1/3 + 1/3", "--format", "json"]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "2/3");

    // REAL mode: the complex fallback never leaks — approx(ln(i)) stays
    // symbolic, approx(ln(-1)) keeps the honest real-domain undefined.
    for (src, expected) in [
        ("approx(ln(i))", "approx(ln(i))"),
        ("approx(ln(-1))", "undefined"),
        ("approx(sqrt(2))", "1.41421356237"),
    ] {
        let (output, _code) = run_cli(&["eval", src, "--format", "json", "--value-domain", "real"]);
        let wire = parse_wire(&output);
        assert_eq!(
            wire["result"], expected,
            "real `{src}` must stay `{expected}`"
        );
    }
}
#[test]
fn complex_value_domain_declines_real_only_zero_identities() {
    // SOUNDNESS (auditoría 2026-07-30, ficha S4-001): los atajos zero-identity
    // colapsaban a 0 identidades SOLO-ℝ bajo --value-domain complex
    // (`sqrt(x^2)-abs(x)` → 0 con required_display=[] mientras el MISMO motor
    // declina `sqrt(x^2)` suelto por real-only — autocontradicción). Tres
    // fugas del eje: las sondas especulativas corrían con
    // SimplifyOptions::default() (RealOnly), los matchers estructurales del
    // par no podían consultar el eje, y LogPerfectSquareRule no lo miraba.
    // En ℂ: sqrt(i²) = i ≠ 1 = |i|; ln(i²) = iπ ≠ 0 = 2·ln|i|.
    for expr in [
        "sqrt(x^2)-abs(x)",
        "abs(x)-sqrt(x^2)",
        "sqrt(x^4)-x^2",
        "sqrt(x^2*y^2)-abs(x*y)",
        "sqrt((x-1)^2)-abs(x-1)",
        "sqrt((x+2)^2)-abs(x+2)",
        "sqrt(4*x^2)-2*abs(x)",
        "sqrt(9*x^2)-3*abs(x)",
        "2*ln(abs(x))-ln(x^2)",
        "ln(x^2)-2*ln(abs(x))",
    ] {
        let (output, _code) = run_cli(&[
            "eval",
            expr,
            "--value-domain",
            "complex",
            "--format",
            "json",
        ]);
        let wire = parse_wire(&output);
        assert_ne!(
            wire["result"], "0",
            "{expr} must NOT collapse to 0 under complex: {output}"
        );
        // The real-domain twin keeps its exact collapse (byte-identical
        // real-mode behavior is the fix's contract).
        let (real_output, _code) = run_cli(&["eval", expr, "--format", "json"]);
        let real_wire = parse_wire(&real_output);
        assert_eq!(
            real_wire["result"], "0",
            "{expr} must still collapse in real mode: {real_output}"
        );
    }
}
#[test]
fn complex_literal_radicands_converge_across_steps_modes() {
    // SOUNDNESS (auditoría 2026-07-30, ficha S4-002, clase literal): con
    // --value-domain complex y steps=off, el atajo «Extract Perfect Square
    // from Radicand» colapsaba radicandos con i LITERAL por razonamiento
    // real-only: sqrt(16·i⁴) publicaba «4·i²» (= −4; el valor verdadero es 4,
    // que steps=on SÍ daba plegando i⁴→1 primero). El gate declina el atajo
    // cuando el radicando porta la unidad imaginaria y el pipeline
    // complex-aware toma el relevo: ambos modos convergen al valor VERDADERO.
    for (expr, expected) in [("sqrt(4*i^2)", "2 * i"), ("sqrt(16*i^4)", "4")] {
        for steps in ["off", "on"] {
            let (output, _code) = run_cli(&[
                "eval",
                expr,
                "--value-domain",
                "complex",
                "--steps",
                steps,
                "--format",
                "json",
                "--no-pretty",
            ]);
            let wire = parse_wire(&output);
            assert_eq!(wire["result"], expected, "{expr} steps={steps}: {output}");
        }
    }
    // La extracción puramente numérica sigue viva en complejo (√ de escala
    // real positiva es complex-válida).
    let (output, _code) = run_cli(&[
        "eval",
        "sqrt(8)",
        "--value-domain",
        "complex",
        "--format",
        "json",
        "--no-pretty",
    ]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "2 * sqrt(2)", "{output}");
}
