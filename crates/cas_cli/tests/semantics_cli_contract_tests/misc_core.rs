use super::*;

#[test]
fn semantics_block_present_in_json() {
    let (output, _code) = run_cli(&["eval", "1+1", "--format", "json"]);
    let wire = parse_wire(&output);

    assert!(
        wire.get("semantics").is_some(),
        "Wire output should have 'semantics' field"
    );
}
#[test]
fn false_equiv_surfaces_residual_diagnostics_without_changing_result() {
    let (output, code) = run_cli(&["eval", "equiv(x^2,x)", "--format", "json"]);
    assert_eq!(code, 0);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "false");
    let diagnostics = wire["equivalence_diagnostics"]
        .as_object()
        .expect("equivalence diagnostics");
    assert_eq!(
        diagnostics.get("residual").and_then(Value::as_str),
        Some("x^2 - x")
    );
}
#[test]
fn false_equiv_surfaces_simplified_residual_diagnostics() {
    let (output, code) = run_cli(&[
        "eval",
        "equiv((1+x)^5, x^5+5*x^4+10*x^3+10*x^2+5*x)",
        "--format",
        "json",
    ]);
    assert_eq!(code, 0);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "false");
    let diagnostics = wire["equivalence_diagnostics"]
        .as_object()
        .expect("equivalence diagnostics");
    assert_eq!(
        diagnostics.get("residual").and_then(Value::as_str),
        Some("1")
    );
}
#[test]
fn semantics_defaults_reflected() {
    let (output, _code) = run_cli(&["eval", "1+1", "--format", "json"]);
    let wire = parse_wire(&output);

    let semantics = &wire["semantics"];
    assert_eq!(semantics["domain_mode"], "generic");
    assert_eq!(semantics["value_domain"], "real");
    assert_eq!(semantics["inv_trig"], "strict");
    assert_eq!(semantics["branch"], "principal");
    assert_eq!(semantics["assume_scope"], "real");
}
#[test]
fn time_budget_zero_surfaces_partial_result_warning_in_wire_output() {
    let (output, _code) = run_cli(&["eval", "a + b", "--format", "json", "--time-budget-ms", "0"]);
    let wire = parse_wire(&output);

    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "a + b");
    let warnings = wire["warnings"].as_array().expect("warnings array");
    assert!(
        warnings.iter().any(|warning| {
            warning["rule"] == "Simplification Time Budget"
                && warning["assumption"]
                    .as_str()
                    .is_some_and(|msg| msg.contains("Partial result"))
        }),
        "expected partial-result timeout warning, got: {warnings:?}"
    );
}
#[test]
fn subtraction_self_cancel_shortcut_handles_abs_sub_mirror_runtime_shape() {
    let (output, _code) = run_cli(&[
        "eval",
        "abs((2*u)/(u^2 - 1) - 1) - abs(1 - 2*u/(u^2 - 1))",
        "--format",
        "json",
        "--steps",
        "off",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    assert_eq!(wire["steps_count"], 0);
}
#[test]
fn cartesian_display_orders_real_part_first() {
    // T4-ciclo3 (C1): within an Add, i-free terms display before i-carrying
    // terms even when the real part is negative — `-1 + 2·i`, never
    // `2·i - 1`. One comparator drives text, hints, and LaTeX.
    for (src, expected) in [
        ("(3+4*i)/(1-2*i)", "-1 + 2·i"),
        ("(1+i)^3", "2·(-1 + i)"),
        ("2*i-3", "-3 + 2·i"),
        ("(2+i)^2", "3 + 4·i"),
    ] {
        let (output, _code) =
            run_cli(&["eval", src, "--format", "json", "--value-domain", "complex"]);
        let wire = parse_wire(&output);
        assert_eq!(wire["result"], expected, "complex `{src}`");
    }

    // Solve sets pick up the same convention.
    let (output, _code) = run_cli(&[
        "eval",
        "solve(x^2+2*x+5,x)",
        "--format",
        "json",
        "--value-domain",
        "complex",
    ]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "{ -1 - 2·i, -1 + 2·i }");

    // The ordering is shape-gated on `i`: i-free expressions keep their
    // established display (affine orientation, sign-first convention).
    for (src, expected) in [("1-x", "1 - x"), ("x^2-3*x+2", "x^2 + 2 - 3·x")] {
        let (output, _code) = run_cli(&["eval", src, "--format", "json"]);
        let wire = parse_wire(&output);
        assert_eq!(wire["result"], expected, "real `{src}` untouched");
    }
}
#[test]
fn parametric_product_inequality_declines_instead_of_inventing_order() {
    // SOUNDNESS (auditoría 2026-07-30, ficha U1b-001 / F9): la ruta factorizada
    // publicaba `(a, b)` SIN condición — el orden salía del comparador
    // ESTRUCTURAL, no del valor (falso para a ≥ b; con a=2,b=1 el intervalo es
    // vacío). La forma EXPANDIDA de la misma inecuación declina honesta; la
    // factorizada debe alcanzar el mismo decline, nunca afirmar de más.
    for expr in [
        "solve((x-a)*(x-b)<0, x)",
        "solve((x-a)*(x-b)>0, x)",
        "solve((x-a)*(x-b)<=0, x)",
        "solve((x-a)*(x-b)>=0, x)",
        "solve((x-z)*(x-a)<0, x)",
        "solve((x-a)*(x-b)*(x-c)<0, x)",
    ] {
        let (output, _code) = run_cli(&["eval", expr, "--format", "json"]);
        let wire = parse_wire(&output);
        assert_eq!(wire["ok"], false, "{expr} must decline, got: {output}");
        assert_eq!(wire["kind"], "SolverError", "{expr}: {output}");
    }
}
#[test]
fn parametric_endpoints_with_decidable_difference_order_exactly() {
    // El gemelo DECIDIBLE (misma ficha): los extremos `a−3` / `a+3` difieren en
    // la constante 6, así que el oráculo de diferencia polinómica los ordena
    // para TODO valor de `a` y el intervalo sale exacto e incondicional —
    // graduado, no declinado. Los vecinos numéricos/surds/mixtos siguen
    // resolviendo, y los moldes abs quedan intactos (F7).
    for (expr, expected) in [
        ("solve((x-(a-3))*(x-(a+3))<=0, x)", "[a - 3, a + 3]"),
        ("solve((x-2)*(x-1)<0, x)", "(1, 2)"),
        ("solve((x-sqrt(2))*(x-sqrt(3))<0, x)", "(sqrt(2), sqrt(3))"),
        ("solve((e^x-1)*(x-2)<0, x)", "(0, 2)"),
        ("solve(abs(x)<b, x)", "(-b, b) if b > 0"),
        ("solve(abs(x-a)<=3, x)", "[a - 3, a + 3]"),
    ] {
        let (output, _code) = run_cli(&["eval", expr, "--format", "json"]);
        let wire = parse_wire(&output);
        assert_eq!(wire["ok"], true, "{expr} must solve, got: {output}");
        assert_eq!(wire["result"], expected, "{expr}: {output}");
    }
}
/// La franja mixta de la cancelación de factoriales: un lado bajo el techo de
/// plegado (1000) y el otro encima, con gap cancelable. El fold del hijo
/// destruía el par antes de que la regla DIV lo viera: `1001!/1000!` producía
/// `(1/1000!)·1001!` con 1000! materializado a 2568 dígitos. El gate declina
/// ese fold usando EL MISMO predicado que acepta el brazo numérico de la
/// regla, así que un par no plegado siempre se cancela después.
#[test]
fn mixed_band_factorial_ratio_cancels_instead_of_folding_one_side() {
    let (output, _code) = run_cli(&["eval", "1001!/1000!", "--format", "json", "--steps", "on"]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "1001", "in: {output}");
    assert_eq!(
        wire["required_display"].as_array().map(Vec::len),
        Some(0),
        "decidable literal condition must discharge: {output}"
    );
    let steps = wire["steps"].as_array().expect("steps");
    assert!(
        steps
            .iter()
            .any(|s| s["rule"] == "Cancelar factoriales consecutivos"),
        "expected ratio narration: {output}"
    );

    // Invertida: denominador mayor.
    let (output, _code) = run_cli(&["eval", "1000!/1001!", "--format", "json"]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "1/1001", "in: {output}");

    // Gap ancho dentro del techo (600): falling factorial exacto de 1845
    // dígitos, referencia externa (Python), sin materializar 1500! ni 900!.
    let (output, _code) = run_cli(&[
        "eval",
        "1500!/900!",
        "--format",
        "json",
        "--max-chars",
        "2000",
    ]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result_chars"], 1845, "in: {output}");
    let result = wire["result"].as_str().expect("result");
    assert!(
        result.starts_with("71260560833908506616"),
        "head: {}",
        &result[..30.min(result.len())]
    );

    // Fuera de la franja no cambia nada: pares pequeños siguen en el fold
    // por lados (misma narración de siempre), y el factorial SUELTO bajo el
    // techo sigue plegando.
    let (output, _code) = run_cli(&["eval", "1000!/999!", "--format", "json"]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "1000", "in: {output}");

    let (output, _code) = run_cli(&["eval", "5!/3!", "--format", "json"]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "20", "in: {output}");

    let (output, _code) = run_cli(&["eval", "1000!", "--format", "json"]);
    let wire = parse_wire(&output);
    assert_eq!(
        wire["result_chars"], 2568,
        "bare 1000! must still fold: {output}"
    );
}
/// Contagio bignum: un producto/cociente que mezcla un literal GIGANTE ya
/// materializado (solo bignum()/combinatoria grande los crea) con un operando
/// numérico simbólico completa la aritmética exacta materializando el nodo
/// entero bajo los gates de bignum. Sobre el techo queda mixto, honesto. El
/// guard de tamaño en el fold de potencias acota el primo de F13 que este
/// flujo destapaba ((2^123456)^500 pasaba el cap de exponente hacia 18M de
/// dígitos).
#[test]
fn giant_literal_contagion_completes_exact_arithmetic() {
    // División: numerador coprimo → racional exacta completa (37164 + / + 863).
    let (output, _code) = run_cli(&[
        "eval",
        "bignum(2^123456)/5^1234",
        "--format",
        "json",
        "--max-chars",
        "100",
    ]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result_chars"], 38028, "in: {output}");
    assert!(
        wire["result"]
            .as_str()
            .expect("result")
            .starts_with("91021647594683810219"),
        "in: {output}"
    );

    // Producto: colapsa a UN entero (2^124456).
    let (output, _code) = run_cli(&[
        "eval",
        "bignum(2^123456)*2^1000",
        "--format",
        "json",
        "--max-chars",
        "100",
    ]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result_chars"], 37465, "in: {output}");

    // Cancelación exacta a través del contagio.
    let (output, _code) = run_cli(&["eval", "bignum(2^123456)/2^123456", "--format", "json"]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "1", "in: {output}");

    // Sobre el techo (8.6M dígitos): mixto intacto, sin minutos de CPU.
    let (output, _code) = run_cli(&[
        "eval",
        "bignum(2^123456)/5^12345678",
        "--format",
        "json",
        "--max-chars",
        "100",
    ]);
    let wire = parse_wire(&output);
    let result = wire["result"].as_str().expect("result");
    assert!(
        result.contains("5^12345678") || wire["result_truncated"] == true,
        "mixed form must survive over the ceiling: {output}"
    );

    // Guard del fold de potencias: base gigante ^500 queda simbólica.
    let (output, _code) = run_cli(&[
        "eval",
        "bignum(2^123456)^500",
        "--format",
        "json",
        "--max-chars",
        "100",
    ]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result_chars"], 37168, "^500 stays symbolic: {output}");
}
#[test]
fn inverse_trig_affine_unit_interval_domains_display_solved_closed_interval() {
    for (expr, result, display, bounded) in [
        ("arcsin(2*x - 1)", "arcsin(2·x - 1)", "0 ≤ x ≤ 1", "x - x^2"),
        ("arccos(1 - x)", "arccos(1 - x)", "0 ≤ x ≤ 2", "2·x - x^2"),
    ] {
        let (output, code) = run_cli(&["eval", expr, "--format", "json"]);
        assert_eq!(
            code, 0,
            "expected successful CLI exit for {expr}, got {code}: {output}"
        );
        let wire = parse_wire(&output);

        assert_eq!(wire["result"], result);
        assert_eq!(wire["required_display"], json!([display]));

        let required_conditions = wire["required_conditions"]
            .as_array()
            .expect("required_conditions");
        assert!(
            required_conditions
                .iter()
                .any(|condition| condition["kind"] == "NonNegative"
                    && condition["expr_canonical"] == bounded),
            "structured bounded-domain guard should remain for {expr}: {required_conditions:?}"
        );
    }
}
/// Los substeps de suma de fracciones (denominador común) publican el título
/// LOCALIZADO vía `desc_key` y LaTeX declarado en `*_latex`, con `before`/`after`
/// en texto plano. El defecto que esto pinnea: títulos hardcodeados en inglés
/// ("Find common denominator: 6") ignorando `--lang`, y LaTeX crudo en los
/// campos plain — el wire lo escapaba como texto (`\text{\unicode{x5C}frac…}`)
/// y la web mostraba símbolos en vez de fracciones renderizadas.
#[test]
fn fraction_sum_substeps_are_localized_and_carry_clean_latex() {
    for (lang_args, common_title, sum_title) in [
        (
            &[][..],
            "Llevar a denominador común: 6",
            "Sumar las fracciones",
        ),
        (
            &["--lang", "en"][..],
            "Put over a common denominator: 6",
            "Add the fractions",
        ),
    ] {
        let mut args = vec!["eval", "1/2+1/3", "--format", "json", "--steps", "on"];
        args.extend_from_slice(lang_args);
        let (output, _code) = run_cli(&args);
        let wire = parse_wire(&output);

        let steps = wire["steps"].as_array().expect("steps array");
        let substeps = steps
            .iter()
            .find_map(|s| {
                let subs = s["substeps"].as_array()?;
                (!subs.is_empty()).then_some(subs)
            })
            .expect("a step with substeps");
        assert_eq!(substeps.len(), 2, "expected 2 substeps: {output}");

        assert_eq!(substeps[0]["title"], common_title, "in: {output}");
        assert_eq!(substeps[0]["before"], "1/2 + 1/3");
        assert_eq!(substeps[0]["after"], "3/6 + 2/6");
        assert_eq!(
            substeps[0]["before_latex"], "\\frac{1}{2} + \\frac{1}{3}",
            "declared LaTeX must reach the wire untouched: {output}"
        );
        assert_eq!(substeps[0]["after_latex"], "\\frac{3}{6} + \\frac{2}{6}");

        assert_eq!(substeps[1]["title"], sum_title, "in: {output}");
        assert_eq!(substeps[1]["before"], "3/6 + 2/6");
        assert_eq!(substeps[1]["after"], "5/6");
        assert_eq!(substeps[1]["before_latex"], "\\frac{3}{6} + \\frac{2}{6}");
        assert_eq!(substeps[1]["after_latex"], "\\frac{5}{6}");

        // Ningún lado publica el escape-de-texto sobre LaTeX (el síntoma web).
        for sub in substeps {
            for field in ["before_latex", "after_latex"] {
                let side = sub[field].as_str().expect("latex side");
                assert!(
                    !side.contains("\\text") && !side.contains("\\unicode"),
                    "{field} leaked text-escaped LaTeX: {side}"
                );
            }
        }
    }
}
/// El par negado SIMBÓLICO: `n!/(n+1)!` → `1/(n+1)` con la condición de
/// dominio sobre el factorial más corto (`n ≥ 0`). Antes solo se cancelaba la
/// dirección con numerador mayor.
#[test]
fn inverted_symbolic_factorial_ratio_cancels_to_reciprocal() {
    let (output, _code) = run_cli(&["eval", "n!/(n+1)!", "--format", "json", "--no-pretty"]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "1 / (n + 1)", "in: {output}");
    let required: Vec<String> = wire["required_display"]
        .as_array()
        .expect("required array")
        .iter()
        .map(|v| v.as_str().unwrap_or_default().to_string())
        .collect();
    assert!(
        required.iter().any(|c| c == "n ≥ 0"),
        "domain condition on the shorter factorial must survive: {required:?} in {output}"
    );
}
