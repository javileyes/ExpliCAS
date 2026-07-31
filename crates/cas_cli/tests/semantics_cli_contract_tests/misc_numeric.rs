use super::*;

/// La cancelación de factoriales consecutivos también cubre el par NUMÉRICO
/// por encima del techo de const-fold (1000!): `12345!/12344!` se cancela como
/// falling factorial exacto sin materializar ningún factorial, en ambas
/// direcciones. El defecto que esto pinnea: el matcher solo veía offsets
/// estructurales (`n+1` vs `n`) y dos literales quedaban sin simplificar.
#[test]
fn numeric_factorial_ratio_cancels_without_materializing_factorials() {
    // Dirección directa: gap 1 → el propio numerador.
    let (output, _code) = run_cli(&["eval", "12345!/12344!", "--format", "json", "--steps", "on"]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "12345", "in: {output}");
    // La condición NonNegative(12344) es decidible y debe evaporarse.
    assert_eq!(
        wire["required_display"].as_array().map(Vec::len),
        Some(0),
        "no spurious conditions: {output}"
    );
    let steps = wire["steps"].as_array().expect("steps array");
    assert!(
        steps
            .iter()
            .any(|s| s["rule"] == "Cancelar factoriales consecutivos"),
        "expected the consecutive-factorial rule to narrate: {output}"
    );

    // Par negado: denominador mayor → racional exacto.
    let (output, _code) = run_cli(&["eval", "12344!/12345!", "--format", "json"]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "1/12345", "in: {output}");

    // Gap 2 → producto de dos factores, plegado.
    let (output, _code) = run_cli(&["eval", "12345!/12343!", "--format", "json"]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "152386680", "in: {output}");

    // Gap astronómico: se queda simbólico (el 3! sí se pliega a 6).
    let (output, _code) = run_cli(&["eval", "12345!/3!", "--format", "json"]);
    let wire = parse_wire(&output);
    let result = wire["result"].as_str().expect("result string");
    assert!(
        result.contains("12345!"),
        "astronomical span must stay unexpanded: {output}"
    );

    // Par pequeño: lo sigue plegando la evaluación normal de factoriales.
    let (output, _code) = run_cli(&["eval", "5!/3!", "--format", "json"]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "20", "in: {output}");
}
/// `approx` de magnitudes fuera del alcance de f64 (potencias y factoriales
/// astronómicos) produce notación científica `mantisa·10^k` por aritmética
/// exacta con cota de error (carril sci), sin materializar el número. El f64
/// sigue siendo dueño de todo su rango histórico. Referencias verificadas
/// externamente con aritmética decimal de 60 dígitos / factorial exacto.
#[test]
fn approx_big_magnitude_produces_scientific_notation() {
    // Potencia gigante: el caso que motivó el carril.
    let (output, _code) = run_cli(&["eval", "approx(5^123456789)", "--format", "json"]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "2.20110600528·10^86292592", "in: {output}");
    assert_eq!(
        wire["result_latex"], "2.20110600528\\cdot {10}^{86292592}",
        "in: {output}"
    );
    assert_eq!(
        wire["required_display"].as_array().map(Vec::len),
        Some(0),
        "approximation must not emit conditions: {output}"
    );

    // Recíproca: exponente decimal negativo (texto plano usa la forma dividida).
    let (output, _code) = run_cli(&["eval", "approx(5^-123456789)", "--format", "json"]);
    let wire = parse_wire(&output);
    assert_eq!(
        wire["result_latex"], "4.5431705588\\cdot {10}^{-86292593}",
        "in: {output}"
    );

    // Factorial más allá del techo de plegado (1000!): dígitos exactos.
    let (output, _code) = run_cli(&["eval", "approx(12345!)", "--format", "json"]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "3.44364246919·10^45150", "in: {output}");

    // Factorial YA materializado (1000! pliega a 2568 dígitos): carril directo.
    let (output, _code) = run_cli(&["eval", "approx(1000!)", "--format", "json"]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "4.02387260077·10^2567", "in: {output}");

    // Producto compuesto dentro de la gramática del carril.
    let (output, _code) = run_cli(&["eval", "approx(3*5^123456789)", "--format", "json"]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "6.60331801585·10^86292592", "in: {output}");

    // evalf es el mismo brazo.
    let (output, _code) = run_cli(&["eval", "evalf(1000000!)", "--format", "json"]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "8.26393168833·10^5565708", "in: {output}");
}
/// La frontera f64 queda cerrada por ambos lados: justo dentro (2^1023) sigue
/// siendo del carril f64 histórico; justo fuera (2^1024, exp10 = 308 pero
/// > f64::MAX) la sirve el carril exacto sin hueco residual.
#[test]
fn approx_f64_borderline_has_no_residual_gap() {
    let (output, _code) = run_cli(&["eval", "approx(2^1023)", "--format", "json"]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "8.98846567431×10^307", "in: {output}");

    let (output, _code) = run_cli(&["eval", "approx(2^1024)", "--format", "json"]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "1.79769313486·10^308", "in: {output}");

    // El rango f64 clásico no cambia ni un glifo.
    let (output, _code) = run_cli(&["eval", "approx(2^100)", "--format", "json"]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "1.26765060023×10^30", "in: {output}");

    // Residual honesto fuera de la gramática (base simbólica).
    let (output, _code) = run_cli(&["eval", "approx(x^123456789)", "--format", "json"]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "approx(x^123456789)", "in: {output}");
}
/// `bignum(x)` materializa el valor exacto de expresiones numéricas gigantes
/// (potencias enteras, factoriales) con presupuesto previo por tamaño: la
/// decisión de coste se toma de los bits/Stirling ANTES de multiplicar nada.
/// Sobre el techo (~600k dígitos) queda como residual instantáneo — la
/// alternativa es approx(). Referencias externas (Python exacto).
#[test]
fn bignum_materializes_exact_giants_with_size_gate() {
    // 2^123456 = 37.164 dígitos: entero exacto, sin truncar con max-chars alto.
    let (output, _code) = run_cli(&[
        "eval",
        "bignum(2^123456)",
        "--format",
        "json",
        "--max-chars",
        "40000",
    ]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result_chars"], 37164, "in: {output}");
    assert_eq!(wire["result_truncated"], false);
    let result = wire["result"].as_str().expect("result");
    assert!(
        result.starts_with("91021647594683810219"),
        "head: {}",
        &result[..30.min(result.len())]
    );
    // 37k chars < cap de LaTeX (50k): el latex existe para la web.
    assert!(wire["result_latex"].is_string(), "latex expected: {output}");

    // Con el max-chars por defecto (2000) el mismo valor llega truncado,
    // con el conteo completo y Sin latex.
    let (output, _code) = run_cli(&["eval", "bignum(2^123456)", "--format", "json"]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result_truncated"], true);
    assert_eq!(wire["result_chars"], 37164);
    assert!(wire["result_latex"].is_null(), "no latex when truncated");

    // 60.206 dígitos: por encima del cap de LaTeX aunque quepa en el wire —
    // el texto plano viaja entero y MathJax no recibe 60k tokens.
    let (output, _code) = run_cli(&[
        "eval",
        "bignum(2^200000)",
        "--format",
        "json",
        "--max-chars",
        "70000",
    ]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result_truncated"], false);
    assert_eq!(wire["result_chars"], 60206);
    assert!(
        wire["result_latex"].is_null(),
        "latex must be capped past 50k chars: {output}"
    );

    // Consistencia con el fold pequeño existente.
    let (output, _code) = run_cli(&["eval", "bignum(100!)", "--format", "json"]);
    let wire = parse_wire(&output);
    let (folded, _code) = run_cli(&["eval", "100!", "--format", "json"]);
    let folded_wire = parse_wire(&folded);
    assert_eq!(
        wire["result"], folded_wire["result"],
        "bignum(100!) == 100!"
    );

    // Sobre el techo: residual instantáneo, jamás minutos de multiplicación.
    for over in ["bignum(5^123456789)", "bignum(300000!)", "bignum(1000000!)"] {
        let (output, _code) = run_cli(&["eval", over, "--format", "json"]);
        let wire = parse_wire(&output);
        let result = wire["result"].as_str().expect("result");
        assert!(
            result.starts_with("bignum("),
            "{over} must stay residual: {output}"
        );
    }

    // Fuera de gramática: simbólico intacto.
    let (output, _code) = run_cli(&["eval", "bignum(x!)", "--format", "json"]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "bignum(x!)", "in: {output}");
}
/// `result_approx` (la lectura `≈` numérica de un resultado EXACTO) es un
/// campo del wire estrictamente opt-in: aparece solo con `--approx-hint`
/// (la web lo envía; los tests jamás), en ASCII re-parseable (`*10^`,
/// exponente negativo entre paréntesis), y solo cuando aporta información:
/// resultado cerrado, no ya-decimal, y distinto de su render exacto.
#[test]
fn approx_hint_field_is_opt_in_and_informative() {
    // Racional exacto → decimal de 12 dígitos.
    let (output, _code) = run_cli(&["eval", "5/6", "--approx-hint", "--format", "json"]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "5/6", "in: {output}");
    assert_eq!(wire["result_approx"], "0.833333333333", "in: {output}");

    // Cerrado simbólico (constantes trascendentales).
    let (output, _code) = run_cli(&["eval", "sqrt(2)+pi", "--approx-hint", "--format", "json"]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result_approx"], "4.55580621596", "in: {output}");

    // Entero gigante materializado: notación científica ASCII.
    let (output, _code) = run_cli(&[
        "eval",
        "2^500",
        "--approx-hint",
        "--format",
        "json",
        "--max-chars",
        "200",
    ]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result_approx"], "3.2733906079*10^150", "in: {output}");

    // Más allá de f64: el carril sci exacto toma el relevo (simbólico intacto
    // + su lectura numérica), con paréntesis en el exponente negativo.
    let (output, _code) = run_cli(&["eval", "5^123456789", "--approx-hint", "--format", "json"]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "5^123456789", "in: {output}");
    assert_eq!(
        wire["result_approx"], "2.20110600528*10^86292592",
        "in: {output}"
    );

    let (output, _code) = run_cli(&["eval", "5^-123456789", "--approx-hint", "--format", "json"]);
    let wire = parse_wire(&output);
    assert_eq!(
        wire["result_approx"], "4.5431705588*10^(-86292593)",
        "in: {output}"
    );

    // Sin información nueva → ausente: entero pequeño idéntico a su render...
    let (output, _code) = run_cli(&["eval", "1001", "--approx-hint", "--format", "json"]);
    let wire = parse_wire(&output);
    assert!(wire.get("result_approx").is_none(), "in: {output}");

    // ...resultado que YA es presentación numérica (nodo decimal)...
    let (output, _code) = run_cli(&["eval", "approx(pi)", "--approx-hint", "--format", "json"]);
    let wire = parse_wire(&output);
    assert!(wire.get("result_approx").is_none(), "in: {output}");

    // ...y variables libres (sin lectura numérica cerrada).
    let (output, _code) = run_cli(&["eval", "x+1", "--approx-hint", "--format", "json"]);
    let wire = parse_wire(&output);
    assert!(wire.get("result_approx").is_none(), "in: {output}");

    // El pin de coste: SIN el flag el campo no existe — ni para el racional
    // que con flag sí lo emite. Los suites nunca pagan este cómputo.
    let (output, _code) = run_cli(&["eval", "5/6", "--format", "json"]);
    let wire = parse_wire(&output);
    assert!(wire.get("result_approx").is_none(), "in: {output}");
}
/// `bignum_available` (la oferta del botón «Calcular bignum») comparte el
/// opt-in de los hints Y los gates de tamaño de bignum: solo aparece para
/// resultados numéricos SIMBÓLICOS que la materialización aceptaría. Nunca
/// promete lo que bignum rechazaría, y nunca existe sin el flag.
#[test]
fn bignum_available_mirrors_materialization_gates() {
    for (expr, expected) in [
        ("2^1234567", true),  // gigante simbólico bajo el techo
        ("12345!", true),     // ídem factorial
        ("300000!", false),   // sobre el techo (1.5M dígitos)
        ("5^123456789", false), // 86M dígitos
        ("2^500", false),     // ya materializado por el fold
        ("sqrt(2)", false),   // fuera de gramática
        ("x^123456789", false), // variable libre
    ] {
        let (output, _code) = run_cli(&["eval", expr, "--approx-hint", "--format", "json"]);
        let wire = parse_wire(&output);
        assert_eq!(
            wire.get("bignum_available").is_some(),
            expected,
            "{expr}: in {output}"
        );
    }

    // Sin flag: ausente incluso para el candidato perfecto.
    let (output, _code) = run_cli(&["eval", "2^1234567", "--format", "json"]);
    let wire = parse_wire(&output);
    assert!(wire.get("bignum_available").is_none(), "in: {output}");
}
