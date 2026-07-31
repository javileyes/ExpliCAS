use super::*;

#[test]
fn dsolve_separable_o0_contract() {
    // Fase 4 · O0: sustrato dsolve — separables con verificación-gate (D5),
    // residual honesto (D8), anti-colapso (D4) y never-fabricate (Z1-Z7).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input])
            .output()
            .expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    };
    let first_line = |input: &str| -> String {
        r(input)
            .lines()
            .next()
            .unwrap_or_default()
            .trim()
            .to_string()
    };
    let err_of = |input: &str| -> String {
        let out = cli()
            .args(["eval", input])
            .output()
            .expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stderr).to_string() + &String::from_utf8_lossy(&out.stdout)
    };

    // --- Graduados S1-S9 + L12 (formas textbook; verificadas a Number(0)
    // exacto antes de emitir — la emisión ES el certificado).
    assert_eq!(
        first_line("dsolve(diff(y,x)=x*y, y, x)"),
        "y = C·e^(x^2 / 2)"
    ); // S1
    assert_eq!(
        first_line("dsolve(diff(y,x)=y^2, y, x)"),
        "y = -1 / (C + x)"
    ); // S2
    assert_eq!(
        first_line("dsolve(diff(y,x)=1+y^2, y, x)"),
        "arctan(y) - x = C"
    ); // S3 implícita honesta (despeje arctan: gap del solve, peldaño nombrado)
    assert_eq!(first_line("dsolve(diff(y,x)=-x/y, y, x)"), "x^2 + y^2 = C"); // S4 implícita textbook
    assert_eq!(first_line("dsolve(diff(y,x)=x/y, y, x)"), "y^2 - x^2 = C"); // S5 implícita (no pierde la rama ±)
    assert_eq!(first_line("dsolve(diff(y,x)=y/x, y, x)"), "y = C·x"); // S6
    assert_eq!(
        first_line("dsolve(diff(y,x)=2*x*y^2, y, x)"),
        "y = -1 / (x^2 + C)"
    ); // S7
    assert_eq!(first_line("dsolve(diff(y,x)=-y, y, x)"), "y = C / e^x"); // L12 (≡ C·e^(−x))
    assert_eq!(first_line("dsolve(diff(y,x)=k*y, y, x)"), "y = C·e^(k·x)"); // S8 paramétrico
    assert_eq!(
        first_line("dsolve(diff(y,x)=cos(x), y, x)"),
        "y = sin(x) + C"
    ); // S9 integración directa

    // Azúcar aridad-2 (pregunta resuelta #1): misma salida que la canónica.
    assert_eq!(first_line("dsolve(diff(y,x)=x*y, y)"), "y = C·e^(x^2 / 2)");

    // Warnings didácticos D7/D12 presentes (constante arbitraria; singular).
    // Los warnings viajan por stderr: capturar ambos streams.
    let s1_full = err_of("dsolve(diff(y,x)=x*y, y, x)");
    assert!(s1_full.contains("constante arbitraria"), "{s1_full}");
    assert!(s1_full.contains("solución singular"), "{s1_full}");

    // --- Z1-Z7 never-fabricate: residual honesto eco re-emitible, jamás valor.
    for z in [
        "dsolve(diff(y,x)=x^2+y^2, y, x)",  // Z1 Riccati
        "dsolve(diff(y,x,2)+x*y=0, y, x)",  // Z2 Airy
        "dsolve(diff(y,x)=sin(x*y), y, x)", // Z3 sin método
        "dsolve(diff(y,x,2)=y^2, y, x)",    // Z4 no-lineal
        "dsolve(diff(y,x)=y^2-x, y, x)",    // Z5 Riccati
        "dsolve(x^2*diff(y,x,2)+x*diff(y,x)+(x^2-1)*y=0, y, x)", // Z6 Bessel
        "dsolve(diff(y,x,2)+sin(y)=0, y, x)", // Z7 péndulo
    ] {
        let out = r(z);
        assert!(
            out.starts_with("dsolve("),
            "never-fabricate: {z} debe declinar a eco residual, no a valor: {out}"
        );
        // El colapso sería el output ENTERO `y = 0` / `{ 0 }` (el eco legítimo
        // puede contener `= 0` como parte de la EDO re-emitida).
        let first = out.lines().next().unwrap_or_default().trim();
        assert!(
            first != "y = 0" && first != "{ 0 }",
            "never-fabricate: {z} jamás emite el colapso: {out}"
        );
    }

    // --- Metamórfico anti-colapso (D4): dsolve JAMÁS emite las formas del
    // colapso de diff, y los contratos de diff/solve planos SIGUEN intactos.
    let s2 = r("dsolve(diff(y,x)=y^2, y, x)");
    assert!(!s2.contains("{ 0 }"), "anti-colapso: {s2}");
    assert_eq!(
        first_line("diff(y,x)"),
        "0",
        "contrato de diff plano intacto"
    );
    assert_eq!(
        first_line("solve(diff(y,x)=y, y)"),
        "{ 0 }",
        "contrato del solve plano (colapso conocido) intacto"
    );

    // --- No-colisión de prefijo solve(/dsolve( en ambas direcciones.
    assert_eq!(first_line("solve(x^2=4, x)"), "{ -2, 2 }");

    // --- Condiciones iniciales: GRADUARON en O3 (el pin de decline de O0
    // migró a resolución con su porqué — V30 pinea en el contrato O3; aquí
    // queda el pin de la RESOLUCIÓN básica).
    assert_eq!(
        first_line("dsolve(diff(y,x)=-y, y, x, y(0)=3)"),
        "y = 3 / e^x"
    );

    // --- 2º orden: GRADUÓ en O4 (el pin de decline migró a resolución).
    assert_eq!(
        first_line("dsolve(diff(y,x,2)+4*y=0, y, x)"),
        "y = C1·sin(2·x) + C2·cos(2·x)"
    );

    // --- Usage-errors del pre-pass (D2): malformado, sympy-style, ambiguo.
    assert!(err_of("dsolve(y, x)").contains("contains no diff"));
    assert!(err_of("dsolve(diff(y(x),x)=y(x), y(x))").contains("not y(x)"));
    assert!(err_of("dsolve(diff(y,x)+diff(y,t)=0, y)").contains("Ambiguous"));
    assert!(err_of("dsolve(diff(y,t)=y, y, x)").contains("with respect to t"));

    // --- Narración D13 keyed es/en: rule names localizan en ambos idiomas.
    let steps_of = |input: &str, lang: Option<&str>| -> String {
        let mut args = vec!["eval", input, "--steps", "on", "--format", "json"];
        if let Some(l) = lang {
            args.extend(["--lang", l]);
        }
        let out = cli().args(&args).output().expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stdout).to_string()
    };
    let es = steps_of("dsolve(diff(y,x)=x*y, y, x)", None);
    assert!(es.contains("Identificar EDO separable"), "{es}");
    assert!(es.contains("Integrar ambos lados"), "{es}");
    assert!(es.contains("Verificar por sustituci"), "{es}");
    let en = steps_of("dsolve(diff(y,x)=x*y, y, x)", Some("en"));
    assert!(en.contains("Identify separable ODE"), "{en}");
    assert!(en.contains("Integrate both sides"), "{en}");
    assert!(en.contains("Verify by substitution"), "{en}");

    // --- Round-trip storability (guardrail #5): la solución wrap_eq se
    // almacena como #N y se recupera sin colapsar ni declinar.
    let repl_out = cli()
        .arg("repl")
        .write_stdin("dsolve(diff(y,x)=x*y, y, x)\n#1\nexit\n")
        .output()
        .expect("Failed to run CLI repl");
    let repl_text = String::from_utf8_lossy(&repl_out.stdout).to_string();
    let occurrences = repl_text.matches("y = C·e^(x^2 / 2)").count();
    assert!(
        occurrences >= 2,
        "round-trip #1: la solución debe emitirse en dsolve Y en el recall: {repl_text}"
    );
}
#[test]
fn dsolve_linear_first_order_o1_contract() {
    // Fase 4 · O1: lineal de primer orden por factor integrante μ = e^(∫p dx),
    // con la emisión gateada por verificación (D5) y el pin D12 de μ (L9:
    // μ = x, jamás |x|).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input])
            .output()
            .expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    };
    let first_line = |input: &str| -> String {
        r(input)
            .lines()
            .next()
            .unwrap_or_default()
            .trim()
            .to_string()
    };
    let err_of = |input: &str| -> String {
        let out = cli()
            .args(["eval", input])
            .output()
            .expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stderr).to_string() + &String::from_utf8_lossy(&out.stdout)
    };

    // L8-L11 (formas split textbook: término particular + C/μ).
    assert_eq!(
        first_line("dsolve(diff(y,x)+y=x, y, x)"),
        "y = C / e^x + x - 1"
    ); // L8
    assert_eq!(
        first_line("dsolve(diff(y,x)+y/x=x^2, y, x)"),
        "y = C / x + 1/4·x^3"
    ); // L9 — el PIN μ-display: μ = x (sin |x| en ninguna parte)
    assert!(
        !r("dsolve(diff(y,x)+y/x=x^2, y, x)").contains('|'),
        "L9: μ=x, no |x|"
    );
    assert_eq!(
        first_line("dsolve(diff(y,x)-y=exp(x), y, x)"),
        "y = C·e^x + x·e^x"
    ); // L10 resonancia 1er orden
    assert_eq!(
        first_line("dsolve(diff(y,x)+2*y=sin(x), y, x)"),
        "y = 1/5·(2·sin(x) - cos(x)) + C / e^(2·x)"
    ); // L11
       // Forma reordenada `y' = x − y` (el matcher lineal la captura tras el
       // decline del separable).
    assert_eq!(
        first_line("dsolve(diff(y,x)=x-y, y, x)"),
        "y = C / e^x + x - 1"
    );
    // Homogénea lineal y coeficiente no-unitario (pelar coef, lección 2026-07-08b).
    assert_eq!(first_line("dsolve(diff(y,x)+y=0, y, x)"), "y = C / e^x");
    assert_eq!(
        first_line("dsolve(2*diff(y,x)+4*y=x, y, x)"),
        "y = 1/4·(x - 1/2) + C / e^(2·x)"
    );
    // Coeficiente variable a(x) = x y parámetro simbólico k.
    assert_eq!(
        first_line("dsolve(x*diff(y,x)+y=x, y, x)"),
        "y = C / x + 1/2·x"
    );
    assert_eq!(
        first_line("dsolve(diff(y,x)+k*y=0, y, x)"),
        "y = C / e^(k·x)"
    );

    // No-lineales declinan honesto nombrando los ciclos dueños.
    for probe in [
        "dsolve(y*diff(y,x)+y=x, y, x)",
        "dsolve(diff(y,x)+y^2=x, y, x)",
    ] {
        let out = r(probe);
        assert!(out.starts_with("dsolve("), "no-lineal declina a eco: {out}");
        assert!(
            err_of(probe).contains("método clásico"),
            "decline nombra el contrato residual"
        );
    }

    // Pins de no-robo: los separables S1/S2 siguen byte-idénticos (el
    // dispatcher prueba separable ANTES de lineal).
    assert_eq!(
        first_line("dsolve(diff(y,x)=x*y, y, x)"),
        "y = C·e^(x^2 / 2)"
    );
    assert_eq!(
        first_line("dsolve(diff(y,x)=y^2, y, x)"),
        "y = -1 / (C + x)"
    );

    // Narración lineal keyed es/en.
    let steps_of = |input: &str, lang: Option<&str>| -> String {
        let mut args = vec!["eval", input, "--steps", "on", "--format", "json"];
        if let Some(l) = lang {
            args.extend(["--lang", l]);
        }
        let out = cli().args(&args).output().expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stdout).to_string()
    };
    let es = steps_of("dsolve(diff(y,x)+y=x, y, x)", None);
    assert!(es.contains("Identificar forma lineal"), "{es}");
    assert!(es.contains("factor integrante"), "{es}");
    let en = steps_of("dsolve(diff(y,x)+y=x, y, x)", Some("en"));
    assert!(en.contains("Identify linear form"), "{en}");
    assert!(en.contains("integrating factor"), "{en}");
}
#[test]
fn dsolve_exact_o2_contract() {
    // Fase 4 · O2: exactas M + N·y' = 0 por la maquinaria de potencial F6
    // (nivel 1 poly_eq) + fallback full-eval del caller (nivel 2, D11) — la
    // emisión gateada POR COMPONENTE: ∂φ/∂x−M → 0 y ∂φ/∂y−N → 0 exactos.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input])
            .output()
            .expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    };
    let first_line = |input: &str| -> String {
        r(input)
            .lines()
            .next()
            .unwrap_or_default()
            .trim()
            .to_string()
    };
    let err_of = |input: &str| -> String {
        let out = cli()
            .args(["eval", input])
            .output()
            .expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stderr).to_string() + &String::from_utf8_lossy(&out.stdout)
    };

    // E14/E15: exactas polinomiales puras (camino exacto nivel 1).
    assert_eq!(
        first_line("dsolve((2*x*y+1) + (x^2+2*y)*diff(y,x) = 0, y, x)"),
        "y·x^2 + y^2 + x = C"
    );
    assert_eq!(
        first_line("dsolve((3*x^2+2*y) + (2*x+3*y^2)*diff(y,x) = 0, y, x)"),
        "x^3 + y^3 + 2·x·y = C"
    );
    // Trascendente exacta (nivel 2 D11: el fallback full-eval del caller —
    // poly_eq no verifica e^y; el evaluador completo sí).
    assert_eq!(
        first_line("dsolve(e^y + (x*e^y+2*y)*diff(y,x) = 0, y, x)"),
        "x·e^y + y^2 = C"
    );
    let exact_warn = err_of("dsolve((2*x*y+1) + (x^2+2*y)*diff(y,x) = 0, y, x)");
    assert!(exact_warn.contains("potencial del campo"), "{exact_warn}");

    // E13 y la E-neg del catálogo se resuelven ANTES por el camino lineal
    // (formas explícitas equivalentes a las implícitas del catálogo — el
    // dispatcher separable→lineal→exacta no roba: gana el método más simple).
    assert_eq!(
        first_line("dsolve(2*x*y + x^2*diff(y,x) = 0, y, x)"),
        "y = C / x^2"
    );
    assert_eq!(
        first_line("dsolve(y + 2*x*diff(y,x) = 0, y, x)"),
        "y = C / sqrt(2·x)"
    );

    // La no-exacta (y+xy²)+x·y' = 0 GRADUÓ en O8: es Bernoulli n=2 (el pin
    // de decline migró a resolución — el camino exacto sigue declinando y el
    // dispatcher cae a Bernoulli).
    assert_eq!(
        first_line("dsolve((y + x*y^2) + x*diff(y,x) = 0, y, x)"),
        "y = 1 / (x·ln(x) + C·x)"
    );

    // Pins de no-robo: separable y lineal byte-idénticos.
    assert_eq!(
        first_line("dsolve(diff(y,x)=x*y, y, x)"),
        "y = C·e^(x^2 / 2)"
    );
    assert_eq!(
        first_line("dsolve(diff(y,x)+y=x, y, x)"),
        "y = C / e^x + x - 1"
    );

    // Pins F6: los verbos potential() NO cambian (metamórfico del dueño).
    assert_eq!(first_line("potential([2*x*y, x^2], [x, y])"), "y·x^2");
    let pot_trig = first_line("potential([cos(x), -sin(y)], [x, y])");
    assert!(
        pot_trig.starts_with("potential("),
        "potential trig sigue residual (el upgrade nivel-2 vive en dsolve, no en el verbo): {pot_trig}"
    );

    // Narración exacta keyed es/en.
    let steps_of = |input: &str, lang: Option<&str>| -> String {
        let mut args = vec!["eval", input, "--steps", "on", "--format", "json"];
        if let Some(l) = lang {
            args.extend(["--lang", l]);
        }
        let out = cli().args(&args).output().expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stdout).to_string()
    };
    let es = steps_of("dsolve((2*x*y+1) + (x^2+2*y)*diff(y,x) = 0, y, x)", None);
    assert!(es.contains("Identificar forma exacta"), "{es}");
    assert!(es.contains("Comprobar exactitud"), "{es}");
    assert!(es.contains("Reconstruir el potencial"), "{es}");
    let en = steps_of(
        "dsolve((2*x*y+1) + (x^2+2*y)*diff(y,x) = 0, y, x)",
        Some("en"),
    );
    assert!(en.contains("Identify exact form"), "{en}");
    assert!(en.contains("Check exactness"), "{en}");
}
#[test]
fn dsolve_initial_conditions_o3_contract() {
    // Fase 4 · O3: la condición inicial fija la constante de la general con
    // verificación DOBLE (la particular verifica la EDO y la condición) antes
    // de emitir; toda inconsistencia declina honesto.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input])
            .output()
            .expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    };
    let first_line = |input: &str| -> String {
        r(input)
            .lines()
            .next()
            .unwrap_or_default()
            .trim()
            .to_string()
    };
    let err_of = |input: &str| -> String {
        let out = cli()
            .args(["eval", input])
            .output()
            .expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stderr).to_string() + &String::from_utf8_lossy(&out.stdout)
    };

    // V30/V32/V33 (separable y lineal, explícitas).
    assert_eq!(
        first_line("dsolve(diff(y,x)=-y, y, x, y(0)=3)"),
        "y = 3 / e^x"
    );
    assert_eq!(
        first_line("dsolve(diff(y,x)=x*y, y, x, y(0)=2)"),
        "y = 2·e^(x^2 / 2)"
    );
    assert_eq!(
        first_line("dsolve(diff(y,x)+y=x, y, x, y(0)=0)"),
        "y = 1 / e^x + x - 1"
    );
    // Implícita IVP: φ(x0,y0) fija C directamente (el círculo clásico).
    assert_eq!(
        first_line("dsolve(diff(y,x)=-x/y, y, x, y(0)=2)"),
        "x^2 + y^2 = 4"
    );
    // La particular NO lleva warning de constante arbitraria.
    let particular = err_of("dsolve(diff(y,x)=-y, y, x, y(0)=3)");
    assert!(
        !particular.contains("constante arbitraria"),
        "la particular no anuncia constante libre: {particular}"
    );

    // Inconsistente (la condición apunta a la singular y=0 descartada):
    // residual honesto, jamás fabricación.
    let bad = r("dsolve(diff(y,x)=y^2, y, x, y(0)=0)");
    assert!(bad.starts_with("dsolve("), "inconsistente declina: {bad}");
    assert!(err_of("dsolve(diff(y,x)=y^2, y, x, y(0)=0)").contains("inconsistente"));

    // Condición de derivada y'(x0) sobre una EDO de 1er orden: inválida por
    // ORDEN (una condición solo fija derivadas de orden menor que la EDO) —
    // el mensaje migró al graduar O4.
    let deriv = r("dsolve(diff(y,x)=-y, y, x, y'(0)=3)");
    assert!(deriv.starts_with("dsolve("), "y'(0) declina: {deriv}");
    assert!(err_of("dsolve(diff(y,x)=-y, y, x, y'(0)=3)").contains("orden MENOR"));

    // Dos condiciones en 1er orden: sobredeterminado → declina honesto.
    let two = r("dsolve(diff(y,x)=-y, y, x, y(0)=3, y(1)=1)");
    assert!(
        two.starts_with("dsolve("),
        "dos condiciones declinan: {two}"
    );

    // Condición malformada → usage-error explícito.
    assert!(err_of("dsolve(diff(y,x)=-y, y, x, y(0)3)").contains("Invalid dsolve"));

    // Pins de no-regresión: la general SIN condición byte-idéntica.
    assert_eq!(first_line("dsolve(diff(y,x)=-y, y, x)"), "y = C / e^x");
    // Pin D16: solve_system acepta ahora c1/C1 (constantes de O4) y los
    // nombres inválidos siguen inválidos.
    assert_eq!(
        first_line("solve_system(c1+c2=3; c1-c2=1; c1; c2)"),
        "{ c1 = 2, c2 = 1 }"
    );
    assert!(err_of("solve_system(a+b=2; a-b=0; 1a; b)").contains("Invalid variable name"));
    // Pin del molde solve_system intacto.
    assert_eq!(
        first_line("solve_system(a+b=3; 2*a-2*b=4; a; b)"),
        "{ a = 5/2, b = 1/2 }"
    );

    // Narración O3 keyed es/en.
    let steps_of = |input: &str, lang: Option<&str>| -> String {
        let mut args = vec!["eval", input, "--steps", "on", "--format", "json"];
        if let Some(l) = lang {
            args.extend(["--lang", l]);
        }
        let out = cli().args(&args).output().expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stdout).to_string()
    };
    let es = steps_of("dsolve(diff(y,x)=-y, y, x, y(0)=3)", None);
    assert!(es.contains("Aplicar la condición inicial"), "{es}");
    assert!(es.contains("Solución particular"), "{es}");
    let en = steps_of("dsolve(diff(y,x)=-y, y, x, y(0)=3)", Some("en"));
    assert!(en.contains("Apply the initial condition"), "{en}");
    assert!(en.contains("Particular solution"), "{en}");
}
#[test]
fn dsolve_second_order_o4_contract() {
    // Fase 4 · O4: 2º orden homogénea de coeficientes constantes por el
    // discriminante exacto INTERNO (D9 — jamás solve desnudo) con la emisión
    // gateada por LINEALIDAD (D5: cada base verifica sola; la combinación con
    // constantes jamás se sustituye — ahí vive el HANG O23).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input])
            .output()
            .expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    };
    let first_line = |input: &str| -> String {
        r(input)
            .lines()
            .next()
            .unwrap_or_default()
            .trim()
            .to_string()
    };
    let err_of = |input: &str| -> String {
        let out = cli()
            .args(["eval", input])
            .output()
            .expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stderr).to_string() + &String::from_utf8_lossy(&out.stdout)
    };

    // O20-O25: las tres ramas del discriminante.
    assert_eq!(
        first_line("dsolve(diff(y,x,2)-y=0, y, x)"),
        "y = C1·e^x + C2 / e^x"
    ); // O20 Δ>0
    assert_eq!(
        first_line("dsolve(diff(y,x,2)+4*y=0, y, x)"),
        "y = C1·sin(2·x) + C2·cos(2·x)"
    ); // O21 Δ<0 α=0
    assert_eq!(
        first_line("dsolve(diff(y,x,2)+2*diff(y,x)+y=0, y, x)"),
        "y = (C2·x + C1) / e^x"
    ); // O22 Δ=0
    assert_eq!(
        first_line("dsolve(diff(y,x,2)+2*diff(y,x)+5*y=0, y, x)"),
        "y = e^(-x)·(C1·sin(2·x) + C2·cos(2·x))"
    ); // O23 — EMITE Y VERIFICA (el pin del ciclo: la linealidad esquiva el hang)
    assert_eq!(
        first_line("dsolve(diff(y,x,2)-3*diff(y,x)+2*y=0, y, x)"),
        "y = C1·e^(2·x) + C2·e^x"
    ); // O24
    assert_eq!(first_line("dsolve(diff(y,x,2)=0, y, x)"), "y = C2·x + C1"); // O25

    // Surds exactos: β=√3 y raíces áureas (verificadas por el gate).
    assert_eq!(
        first_line("dsolve(diff(y,x,2)+3*y=0, y, x)"),
        "y = C1·sin(x·sqrt(3)) + C2·cos(x·sqrt(3))"
    );
    let golden = first_line("dsolve(diff(y,x,2)-diff(y,x)-y=0, y, x)");
    assert!(golden.starts_with("y = C1·e^(phi·x)"), "{golden}");

    // Shape anidado diff(diff(y,x),x) ≡ diff(y,x,2).
    assert_eq!(
        first_line("dsolve(diff(diff(y,x),x)+4*y=0, y, x)"),
        "y = C1·sin(2·x) + C2·cos(2·x)"
    );
    // Coeficientes pelados (no-unitarios y divisor).
    assert_eq!(
        first_line("dsolve(2*diff(y,x,2)+4*diff(y,x)+2*y=0, y, x)"),
        "y = (C2·x + C1) / e^x"
    );
    assert_eq!(
        first_line("dsolve(diff(y,x,2)/4+y=0, y, x)"),
        "y = C1·sin(2·x) + C2·cos(2·x)"
    );

    // IVP de 2º orden en las tres ramas (verificación de AMBAS condiciones).
    assert_eq!(
        first_line("dsolve(diff(y,x,2)+4*y=0, y, x, y(0)=0, y'(0)=2)"),
        "y = sin(2·x)"
    ); // V31
    assert_eq!(
        first_line("dsolve(diff(y,x,2)=0, y, x, y(0)=1, y'(0)=2)"),
        "y = 2·x + 1"
    );
    assert_eq!(
        first_line("dsolve(diff(y,x,2)-3*diff(y,x)+2*y=0, y, x, y(0)=1, y'(0)=0)"),
        "y = 2·e^x - e^(2·x)"
    );
    assert_eq!(
        first_line("dsolve(diff(y,x,2)+2*diff(y,x)+y=0, y, x, y(0)=1, y'(0)=0)"),
        "y = (x + 1) / e^x"
    );
    // La envelope compleja: el caso que quemaba el budget derivando la
    // general con constantes — las ecuaciones salen de las BASES.
    assert_eq!(
        first_line("dsolve(diff(y,x,2)+2*diff(y,x)+5*y=0, y, x, y(0)=1, y'(0)=1)"),
        "y = (sin(2·x) + cos(2·x)) / e^x"
    );

    // Constantes frescas cuando la entrada usa C1/C2 (D7).
    let fresh = first_line("dsolve(diff(y,x,2)+C1*y=0, y, x)");
    assert!(
        fresh.starts_with("dsolve("),
        "coef simbólico declina: {fresh}"
    );

    // Declines honestos: 1 condición (faltan), coef variables (Airy/Bessel),
    // no-lineal (y''=y²), no-homogénea (O5), orden ≥3.
    assert!(r("dsolve(diff(y,x,2)-y=0, y, x, y(0)=1)").starts_with("dsolve("));
    assert!(err_of("dsolve(diff(y,x,2)-y=0, y, x, y(0)=1)").contains("DOS condiciones"));
    assert!(r("dsolve(diff(y,x,2)+x*y=0, y, x)").starts_with("dsolve(")); // Z2 Airy
    assert!(r("dsolve(x^2*diff(y,x,2)+x*diff(y,x)+(x^2-1)*y=0, y, x)").starts_with("dsolve(")); // Z6 Bessel
    assert!(r("dsolve(diff(y,x,2)=y^2, y, x)").starts_with("dsolve(")); // Z4
    assert!(r("dsolve(diff(y,x,2)+sin(y)=0, y, x)").starts_with("dsolve(")); // Z7
                                                                             // La no-homogénea GRADUÓ en O5 (el pin de decline migró a resolución).
    assert_eq!(
        first_line("dsolve(diff(y,x,2)+y=x, y, x)"),
        "y = C1·sin(x) + C2·cos(x) + x"
    );
    assert!(err_of("dsolve(diff(y,x,3)+y=0, y, x)").contains("orden ≥3"));

    // El verbo solve NO cambia (D9: el set colapsa multiplicidad — contrato).
    assert_eq!(first_line("solve(r^2+2*r+1=0, r)"), "{ -1 }");
    // Spot checks O0-O3 byte-idénticos.
    assert_eq!(
        first_line("dsolve(diff(y,x)=x*y, y, x)"),
        "y = C·e^(x^2 / 2)"
    );
    assert_eq!(
        first_line("dsolve(diff(y,x)+y=x, y, x)"),
        "y = C / e^x + x - 1"
    );

    // Narración O4 keyed es/en.
    let steps_of = |input: &str, lang: Option<&str>| -> String {
        let mut args = vec!["eval", input, "--steps", "on", "--format", "json"];
        if let Some(l) = lang {
            args.extend(["--lang", l]);
        }
        let out = cli().args(&args).output().expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stdout).to_string()
    };
    let es = steps_of("dsolve(diff(y,x,2)+4*y=0, y, x)", None);
    assert!(es.contains("ecuación característica"), "{es}");
    assert!(es.contains("discriminante"), "{es}");
    assert!(es.contains("Raíces complejas conjugadas"), "{es}");
    let en = steps_of("dsolve(diff(y,x,2)+4*y=0, y, x)", Some("en"));
    assert!(en.contains("characteristic equation"), "{en}");
    assert!(en.contains("Complex conjugate roots"), "{en}");
}
#[test]
fn dsolve_undetermined_coefficients_o5_contract() {
    // Fase 4 · O5: no-homogénea por coeficientes indeterminados — collector
    // de coeficientes por función-base sobre derivadas ESTRUCTURALES (jamás
    // el simplificador: la familia fresh×exp×trig es C5-hostil), Gauss
    // racional exacto, shift de resonancia x^s por multiplicidad
    // característica exacta, y el gate afín L[y_p]−RHS → 0.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input])
            .output()
            .expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    };
    let first_line = |input: &str| -> String {
        r(input)
            .lines()
            .next()
            .unwrap_or_default()
            .trim()
            .to_string()
    };
    let err_of = |input: &str| -> String {
        let out = cli()
            .args(["eval", input])
            .output()
            .expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stderr).to_string() + &String::from_utf8_lossy(&out.stdout)
    };

    // N26-N29 (el catálogo del scoping).
    assert_eq!(
        first_line("dsolve(diff(y,x,2)+y=x, y, x)"),
        "y = C1·sin(x) + C2·cos(x) + x"
    ); // N26 UC polinomio
    assert_eq!(
        first_line("dsolve(diff(y,x,2)-y=exp(2*x), y, x)"),
        "y = 1/3·e^(2·x) + C1·e^x + C2 / e^x"
    ); // N27 UC exponencial
    assert_eq!(
        first_line("dsolve(diff(y,x,2)+y=cos(x), y, x)"),
        "y = 1/2·x·sin(x) + C1·sin(x) + C2·cos(x)"
    ); // N28 resonancia trig (s=1)
    assert_eq!(
        first_line("dsolve(diff(y,x,2)-3*diff(y,x)+2*y=exp(x), y, x)"),
        "y = C1·e^(2·x) + C2·e^x - x·e^x"
    ); // N29 resonancia raíz simple

    // Siblings del barrido: resonancia DOBLE (s=2), trig con β resonante,
    // RHS constante, polinomio cuadrático, producto x·e^x.
    assert_eq!(
        first_line("dsolve(diff(y,x,2)+2*diff(y,x)+y=exp(-x), y, x)"),
        "y = x^2 / (2·e^x) + (C2·x + C1) / e^x"
    );
    assert_eq!(
        first_line("dsolve(diff(y,x,2)+4*y=sin(2*x), y, x)"),
        "y = C1·sin(2·x) + C2·cos(2·x) - 1/4·x·cos(2·x)"
    );
    assert_eq!(
        first_line("dsolve(diff(y,x,2)+y=5, y, x)"),
        "y = C1·sin(x) + C2·cos(x) + 5"
    );
    assert_eq!(
        first_line("dsolve(diff(y,x,2)-y=x^2+1, y, x)"),
        "y = C1·e^x + C2 / e^x - x^2 - 3"
    );
    assert_eq!(
        first_line("dsolve(diff(y,x,2)+4*y=4*x*exp(x), y, x)"),
        "y = C1·sin(2·x) + C2·cos(2·x) + e^x·(4/5·x - 8/25)"
    );

    // IVP no-homogéneo: las ecuaciones de condición incluyen y_p^(k)(x0).
    assert_eq!(
        first_line("dsolve(diff(y,x,2)+y=x, y, x, y(0)=0, y'(0)=2)"),
        "y = sin(x) + x"
    );

    // Fuera de la tabla UC → residual honesto nombrando variación de parámetros.
    let tan = r("dsolve(diff(y,x,2)+y=tan(x), y, x)");
    assert!(tan.starts_with("dsolve("), "tan(x) declina: {tan}");
    assert!(err_of("dsolve(diff(y,x,2)+y=tan(x), y, x)").contains("variación de parámetros"));

    // Pins de no-regresión: homogéneas O4 y primer orden byte-idénticos.
    assert_eq!(
        first_line("dsolve(diff(y,x,2)+4*y=0, y, x)"),
        "y = C1·sin(2·x) + C2·cos(2·x)"
    );
    assert_eq!(
        first_line("dsolve(diff(y,x)=x*y, y, x)"),
        "y = C·e^(x^2 / 2)"
    );

    // Narración O5 keyed es/en.
    let steps_of = |input: &str, lang: Option<&str>| -> String {
        let mut args = vec!["eval", input, "--steps", "on", "--format", "json"];
        if let Some(l) = lang {
            args.extend(["--lang", l]);
        }
        let out = cli().args(&args).output().expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stdout).to_string()
    };
    let es = steps_of("dsolve(diff(y,x,2)+y=x, y, x)", None);
    assert!(es.contains("coeficientes indeterminados"), "{es}");
    let en = steps_of("dsolve(diff(y,x,2)+y=x, y, x)", Some("en"));
    assert!(en.contains("undetermined coefficients"), "{en}");
}
#[test]
fn dsolve_bernoulli_homogeneous_o8_contract() {
    // Fase 4 · O8: Bernoulli (v = y^(1−n) → lineal compartido → back-subs) y
    // homogéneas (v = y/x → separable en v) — composición de métodos ya
    // graduados, emisión gateada (D5) en todas las rutas.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input])
            .output()
            .expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    };
    let first_line = |input: &str| -> String {
        r(input)
            .lines()
            .next()
            .unwrap_or_default()
            .trim()
            .to_string()
    };
    let err_of = |input: &str| -> String {
        let out = cli()
            .args(["eval", input])
            .output()
            .expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stderr).to_string() + &String::from_utf8_lossy(&out.stdout)
    };

    // B16/B17 (n = 2, la forma explícita textbook y = 1/v).
    assert_eq!(
        first_line("dsolve(diff(y,x)+y=y^2, y, x)"),
        "y = 1 / (C·e^x + 1)"
    );
    assert_eq!(
        first_line("dsolve(diff(y,x)+y/x=x*y^2, y, x)"),
        "y = 1 / (C·x - x^2)"
    );
    // Warning de solución singular (D12: y = 0 descartada al dividir por yⁿ).
    assert!(err_of("dsolve(diff(y,x)+y=y^2, y, x)").contains("solución singular"));

    // H18: homogénea con despeje explícito textbook.
    assert_eq!(
        first_line("dsolve(diff(y,x)=(x+y)/x, y, x)"),
        "y = x·ln(x) + C·x"
    );
    // H19: la ruta implícita RACIONAL esquiva el residuo surd del scoping.
    assert_eq!(
        first_line("dsolve(diff(y,x)=(x^2+y^2)/(x*y), y, x)"),
        "y^2 / (2·x^2) - ln(x) = C"
    );

    // Bernoulli + IVP (la condición fija C con verificación doble).
    let ivp = first_line("dsolve(diff(y,x)+y=y^2, y, x, y(0)=2)");
    assert!(ivp.starts_with("y = 1 /"), "Bernoulli IVP resuelve: {ivp}");

    // n ≠ 2: decline honesto nombrando el peldaño (verificación por rama).
    let n3 = r("dsolve(diff(y,x)+y=y^3, y, x)");
    assert!(n3.starts_with("dsolve("), "n=3 declina: {n3}");
    assert!(err_of("dsolve(diff(y,x)+y=y^3, y, x)").contains("verificación por rama"));

    // Riccati sigue residual honesto permanente (Z1/Z5 never-fabricate).
    let riccati = r("dsolve(diff(y,x)=x^2+y^2, y, x)");
    assert!(
        riccati.starts_with("dsolve("),
        "Riccati residual: {riccati}"
    );
    assert!(err_of("dsolve(diff(y,x)=x^2+y^2, y, x)").contains("Riccati"));

    // Homogénea cuya reducción no cierra en el integrador → residual honesto
    // CON DUEÑO (el peldaño es del integrador racional, no de dsolve).
    let hard = r("dsolve(diff(y,x)=(2*x*y)/(x^2-y^2), y, x)");
    assert!(hard.starts_with("dsolve("), "reducción no cierra: {hard}");
    assert!(err_of("dsolve(diff(y,x)=(2*x*y)/(x^2-y^2), y, x)").contains("F(v)−v"));

    // Pins de no-robo del dispatcher: separable/lineal byte-idénticos.
    assert_eq!(first_line("dsolve(diff(y,x)=y/x, y, x)"), "y = C·x");
    assert_eq!(
        first_line("dsolve(diff(y,x)=x*y, y, x)"),
        "y = C·e^(x^2 / 2)"
    );
    assert_eq!(
        first_line("dsolve(diff(y,x)+y=x, y, x)"),
        "y = C / e^x + x - 1"
    );

    // Narración O8 keyed es/en.
    let steps_of = |input: &str, lang: Option<&str>| -> String {
        let mut args = vec!["eval", input, "--steps", "on", "--format", "json"];
        if let Some(l) = lang {
            args.extend(["--lang", l]);
        }
        let out = cli().args(&args).output().expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stdout).to_string()
    };
    let es = steps_of("dsolve(diff(y,x)+y=y^2, y, x)", None);
    assert!(es.contains("forma de Bernoulli"), "{es}");
    assert!(es.contains("se vuelve lineal"), "{es}");
    let en = steps_of("dsolve(diff(y,x)+y=y^2, y, x)", Some("en"));
    assert!(en.contains("Bernoulli form"), "{en}");
    let es_h = steps_of("dsolve(diff(y,x)=(x+y)/x, y, x)", None);
    assert!(es_h.contains("EDO homogénea"), "{es_h}");
    let en_h = steps_of("dsolve(diff(y,x)=(x+y)/x, y, x)", Some("en"));
    assert!(en_h.contains("homogeneous ODE"), "{en_h}");
}
#[test]
fn dsolve_systems_2x2_o6_contract() {
    // Fase 4 · O6: sistemas 2×2 X' = A·X por la ruta eigen INTERNA (D17: los
    // verbos eigenvalues/eigenvectors NO se tocan); autovalores complejos
    // emiten soluciones REALES; defectiva por vector generalizado; cada base
    // verifica contra AMBAS ecuaciones (D5 por componente).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input])
            .output()
            .expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    };
    let first_line = |input: &str| -> String {
        r(input)
            .lines()
            .next()
            .unwrap_or_default()
            .trim()
            .to_string()
    };
    let err_of = |input: &str| -> String {
        let out = cli()
            .args(["eval", input])
            .output()
            .expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stderr).to_string() + &String::from_utf8_lossy(&out.stdout)
    };

    // Y34: autovalores reales ±1.
    assert_eq!(
        first_line("dsolve([diff(x,t)=y, diff(y,t)=x], [x,y], t)"),
        "{ x = C1·e^t + C2 / e^t, y = C1·e^t - C2 / e^t }"
    );
    // Y35: λ = ±i → soluciones REALES (sin `i` en la salida).
    // (El pin de igualdad exacta garantiza la forma REAL — sin unidad
    // imaginaria en la salida.)
    assert_eq!(
        first_line("dsolve([diff(x,t)=-y, diff(y,t)=x], [x,y], t)"),
        "{ x = -C1·cos(t) - C2·sin(t), y = C2·cos(t) - C1·sin(t) }"
    );
    // Y-def: doble defectiva λ=2 con vector generalizado.
    assert_eq!(
        first_line("dsolve([diff(x,t)=2*x+y, diff(y,t)=2*y], [x,y], t)"),
        "{ x = C1·e^(2·t) + C2·t·e^(2·t), y = C2·e^(2·t) }"
    );

    // Declines honestos: no-lineal, IVP de sistemas, autovalores irracionales.
    let nl = r("dsolve([diff(x,t)=x+y, diff(y,t)=x*y], [x,y], t)");
    assert!(nl.starts_with("dsolve("), "no-lineal declina: {nl}");
    let ivp = r("dsolve([diff(x,t)=y, diff(y,t)=x], [x,y], t, x(0)=1)");
    assert!(ivp.starts_with("dsolve("), "IVP de sistemas declina: {ivp}");
    assert!(
        err_of("dsolve([diff(x,t)=y, diff(y,t)=x], [x,y], t, x(0)=1)").contains("ciclo futuro")
    );
    let irr = r("dsolve([diff(x,t)=x+y, diff(y,t)=x], [x,y], t)");
    assert!(
        irr.starts_with("dsolve("),
        "autovalores irracionales declinan: {irr}"
    );

    // Pin D17: el verbo eigenvalues NO cambia (su decline es contrato propio).
    let ev = first_line("eigenvalues([[0,-1],[1,0]])");
    assert!(
        ev.starts_with("eigenvalues("),
        "el verbo eigenvalues sigue residual: {ev}"
    );

    // Pins de no-regresión escalar: dsolve escalar byte-idéntico.
    assert_eq!(
        first_line("dsolve(diff(y,x)=x*y, y, x)"),
        "y = C·e^(x^2 / 2)"
    );
    assert_eq!(
        first_line("dsolve(diff(y,x,2)+4*y=0, y, x)"),
        "y = C1·sin(2·x) + C2·cos(2·x)"
    );

    // Usage-error del pre-pass para la forma lista malformada.
    assert!(err_of("dsolve([diff(x,t)=y], [x,y], t)").contains("dsolve system"));

    // Narración O6 keyed es/en.
    let steps_of = |input: &str, lang: Option<&str>| -> String {
        let mut args = vec!["eval", input, "--steps", "on", "--format", "json"];
        if let Some(l) = lang {
            args.extend(["--lang", l]);
        }
        let out = cli().args(&args).output().expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stdout).to_string()
    };
    let es = steps_of("dsolve([diff(x,t)=-y, diff(y,t)=x], [x,y], t)", None);
    assert!(es.contains("sistema lineal X'"), "{es}");
    assert!(es.contains("Autovalores complejos conjugados"), "{es}");
    let en = steps_of("dsolve([diff(x,t)=-y, diff(y,t)=x], [x,y], t)", Some("en"));
    assert!(en.contains("linear system X'"), "{en}");
    assert!(en.contains("Complex conjugate eigenvalues"), "{en}");
}
#[test]
fn dsolve_surface_o7_contract() {
    // Fase 4 · O7: superficie de usuario — help topic, LaTeX d/dx (D14: la
    // incógnita es DEPENDIENTE, jamás ∂ en el canal dsolve), y las claves de
    // narración es/en completas (grep-gate por familia en el ciclo).
    let repl_out = |input: &str| -> String {
        let out = cli()
            .arg("repl")
            .write_stdin(format!("{input}\nexit\n"))
            .output()
            .expect("Failed to run CLI repl");
        String::from_utf8_lossy(&out.stdout).to_string()
    };
    // Help topic presente con las secciones clave.
    let help = repl_out("help dsolve");
    assert!(help.contains("Command: dsolve"), "{help}");
    assert!(help.contains("Supported families"), "{help}");
    assert!(help.contains("Honest residuals"), "{help}");
    assert!(help.contains("VERIFIED"), "{help}");

    // D14: el input LaTeX usa d/dx (ordinaria), jamás ∂, en escalar y sistema.
    let latex_of = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let v: serde_json::Value =
            serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        v["input_latex"].as_str().unwrap_or_default().to_string()
    };
    let scalar = latex_of("dsolve(diff(y,x,2)+4*y=0, y, x)");
    assert!(scalar.contains("\\frac{d^{2}}{d x^{2}}"), "{scalar}");
    assert!(
        !scalar.contains("\\partial"),
        "sin ∂ en el canal dsolve: {scalar}"
    );
    let system = latex_of("dsolve([diff(x,t)=-y, diff(y,t)=x], [x,y], t)");
    assert!(!system.contains("\\partial"), "sin ∂ en sistemas: {system}");
    // El render general de diff multivariable FUERA de dsolve conserva ∂
    // (decisión vectorial #3 — contrato del formatter, no de dsolve).
    let generic = latex_of("diff(x^2*y, x)");
    assert!(
        generic.contains("\\partial"),
        "∂ intacto fuera de dsolve: {generic}"
    );
}
#[test]
fn dsolve_cauchy_euler_o9_contract() {
    // Fase 4 · O9 (opcional pre-aprobado): Cauchy-Euler por la ecuación
    // indicial r(r−1)+a·r+b sobre el molde D9 — bases x^r / x^r·ln x /
    // x^α·trig(β·ln x), emisión gateada por base (D5).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input])
            .output()
            .expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    };
    let first_line = |input: &str| -> String {
        r(input)
            .lines()
            .next()
            .unwrap_or_default()
            .trim()
            .to_string()
    };
    let err_of = |input: &str| -> String {
        let out = cli()
            .args(["eval", input])
            .output()
            .expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stderr).to_string() + &String::from_utf8_lossy(&out.stdout)
    };

    // Las tres ramas indiciales.
    assert_eq!(
        first_line("dsolve(x^2*diff(y,x,2)+x*diff(y,x)-y=0, y, x)"),
        "y = C2 / x + C1·x"
    ); // r = ±1 — el colapso probado del scoping MIGRÓ a resolución
    assert_eq!(
        first_line("dsolve(x^2*diff(y,x,2)-x*diff(y,x)+y=0, y, x)"),
        "y = x·(C2·ln(x) + C1)"
    ); // doble r = 1
    assert_eq!(
        first_line("dsolve(x^2*diff(y,x,2)+x*diff(y,x)+y=0, y, x)"),
        "y = C1·cos(ln(x)) + C2·sin(ln(x))"
    ); // complejas ±i
    assert_eq!(
        first_line("dsolve(x^2*diff(y,x,2)-2*y=0, y, x)"),
        "y = C2 / x + C1·x^2"
    ); // sin término y'
    assert!(err_of("dsolve(x^2*diff(y,x,2)+x*diff(y,x)-y=0, y, x)").contains("dominio x > 0"));

    // Z6 Bessel SIGUE residual ((x²−1)·y no es estructura Euler — pin
    // never-fabricate intacto).
    let bessel = r("dsolve(x^2*diff(y,x,2)+x*diff(y,x)+(x^2-1)*y=0, y, x)");
    assert!(bessel.starts_with("dsolve("), "Bessel residual: {bessel}");
    // IVP de Cauchy-Euler declina honesto (x0 = 0 singular).
    let ivp = r("dsolve(x^2*diff(y,x,2)+x*diff(y,x)-y=0, y, x, y(1)=2, y'(1)=0)");
    assert!(ivp.starts_with("dsolve("), "CE IVP declina: {ivp}");

    // Pins de no-robo: coeficientes constantes byte-idénticos.
    assert_eq!(
        first_line("dsolve(diff(y,x,2)+4*y=0, y, x)"),
        "y = C1·sin(2·x) + C2·cos(2·x)"
    );

    // Narración O9 keyed es/en.
    let steps_of = |input: &str, lang: Option<&str>| -> String {
        let mut args = vec!["eval", input, "--steps", "on", "--format", "json"];
        if let Some(l) = lang {
            args.extend(["--lang", l]);
        }
        let out = cli().args(&args).output().expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stdout).to_string()
    };
    let es = steps_of("dsolve(x^2*diff(y,x,2)+x*diff(y,x)+y=0, y, x)", None);
    assert!(es.contains("Cauchy-Euler"), "{es}");
    assert!(es.contains("ecuación indicial"), "{es}");
    let en = steps_of("dsolve(x^2*diff(y,x,2)+x*diff(y,x)+y=0, y, x)", Some("en"));
    assert!(en.contains("indicial equation"), "{en}");
}
#[test]
fn dsolve_integrating_factor_mu_contract() {
    // Fase 4 · O9-μ (opcional pre-aprobado): no-exactas con factor
    // integrante SIMPLE — μ(x) si (M_y−N_x)/N es solo-de-x, μ(y) si
    // (N_x−M_y)/M es solo-de-y; multiplicar y DELEGAR en el handler exacto
    // (la emisión hereda los gates D5/D11 del potencial).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input])
            .output()
            .expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    };
    let full = |input: &str| -> String {
        let out = cli()
            .args(["eval", input])
            .output()
            .expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr)
    };

    // μ(x) = x (Boyce clásico).
    let mu_x = full("dsolve((3*x*y+y^2) + (x^2+x*y)*diff(y,x) = 0, y, x)");
    assert!(mu_x.contains("2·y·x^3 + x^2·y^2 = C"), "{mu_x}");
    assert!(mu_x.contains("μ(x) = x"), "{mu_x}");
    assert!(mu_x.contains("soluciones singulares"), "{mu_x}");
    // μ(y) = y con integral por partes (y²·e^y).
    let mu_y = full("dsolve(y + (2*x-y*exp(y))*diff(y,x) = 0, y, x)");
    assert!(
        mu_y.contains("2·y·e^y + e^y·(-y^2 - 2) + x·y^2 = C"),
        "{mu_y}"
    );
    assert!(mu_y.contains("μ(y) = y"), "{mu_y}");

    // Techo honesto: μ(y) racional lleva a potencial racional fuera del
    // reconstructor — residual honesto POR COMPOSICIÓN (la forma exacta
    // equivalente directa también declina hoy).
    let ceiling = r("dsolve(2*x*y + (y^2-x^2)*diff(y,x) = 0, y, x)");
    assert!(ceiling.starts_with("dsolve("), "{ceiling}");

    // Pins de no-robo: separable/lineal/exacta byte-idénticas.
    assert!(r("dsolve(diff(y,x)=x*y, y, x)").starts_with("y = C·e^(x^2 / 2)"));
    assert!(r("dsolve(diff(y,x)+y=x, y, x)").starts_with("y = C / e^x + x - 1"));
    assert!(
        r("dsolve((2*x*y+1) + (x^2+2*y)*diff(y,x) = 0, y, x)").starts_with("y·x^2 + y^2 + x = C")
    );

    // Narración en inglés vía la tabla es/en.
    let out = cli()
        .args([
            "eval",
            "dsolve((3*x*y+y^2) + (x^2+x*y)*diff(y,x) = 0, y, x)",
            "--steps",
            "on",
            "--lang",
            "en",
            "--format",
            "json",
        ])
        .output()
        .expect("Failed to run CLI");
    let en = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(en.contains("simple integrating factor"), "{en}");
}
