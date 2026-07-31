use super::*;

#[test]
fn test_eval_gaussian_reciprocal_clean_form() {
    // Fase 2 · residual C1 del frente complejo, cerrado: `(z)^(-1)` llegaba mangled
    // (`(1/2·2 - i)/(2)`) SOLO por la ruta Pow(z,-1) — AddFractions construía el
    // numerador cruzado con `Mul(Number, Number)` crudo (mul2_raw) y el pipeline
    // abandonaba el ciclo combina↔separa en el lado sin plegar (los gemelos reales
    // se limpiaban vía factor-out, que declina con `i`). El builder ahora pliega
    // Number×Number exacto en la emisión.
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
    assert_eq!(rc("(1+i)^(-1)"), "1/2 - 1/2·i");
    assert_eq!(rc("(2+i)^(-1)"), "2/5 - 1/5·i");
    assert_eq!(rc("(1-i)^(-1)"), "1/2 + 1/2·i");
    assert_eq!(rc("(3+4i)^(-1)"), "3/25 - 4/25·i");
    // Verificación de valor: z·z^(-1) = 1 exacto.
    assert_eq!(rc("(1+i)*(1+i)^(-1)"), "1");
    assert_eq!(rc("(2+i)*(2+i)^(-1)"), "1");
}
#[test]
fn test_eval_complex_rule_names_localized() {
    // Fase 2 · C2: los nombres de regla del frente complejo llegan al wire
    // LOCALIZADOS (es fuente, en vía tabla) — la "barra baja" del frente elevada al
    // patrón de los verbos vectoriales. Solo AÑADIR claves, jamás editar existentes.
    let rules_of = |input: &str, lang: Option<&str>| -> Vec<String> {
        let mut args = vec![
            "eval",
            input,
            "--value-domain",
            "complex",
            "--steps",
            "on",
            "--format",
            "json",
        ];
        if let Some(l) = lang {
            args.extend(["--lang", l]);
        }
        let out = cli().args(&args).output().expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["steps"]
            .as_array()
            .map(|steps| {
                steps
                    .iter()
                    .map(|s| s["rule"].as_str().unwrap_or("").to_string())
                    .collect()
            })
            .unwrap_or_default()
    };
    assert_eq!(
        rules_of("(3+4i)/(1-2i)", None),
        vec!["Dividir números complejos"]
    );
    assert_eq!(
        rules_of("(3+4i)/(1-2i)", Some("en")),
        vec!["Divide complex numbers"]
    );
    assert_eq!(
        rules_of("sin(i)", None),
        vec!["Aplicar trigonometría de argumento imaginario"]
    );
    let euler_unimodular = rules_of("abs(e^(2*i))", Some("en"));
    assert_eq!(
        euler_unimodular,
        vec![
            "Apply Euler's formula",
            "Apply the unimodular absolute value"
        ]
    );
}
#[test]
fn test_eval_reciprocal_cis() {
    // Tanda-3 ciclo 4: n/(cos u ± i·sin u) → n·(cos u ∓ i·sin u) — identidad ENTERA
    // (cis·cis̄ = cos²+sin² = 1 en todo ℂ), sin guard de realidad. Cierra el residual
    // B2: la canonicalización de exponente negativo convertía e^(-ix) en 1/e^(ix)
    // ANTES de Euler, y Euler expandía solo el denominador.
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
    assert_eq!(rc("e^(-i*x)"), "cos(x) - i·sin(x)");
    assert_eq!(rc("exp(-i*x)"), "cos(x) - i·sin(x)");
    assert_eq!(rc("1/(cos(x)+i*sin(x))"), "cos(x) - i·sin(x)");
    assert_eq!(rc("2/(cos(x)+i*sin(x))"), "2·cos(x) - 2·i·sin(x)");
    assert_eq!(rc("1/(cos(x)-i*sin(x))"), "cos(x) + i·sin(x)");
    // Pins: Euler directo intacto; la unimodularidad sigue viva sobre el matcher
    // compartido refactorizado; denominador no-cis intacto; real mode gated.
    assert_eq!(rc("e^(i*x)"), "cos(x) + i·sin(x)");
    assert_eq!(rc("abs(e^(2*i))"), "1");
    assert_eq!(rc("1/(cos(x)+sin(x))"), "1 / (sin(x) + cos(x))");
    assert_eq!(r("e^(-i*x)"), "1 / e^(i·x)");
}
#[test]
fn test_eval_gaussian_surd_modulus() {
    // Tanda-3 ciclo 2: |a+b·i| con componentes reales DECIDIBLES (provable_const_sign
    // — surds, e/π; la disciplina V0 hace declinar los símbolos). Cierra la familia
    // π-racional que la unimodularidad dejó nombrada (el trig pliega a surds ANTES
    // del abs). El caso ambos-racionales conserva su dueño exacto (GaussianRational).
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
    // La familia π-racional cierra: unimodulares por la vía surd.
    assert_eq!(rc("abs(1/2 + i*sqrt(3)/2)"), "1");
    assert_eq!(rc("abs(e^(i*pi/3))"), "1");
    assert_eq!(rc("abs(e^(i*pi/4))"), "1");
    // Módulos surd generales, forma factorizada incluida.
    assert_eq!(rc("abs(1 + i*sqrt(3))"), "2");
    assert_eq!(rc("abs(sqrt(2) + i*sqrt(2))"), "2");
    // Puro-imaginario decidible: el signo ya está decidido — emite ±b directo.
    assert_eq!(rc("abs(i*sqrt(3))"), "sqrt(3)");
    assert_eq!(rc("abs(-i*sqrt(3))"), "sqrt(3)");
    assert_eq!(rc("abs(i*pi)"), "pi");
    // Transcendentales: forma exacta sin plegar, sound.
    assert_eq!(rc("abs(e + i*pi)"), "sqrt(pi^2 + e^2)");
    // Ownership: racionales al dueño exacto; símbolos declinan (disciplina V0).
    assert_eq!(rc("abs(3+4*i)"), "5");
    assert_eq!(rc("abs(x + i*sqrt(3))"), "|x + i·sqrt(3)|");
}
#[test]
fn test_eval_trig_of_imaginary_bridge() {
    // Fase 2 · trig-de-i (residual B4b): el puente trig↔hiperbólico de argumento
    // puro-imaginario — identidades ENTERAS (válidas para y complejo arbitrario, sin
    // guard de realidad, a diferencia de la unimodularidad). ONE-DIRECTION.
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
    // Los 6 brazos, literal y simbólico.
    assert_eq!(rc("sin(i)"), "i·sinh(1)");
    assert_eq!(rc("cos(i)"), "cosh(1)");
    assert_eq!(rc("tan(i)"), "i·tanh(1)");
    assert_eq!(rc("sin(i*x)"), "i·sinh(x)");
    assert_eq!(rc("sinh(i)"), "i·sin(1)");
    assert_eq!(rc("cosh(i*x)"), "cos(x)");
    assert_eq!(rc("tanh(3*i)"), "i·tan(3)");
    // Composición exacta a través del puente: cosh(iπ) = cos(π) = -1.
    assert_eq!(rc("cosh(i*pi)"), "-1");
    // El decline de argumento mixto GRADUÓ en tanda-3 ciclo 1 (ComplexAngleSumRule);
    // el pin migra a la forma expandida — la propiedad de ESTE test (puro-imaginario
    // es del puente) se conserva en los asserts de arriba.
    assert_eq!(rc("sin(1+i)"), "sin(1)·cosh(1) + i·cos(1)·sinh(1)");
    assert_eq!(r("sin(i)"), "sin(i)");
    // La red B1 con los brazos hiperbólicos nuevos del walker: refuta la identidad
    // FALSA (jamás confirma la verdadera desde probe — el wire equiv es Bool
    // exacto-solo, "false" = no-probado; residual nombrado).
    assert_eq!(rc("equiv(sin(i), i*sinh(2))"), "false");
    // approx compone con el puente (el walker evalúa sinh complejo).
    assert_eq!(rc("approx(sin(i))"), "1.17520119364·i");
}
#[test]
fn test_eval_complex_negative_base_odd_root_principal_branch() {
    // In complex mode, a negative base under a rational `p/q` with ODD denominator is the PRINCIPAL
    // value `r^(p/q)·(cos(πp/q) + i·sin(πp/q))`, not the real odd root: `(-1)^(1/3) = 1/2 + (√3/2)i`,
    // not `-1`. The real-odd-root literal value was leaking into complex mode (Round-5 audit, P0).
    let cx = |input: &str| -> String {
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
    let re = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Complex principal branch: a non-real value with the correct real part, NOT the real odd root.
    let r13 = cx("(-1)^(1/3)");
    assert!(
        r13.contains('i') && r13.contains("1/2") && r13 != "-1",
        "(-1)^(1/3) complex -> {r13}"
    );
    assert!(cx("(-8)^(1/3)").contains('i') && cx("(-8)^(1/3)") != "-2");
    assert!(cx("(-1)^(2/3)").contains('i') && cx("(-1)^(2/3)") != "1");
    // Even-root complex (sqrt(-n) → i·sqrt(n)) and positive bases are unaffected.
    assert_eq!(cx("(-4)^(1/2)"), "2·i");
    assert_eq!(cx("8^(1/3)"), "2");
    // REAL mode keeps the engine's real-odd-root convention.
    assert_eq!(re("(-8)^(1/3)"), "-2");
    assert_eq!(re("(-1)^(1/3)"), "-1");
}
#[test]
fn test_eval_non_real_solution_rejected_in_real_domain() {
    // In the RealOnly domain, a provably NON-REAL solution (the imaginary unit `i`, `√(negative)`,
    // an even root of a negative `(-1)^(1/2)`, or anything carrying them) has no real solution. The
    // `ln`/`exp` inversion did not re-check reality, so `solve(ln(x)=√(-1)) → {e^((-1)^(1/2))}` (= e^i)
    // and `solve(x=i) → {i}` slipped through. An ODD root of a negative (`(-8)^(1/3) = -2`) is REAL.
    for input in [
        "solve(ln(x)=sqrt(-1), x)",
        "solve(x=sqrt(-1), x)",
        "solve(x=e^(sqrt(-1)), x)",
        "solve(ln(x)=sqrt(-4), x)",
        "solve(x=i, x)",
        "solve(x=2*i, x)",
        "solve(x=1+i, x)",
        "solve(x^2=e^(sqrt(-1)), x)",
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some("No solution"), "{input}");
    }
    // REAL solutions (incl. odd roots of negatives) must survive.
    for (input, expected) in [
        ("solve(x=5, x)", "{ 5 }"),
        ("solve(x^2=4, x)", "{ -2, 2 }"),
        ("solve(x=(-8)^(1/3), x)", "{ -2 }"),
        ("solve(x^3=-8, x)", "{ -2 }"),
        ("solve(ln(x)=2, x)", "{ e^2 }"),
        ("solve(x=sqrt(2), x)", "{ sqrt(2) }"),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
    }
    // Complex domain keeps an imaginary solution.
    let output = cli()
        .args([
            "eval",
            "solve(x=i, x)",
            "--format",
            "json",
            "--value-domain",
            "complex",
        ])
        .output()
        .expect("Failed to run CLI");
    assert!(output.status.success());
    let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
    assert_eq!(wire["result"].as_str(), Some("{ i }"));
}
