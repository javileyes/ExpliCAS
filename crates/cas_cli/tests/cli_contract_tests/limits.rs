use super::*;

#[test]
fn test_eval_infinity_over_infinity_is_undefined() {
    // `∞/∞` is indeterminate; the generic `a/a -> 1` / `(a·X)/(b·X) -> a/b` cancellation used to
    // treat `∞` as a cancellable factor and fabricate a finite value. A dedicated rule now folds it
    // to `undefined` (including finite-scaled, symbolic-scaled, and multi-factor forms and signs).
    for (input, expected) in [
        ("inf/inf", "undefined"),
        ("(2*inf)/inf", "undefined"),
        ("(-inf)/inf", "undefined"),
        ("inf/(2*inf)", "undefined"),
        ("(3*inf)/(-inf)", "undefined"),
        // Finite-scaled `(c·∞)/(d·∞)` must NOT cancel `∞` to `c/d`.
        ("(2*inf)/(5*inf)", "undefined"),
        ("(2*inf)/(2*inf)", "undefined"),
        ("(10*inf)/(4*inf)", "undefined"),
        ("(-2*inf)/(-3*inf)", "undefined"),
        // Symbolic-scaled `(x·∞)/(k·x·∞)` and identical `(x·∞)/(x·∞)` are still `∞/∞`, not `1`.
        ("(x*inf)/(2*x*inf)", "undefined"),
        ("(x*inf)/(x*inf)", "undefined"),
        // Multi-factor products: the shared finite cofactor does not make it finite.
        ("(2*inf*sin(x))/(5*inf*sin(x))", "undefined"),
        ("(inf*sin(x))/(inf*cos(x))", "undefined"),
        // `∞^p` with a positive literal exponent is `∞`: `∞^2/∞^2` is NOT `1`, `∞^3/∞^2` is NOT `∞`.
        ("inf^2/inf^2", "undefined"),
        ("(inf^3)/(inf^2)", "undefined"),
        ("(inf^2)/(inf^3)", "undefined"),
        ("(2*inf^2)/(inf^2)", "undefined"),
        ("(inf^2*x)/(inf^2*y)", "undefined"),
        ("sqrt(inf)/sqrt(inf)", "undefined"),
        // Additive: `∞ + finite = ∞`, so `(∞+1)/(∞+1)` is `∞/∞`, NOT `1`. `∞ − ∞` stays indeterminate.
        ("(inf+1)/(inf+1)", "undefined"),
        ("(inf+inf)/(inf+inf)", "undefined"),
        ("(2*inf+2*inf)/(inf+inf)", "undefined"),
        ("(inf+x)/(inf+x)", "undefined"),
        ("((-inf)+5)/((-inf)+5)", "undefined"),
        // Finite divisions are unaffected.
        ("1/inf", "0"),
        ("2/inf", "0"),
        ("inf/2", "infinity"),
        ("inf/0", "undefined"),
        ("inf-inf", "undefined"),
        ("0*inf", "undefined"),
        ("x/x", "1"),
        // Non-positive / symbolic exponents stay finite or unevaluated (NOT folded).
        ("inf^0", "1"),
        ("inf^(-1)", "0"),
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
fn test_eval_infinity_quotient_plain_matches_steps() {
    // CONSISTENCY: an `∞/∞` quotient must evaluate identically whether or not steps are requested.
    // Several cancellation primitives (plain-mode root shortcuts AND per-node Core rules) used to
    // race the `∞/∞ -> undefined` fold; in the default (no-step-listener) path a cancellation won,
    // so `(2·∞)/(5·∞)` returned `2/5` plain but `undefined` with `--steps`. The fold now runs up
    // front in both modes.
    for input in [
        "(2*inf)/(5*inf)",
        "(2*inf)/(2*inf)",
        "(x*inf)/(2*x*inf)",
        "(2*inf*sin(x))/(5*inf*sin(x))",
        "(inf*sin(x))/(inf*cos(x))",
        "inf/inf",
        // Nested `∞/∞`: the fold is recursive, so an enclosing power/root/log/product/sum cannot
        // let the inner quotient escape via a cancellation that runs before the indeterminate fold.
        "((2*inf)/(5*inf))^2",
        "sqrt((2*inf)/(5*inf))",
        "ln((2*inf)/(5*inf))",
        "(2*inf)/(5*inf)*5",
        "2*((2*inf)/(5*inf))",
        "1+(2*inf)/(3*inf)",
        "abs((2*inf)/(5*inf))",
        // Additive ∞ in the quotient (was `1`/`2` plain vs `undefined` steps).
        "(inf+inf)/(inf+inf)",
        "(2*inf+2*inf)/(inf+inf)",
        "(inf+1)/(inf+1)",
    ] {
        let plain = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(plain.status.success(), "plain {input}");
        let plain_wire: Value = serde_json::from_slice(&plain.stdout).expect("Invalid wire output");

        let steps = cli()
            .args(["eval", input, "--format", "json", "--steps", "on"])
            .output()
            .expect("Failed to run CLI");
        assert!(steps.status.success(), "steps {input}");
        let steps_wire: Value = serde_json::from_slice(&steps.stdout).expect("Invalid wire output");

        assert_eq!(
            plain_wire["result"].as_str(),
            Some("undefined"),
            "plain {input}"
        );
        assert_eq!(
            plain_wire["result"].as_str(),
            steps_wire["result"].as_str(),
            "plain vs --steps divergence for {input}"
        );
    }
}
#[test]
fn test_eval_limit_oscillation_dne() {
    // Tanda-3 ciclo 3 (item del frontier-audit estrechado a exactamente esto por el
    // meta-audit): sin/cos/tan(g) con lateral de g probadamente ±∞ → el límite NO
    // EXISTE por oscilación, con motivo educativo — como ya hacían los laterales
    // discrepantes. Conservador: sin divergencia probada del argumento, residual.
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
    for probe in [
        "limit(sin(1/x), x, 0)",
        "limit(cos(1/x), x, 0)",
        "limit(tan(1/x), x, 0)",
        "limit(sin(1/(x-2)), x, 2)",
        "limit(cos(1/x^2), x, 0)",
    ] {
        let (result, warnings) = eval_full(probe);
        assert_eq!(result, "undefined", "{probe}");
        assert!(
            warnings.contains("OSCILLATES"),
            "{probe} debe llevar el motivo de oscilación, got: {warnings}"
        );
    }
    // Pins: el sandwich, el notable, los laterales discrepantes (SU motivo propio),
    // el infinito y el continuo quedan intactos.
    assert_eq!(eval_full("limit(x*sin(1/x), x, 0)").0, "0");
    assert_eq!(eval_full("limit(sin(x)/x, x, 0)").0, "1");
    let (r, w) = eval_full("limit(abs(x)/x, x, 0)");
    assert_eq!(r, "undefined");
    assert!(w.contains("one-sided limits disagree"));
    assert_eq!(eval_full("limit(sin(1/x), x, infinity)").0, "0");
    assert_eq!(eval_full("limit(sin(x), x, 0)").0, "0");
}
#[test]
fn test_eval_limit_complex_domain_kill_switch() {
    // Fase 3 · F0 (P0): el motor de límites razona con el ORDEN REAL — bajo
    // `--value-domain complex` fabricaba valores (`e^(-1/z²)→0` y `z·sin(1/z)→0`
    // cuando en ℂ NINGUNO existe: singularidad esencial). El kill-switch de
    // entrada declina TODO límite complejo a residual honesto; F10/F11 re-otorgan
    // selectivamente con justificación analítica. Los 7 WRONG del scoping:
    let eval_complex = |input: &str| -> (String, String) {
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
    for probe in [
        "limit(e^(-1/z^2), z, 0)",
        "limit(z*sin(1/z), z, 0)",
        "limit(tanh(z), z, i*pi/2)",
        "limit(atan(z), z, 2*i)",
        "limit(1/(z^2+1), z, i*1)",
        "limit(1/z^2, z, 0)",
        "limit(e^z, z, infinity)",
    ] {
        let (result, warnings) = eval_complex(probe);
        assert!(
            result.starts_with("limit("),
            "{probe} debe declinar a residual bajo complex, got: {result}"
        );
        assert!(
            warnings.contains("complex value domain"),
            "{probe} debe llevar el motivo del kill-switch, got: {warnings}"
        );
    }
    // Never-fabricate (cuerpo-con-I): la protección era coincidental (declinaban
    // por Polynomial-sobre-ℚ); estos pins la vuelven CONTRATO — F11 los hereda.
    for probe in ["limit(e^(i*x), x, infinity)", "limit(i*sin(x)/x, x, 0)"] {
        let (result, _) = eval_complex(probe);
        assert!(
            result.starts_with("limit("),
            "{probe} jamás fabrica bajo complex, got: {result}"
        );
    }
}
#[test]
fn test_eval_limit_imaginary_point_real_domain_residual() {
    // Fase 3 · F0, mitad real: el gate léxico del wire es un colador (`i` desnudo
    // rechaza pero `2*i`/`i*pi`/`i*1` PASAN) y el motor sustituía el punto — en el
    // polo de tanh en iπ/2 emitía el VALOR `tanh(pi·i/2)`. Punto-con-I en dominio
    // real → residual honesto + el Imaginary Usage Warning estándar (el mismo
    // escape-hatch que ofrece simplify).
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
    for probe in [
        "limit(tanh(z), z, i*pi/2)",
        "limit(1/(z^2+1), z, i*1)",
        "limit(atan(z), z, 2*i)",
        // F0b: evasión del barrido adversarial — punto imaginario deletreado como
        // raíz par de constante negativa NO-racional (sqrt(-pi^2) = i·pi alcanzaba
        // el polo de tanh); el detector decide con provable_const_sign, exacto.
        "limit(tanh(x), x, sqrt(-pi^2)/2)",
    ] {
        let (result, warnings) = eval_full(probe);
        assert!(
            result.starts_with("limit("),
            "{probe} debe declinar a residual con punto imaginario, got: {result}"
        );
        // El aviso viaja en el idioma pedido (C5.3: los warnings pasaron a ir
        // por catálogo bidireccional). Este pin fijaba el TEXTO INGLÉS corriendo
        // en modo español, o sea el defecto; ahora fija el motivo en el idioma
        // correcto, que es un contrato más fuerte.
        assert!(
            warnings.contains("unidad imaginaria"),
            "{probe} debe llevar el motivo del punto imaginario, got: {warnings}"
        );
        assert!(
            warnings.contains("semantics set value complex"),
            "{probe} debe ofrecer el escape-hatch estándar, got: {warnings}"
        );
    }
    // Never-fabricate en real (cuerpo-con-I, punto real): siguen residual.
    for probe in ["limit(e^(i*x), x, infinity)", "limit(i*sin(x)/x, x, 0)"] {
        let (result, _) = eval_full(probe);
        assert!(
            result.starts_with("limit("),
            "{probe} jamás fabrica en real, got: {result}"
        );
    }
    // Pins real byte-idénticos: el guard es invisible sin `i` en el punto.
    assert_eq!(eval_full("limit(sin(x)/x, x, 0)").0, "1");
    // F10 canonizó el output (branch-hop eliminado: el eval directo da e^2 y
    // el límite emitía exp(2) — misma clase de valor, forma unificada).
    assert_eq!(eval_full("limit(e^z, z, 2)").0, "e^2");
    let (r, w) = eval_full("limit(1/x, x, 0)");
    assert_eq!(r, "undefined");
    assert!(w.contains("one-sided limits disagree"));
}
#[test]
fn test_eval_limit_multivar_dne_by_paths_f8() {
    // Fase 3 · F8: DNE-por-caminos con testigos CITADOS en el warning; el pin
    // central: caminos que coinciden JAMÁS prueban existencia (residual).
    let eval_wire = |input: &str| -> Value {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        serde_json::from_slice(&out.stdout).expect("Invalid wire output")
    };
    let warnings_of = |wire: &Value| -> String {
        wire["warnings"]
            .as_array()
            .map(|w| {
                w.iter()
                    .filter_map(|x| x["assumption"].as_str())
                    .collect::<Vec<_>>()
                    .join(" | ")
            })
            .unwrap_or_default()
    };
    // Los 4 clásicos: undefined + AMBOS testigos citados.
    for (probe, frag_a, frag_b) in [
        (
            "limit(x*y/(x^2+y^2), [x,y], [0,0])",
            "por y = 0 el límite es 0",
            "por y = x es 1/2",
        ),
        (
            "limit(x^2*y/(x^4+y^2), [x,y], [0,0])",
            "por y = 0 el límite es 0",
            "por y = x^2 es 1/2",
        ),
        (
            "limit((x^2-y^2)/(x^2+y^2), [x,y], [0,0])",
            "por y = 0 el límite es 1",
            "por x = 0 es -1",
        ),
        (
            "limit(x*y^2/(x^2+y^4), [x,y], [0,0])",
            "por y = 0 el límite es 0",
            "por x = y^2 es 1/2",
        ),
    ] {
        let wire = eval_wire(probe);
        assert_eq!(
            wire["result"].as_str().unwrap_or(""),
            "undefined",
            "{probe}"
        );
        let w = warnings_of(&wire);
        assert!(w.contains(frag_a), "{probe}: falta testigo A en {w}");
        assert!(w.contains(frag_b), "{probe}: falta testigo B en {w}");
    }
    // EL PIN CENTRAL, GRADUADO POR F8b: x²y/(x²+y²) ahora da 0 PROBADO por
    // squeeze (cota polar |f| ≤ C·r — teorema, no caminos): la cláusula
    // "jamás existencia desde finitos caminos" SIGUE INTACTA — el valor viene
    // del prover de acotación, y la cita del warning lo dice.
    let wire = eval_wire("limit(x^2*y/(x^2+y^2), [x,y], [0,0])");
    assert_eq!(wire["result"].as_str().unwrap_or(""), "0");
    let w = warnings_of(&wire);
    assert!(
        w.contains("cota polar"),
        "el 0 debe venir CITADO por squeeze, no por caminos: {w}"
    );
    // Pins F7: continuidad intacta.
    assert_eq!(
        eval_wire("limit(x*y/(x^2+y^2), [x,y], [1,1])")["result"]
            .as_str()
            .unwrap_or(""),
        "1/2"
    );
}
#[test]
fn test_eval_limit_multivar_continuity_f7() {
    // Fase 3 · F7: limit(f,[vars],[punto]) por continuidad PROBADA (den
    // sustituido pliega a racional ≠0, exacto) — lo demás queda residual
    // honesto (la existencia multivar es path-dependent; F8 posee el lado
    // negativo). Bajo complex hereda la disciplina del kill-switch F0.
    let eval_result = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(eval_result("limit(x^2+y^2, [x,y], [1,2])"), "5");
    assert_eq!(eval_result("limit(x*y/(x^2+y^2), [x,y], [1,1])"), "1/2");
    assert_eq!(
        eval_result("limit(sin(x+y)/(1+x^2+y^2), [x,y], [0,0])"),
        "0"
    );
    // Residuales honestos: punto con infinity (at-infinity multivar fuera de
    // scope), punto imaginario. (El singular [0,0] con caminos discrepantes
    // GRADUÓ en F8 a undefined+testigos — pin migrado a su test; el que
    // COINCIDE en caminos sigue residual y lo pinea el test de F8.)
    for probe in [
        "limit(x^2*y, [x,y], [infinity,1])",
        "limit(x+y, [x,y], [i,0])",
    ] {
        let r = eval_result(probe);
        assert!(
            r.starts_with("limit("),
            "{probe} debe quedar eco residual, got: {r}"
        );
    }
    // Bajo complex: residual (la continuidad razona con orden real).
    let out = cli()
        .args([
            "eval",
            "limit(x^2+y^2, [x,y], [1,2])",
            "--value-domain",
            "complex",
            "--format",
            "json",
        ])
        .output()
        .expect("Failed to run CLI");
    let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
    assert!(wire["result"].as_str().unwrap_or("").starts_with("limit("));
    // Pins: univar paramétrico intacto; comando wire univar intacto; el
    // anidado GRADUÓ en F9 (evalúa — pin migrado de eco a valor).
    assert!(eval_result("limit(x*y/(x^2+y^2), x, 0)").starts_with("limit("));
    assert_eq!(eval_result("limit(x^2, x, 2)"), "4");
    assert_eq!(eval_result("1+limit(x^2, x, 2)"), "5");
}
#[test]
fn test_eval_limit_complex_selective_regrant_f11() {
    // Fase 3 · F11: re-otorgo SELECTIVO bajo complex para formas ANALÍTICAS
    // (meromorfas sin singularidades esenciales), decidido exacto: sustitución
    // directa con den≠0 Gaussiano probado; forma ENTERA en cualquier punto
    // finito; punto real 0/0 delega al motor real (meromorfa + límite real
    // finito ⇒ límite complejo). Los 7 WRONG de F0 son never-fabricate
    // PERMANENTES — sus formas fallan el shape por construcción.
    let eval_complex = |input: &str| -> String {
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
    // Re-otorgados con justificación analítica.
    assert_eq!(eval_complex("limit(z^2+1, z, i)"), "0");
    assert_eq!(eval_complex("limit(sin(z), z, i)"), "i·sinh(1)");
    assert_eq!(eval_complex("limit(1/(z^2+1), z, 2*i)"), "-1/3");
    assert_eq!(eval_complex("limit(sin(z)/z, z, 0)"), "1");
    assert_eq!(eval_complex("limit((z^2-1)/(z-1), z, 1)"), "2");
    assert_eq!(eval_complex("limit(exp(z), z, i*pi)"), "-1");
    assert_eq!(eval_complex("limit(e^(2*z), z, i*pi)"), "1");
    // NEVER-FABRICATE permanentes (los 7 WRONG de F0 + conjugate).
    for probe in [
        "limit(e^(-1/z^2), z, 0)",
        "limit(z*sin(1/z), z, 0)",
        "limit(tanh(z), z, i*pi/2)",
        "limit(atan(z), z, 2*i)",
        "limit(1/(z^2+1), z, i)",
        "limit(1/z^2, z, 0)",
        "limit(e^z, z, infinity)",
        "limit(conjugate(z)/z, z, 0)",
    ] {
        let r = eval_complex(probe);
        assert!(
            r.starts_with("limit("),
            "{probe} JAMÁS fabrica bajo complex, got: {r}"
        );
    }
    // Unlock del wire: `i` desnudo ya parsea; en REAL declina con warning (F0).
    let out = cli()
        .args(["eval", "limit(x, x, i)", "--format", "json"])
        .output()
        .expect("Failed to run CLI");
    let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
    assert!(wire["result"].as_str().unwrap_or("").starts_with("limit("));
    // Real byte-idéntico.
    let out = cli()
        .args(["eval", "limit(sin(x)/x, x, 0)", "--format", "json"])
        .output()
        .expect("Failed to run CLI");
    let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
    assert_eq!(wire["result"].as_str().unwrap_or(""), "1");
}
#[test]
fn test_eval_limit_output_fold_f10() {
    // Fase 3 · F10: el output RESUELTO del límite pasa por el pipeline de
    // simplify antes de emitirse (la rama residual conserva su cleanup — pin
    // round-trip); el input PreSimplifyMode::Off no se toca. En real: byte-
    // idéntico o canonización (exp(2)→e^2, unificado con eval directo).
    let eval_result = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(eval_result("limit(e^z, z, 2)"), "e^2");
    assert_eq!(eval_result("limit((x^2-1)/(x-1), x, 1)"), "2");
    assert_eq!(eval_result("limit((1+1/x)^x, x, infinity)"), "e");
    assert_eq!(eval_result("limit(x*y, x, 2)"), "2·y");
    // La rama RESIDUAL conserva su forma re-emitible (round-trip intacto).
    assert_eq!(
        eval_result("limit(e^(i*x), x, infinity)"),
        "limit(e^(i·x), x, infinity)"
    );
    // Los DNE con testigos no cambian de shape.
    assert_eq!(eval_result("limit(1/x, x, 0)"), "undefined");
    assert_eq!(eval_result("limit(sin(1/x), x, 0)"), "undefined");
}
#[test]
fn test_eval_limit_iterated_and_composed_f9() {
    // Fase 3 · F9: limit anidado/compuesto/iterado evalúa vía el motor
    // univariado y SOLO emite en resolución completa (inner residual → declina
    // all-or-nothing); el wire clasifica la forma (single call / compuesta /
    // malformada) por matching de paréntesis — las compuestas van a eval.
    let eval_result = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Iterados (la narración dice ITERADO, jamás doble).
    assert_eq!(eval_result("limit(limit(x^2+y^2,x,1),y,2)"), "5");
    assert_eq!(eval_result("limit(limit(x*y,x,2),y,3)"), "6");
    // Compuestos.
    assert_eq!(eval_result("limit(sin(x)/x,x,0)*2"), "2");
    assert_eq!(eval_result("limit(x^2,x,2)+limit(x,x,1)"), "5");
    assert_eq!(eval_result("diff(limit(x^2,x,2)*t,t)"), "4");
    // Inner residual/DNE → all-or-nothing (eco completo, jamás operar sobre
    // un residual como valor).
    assert!(eval_result("limit(limit(x/y,y,0),x,0)").starts_with("limit("));
    // Complex declina (disciplina F0; F11 re-otorga).
    let out = cli()
        .args([
            "eval",
            "1+limit(x^2,x,2)",
            "--value-domain",
            "complex",
            "--format",
            "json",
        ])
        .output()
        .expect("Failed to run CLI");
    let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
    assert!(wire["result"].as_str().unwrap_or("").contains("limit("));
    // Pins del comando wire: las formas single-call byte-idénticas.
    assert_eq!(eval_result("limit(x^2, x, 2)"), "4");
    assert_eq!(eval_result("limit(sin(x)/x, x, 0)"), "1");
    assert_eq!(eval_result("limit(1/x, x, 0)"), "undefined");
    assert_eq!(eval_result("limit(1/x, x, 0, right)"), "infinity");
    // Never-fabricate heredados: cuerpo-con-I sigue residual en ambos dominios.
    assert!(eval_result("limit(e^(i*x), x, infinity)").starts_with("limit("));
}
#[test]
fn test_eval_log_sum_limit_at_infinity_and_convergent_degree_n_improper_integral() {
    // `lim_{x→∞} Σ cᵢ·ln(pᵢ(x))` with `Σ cᵢ·deg pᵢ = 0` is the finite `Σ cᵢ·ln(lead pᵢ)`, not the
    // `+∞−∞` residual the limit engine left for N≥3 terms (it only combined a two-term `ln p − ln q`).
    // `log_sum_limit_at_infinity` decides it from polynomial growth, which lets a partial-fraction log
    // antiderivative of an `∫_a^∞ p/q` with a degree-n denominator that splits into LINEAR factors over
    // ℚ resolve at the boundary. A leftover irreducible quadratic factor (an arctan term) still
    // declines — the next peldaño.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // The N-term sum-of-logs limit (degree sum 0 -> finite; nonzero -> ±∞).
    assert_eq!(r("limit(ln(x-1)+ln(x+1)-2*ln(x), x, infinity)"), "0");
    assert_eq!(r("limit(1/2*ln(x-1)+1/2*ln(x+1)-ln(x), x, infinity)"), "0");
    assert_eq!(r("limit(ln(2*x^2+1)-ln(x^2), x, infinity)"), "ln(2)");
    assert_eq!(r("limit(3*ln(x)-ln(x^3), x, infinity)"), "0");
    // Convergent improper integrals with a degree-n denominator factoring over ℚ into linears.
    // ∫_2^∞ 1/(x³−x) = ln2 − ½ln3; ∫_3^∞ 1/(x³−4x) = ⅛ln(9/5).
    assert_eq!(r("integrate(1/(x^3-x), x, 2, oo)"), "1/2·(2·ln(2) - ln(3))");
    assert_eq!(
        r("integrate(1/(x^3-4*x), x, 3, oo)"),
        "1/8·(2·ln(3) - ln(5))"
    );
    // Soundness preserved: the single bare log still diverges, and the `−∞` side stays residual.
    assert_eq!(r("limit(ln(x^2-1), x, infinity)"), "infinity");
}
#[test]
fn test_eval_arctan_plus_log_boundary_limit_and_irreducible_quadratic_improper_integral() {
    // A rational partial-fraction antiderivative with an irreducible quadratic factor mixes an
    // `arctan` term with the logs. When the `arctan` sits BETWEEN the logs in the Add tree the
    // additive fallback splits the logs individually into `+∞ − ∞` and stalls. `log_sum_limit_at_infinity`
    // now absorbs the arctan terms (`arctan(q) → sign(lead q)·π/2`) alongside the log block, so the
    // boundary limit resolves regardless of order and `∫_a^∞ 1/(x⁴−1)` computes.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // The mixed arctan+log boundary limit (arctan interleaved between the two logs) -> finite.
    assert_eq!(
        r("limit(1/4*ln(x-1)-1/2*arctan(x)-1/4*ln(x+1), x, infinity)"),
        "-1/4·pi"
    );
    // ∫_2^∞ 1/(x⁴−1): denominator (x²−1)(x²+1) -> two linear logs + an arctan. Numerically ≈ 0.04283.
    assert_eq!(
        r("integrate(1/(x^4-1), x, 2, oo)"),
        "1/4·(ln(3) + 2·arctan(2)) - 1/4·pi"
    );
    // Soundness preserved for the irreducible-quadratic family: pole in range -> divergent, ~1/x tail -> +∞.
    assert_eq!(r("integrate(1/(x^4-1), x, 0, oo)"), "undefined"); // pole at x=1
    assert_eq!(r("integrate(x^3/(x^4-1), x, 2, oo)"), "infinity"); // ~1/x tail
                                                                   // A PRE-FACTORED denominator with an irreducible quadratic factor: the antiderivative is
                                                                   // `Add(__hold(−½·arctan x − ¼·ln(x²+1)), ½·ln|x−1|)`; the surviving inner `__hold` used to block
                                                                   // the boundary limit. Stripping ALL holds first lets it fold. Numerically ≈ 0.170535673.
    assert_eq!(
        r("integrate(1/((x-1)*(x^2+1)), x, 2, oo)"),
        "1/4·(ln(5) + 2·arctan(2)) - 1/4·pi"
    );
    // The expanded-equivalent denominator computes to the SAME value.
    assert_eq!(
        r("integrate(1/(x^3-x^2+x-1), x, 2, oo)"),
        "1/4·(ln(5) + 2·arctan(2)) - 1/4·pi"
    );
    // Soundness preserved here too: pole at x=1 in range -> undefined; ~1/x tail -> +∞.
    assert_eq!(r("integrate(1/((x-1)*(x^2+1)), x, 0, oo)"), "undefined");
    assert_eq!(r("integrate(x^2/((x-1)*(x^2+1)), x, 2, oo)"), "infinity");
    // Edge: a lone arctan and a pure arctan pair are left to the unary/additive rules.
    // F10 canonizó el output del límite (branch-hop eliminado: eval directo de
    // pi/2 da 1/2·pi — la forma canónica del pipeline).
    assert_eq!(r("limit(arctan(x), x, infinity)"), "1/2·pi");
}
#[test]
fn test_eval_divergent_p_series_is_infinity() {
    // A divergent p-series `Σ c/n^p` with `0 < p ≤ 1` (the harmonic series and slower) now reports its
    // divergence as `±infinity` instead of a residual: every term eventually shares the sign of `c`.
    // The ζ-convergent `p > 1` cases, alternating series, and a sum that includes the `n = 0` pole are
    // unchanged.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Harmonic and slower (p ≤ 1) -> diverges with the sign of the coefficient.
    assert_eq!(r("sum(1/n, n, 1, oo)"), "infinity");
    assert_eq!(r("sum(2/n, n, 1, oo)"), "infinity");
    assert_eq!(r("sum(1/(2*n), n, 1, oo)"), "infinity");
    assert_eq!(r("sum(1/sqrt(n), n, 1, oo)"), "infinity"); // p = 1/2
    assert_eq!(r("sum(-1/n, n, 1, oo)"), "-infinity");
    assert_eq!(r("sum(1/n, n, 5, oo)"), "infinity"); // tail from any start ≥ 1 still diverges
                                                     // MUST NOT regress: p > 1 converges (ζ), alternating is conditionally convergent, n = 0 is a pole.
    assert_eq!(r("sum(1/n^2, n, 1, oo)"), "1/6·pi^2");
    assert_eq!(
        r("sum(1/n^(3/2), n, 1, oo)"),
        "sum(1 / n^(3/2), n, 1, infinity)"
    ); // ζ(3/2), no closed form
    assert_eq!(r("sum(1/n^3, n, 1, oo)"), "sum(1 / n^3, n, 1, infinity)"); // ζ(3), deliberate residual
    assert_eq!(
        r("sum((-1)^n/n, n, 1, oo)"),
        "sum((-1)^n / n, n, 1, infinity)"
    ); // alternating
    assert_eq!(r("sum(1/n, n, 0, oo)"), "undefined"); // n = 0 pole in range
}
#[test]
fn test_eval_limit_abs_finite_tail_at_infinity() {
    // `lim_{x→∞} |u(x)| = |L|` when the rational argument has a finite tail L — previously only the
    // divergent case (`abs → +∞`) was handled, so `|(x-1)/(x+1)|` stayed an unevaluated residual.
    // Composing through `ln` (`lim ln(|u|) = ln(|L|)`) is what an improper rational integral with a
    // log antiderivative needs at its infinite bound.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("limit(abs((x-1)/(x+1)), x, inf)"), "1");
    assert_eq!(r("limit(abs((2*x+1)/(x-3)), x, inf)"), "2");
    assert_eq!(r("limit(ln(abs((x-1)/(x+1))), x, inf)"), "0");
    assert_eq!(r("limit(ln(abs((3*x+1)/(x+1))), x, inf)"), "ln(3)");
    // Improper integral unlocked by the composition: ∫₁^∞ 1/(x(x+1)) = ln 2.
    assert_eq!(r("integrate(1/(x*(x+1)), x, 1, inf)"), "-ln(1/2)");
    // Controls: a divergent abs still → ∞, a finite-point abs is unchanged, plain ln/sqrt unaffected.
    assert_eq!(r("limit(abs(x^2-x), x, inf)"), "infinity");
    assert_eq!(r("limit(abs(x-3), x, 5)"), "2");
    assert_eq!(r("limit(ln(x^2+1), x, inf)"), "infinity");
    assert_eq!(r("limit(sqrt((x^2+1)/x^2), x, inf)"), "1");
}
#[test]
fn test_eval_finite_plus_infinity_absorbs_in_both_modes() {
    // `finite + ∞ = ∞` (absorption). In plain mode `∞` was treated as a symbolic atom, so the
    // "symbolic atom + literal" shortcut returned `∞ + 1` UNEVALUATED — diverging from `--steps`,
    // which absorbs it. `∞`/`undefined` are no longer symbolic atoms; finite constants (`π`,`e`,`i`)
    // still are.
    for (input, expected) in [
        ("inf+1", "infinity"),
        ("1+inf", "infinity"),
        ("inf+5", "infinity"),
        ("inf-1", "infinity"),
        ("2+inf+3", "infinity"),
        ("(-inf)+3", "-infinity"),
        // Finite atoms stay symbolic; undefined still propagates.
        ("pi+1", "1 + pi"),
        ("e+2", "2 + e"),
        ("undefined+1", "undefined"),
        ("inf+x", "x + infinity"),
    ] {
        let plain = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(plain.status.success(), "plain {input}");
        let plain_wire: Value = serde_json::from_slice(&plain.stdout).expect("Invalid wire output");
        let steps = cli()
            .args(["eval", input, "--format", "json", "--steps", "on"])
            .output()
            .expect("Failed to run CLI");
        assert!(steps.status.success(), "steps {input}");
        let steps_wire: Value = serde_json::from_slice(&steps.stdout).expect("Invalid wire output");
        assert_eq!(
            plain_wire["result"].as_str(),
            Some(expected),
            "plain {input}"
        );
        assert_eq!(
            plain_wire["result"].as_str(),
            steps_wire["result"].as_str(),
            "plain vs --steps divergence for {input}"
        );
    }
}
