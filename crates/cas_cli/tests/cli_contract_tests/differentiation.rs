use super::*;

#[test]
fn test_eval_diff_sign_polynomial_returns_zero_with_nonzero_domain_json() {
    let output = cli()
        .args([
            "eval",
            "diff(sign(x), x)",
            "--format",
            "json",
            "--steps",
            "on",
        ])
        .output()
        .expect("Failed to run CLI");

    assert!(
        output.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).unwrap();
    let wire: Value = serde_json::from_str(&stdout).expect("Invalid wire output");
    let steps = wire["steps"].as_array().expect("steps array");
    let first_substeps = steps[0]["substeps"].as_array().expect("substeps array");

    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "0");
    assert_eq!(wire["required_display"], serde_json::json!(["x ≠ 0"]));
    assert_eq!(steps[0]["rule"], "Calcular la derivada");
    assert!(
        first_substeps
            .iter()
            .any(|substep| substep["title"] == "Usar derivada de sign(u) fuera de u = 0"),
        "missing sign derivative substep: {first_substeps:?}"
    );
}
#[test]
fn test_eval_wire_returns_undefined_for_empty_diff_domain() {
    let output = cli()
        .args(["eval", "diff(atanh(sqrt(x^2+2)), x)", "--format", "json"])
        .output()
        .expect("Failed to run CLI");

    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).unwrap();
    let wire: Value = serde_json::from_str(&stdout).expect("Invalid wire output");
    assert_eq!(wire["result"], "undefined");
    assert_eq!(
        wire["blocked_hints"],
        serde_json::Value::Null,
        "undefined domain is now explained by the derivative step, not a blocked residual"
    );
}
#[test]
fn test_eval_scaled_arctan_surd_diff_stays_off_rationalize_overflow_route() {
    let output = cli()
        .args([
            "eval",
            "diff(7*arctan((2*x+1)/sqrt(3))/sqrt(3), x)",
            "--format",
            "json",
        ])
        .output()
        .expect("Failed to run CLI");

    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    let wire: Value = serde_json::from_str(&stdout).expect("Invalid wire output");

    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "7 / (2\u{00b7}(x^2 + x + 1))");
    assert!(
        !stderr.contains("depth_overflow") && !stderr.contains("WARN"),
        "scaled arctan-surd diff should use the compact route, got stderr:\n{stderr}"
    );
}
#[test]
fn test_eval_atanh_exact_square_symbolic_denominator_diff_stays_compact_without_overflow() {
    let input = "diff(atanh(sqrt(4*x+4)/a), x)";
    let output = cli()
        .args(["eval", input, "--format", "json", "--steps", "on"])
        .output()
        .expect("Failed to run CLI");

    assert!(
        output.status.success(),
        "CLI failed for {input}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    let wire: Value = serde_json::from_str(&stdout).expect("Invalid wire output");

    assert_eq!(wire["ok"], true);
    assert_eq!(
        wire["result"],
        "a / (sqrt(x + 1)\u{00b7}(a^2 - 4\u{00b7}x - 4))"
    );
    assert_eq!(
        wire["required_display"],
        serde_json::json!(["a \u{2260} 0", "a^2 - 4\u{00b7}x - 4 > 0", "x > -1"])
    );
    assert!(
        !stderr.contains("depth_overflow") && !stderr.contains("WARN"),
        "atanh exact-square symbolic denominator diff should stay off the overflow route, got stderr:\n{stderr}"
    );

    let steps = wire["steps"].as_array().expect("steps should be present");
    for expected_rule in [
        "Reconocer un cuadrado perfecto bajo la raíz",
        "Sacar constante de una fracción",
        "Calcular la derivada",
    ] {
        assert!(
            steps.iter().any(|step| step["rule"] == expected_rule),
            "expected rule {expected_rule} in public trace for {input}, got {steps:?}"
        );
    }
    let derivative_step = steps
        .iter()
        .find(|step| step["rule"] == "Calcular la derivada")
        .expect("expected derivative step");
    let substeps = derivative_step["substeps"]
        .as_array()
        .expect("derivative step should expose substeps");
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Usar regla de la cadena"),
        "expected chain-rule substep for {input}, got {substeps:?}"
    );
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Identificar u y du"),
        "expected u/du substep for {input}, got {substeps:?}"
    );
}
#[test]
fn test_eval_diff_periodic_required_display_preserves_argument_scale() {
    let input = "diff(sec((3*x+2)/2), x)";
    let output = cli()
        .args(["eval", input, "--format", "json", "--steps", "on"])
        .output()
        .expect("Failed to run CLI");

    assert!(
        output.status.success(),
        "CLI failed for {input}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    let wire: Value = serde_json::from_str(&stdout).expect("Invalid wire output");

    assert_eq!(wire["ok"], true);
    assert_eq!(
        wire["result"],
        "3/2\u{00b7}sec((3\u{00b7}x + 2) / 2)\u{00b7}tan((3\u{00b7}x + 2) / 2)"
    );
    assert_eq!(
        wire["required_display"],
        serde_json::json!(["cos((3\u{00b7}x + 2) / 2) \u{2260} 0"])
    );
    let required_text = wire["required_display"]
        .as_array()
        .expect("required_display array")
        .iter()
        .map(|value| value.as_str().expect("required_display string"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        !required_text.contains("cos(3\u{00b7}x + 2) \u{2260} 0"),
        "periodic zero-set display must not scale-normalize the argument for {input}: {required_text}"
    );
    assert!(
        !stderr.contains("depth_overflow") && !stderr.contains("WARN"),
        "periodic required-display diff should stay off fragile routes for {input}, got stderr:\n{stderr}"
    );

    let steps = wire["steps"].as_array().expect("steps should be present");
    assert!(
        steps
            .iter()
            .any(|step| step["rule"] == "Calcular la derivada"),
        "expected derivative step in public trace for {input}, got {steps:?}"
    );
}
#[test]
fn test_eval_plain_reciprocal_trig_log_diff_stays_off_depth_overflow_route() {
    let cases = [
        (
            "diff(ln(sec(sqrt(x))+tan(sqrt(x))), x)",
            "1 / (2\u{00b7}sqrt(x)\u{00b7}cos(sqrt(x)))",
            vec![
                "cos(sqrt(x)) \u{2260} 0",
                "tan(sqrt(x)) + sec(sqrt(x)) > 0",
                "x > 0",
            ],
        ),
        (
            "diff(ln(csc(sqrt(x))-cot(sqrt(x))), x)",
            "1 / (2\u{00b7}sqrt(x)\u{00b7}sin(sqrt(x)))",
            vec![
                "sin(sqrt(x)) \u{2260} 0",
                "csc(sqrt(x)) - cot(sqrt(x)) > 0",
                "x > 0",
            ],
        ),
        (
            "diff(ln(sec(sqrt(3*x+1))+tan(sqrt(3*x+1))), x)",
            "3 / (2\u{00b7}sqrt(3\u{00b7}x + 1)\u{00b7}cos(sqrt(3\u{00b7}x + 1)))",
            vec![
                "cos(sqrt(3\u{00b7}x + 1)) \u{2260} 0",
                "tan(sqrt(3\u{00b7}x + 1)) + sec(sqrt(3\u{00b7}x + 1)) > 0",
                "x > -1/3",
            ],
        ),
        (
            "diff(ln(csc(sqrt(3*x+1))-cot(sqrt(3*x+1))), x)",
            "3 / (2\u{00b7}sqrt(3\u{00b7}x + 1)\u{00b7}sin(sqrt(3\u{00b7}x + 1)))",
            vec![
                "sin(sqrt(3\u{00b7}x + 1)) \u{2260} 0",
                "csc(sqrt(3\u{00b7}x + 1)) - cot(sqrt(3\u{00b7}x + 1)) > 0",
                "x > -1/3",
            ],
        ),
    ];

    for (input, expected_result, expected_required_display) in cases {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .unwrap_or_else(|err| panic!("Failed to run CLI for {input}: {err}"));

        assert!(output.status.success(), "CLI failed for {input}");

        let stdout = String::from_utf8(output.stdout).unwrap();
        let stderr = String::from_utf8(output.stderr).unwrap();
        let wire: Value = serde_json::from_str(&stdout).expect("Invalid wire output");

        assert_eq!(wire["ok"], true, "input: {input}");
        assert_eq!(wire["result"], expected_result, "input: {input}");
        let result_latex = wire["result_latex"]
            .as_str()
            .expect("result_latex should be a string");
        assert!(
            result_latex.contains("\\sqrt") && !result_latex.contains("}^{-"),
            "post-calculus LaTeX should mirror the compact reciprocal-root display for {input}, got: {result_latex}"
        );
        let mut actual_required = wire["required_display"]
            .as_array()
            .expect("required_display array")
            .iter()
            .map(|condition| condition.as_str().expect("required condition").to_owned())
            .collect::<Vec<_>>();
        let mut expected_required = expected_required_display
            .iter()
            .map(|condition| (*condition).to_string())
            .collect::<Vec<_>>();
        actual_required.sort();
        expected_required.sort();
        assert_eq!(actual_required, expected_required, "input: {input}");
        assert!(
            wire.get("blocked_hints").is_none(),
            "successful reciprocal-trig log diff should not surface non-actionable cycle hints for {input}: {:?}",
            wire["blocked_hints"]
        );
        assert!(
            !stderr.contains("depth_overflow") && !stderr.contains("WARN"),
            "plain reciprocal-trig log diff should use the direct route for {input}, got stderr:\n{stderr}"
        );
    }

    let output = cli()
        .args([
            "eval",
            "diff(ln(sec(sqrt(x))+tan(sqrt(x))), x)",
            "--format",
            "json",
            "--steps",
            "on",
        ])
        .output()
        .expect("Failed to run CLI with steps enabled");
    assert!(output.status.success());
    let wire: Value =
        serde_json::from_slice(&output.stdout).expect("Invalid wire output with steps enabled");
    let final_step = wire["steps"]
        .as_array()
        .expect("steps array")
        .last()
        .expect("final step");
    let final_after_latex = final_step["after_latex"]
        .as_str()
        .expect("final after_latex string");
    assert!(
        final_after_latex.contains("\\sqrt{x}") && !final_after_latex.contains("{x}^{-"),
        "post-calculus step LaTeX should mirror the compact reciprocal-root display, got: {final_after_latex}"
    );
    for step in wire["steps"].as_array().expect("steps array") {
        for field in ["rule_latex", "before_latex", "after_latex"] {
            let Some(latex) = step[field].as_str() else {
                continue;
            };
            assert!(
                !latex.contains("{x}^{-"),
                "calculus step {field} should not leak reciprocal-root power notation, got: {latex}"
            );
        }
    }
}
#[test]
fn test_eval_plain_log_tan_sqrt_diff_uses_sqrt_in_scaled_trig_argument() {
    let output = cli()
        .args(["eval", "diff(ln(tan(sqrt(x))), x)", "--format", "json"])
        .output()
        .expect("Failed to run CLI");

    assert!(output.status.success());
    let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");

    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "1 / (sin(2\u{00b7}sqrt(x))\u{00b7}sqrt(x))");
    assert!(
        !wire["result"]
            .as_str()
            .expect("result string")
            .contains("x^(1/2)"),
        "post-calculus display should not leak half-power notation in scaled trig arguments"
    );
    let mut required_display = wire["required_display"]
        .as_array()
        .expect("required_display array")
        .iter()
        .map(|value| value.as_str().expect("required_display string"))
        .collect::<Vec<_>>();
    required_display.sort_unstable();
    // `cos(sqrt(x)) ≠ 0` is now listed explicitly (tan(sqrt(x)) requires it), matching the sibling
    // `diff(ln(tan(sqrt(x))+1))` case below — the derivative is only valid where the original
    // tan-containing function is defined, even though `cos≠0` is implied by `tan(sqrt(x)) > 0`.
    assert_eq!(
        required_display,
        ["cos(sqrt(x)) \u{2260} 0", "tan(sqrt(x)) > 0", "x > 0"]
    );
}
#[test]
fn test_diff_cancelling_reciprocal_trig_product_keeps_domain_condition() {
    // A reciprocal-trig factor (tan/sec → cos≠0, cot/csc → sin≠0) that CANCELS away in a product
    // must still impose its domain condition on the derivative: the original function is undefined
    // where the cancelled factor blew up, so the derivative does not exist there either. Before the
    // fix these returned the derivative with NO condition (e.g. diff(tan(x)*cos(x)) → cos(x) on all
    // of ℝ, though tan(x)·cos(x) is undefined at cos(x)=0).
    for (input, expected) in [
        ("diff(sec(x)*cos(x), x)", "cos(x) \u{2260} 0"),
        ("diff(tan(x)*cos(x), x)", "cos(x) \u{2260} 0"),
        ("diff(cot(x)*sin(x), x)", "sin(x) \u{2260} 0"),
        ("diff(sin(x)*cot(x), x)", "sin(x) \u{2260} 0"),
        ("diff(csc(x)*sin(x), x)", "sin(x) \u{2260} 0"),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        let displays = wire["required_display"]
            .as_array()
            .expect("required_display array");
        assert!(
            displays.iter().any(|v| v.as_str() == Some(expected)),
            "{input}: expected required condition `{expected}`, got {displays:?}"
        );
    }

    // Already-conditioned single function must NOT gain a duplicate: diff(tan(x)) carries exactly
    // one cos(x) ≠ 0 (from the 1/cos² result), and the differand re-attachment dedupes against it.
    let output = cli()
        .args(["eval", "diff(tan(x), x)", "--format", "json"])
        .output()
        .expect("Failed to run CLI");
    let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
    let cos_conditions = wire["required_display"]
        .as_array()
        .expect("required_display array")
        .iter()
        .filter(|v| v.as_str() == Some("cos(x) \u{2260} 0"))
        .count();
    assert_eq!(
        cos_conditions, 1,
        "diff(tan(x)) must not duplicate cos(x) ≠ 0"
    );
}
#[test]
fn test_eval_cosh_cube_difference_does_not_collapse_to_zero() {
    // cosh³(x) − cosh(x) = cosh(x)·(cosh²(x) − 1) = cosh(x)·sinh²(x), which is
    // NOT identically 0. The "Hyperbolic Pythagorean Identity Cancellation
    // Bridge" rule recognised the FactorThenRewrite pattern and, at the root,
    // unconditionally collapsed it to 0 (a wrong-answer, e.g. cosh(3x)−cosh(x)
    // → 0). The fix declines that standalone case, leaving the correct expanded
    // form (just as a plain polynomial y³−y is left unfactored). (The sin/cos
    // analogues already worked.)
    for (input, expected) in [
        ("cosh(x)^3 - cosh(x)", "cosh(x)^3 - cosh(x)"),
        ("4*cosh(x)^3 - 4*cosh(x)", "4·cosh(x)^3 - 4·cosh(x)"),
        // cosh(3x) expands (triple angle) to 4cosh³−3cosh; the difference is
        // 4cosh³−4cosh = 4cosh·sinh², never 0.
        ("cosh(3*x) - cosh(x)", "4·cosh(x)^3 - 4·cosh(x)"),
    ] {
        for mode in ["off", "on"] {
            let output = cli()
                .args(["eval", input, "--format", "json", "--steps", mode])
                .output()
                .expect("Failed to run CLI");
            assert!(output.status.success(), "{input} (steps={mode})");
            let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
            let result = wire["result"].as_str();
            assert_ne!(
                result,
                Some("0"),
                "{input} (steps={mode}): hyperbolic cube difference must not collapse to 0"
            );
            assert_eq!(result, Some(expected), "{input} (steps={mode})");
        }
    }

    // The genuinely-zero hyperbolic Pythagorean identities must still collapse:
    // 4cosh·sinh² + 4cosh − 4cosh³ = 4cosh(sinh² + 1 − cosh²) = 0.
    for input in [
        "4*cosh(x)*sinh(x)^2 + 4*cosh(x) - 4*cosh(x)^3",
        "sinh(2*x+1)*(cosh(2*x+1)^2 - 1) - sinh(2*x+1)^3",
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(
            wire["result"].as_str(),
            Some("0"),
            "{input}: genuine hyperbolic Pythagorean zero identity must still collapse"
        );
    }
}
#[test]
fn test_eval_second_derivative_of_sin_tan_is_numerically_equivalent() {
    // P0-G: `diff(sin(x)·tan(x), x, 2)` returned a NON-equivalent tree (wrong at
    // every sample point). Root cause: `collect_mul_factors_int_pow` returned a
    // repeated base (`2·sin·sin·cos`, a legal mid-pipeline non-canonical tree from
    // the double-angle expansion) as TWO entries, and the factor-from-Add
    // subtraction removed the common exponent from each — over-cancelling a
    // factor. The collector now aggregates duplicates. Same root cause fixed the
    // C5 family `diff((x+tan(x))^n, x)` for n = 3, 4 (dropped a cos / hung).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // (8 - (2·sin·cos)²)/(4·cos³) = (2 - sin²cos²·... ) — numerically f'' of sin·tan.
    assert_eq!(
        r("diff(sin(x)*tan(x), x, 2)"),
        "(8 - (2\u{b7}sin(x)\u{b7}cos(x))^2) / (4\u{b7}cos(x)^3)"
    );
    // The minimal over-cancel repro keeps its sin factor.
    assert_eq!(
        r("simplify((-sin(x)^3*cos(x) + 2*sin(x)^2*cos(x)) / (cos(x)^2*sin(x)))"),
        "(2\u{b7}sin(x) - sin(x)^2) / cos(x)"
    );
    // C5 siblings: n = 3 and n = 4 produce the correct 3(x+tan)²·(1+sec²) shape.
    assert_eq!(
        r("diff((x+tan(x))^3, x)"),
        "3\u{b7}(sin(x) / cos(x) + x)^2\u{b7}(2\u{b7}cos(x)^2 - 1 + 3) / (2\u{b7}cos(x)^2)"
    );
}
#[test]
fn test_eval_diff_multivar_input_latex() {
    // Tanda-2 ciclo 5 arregló el DROP de variables; el cierre vectorial C aplica la
    // decisión del usuario (pregunta abierta #3): ∂ GLOBAL cuando la derivación
    // involucra más de una variable (mixtas O target multivariable); el univariable
    // conserva `d` BYTE-idéntico. Denominador derecha-a-izquierda (convención de
    // parciales mixtas).
    let latex_of = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["input_latex"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(
        latex_of("diff(x^2*y^3, x, y)"),
        "\\frac{\\partial^{2}}{\\partial y \\, \\partial x}({x}^{2}\\cdot {y}^{3})"
    );
    assert_eq!(
        latex_of("diff(x^2*y, x)"),
        "\\frac{\\partial}{\\partial x}(y\\cdot {x}^{2})"
    );
    assert_eq!(
        latex_of("diff(x^5, x, 2)"),
        "\\frac{d^{2}}{dx^{2}}({x}^{5})"
    );
    assert_eq!(
        latex_of("diff(x^2*y^2, x, 2, y)"),
        "\\frac{\\partial^{3}}{\\partial y \\, \\partial x^{2}}({x}^{2}\\cdot {y}^{2})"
    );
    // El 2-args queda BYTE-IDÉNTICO (ambos renderers).
    assert_eq!(latex_of("diff(x^2, x)"), "\\frac{d}{dx}({x}^{2})");
    assert_eq!(
        latex_of("sqrt(diff(x^2,x))"),
        "\\sqrt{\\frac{d}{dx}({x}^{2})}"
    );
}
#[test]
fn test_eval_taylor_substrate_f1() {
    // Fase 3 · F1: (1) singularidad EVITABLE computa vía cancelación de la
    // potencia común re-expandida a order+s; (2) punto singular → residual
    // honesto (eco), JAMÁS `undefined` como respuesta; (3) cap de orden 32.
    let eval_result = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Evitables — los tres clásicos de libro.
    assert_eq!(
        eval_result("taylor(sin(x)/x, x, 0, 4)"),
        "1/120·(x^4 + 120 - 20·x^2)"
    );
    assert_eq!(
        eval_result("taylor((1-cos(x))/x^2, x, 0, 4)"),
        "1/720·(x^4 + 360 - 30·x^2)"
    );
    assert_eq!(
        eval_result("taylor(x/sin(x), x, 0, 4)"),
        "1/6·(7/60·x^4 + x^2 + 6)"
    );
    // Singulares → eco residual (la política V5), nunca `undefined`.
    for probe in [
        "taylor(ln(x), x, 0, 2)",
        "taylor(1/x, x, 0, 3)",
        "taylor(sin(x)/x^2, x, 0, 3)",
        "taylor(cos(x)/x, x, 0, 3)",
    ] {
        let r = eval_result(probe);
        assert!(
            r.starts_with("taylor("),
            "{probe} debe quedar eco residual, got: {r}"
        );
    }
    // Cap explícito: 33 declina, los pins de siempre intactos.
    assert!(eval_result("taylor(exp(x), x, 0, 33)").starts_with("taylor("));
    assert_eq!(
        eval_result("taylor(exp(x), x, 0, 6)"),
        "1/720·(x^6 + 6·x^5 + 30·x^4 + 120·x^3 + 360·x^2 + 720·x + 720)"
    );
    assert_eq!(
        eval_result("taylor(e^(x+y), x, 0, 2)"),
        "1/2·e^y·(x^2 + 2·x + 2)"
    );
    assert_eq!(
        eval_result("series(sin(x), x, 0, 5)"),
        "1/120·(x^5 + 120·x - 20·x^3)"
    );
}
#[test]
fn test_eval_taylor_multivar_f2() {
    // Fase 3 · F2: multi-índice por grado TOTAL (e^(x+y) a orden 2 SIN x²y²),
    // punto singular/lista malformada/cap → residual honesto; univar intacto.
    let eval_result = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(eval_result("taylor(x^2+y^2, [x,y], [0,0], 2)"), "x^2 + y^2");
    assert_eq!(
        eval_result("taylor(e^(x+y), [x,y], [0,0], 2)"),
        "1/2·(x^2 + y^2 + 2·x·y + 2·x + 2·y + 2)"
    );
    assert_eq!(eval_result("taylor(sin(x*y), [x,y], 2)"), "x·y");
    // 2-args: el default multivar es la aproximación cuadrática.
    assert_eq!(
        eval_result("taylor(e^(x+y), [x,y])"),
        "1/2·(x^2 + y^2 + 2·x·y + 2·x + 2·y + 2)"
    );
    // Declines honestos: singular, lista malformada, cap de términos.
    for probe in [
        "taylor(ln(x*y), [x,y], [0,0], 2)",
        "taylor(x*y, [x, 2*y], 2)",
        "taylor(e^(x+y), [x,y], [x,0], 2)",
        "taylor(e^(x+y), [x,y], 20)",
    ] {
        let r = eval_result(probe);
        assert!(
            r.starts_with("taylor("),
            "{probe} debe declinar a eco residual, got: {r}"
        );
    }
    // Pin: el univar no se mueve.
    assert_eq!(
        eval_result("taylor(exp(x), x, 0, 6)"),
        "1/720·(x^6 + 6·x^5 + 30·x^4 + 120·x^3 + 360·x^2 + 720·x + 720)"
    );
}
#[test]
fn test_eval_gradient_verb() {
    // Fase 2 V3: `gradient(f, [vars])` / alias `grad` — the first vectorial verb. Output
    // is an n×1 COLUMN (the parser's own [x,y] convention), re-enterable and composable.
    // The list-of-vars arity is EXCLUSIVE to the verbs (arity-2 exact); `diff`'s 3+-arity
    // SymPy convention keeps its own owner.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("gradient(x^2+y^2,[x,y])"), "[[2·x], [2·y]]");
    assert_eq!(r("gradient(x^2*y,[x,y])"), "[[2·x·y], [x^2]]");
    assert_eq!(r("grad(x*y*z,[x,y,z])"), "[[y·z], [x·z], [x·y]]");
    assert_eq!(
        r("gradient(sin(x*y),[x,y])"),
        "[[y·cos(x·y)], [x·cos(x·y)]]"
    );
    // Composition (the result is a live Matrix): norm — pin THIS form, the engine does
    // not extract the square factor from the radical — and the directional derivative.
    assert_eq!(r("norm(gradient(x^2+y^2,[x,y]))"), "sqrt(4·x^2 + 4·y^2)");
    assert_eq!(r("dot(gradient(x^2*y,[x,y]),[1,0])"), "2·x·y");
    // Honest declines: non-variable list entry, Matrix field (jacobian territory, V4),
    // over-cap var list (VERB_MAX_VARS=8).
    assert_eq!(r("gradient(x^2,[x,2])"), "gradient(x^2, [[x], [2]])");
    assert_eq!(
        r("gradient([x,y],[x,y])"),
        "gradient([[x], [y]], [[x], [y]])"
    );
    assert_eq!(
        r("gradient(x^2, [a,b,c,d,e,f,g,h,k])"),
        "gradient(x^2, [[a], [b], [c], [d], [e], [f], [g], [h], [k]])"
    );
    // Never-confirm fixture: the OTHER verbs stay unregistered until their cycle lands
    // (detector of the gate-without-rule gotcha, in both directions).
    let err_of = |input: &str| -> String {
        let out = cli()
            .args(["eval", input])
            .output()
            .expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stderr).to_string() + &String::from_utf8_lossy(&out.stdout)
    };
    // lineintegral GRADUÓ en F4 (Fase 3) con su aridad REAL 6 — el detector
    // never-confirm se actualiza INTENCIONALMENTE (decisión D10 del scoping:
    // el assert por aridad quedaría verde por accidente si no se migra): la
    // aridad-6 computa, cualquier OTRA aridad sigue "función no definida", y
    // los nombres aún fuera de alcance conservan su decline.
    // Aridades NO registradas de los verbos Fase-3 (lineintegral/surface son
    // arity-6; estas formas arity-2 declinan) — el detector por (nombre,aridad).
    for probe in ["lineintegral(x^2, [x,y])", "surface_integral(x*y, [x,y,z])"] {
        assert!(
            err_of(probe).contains("no definida"),
            "Fase-3 wrong-arity must stay 'función no definida': {probe}"
        );
    }
    // potential GRADUÓ en F6 con su aridad real 2 (assert separado por
    // semántica, decisión D10); las presas del never-confirm pasan a los
    // nombres FUERA del norte, que jamás deben registrarse por accidente.
    // dsolve GRADUÓ en Fase 4 · O0 como special-command (decisión D10 del
    // scoping Fase 4): su pin migra BIDIRECCIONAL abajo — la forma comando
    // computa, el malformado da usage-error EXPLÍCITO, y la forma embebida
    // en expresión sigue declinando por el eval genérico.
    for probe in ["erf(1)", "gamma(5)", "residue(1/z, z, 0)"] {
        assert!(
            err_of(probe).contains("no definida"),
            "nombre fuera del norte debe seguir 'función no definida': {probe}"
        );
    }
    // (a) the well-formed command computes (verification-gated emission).
    let out = err_of("dsolve(diff(y,x)=y^2, y, x)");
    assert!(
        out.contains("y = -1 / (C + x)"),
        "dsolve command form must compute: {out}"
    );
    // (b) malformed `dsolve(y, x)` is an EXPLICIT usage error (pre-pass), no
    // longer the cryptic unknown-function decline.
    let err = err_of("dsolve(y, x)");
    assert!(
        err.contains("contains no diff"),
        "malformed dsolve must be an explicit usage error: {err}"
    );
    // (c) embedded-in-expression form still declines through generic eval.
    assert!(
        err_of("dsolve(y)+1").contains("no definida"),
        "embedded dsolve stays declined by generic eval"
    );
    // Narration: rule name localizes es/en; one keyed substep per component.
    let steps_json = |input: &str, lang: Option<&str>| -> Value {
        let mut args = vec!["eval", input, "--steps", "on", "--format", "json"];
        if let Some(l) = lang {
            args.extend(["--lang", l]);
        }
        let out = cli().args(&args).output().expect("Failed to run CLI");
        serde_json::from_slice(&out.stdout).expect("Invalid wire output")
    };
    let es = steps_json("gradient(x^2+y^2,[x,y])", None);
    assert_eq!(
        es["steps"][0]["rule"].as_str().unwrap(),
        "Calcular el gradiente"
    );
    let subs = es["steps"][0]["substeps"].as_array().expect("substeps");
    assert_eq!(subs.len(), 2, "one substep per component");
    assert!(subs[0]["title"]
        .as_str()
        .unwrap()
        .contains("Derivar respecto de x"));
    let en = steps_json("gradient(x^2+y^2,[x,y])", Some("en"));
    assert_eq!(
        en["steps"][0]["rule"].as_str().unwrap(),
        "Compute the gradient"
    );
    assert!(en["steps"][0]["substeps"][0]["title"]
        .as_str()
        .unwrap()
        .contains("Differentiate with respect to x"));
}
#[test]
fn test_eval_jacobian_hessian_verbs() {
    // Fase 2 V4: jacobian (ROWS = functions, COLUMNS = variables — the orientation pin)
    // and hessian (n×n symmetric, computed as the jacobian of the internal gradient),
    // plus the bracket-aware `equiv` micro-cable that powers the metamorphic fixture.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("jacobian([x^2*y, x+y],[x,y])"), "[[2·x·y, x^2], [1, 1]]");
    assert_eq!(
        r("jacobian([x*y, x+y, sin(x)],[x,y])"),
        "[[y, x], [1, 1], [cos(x), 0]]"
    );
    assert_eq!(r("hessian(x^2*y,[x,y])"), "[[2·y, 2·x], [2·x, 0]]");
    assert_eq!(r("hessian(x^2+y^2,[x,y])"), "[[2, 0], [0, 2]]");
    // det(hessian) composes — the second-derivative-test discriminant.
    assert_eq!(r("det(hessian(x^2*y,[x,y]))"), "-4·x^2");
    // Metamorphic fixture via bracket-aware equiv: hessian ≡ jacobian ∘ gradient.
    assert_eq!(
        r("equiv(jacobian(gradient(x^3*y^2,[x,y]),[x,y]), hessian(x^3*y^2,[x,y]))"),
        "true"
    );
    assert_eq!(r("equiv([x,y],[y,x])"), "false");
    // Scalar equiv pins intact (the shared splitter now tracks brackets).
    assert_eq!(r("equiv(diff(x^2,x), 2*x)"), "true");
    assert_eq!(r("equiv(x+1, x+2)"), "false");
    // Honest declines: scalar target for jacobian (gradient owns scalars), matrix field
    // for hessian, general matrix target for jacobian.
    assert_eq!(r("jacobian(x^2,[x,y])"), "jacobian(x^2, [[x], [y]])");
    assert_eq!(r("hessian([x,y],[x,y])"), "hessian([[x], [y]], [[x], [y]])");
    assert_eq!(
        r("jacobian([[1,2],[3,4]],[x,y])"),
        "jacobian([[1, 2], [3, 4]], [[x], [y]])"
    );
    // Narration: localized rule names + one keyed substep per row.
    let steps_json = |input: &str, lang: Option<&str>| -> Value {
        let mut args = vec!["eval", input, "--steps", "on", "--format", "json"];
        if let Some(l) = lang {
            args.extend(["--lang", l]);
        }
        let out = cli().args(&args).output().expect("Failed to run CLI");
        serde_json::from_slice(&out.stdout).expect("Invalid wire output")
    };
    let es = steps_json("jacobian([x^2*y, x+y],[x,y])", None);
    assert_eq!(
        es["steps"][0]["rule"].as_str().unwrap(),
        "Calcular el jacobiano"
    );
    let subs = es["steps"][0]["substeps"].as_array().expect("substeps");
    assert_eq!(subs.len(), 2, "one substep per row");
    assert!(subs[0]["title"].as_str().unwrap().contains("Fila 1"));
    let en = steps_json("hessian(x^2*y,[x,y])", Some("en"));
    assert_eq!(
        en["steps"][0]["rule"].as_str().unwrap(),
        "Compute the Hessian"
    );
    assert!(en["steps"][0]["substeps"][0]["title"]
        .as_str()
        .unwrap()
        .contains("Row 1"));
}
#[test]
fn test_eval_fractional_binomial_taylor_at_zero() {
    // `taylor((1+x)^α, x, 0, n)` for a fractional α declined at center 0 (the analytic Maclaurin
    // engine has no binomial-series case), although the SAME expansion works at a nonzero center.
    // Falling back to the definition-by-differentiation method at 0 now produces the binomial series.
    // The coefficients are the exact generalized binomials C(α, k).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // sqrt(1+x) = 1 + x/2 - x^2/8 + x^3/16 - 5x^4/128.
    assert_eq!(
        r("taylor(sqrt(1+x),x,0,4)"),
        "1/128·(8·x^3 + 64·x + 128 - 5·x^4 - 16·x^2)"
    );
    // 1/sqrt(1+x) = 1 - x/2 + 3x^2/8 - 5x^3/16.
    assert_eq!(
        r("taylor(1/sqrt(1+x),x,0,3)"),
        "1/2·(3/4·x^2 + 2 - 5/8·x^3 - x)"
    );
    // (1+x)^(1/3) = 1 + x/3 - x^2/9 + 5x^3/81.
    assert_eq!(
        r("taylor((1+x)^(1/3),x,0,3)"),
        "1/9·(5/9·x^3 + 3·x + 9 - x^2)"
    );
    // The analytic-engine cases keep their canonical Maclaurin forms (tried first).
    assert_eq!(
        r("taylor(exp(x),x,0,4)"),
        "1/24·(x^4 + 4·x^3 + 12·x^2 + 24·x + 24)"
    );
    assert_eq!(
        r("taylor(log(1+x),x,0,4)"),
        "1/12·(4·x^3 + 12·x - 3·x^4 - 6·x^2)"
    );
    // The 2-argument form `taylor(f, x)` / `series(f, x)` defaults to a Maclaurin expansion of
    // the default order (6) — the most natural invocation, previously an "undefined" arity error.
    assert_eq!(
        r("taylor(exp(x),x)"),
        "1/720·(x^6 + 6·x^5 + 30·x^4 + 120·x^3 + 360·x^2 + 720·x + 720)"
    );
    assert_eq!(r("taylor(exp(x),x)"), r("taylor(exp(x),x,6)"));
    assert_eq!(
        r("series(1/(1-x),x)"),
        "x^6 + x^5 + x^4 + x^3 + x^2 + x + 1"
    );
    // Textbook / SymPy / Mathematica command aliases parse to the canonical command.
    assert_eq!(r("Taylor(exp(x),x,4)"), r("taylor(exp(x),x,4)"));
    assert_eq!(r("Series(sin(x),x)"), r("series(sin(x),x)"));
    assert_eq!(r("Sum(k,k,1,n)"), r("sum(k,k,1,n)"));
    assert_eq!(r("summation(k^2,k,1,n)"), r("sum(k^2,k,1,n)"));
    assert_eq!(r("prod(k,k,1,5)"), r("product(k,k,1,5)"));
}
#[test]
fn test_eval_two_different_base_exponential_divides_to_a_log() {
    // Two exponentials with DIFFERENT (incompatible-prime) bases: `A·M^x + B·N^x = 0 ⟺ (M/N)^x = −B/A`,
    // i.e. `x = ln(−B/A)/ln(M/N)`. The A=B forms happened to isolate, but the one-sided
    // (`4^x − 9^x = 0`) and both-coefficiented (`5·2^x = 3^x`) forms errored with "Cannot isolate 'x'".
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // `(4/9)^x = 1 ⟹ x = 0` (the ratio is 1, so `ln(1) = 0`).
    assert_eq!(r("solve(4^x - 9^x = 0, x)"), "{ 0 }");
    assert_eq!(r("solve(2^x - 5^x = 0, x)"), "{ 0 }");
    // Non-unit coefficients ⟹ a genuine log. `ln(3/2)/ln(4/9)` folds to the
    // exact rational −1/2 (`4/9 = (2/3)²`, `3/2 = (2/3)⁻¹`); a truly irrational
    // ratio (`ln(1/5)/ln(2/3)`, distinct primes) stays symbolic.
    assert_eq!(r("solve(2*4^x = 3*9^x, x)"), "{ -1/2 }");
    assert_eq!(r("solve(5*2^x = 3^x, x)"), "{ ln(1/5) / ln(2/3) }");
    // SOUNDNESS: `(M/N)^x > 0`, so a non-positive ratio has no real solution.
    assert_eq!(r("solve(4^x + 9^x = 0, x)"), "No solution");
    assert_eq!(r("solve(2^x = -3^x, x)"), "No solution");
    // Controls: same-base polynomial forms and a nonzero-constant RHS are NOT this shape.
    assert_eq!(r("solve(4^x - 3*2^x + 2 = 0, x)"), "{ 0, 1 }");
    assert_eq!(r("solve(2^x = 8, x)"), "{ 3 }");
}
#[test]
fn test_eval_diff_cancelling_bounded_inverse_keeps_domain_condition() {
    // `diff(2·arcsin(x)+2·arccos(x)) → 0` silently dropped the `-1<x<1` differentiability
    // interval when the derivative cancelled (the condition vanished with the √(1-x²) radical).
    // The differand is now walked for bounded-inverse subterms, re-emitting each one's OPEN
    // derivative-domain condition even on cancellation.
    let cond = |input: &str| -> Vec<String> {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["required_display"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    };
    assert!(cond("diff(2*arcsin(x)+2*arccos(x), x)").contains(&"-1 < x < 1".to_string()));
    assert!(cond("diff(arccosh(x)-arccosh(x), x)").contains(&"x > 1".to_string()));
    assert!(cond("diff(arctanh(x)-arctanh(x), x)").contains(&"-1 < x < 1".to_string()));
    // Non-cancelling stays exactly one condition (no duplicate from the new walker).
    assert_eq!(cond("diff(arcsin(x), x)"), vec!["-1 < x < 1".to_string()]);
    // All-real derivative domains gain no spurious condition; plain (non-diff) is untouched.
    assert!(cond("diff(arctan(x)-arctan(x), x)").is_empty());
    assert_eq!(cond("arcsin(x)-arcsin(x)"), vec!["-1 ≤ x ≤ 1".to_string()]);
}
#[test]
fn test_eval_shifted_log_tan_sqrt_diff_finishes_without_depth_overflow() {
    let cases = [
        (
            "diff(ln(tan(sqrt(x))+1), x)",
            "1 / (2\u{00b7}sqrt(x)\u{00b7}cos(sqrt(x))^2\u{00b7}(tan(sqrt(x)) + 1))",
            vec!["x > 0", "cos(sqrt(x)) \u{2260} 0", "tan(sqrt(x)) + 1 > 0"],
        ),
        (
            "diff(ln(1-tan(sqrt(x))), x)",
            "-1 / (2\u{00b7}sqrt(x)\u{00b7}cos(sqrt(x))^2\u{00b7}(1 - tan(sqrt(x))))",
            vec!["x > 0", "cos(sqrt(x)) \u{2260} 0", "1 - tan(sqrt(x)) > 0"],
        ),
        (
            "diff(ln(tan(sqrt(x))-1), x)",
            "1 / (2\u{00b7}sqrt(x)\u{00b7}cos(sqrt(x))^2\u{00b7}(tan(sqrt(x)) - 1))",
            vec!["x > 0", "cos(sqrt(x)) \u{2260} 0", "tan(sqrt(x)) - 1 > 0"],
        ),
        (
            "diff(ln(2+tan(sqrt(x))), x)",
            "1 / (2\u{00b7}sqrt(x)\u{00b7}cos(sqrt(x))^2\u{00b7}(tan(sqrt(x)) + 2))",
            vec!["x > 0", "cos(sqrt(x)) \u{2260} 0", "tan(sqrt(x)) + 2 > 0"],
        ),
        (
            "diff(ln(1+tan(sqrt(2*x+3))), x)",
            "1 / (sqrt(2\u{00b7}x + 3)\u{00b7}cos(sqrt(2\u{00b7}x + 3))^2\u{00b7}(tan(sqrt(2\u{00b7}x + 3)) + 1))",
            vec![
                "x > -3/2",
                "cos(sqrt(2\u{00b7}x + 3)) \u{2260} 0",
                "tan(sqrt(2\u{00b7}x + 3)) + 1 > 0",
            ],
        ),
    ];

    for (input, expected_result, expected_required) in cases {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");

        assert!(output.status.success(), "input: {input}");
        let stderr = String::from_utf8(output.stderr).expect("stderr utf8");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");

        assert_eq!(wire["ok"], true, "input: {input}");
        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert!(
            !stderr.contains("depth_overflow"),
            "shifted tan sqrt diff should stay off the fragile simplification route for {input}, got stderr:\n{stderr}"
        );
        let mut actual_required = wire["required_display"]
            .as_array()
            .expect("required_display array")
            .iter()
            .map(|value| value.as_str().expect("required string").to_owned())
            .collect::<Vec<_>>();
        let mut expected_required = expected_required
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        actual_required.sort();
        expected_required.sort();
        assert_eq!(actual_required, expected_required, "input: {input}");
    }
}
#[test]
fn test_eval_diff_requires_explicit_variable_diagnostic() {
    let output = cli()
        .args(["eval", "diff(sin(e^(x^2)))", "--format", "json"])
        .output()
        .expect("Failed to run CLI");

    assert!(
        output.status.success(),
        "eval should surface the diagnostic as JSON, not a process failure"
    );

    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    let wire: Value = serde_json::from_str(&stdout).expect("Invalid wire output");

    assert!(stderr.is_empty(), "unexpected stderr: {stderr}");
    assert_eq!(wire["ok"], false);
    assert_eq!(wire["kind"], "InternalError");
    assert_eq!(wire["code"], "E_INTERNAL");
    assert_eq!(
        wire["error"],
        "diff requiere variable explícita: diff(expr, x)"
    );
}
#[test]
fn test_eval_text_returns_undefined_for_empty_diff_domain_without_blocked_stderr() {
    cli()
        .args(["eval", "diff(atanh(sqrt(x^2+2)), x)"])
        .assert()
        .success()
        .stdout(predicate::str::contains("undefined"))
        .stderr(predicate::str::is_empty());
}
