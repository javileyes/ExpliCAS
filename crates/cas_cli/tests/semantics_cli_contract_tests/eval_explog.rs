use super::*;

#[test]
fn eval_binomial_expansion_cancel_common_denominator_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "(((a+b)^2 - a^2 - 2*a*b)/q) - ((b^2)/q)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert!(required.iter().any(|item| item == "q ≠ 0"));
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(
        steps[0]["rule"],
        "Collapse Common-Scale Equivalent Difference"
    );
}
#[test]
fn eval_binomial_expansion_cancel_shifted_quotient_collapses_to_one() {
    let (output, _code) = run_cli(&[
        "eval",
        "(((a+b)^2 - a^2 - 2*a*b) + 1)/((b^2) + 1)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "1");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(
        steps[0]["rule"],
        "Cancelar el cociente de expresiones equivalentes"
    );
}
#[test]
fn eval_diff_chain_rule_exposes_public_didactic_substeps() {
    let (output, code) = run_cli(&[
        "eval",
        "diff((x^2+1)^3, x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    assert_eq!(code, 0);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "6·x·(x^2 + 1)^2");
    let steps = wire["steps"].as_array().expect("steps array");
    let substeps = steps[0]["substeps"].as_array().expect("substeps array");
    assert_eq!(substeps[0]["title"], "Usar regla de la potencia con cadena");
    assert_eq!(substeps[1]["title"], "Identificar u y du");
    let u_latex = substeps[1]["before_latex"].as_str().expect("before_latex");
    assert!(
        u_latex.contains("{x}^{2}") && u_latex.contains("1"),
        "expected u = x^2 + 1, got {:?}",
        substeps[1]
    );
    assert!(
        substeps[1]["after_latex"]
            .as_str()
            .expect("after_latex")
            .contains("2\\cdot x"),
        "expected du = 2*x, got {:?}",
        substeps[1]
    );
}
#[test]
fn eval_diff_exponential_affine_chain_exposes_public_u_du_substep() {
    let (output, code) = run_cli(&[
        "eval",
        "diff(exp(2*x+1), x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    assert_eq!(code, 0);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "2·e^(2·x + 1)");
    let steps = wire["steps"].as_array().expect("steps array");
    let substeps = steps[0]["substeps"].as_array().expect("substeps array");
    assert_eq!(substeps[0]["title"], "Usar regla exponencial");
    assert_eq!(substeps[1]["title"], "Identificar u y du");
    assert!(
        substeps[1]["before_latex"]
            .as_str()
            .expect("before_latex")
            .contains("2\\cdot x + 1"),
        "expected u = 2*x + 1, got {:?}",
        substeps[1]
    );
    assert!(
        substeps[1]["after_latex"]
            .as_str()
            .expect("after_latex")
            .contains("2\\,dx"),
        "expected du = 2 dx, got {:?}",
        substeps[1]
    );
}
#[test]
fn eval_diff_zero_base_variable_exponent_exposes_positive_exponent_domain() {
    let (output, code) = run_cli(&[
        "eval",
        "diff(0^(2*x+1), x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    assert_eq!(code, 0);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    assert_eq!(wire["required_display"], json!(["x > -1/2"]));
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1, "unexpected steps: {steps:?}");
    let substeps = steps[0]["substeps"].as_array().expect("substeps array");
    assert_eq!(
        substeps[0]["title"],
        "Detectar base cero con exponente variable"
    );
    assert!(
        !output.contains("Evaluate Logarithms")
            && !output.contains("infinity")
            && !output.contains("ln(0)")
            && !output.contains("Usar regla exponencial"),
        "zero-base boundary should not first apply the exponential rule: {output}"
    );
}
#[test]
fn eval_diff_zero_base_variable_exponent_rejects_empty_positive_exponent_domain() {
    let (output, code) = run_cli(&[
        "eval",
        "diff(0^(-x^2-1), x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    assert_eq!(code, 0);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "undefined");
    assert_eq!(wire["required_display"], json!([]));
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1, "unexpected steps: {steps:?}");
    let substeps = steps[0]["substeps"].as_array().expect("substeps array");
    assert_eq!(
        substeps[0]["title"],
        "Detectar dominio real vacío de base cero"
    );
    assert!(
        !output.contains("Evaluate Logarithms")
            && !output.contains("infinity")
            && !output.contains("ln(0)")
            && !output.contains("-x^2 - 1 > 0")
            && !output.contains("Usar regla exponencial"),
        "empty zero-base domain should not expose an impossible condition or exponential rule: {output}"
    );
}
#[test]
fn eval_diff_logarithm_rejects_empty_positive_argument_domain() {
    let (output, code) = run_cli(&[
        "eval",
        "diff(ln(-x^2-1), x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    assert_eq!(code, 0);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "undefined");
    assert_eq!(wire["required_display"], json!([]));
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1, "unexpected steps: {steps:?}");
    let substeps = steps[0]["substeps"].as_array().expect("substeps array");
    assert_eq!(
        substeps[0]["title"],
        "Detectar dominio real vacío del logaritmo"
    );
    assert!(
        !output.contains("Usar regla de ln(u)")
            && !output.contains("Identificar u y du")
            && !output.contains("-x^2 - 1 > 0"),
        "empty logarithm domain should not expose a chain rule or impossible condition: {output}"
    );
}
#[test]
fn eval_diff_product_and_quotient_expose_public_component_substeps() {
    let (product_output, product_code) = run_cli(&[
        "eval",
        "diff(x*ln(x), x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    assert_eq!(product_code, 0);
    let product_wire = parse_wire(&product_output);

    assert_eq!(product_wire["result"], "ln(x) + 1");
    let product_steps = product_wire["steps"].as_array().expect("steps array");
    let product_substeps = product_steps[0]["substeps"]
        .as_array()
        .expect("substeps array");
    for title in [
        "Usar regla del producto",
        "Derivar el primer factor",
        "Derivar el segundo factor",
    ] {
        let substep = product_substeps
            .iter()
            .find(|substep| substep["title"] == title)
            .unwrap_or_else(|| panic!("missing product substep title: {title}"));
        assert!(
            substep["before_latex"]
                .as_str()
                .is_some_and(|latex| !latex.is_empty()),
            "expected concrete before_latex for {title}: {substep:?}"
        );
        assert!(
            substep["after_latex"]
                .as_str()
                .is_some_and(|latex| !latex.is_empty()),
            "expected concrete after_latex for {title}: {substep:?}"
        );
    }

    let (quotient_output, quotient_code) = run_cli(&[
        "eval",
        "diff(x/(x+1), x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    assert_eq!(quotient_code, 0);
    let quotient_wire = parse_wire(&quotient_output);

    assert_eq!(quotient_wire["result"], "1 / (x + 1)^2");
    assert_eq!(
        quotient_wire["required_display"]
            .as_array()
            .expect("required display")[0],
        "x ≠ -1"
    );
    let quotient_steps = quotient_wire["steps"].as_array().expect("steps array");
    let quotient_substeps = quotient_steps[0]["substeps"]
        .as_array()
        .expect("substeps array");
    for title in [
        "Usar regla del cociente",
        "Derivar el numerador",
        "Derivar el denominador",
    ] {
        let substep = quotient_substeps
            .iter()
            .find(|substep| substep["title"] == title)
            .unwrap_or_else(|| panic!("missing quotient substep title: {title}"));
        assert!(
            substep["before_latex"]
                .as_str()
                .is_some_and(|latex| !latex.is_empty()),
            "expected concrete before_latex for {title}: {substep:?}"
        );
        assert!(
            substep["after_latex"]
                .as_str()
                .is_some_and(|latex| !latex.is_empty()),
            "expected concrete after_latex for {title}: {substep:?}"
        );
    }
}
#[test]
fn eval_log_cancellation_exponential_step_keeps_full_additive_before_highlight() {
    let (output, _code) = run_cli(&[
        "eval",
        "e^(ln(x^3) + ln(y^2) - ln(x^3 * y^2) + 1)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 3);
    let expand_log_step = &steps[2];
    assert_rule_matches_any(
        &expand_log_step["rule"],
        &[
            "Expandir logaritmos y cancelar términos iguales",
            "Expand Log Product Power",
            "Cancelar la subexpresión idénticamente nula",
        ],
    );
    let before_latex = expand_log_step["before_latex"]
        .as_str()
        .expect("before_latex");
    assert!(
        before_latex.contains("\\color{red}")
            && before_latex.contains("3\\cdot \\ln(x)")
            && before_latex.contains("2\\cdot \\ln(|y|)")
            && before_latex.contains("\\ln({x}^{3}\\cdot {y}^{2})"),
        "expected final log-cancellation step to highlight the full additive scope including ln(...), got: {before_latex}"
    );
    assert!(
        !before_latex.contains("{\\color{red}{1 +"),
        "expected additive passthrough term to stay outside the highlight, got: {before_latex}"
    );

    let after_latex = expand_log_step["after_latex"]
        .as_str()
        .expect("after_latex");
    assert!(
        matches!(after_latex, "{e}^{1 + {\\color{green}{0}}}" | "e"),
        "unexpected after_latex: {after_latex}"
    );
}
#[test]
fn eval_log_contraction_step_keeps_full_before_scope_highlight() {
    let (output, _code) = run_cli(&[
        "eval",
        "log(x-y)+log(x+y)-log(u)-log(v)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    let step = &steps[1];
    let before_latex = step["before_latex"].as_str().expect("before_latex");
    assert!(
        before_latex.contains("{\\color{red}{\\ln({x}^{2} - {y}^{2}) - \\ln(u)}}"),
        "expected step 2 before_latex to highlight the full log-contraction scope, got: {before_latex}"
    );
}
#[test]
fn eval_log_inverse_power_chain_highlights_power_steps_in_mixed_sum() {
    let (output, _code) = run_cli(&[
        "eval",
        "(asin(x/sqrt(x^2 + 1)) - atan(x)) + (factorial(n+1)/factorial(n-1) - n^2 - n) + (x^(ln(ln(x))/ln(x)) - ln(x)) + ((x^3 + y^3)/(x^2 - x*y + y^2) - x - y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    let steps = wire["steps"].as_array().expect("steps array");
    // No-op "Agrupar términos semejantes" steps (before == after) are pruned, so the real
    // transformations are 9 (three cosmetic grouping no-ops were dropped from the tail).
    assert_eq!(steps.len(), 9);

    let log_inverse_power = &steps[2];
    assert_eq!(
        log_inverse_power["rule"],
        "Convertir potencia logarítmica inversa"
    );
    let step3_before = log_inverse_power["before_latex"]
        .as_str()
        .expect("step3 before_latex");
    let step3_after = log_inverse_power["after_latex"]
        .as_str()
        .expect("step3 after_latex");
    assert!(
        step3_before.contains("{\\color{red}{{x}^{\\frac{\\ln(\\ln(x))}{\\ln(x)}}}}"),
        "step 3 should highlight the log-inverse power, got: {step3_before}"
    );
    assert!(
        step3_after.contains("{\\color{green}{{e}^{\\ln(\\ln(x))}}}"),
        "step 3 should highlight the rewritten exponential, got: {step3_after}"
    );
    assert!(
        !step3_before.contains("{\\color{red}{\\frac{(1 + n)!}")
            && !step3_after.contains("{\\color{green}{\\frac{(1 + n)!}"),
        "step 3 should not highlight the factorial chunk: before={step3_before}; after={step3_after}"
    );

    let exp_log_inverse = &steps[3];
    assert_eq!(
        exp_log_inverse["rule"],
        "Cancelar exponencial y logaritmo inversos"
    );
    let step4_before = exp_log_inverse["before_latex"]
        .as_str()
        .expect("step4 before_latex");
    let step4_after = exp_log_inverse["after_latex"]
        .as_str()
        .expect("step4 after_latex");
    assert!(
        step4_before.contains("{\\color{red}{{e}^{\\ln(\\ln(x))}}}"),
        "step 4 should highlight the exp-log inverse, got: {step4_before}"
    );
    assert!(
        step4_after.contains("{\\color{green}{\\ln(x)}}"),
        "step 4 should highlight ln(x), got: {step4_after}"
    );
    assert!(
        !step4_before.contains("{\\color{red}{\\frac{(1 + n)!}")
            && !step4_after.contains("{\\color{green}{\\frac{(1 + n)!}"),
        "step 4 should not highlight the factorial chunk: before={step4_before}; after={step4_after}"
    );

    let hidden_fraction_cancellation = &steps[7];
    assert_rule_eq(
        &hidden_fraction_cancellation["rule"],
        "Cancel Exact Additive Pairs",
    );
    let step8_before = hidden_fraction_cancellation["before_latex"]
        .as_str()
        .expect("step8 before_latex");
    assert!(
        step8_before.contains("{\\color{red}{\\frac{{x}^{3} + {y}^{3}}")
            && step8_before.contains("- {\\color{red}{x}} - y"),
        "step 9 should highlight the changed fraction and x term, got: {step8_before}"
    );
    assert!(
        !step8_before.contains("{{\\color{red}{{y}^{2}"),
        "step 9 should not highlight only the denominator scope, got: {step8_before}"
    );
}
#[test]
fn eval_odd_half_power_difference_to_zero_uses_extract_then_self_cancel_steps() {
    let (output, _code) = run_cli(&[
        "eval",
        "sqrt(x^5) - x^2*sqrt(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    assert_eq!(wire["required_display"], json!(["x ≥ 0"]));

    let steps = wire["steps"].as_array().expect("steps array");
    assert!(matches!(steps.len(), 1 | 2));
    if steps.len() == 1 {
        assert_rule_matches_any(
            &steps[0]["rule"],
            &[
                "Restar dos expresiones iguales",
                "Cancel Exact Additive Pairs",
            ],
        );
    } else {
        assert_rule_eq(&steps[0]["rule"], "Extraer potencia par de la raíz");
        assert_rule_matches_any(
            &steps[1]["rule"],
            &[
                "Restar dos expresiones iguales",
                "Cancel Exact Additive Pairs",
            ],
        );
        let substeps = steps[0]["substeps"].as_array().expect("substeps array");
        assert_eq!(substeps.len(), 2);
    }
}
#[test]
fn eval_higher_odd_half_power_difference_to_zero_uses_extract_then_self_cancel_steps() {
    let (output, _code) = run_cli(&[
        "eval",
        "sqrt(x^7) - x^3*sqrt(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    assert_eq!(wire["required_display"], json!(["x ≥ 0"]));

    let steps = wire["steps"].as_array().expect("steps array");
    assert!(matches!(steps.len(), 1 | 2));
    if steps.len() == 1 {
        assert_rule_matches_any(
            &steps[0]["rule"],
            &[
                "Restar dos expresiones iguales",
                "Cancel Exact Additive Pairs",
            ],
        );
    } else {
        assert_rule_eq(&steps[0]["rule"], "Extraer potencia par de la raíz");
        assert_rule_matches_any(
            &steps[1]["rule"],
            &[
                "Restar dos expresiones iguales",
                "Cancel Exact Additive Pairs",
            ],
        );
        let substeps = steps[0]["substeps"].as_array().expect("substeps array");
        assert_eq!(substeps.len(), 2);
    }
}
#[test]
fn eval_log_cancellation_drops_redundant_exponent_log_substeps() {
    let (output, _code) = run_cli(&[
        "eval",
        "ln(x^3) + ln(y^2) - ln(x^3 * y^2)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_matches_any(
        &steps[0]["rule"],
        &[
            "Expandir logaritmos y cancelar términos iguales",
            "Cancelar la subexpresión idénticamente nula",
        ],
    );
}
#[test]
fn eval_log_power_product_difference_to_zero_uses_combined_log_cancellation_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "ln(x^3) + ln(y^2) - ln(x^3 * y^2)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    assert_eq!(wire["steps_count"], 1);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    let rule = steps[0]["rule"].as_str().expect("rule");
    assert!(
        matches!(
            rule,
            "Expandir logaritmos y cancelar términos iguales"
                | "Cancelar la subexpresión idénticamente nula"
        ),
        "unexpected rule {rule:?}"
    );
    if rule == "Expandir logaritmos y cancelar términos iguales" {
        let substeps = steps[0]["substeps"].as_array().expect("substeps array");
        assert_eq!(substeps.len(), 3);
        assert_eq!(
            substeps[0]["title"],
            "Expandir el logaritmo del producto o del cociente"
        );
        assert_eq!(
            substeps[1]["title"],
            "Sacar exponentes fuera del logaritmo cuando sea necesario"
        );
        assert_eq!(substeps[2]["title"], "Cancelar términos iguales");
    }
}
#[test]
fn eval_unary_log_inverse_power_difference_collapses_to_zero() {
    let (output, code) = run_cli(&[
        "eval",
        "x^(log(log(x))/log(x)) - log(x)",
        "--format",
        "json",
    ]);
    assert_eq!(
        code, 0,
        "expected successful CLI exit, got {code} with output: {output}"
    );

    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "0");
}
#[test]
fn eval_exp_natural_log_power_difference_collapses_to_zero() {
    let (output, code) = run_cli(&["eval", "exp(y*log(x)) - x^y", "--format", "json"]);
    assert_eq!(
        code, 0,
        "expected successful CLI exit, got {code} with output: {output}"
    );

    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "0");
}
#[test]
fn eval_log10_power_difference_collapses_to_zero() {
    let (output, code) = run_cli(&["eval", "10^(y*log10(x)) - x^y", "--format", "json"]);
    assert_eq!(
        code, 0,
        "expected successful CLI exit, got {code} with output: {output}"
    );

    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "0");
}
#[test]
fn eval_log_and_ln_alias_difference_collapses_to_zero() {
    let (output, code) = run_cli(&["eval", "log(x) - ln(x)", "--format", "json"]);
    assert_eq!(
        code, 0,
        "expected successful CLI exit, got {code} with output: {output}"
    );

    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "0");
}
#[test]
fn eval_scaled_log_power_product_difference_collapses_in_one_common_scale_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "k*(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) - k*(0)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    assert_eq!(wire["steps_count"], 1);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Collapse Common-Scale Equivalent Difference"
    );
}
#[test]
fn eval_even_log_product_difference_to_zero_finishes_with_didactic_log_cancellation_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "log((x*y)^2) - log(x^2) - log(y^2)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    assert_eq!(wire["steps_count"], 1);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Expandir logaritmos y cancelar términos iguales"
    );
    assert_eq!(steps[0]["after"], "0");
}
#[test]
fn eval_factored_log_difference_to_zero_keeps_global_log_context_through_preorder_cancel() {
    let (output, _code) = run_cli(&[
        "eval",
        "log(x^2 - y^2) - log(x-y) - log(x+y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    assert_eq!(wire["steps_count"], 3);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 3);
    assert_eq!(
        steps[1]["rule"],
        "Factorizar una diferencia de cuadrados y cancelar"
    );
    assert_eq!(steps[1]["before"], "ln((x^2 - y^2)/(x - y)) - ln(x + y)");
    assert_eq!(steps[1]["after"], "ln(x + y) - ln(x + y)");
    assert_rule_matches_any(
        &steps[2]["rule"],
        &[
            "Collapse Common-Scale Equivalent Difference",
            "Cancel Exact Additive Pairs",
        ],
    );
    let before_latex = steps[1]["before_latex"].as_str().expect("before_latex");
    let after_latex = steps[1]["after_latex"].as_str().expect("after_latex");
    assert!(
        before_latex.contains("\\ln(") && before_latex.contains("\\frac{{x}^{2} - {y}^{2}}{x - y}"),
        "expected full logarithmic before context, got: {before_latex}"
    );
    assert!(
        after_latex.contains("\\ln(") && after_latex.contains("{\\color{green}{x + y}}"),
        "expected full logarithmic after context, got: {after_latex}"
    );
}
#[test]
fn eval_general_base_log_product_square_passthrough_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "((log(b,(x*y)^2)+a) + m) - ((2*log(b,x)+2*log(b,y)+a) + m)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Expandir logaritmos y cancelar términos iguales"
    );
}
#[test]
fn eval_general_base_log_power_quotient_common_denominator_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "((log(b, (x^2*y^3)/(z^2*t)))/q) - ((2*log(b, x) + 3*log(b, y) - 2*log(b, z) - log(b, t))/q)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Collapse Common-Scale Equivalent Difference"
    );
}
#[test]
fn eval_general_base_logs_to_grouped_power_passthrough_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "((2*log(b,x)+2*log(b,y)+a) + m) - ((log(b,(x*y)^2)+a) + m)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Expandir logaritmos y cancelar términos iguales"
    );
}
#[test]
fn eval_abs_product_denominator_expands_to_atomic_nonzero_guards() {
    let (output, _code) = run_cli(&["eval", "1/abs(x*y)", "--format", "json"]);
    let wire = parse_wire(&output);

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert_eq!(
        required
            .iter()
            .filter_map(|item| item.as_str())
            .collect::<Vec<_>>(),
        vec!["x ≠ 0", "y ≠ 0"]
    );
}
#[test]
fn eval_abs_factored_product_denominator_expands_to_atomic_nonzero_guards() {
    let (output, _code) = run_cli(&["eval", "1/abs((x-1)*(x+1))", "--format", "json"]);
    let wire = parse_wire(&output);

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert_eq!(
        required
            .iter()
            .filter_map(|item| item.as_str())
            .collect::<Vec<_>>(),
        vec!["x ≠ 1", "x ≠ -1"]
    );
}
#[test]
fn eval_log_abs_factored_product_expands_to_atomic_nonzero_guards() {
    let (output, _code) = run_cli(&["eval", "ln(abs((x-1)*(x+1)))", "--format", "json"]);
    let wire = parse_wire(&output);

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert_eq!(
        required
            .iter()
            .filter_map(|item| item.as_str())
            .collect::<Vec<_>>(),
        vec!["x ≠ 1", "x ≠ -1"]
    );
}
#[test]
fn eval_abs_quotient_expands_to_atomic_numerator_and_denominator_guards() {
    let (output, _code) = run_cli(&["eval", "1/abs(x/(x+1))", "--format", "json"]);
    let wire = parse_wire(&output);

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    let required: Vec<_> = required.iter().filter_map(|item| item.as_str()).collect();
    assert_eq!(required.len(), 2);
    assert!(required.contains(&"x ≠ 0"));
    assert!(required.contains(&"x ≠ -1"));
}
#[test]
fn eval_log_abs_quotient_expands_to_atomic_numerator_and_denominator_guards() {
    let (output, _code) = run_cli(&["eval", "ln(abs((x-1)/(x+1)))", "--format", "json"]);
    let wire = parse_wire(&output);

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert_eq!(
        required
            .iter()
            .filter_map(|item| item.as_str())
            .collect::<Vec<_>>(),
        vec!["x ≠ 1", "x ≠ -1"]
    );
}
#[test]
fn eval_general_base_log_surfaces_positive_base_and_argument_requires() {
    let (output, _code) = run_cli(&["eval", "log(b, x)", "--format", "json"]);
    let wire = parse_wire(&output);

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert_eq!(required.len(), 3);
    assert!(required.iter().any(|item| item == "b ≠ 1"));
    assert!(required.iter().any(|item| item == "b > 0"));
    assert!(required.iter().any(|item| item == "x > 0"));
}
