use super::*;

#[test]
fn eval_cos_diff_sin_diff_quotient_difference_collapses_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "(cos(x)-cos(3*x))/(sin(3*x)-sin(x)) - tan(2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    assert_eq!(wire["steps_count"], 1);
}
#[test]
fn eval_cos_diff_sin_diff_quotient_passthrough_difference_collapses_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "(((cos(x)-cos(3*x))/(sin(3*x)-sin(x))) + m) - ((tan(2*x)) + m)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    assert_eq!(wire["steps_count"], 1);
}
#[test]
fn eval_cos_diff_sin_diff_quotient_shifted_quotient_collapses_to_one_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "(((cos(x)-cos(3*x))/(sin(3*x)-sin(x))) + 1)/((tan(2*x)) + 1)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "1");
    assert_eq!(wire["steps_count"], 1);
}
#[test]
fn eval_fraction_sum_to_sec_squared_keeps_faithful_pythagorean_intermediate() {
    let (output, _code) = run_cli(&[
        "eval",
        "1/(1 + sin(x)) + 1/(1 - sin(x))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 4);
    let pythagorean_step = steps.last().expect("pythagorean step");
    assert_rule_eq(&pythagorean_step["rule"], "Aplicar identidad pitagórica");
    assert_eq!(pythagorean_step["before"], "2/(1 - sin(x)^2)");
    assert!(
        pythagorean_step["before_latex"]
            .as_str()
            .expect("before latex")
            .contains("^{2}"),
        "expected squared sine to survive in before_latex: {:?}",
        pythagorean_step["before_latex"]
    );
}
#[test]
fn eval_tangent_expansion_uses_visible_step_name() {
    let (output, _code) = run_cli(&["eval", "tan(x)", "--format", "json", "--steps", "on"]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "tan(x)");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 2);
    assert_rule_eq(&steps[0]["rule"], "Tan to Sin/Cos");
    assert!(
        steps
            .iter()
            .all(|step| step["rule"].as_str() != Some("Tan to Sin/Cos")),
        "public steps should not expose internal Tan to Sin/Cos label: {steps:?}"
    );
}
#[test]
fn eval_reciprocal_trig_expansions_use_visible_step_names() {
    for (expr, expected_internal_rule) in [
        ("sec(x)", "Secant to Reciprocal Cosine"),
        ("csc(x)", "Cosecant to Reciprocal Sine"),
        ("cot(x)", "Cotangent to Cosine over Sine"),
    ] {
        let (output, _code) = run_cli(&["eval", expr, "--format", "json", "--steps", "on"]);
        let wire = parse_wire(&output);

        assert_eq!(wire["result"], expr);
        let steps = wire["steps"].as_array().expect("steps array");
        assert_eq!(steps.len(), 2, "unexpected steps for {expr}: {steps:?}");
        assert_rule_eq(&steps[0]["rule"], expected_internal_rule);
        assert!(
            steps
                .iter()
                .all(|step| step["rule"].as_str() != Some(expected_internal_rule)),
            "public steps should not expose internal {expected_internal_rule} label for {expr}: {steps:?}"
        );
    }
}
#[test]
fn eval_reciprocal_cosine_avoids_ping_pong_and_keeps_single_step() {
    let (output, _code) = run_cli(&["eval", "1/cos(x)", "--format", "json", "--steps", "on"]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "sec(x)");
    assert_eq!(wire["steps_count"], 1);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Reconocer secante desde un recíproco");
    assert_eq!(steps[0]["before"], "1/cos(x)");
    assert_eq!(steps[0]["after"], "sec(x)");
}
#[test]
fn eval_diff_negative_constant_base_variable_exponent_explains_real_domain_boundary() {
    let (output, code) = run_cli(&[
        "eval",
        "diff((-2)^(2*x+1), x)",
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
        "Detectar base negativa con exponente variable"
    );
    assert!(
        !output.contains("Evaluate Logarithms")
            && !output.contains("Undefined Times Anything")
            && !output.contains("Usar regla exponencial"),
        "negative-base boundary should not first apply the exponential rule: {output}"
    );
}
#[test]
fn eval_diff_inverse_functions_reject_symbolic_constants_outside_real_domain() {
    for expr in [
        "diff(arcsin(pi), x)",
        "diff(arccos(e), x)",
        "diff(arccos(-e), x)",
        "diff(atanh(pi), x)",
    ] {
        let (output, code) = run_cli(&["eval", expr, "--format", "json", "--steps", "on"]);
        assert_eq!(code, 0, "unexpected command failure for {expr}: {output}");
        let wire = parse_wire(&output);

        assert_eq!(
            wire["result"], "undefined",
            "symbolic constant outside real domain should be undefined for {expr}: {output}"
        );
        assert_eq!(wire["required_display"], json!([]));
        assert_optional_empty_domain_blocked_hint(&wire, &output);
        assert!(
            !output.contains("Usar regla de arcsin(u)")
                && !output.contains("Usar regla de arccos(u)")
                && !output.contains("Usar regla de atanh(u)")
                && !output.contains("Identificar u y du"),
            "symbolic constant outside domain should not expose a chain rule: {output}"
        );
        assert!(
            output.contains("Detectar dominio real vacío de la función inversa"),
            "symbolic constant outside domain should explain the undefined derivative: {output}"
        );
    }

    for expr in [
        "diff(arcsin(1/2), x)",
        "diff(arccos(0), x)",
        "diff(atanh(0), x)",
    ] {
        let (output, code) = run_cli(&["eval", expr, "--format", "json", "--steps", "on"]);
        assert_eq!(code, 0, "unexpected command failure for {expr}: {output}");
        let wire = parse_wire(&output);

        assert_eq!(wire["result"], "0", "unexpected result for {expr}");
        assert_eq!(wire["required_display"], json!([]));
        assert!(
            wire["blocked_hints"].is_null(),
            "valid finite constant should not be blocked for {expr}: {output}"
        );
    }
}
#[test]
fn eval_symbolic_sine_sum_to_product_difference_to_zero_uses_two_didactic_steps() {
    let (output, _code) = run_cli(&[
        "eval",
        "sin(x) + sin(y) - 2*sin((x+y)/2)*cos((x-y)/2)",
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
    assert_rule_eq(&steps[0]["rule"], "Aplicar suma a producto");
    assert!(steps[0].get("substeps").is_none());
}
#[test]
fn eval_symbolic_sine_sum_to_product_negated_orientation_still_reaches_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "2*sin((x+y)/2)*cos((x-y)/2) - sin(x) - sin(y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_matches_any(
        &steps[0]["rule"],
        &[
            "Aplicar producto a suma",
            "Cancelar la subexpresión idénticamente nula",
        ],
    );
}
#[test]
fn eval_symbolic_cosine_difference_sum_to_product_difference_to_zero_uses_two_steps() {
    let (output, _code) = run_cli(&[
        "eval",
        "cos(x) - cos(y) + 2*sin((x+y)/2)*sin((x-y)/2)",
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
    assert_rule_eq(&steps[0]["rule"], "Aplicar suma a producto");
    assert!(steps[0].get("substeps").is_none());
}
#[test]
fn eval_general_sine_sum_to_product_scaled_difference_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "k*(sin(5*x)+sin(x)) - k*(2*sin(3*x)*cos(2*x))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Collapse Common-Scale Equivalent Difference"
    );
}
#[test]
fn eval_general_sine_sum_to_product_passthrough_difference_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "((sin(5*x)+sin(x)) + m) - ((2*sin(3*x)*cos(2*x)) + m)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_matches_any(
        &steps[0]["rule"],
        &[
            "Aplicar suma a producto",
            "Cancelar la subexpresión idénticamente nula",
        ],
    );
}
#[test]
fn eval_special_sine_difference_to_product_difference_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "sin(3*x)-sin(x) - (2*cos(2*x)*sin(x))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_matches_any(
        &steps[0]["rule"],
        &["Aplicar suma a producto", "Expandir ángulo doble"],
    );
}
#[test]
fn eval_special_sine_difference_to_product_passthrough_difference_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "((sin(3*x)-sin(x)) + m) - ((2*cos(2*x)*sin(x)) + m)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Aplicar suma a producto");
}
#[test]
fn eval_general_cosine_sum_to_product_shifted_quotient_collapses_to_one() {
    let (output, _code) = run_cli(&[
        "eval",
        "((cos(5*x)+cos(x)) + 1)/((2*cos(3*x)*cos(2*x)) + 1)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "1");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Cancelar el cociente de expresiones equivalentes"
    );
}
#[test]
fn eval_general_cosine_sum_to_product_passthrough_difference_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "((cos(5*x)+cos(x)) + m) - ((2*cos(3*x)*cos(2*x)) + m)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_matches_any(
        &steps[0]["rule"],
        &[
            "Aplicar suma a producto",
            "Cancelar la subexpresión idénticamente nula",
        ],
    );
}
#[test]
fn eval_recursive_six_sine_shifted_quotient_collapses_to_one() {
    let (output, _code) = run_cli(&[
        "eval",
        "((sin(6*x)) + 1)/((sin(5*x)*cos(x)+cos(5*x)*sin(x)) + 1)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "1");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(
        steps[0]["rule"],
        "Cancelar el cociente de expresiones equivalentes"
    );
    assert_eq!(steps.len(), 1);
}
#[test]
fn eval_recursive_six_sine_difference_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "sin(6*x) - (sin(5*x)*cos(x)+cos(5*x)*sin(x))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Angle Sum/Diff Identity");
}
#[test]
fn eval_recursive_six_sine_scaled_difference_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "k*sin(6*x) - k*(sin(5*x)*cos(x)+cos(5*x)*sin(x))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Collapse Common-Scale Equivalent Difference"
    );
}
#[test]
fn eval_mixed_trig_double_angle_product_difference_collapses_to_zero_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x)-2*sin(x))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_matches_any(
        &steps[0]["rule"],
        &[
            "Cancelar la subexpresión idénticamente nula",
            "Expandir ángulo doble",
        ],
    );
}
#[test]
fn eval_expanded_affine_sine_double_angle_product_difference_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "sin(4*x+2) - 2*sin(2*x+1)*cos(2*x+1)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_matches_any(
        &steps[0]["rule"],
        &[
            "Cancelar la subexpresión idénticamente nula",
            "Expandir ángulo doble",
        ],
    );
}
#[test]
fn eval_mixed_trig_double_angle_product_scaled_difference_collapses_to_zero_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "k*(2*cos(2*x)*sin(x)) - k*(4*cos(x)^2*sin(x)-2*sin(x))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Collapse Common-Scale Equivalent Difference"
    );
}
#[test]
fn eval_mixed_trig_double_angle_product_passthrough_collapses_to_zero_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "((2*cos(2*x)*sin(x)) + m) - ((4*cos(x)^2*sin(x)-2*sin(x)) + m)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Cancelar la subexpresión idénticamente nula"
    );
}
#[test]
fn eval_sine_cosine_square_product_difference_collapses_to_zero_with_power_reduction() {
    let (output, _code) = run_cli(&[
        "eval",
        "sin(x)^2*cos(x)^2 - ((1-cos(4*x))/8)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Aplicar reducción de potencias");
}
#[test]
fn eval_sine_cosine_square_product_shifted_quotient_collapses_to_one() {
    let (output, _code) = run_cli(&[
        "eval",
        "((sin(x)^2*cos(x)^2) + 1)/(((1-cos(4*x))/8) + 1)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "1");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Cancelar el cociente de expresiones equivalentes"
    );
}
#[test]
fn eval_sine_sixth_power_difference_collapses_to_zero_with_power_reduction() {
    let (output, _code) = run_cli(&[
        "eval",
        "sin(x)^6 - ((10-15*cos(2*x)+6*cos(4*x)-cos(6*x))/32)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Aplicar reducción de potencias");
}
#[test]
fn eval_sine_sixth_power_common_denominator_difference_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "((sin(x)^6)/q) - (((10-15*cos(2*x)+6*cos(4*x)-cos(6*x))/32)/q)",
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
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Collapse Common-Scale Equivalent Difference"
    );
}
#[test]
fn eval_trig_product_to_sum_difference_stays_direct_and_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "2*sin(x)*cos(y) - (sin(x+y) + sin(x-y))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Aplicar producto a suma");
}
#[test]
fn eval_trig_product_to_sum_common_denominator_difference_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "((2*cos(x)*sin(y))/q) - ((sin(x+y) - sin(x-y))/q)",
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
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Collapse Common-Scale Equivalent Difference"
    );
}
#[test]
fn eval_recursive_six_cosine_passthrough_difference_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "((cos(5*x)*cos(x)-sin(5*x)*sin(x)) + m) - ((cos(6*x)) + m)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Angle Sum/Diff Identity");
}
#[test]
fn eval_recursive_six_cosine_common_denominator_difference_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "((cos(5*x)*cos(x)-sin(5*x)*sin(x))/q) - ((cos(6*x))/q)",
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
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Collapse Common-Scale Equivalent Difference"
    );
}
#[test]
fn eval_sec_squared_shifted_quotient_collapses_to_one() {
    let (output, _code) = run_cli(&[
        "eval",
        "((sec(x)^2) + 1)/((1 + tan(x)^2) + 1)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "1");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(
        steps[0]["rule"],
        "Cancelar el cociente de expresiones equivalentes"
    );
}
#[test]
fn eval_half_angle_square_shifted_quotient_collapses_to_one() {
    let (output, _code) = run_cli(&[
        "eval",
        "((sin(x)^2) + 1)/(((1-cos(2*x))/2) + 1)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "1");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(
        steps[0]["rule"],
        "Cancelar el cociente de expresiones equivalentes"
    );
}
#[test]
fn eval_exact_shifted_sine_cosine_difference_to_zero_survives_common_factorization() {
    let (output, _code) = run_cli(&[
        "eval",
        "sqrt(2)*sin(x+pi/4) - sqrt(2)*cos(x-pi/4)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(
        steps[0]["rule"],
        "Collapse Common-Scale Equivalent Difference"
    );
}
#[test]
fn eval_exact_shifted_sine_cosine_scaled_difference_collapses_after_common_factor() {
    let (output, _code) = run_cli(&[
        "eval",
        "k*(sqrt(2)*sin(x+pi/4)) - k*(sqrt(2)*cos(x-pi/4))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(
        steps[0]["rule"],
        "Collapse Common-Scale Equivalent Difference"
    );
}
#[test]
fn eval_exact_shifted_sine_cosine_shifted_quotient_collapses_to_one() {
    let (output, _code) = run_cli(&[
        "eval",
        "((sqrt(2)*sin(x+pi/4)+a) + 1)/((sqrt(2)*cos(x-pi/4)+a) + 1)",
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
fn eval_exact_shifted_sine_cosine_common_denominator_difference_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "((sqrt(2)*sin(x+pi/4)+a)/q) - ((sqrt(2)*cos(x-pi/4)+a)/q)",
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
fn eval_general_shifted_sine_cosine_difference_to_zero_survives_common_factorization() {
    let (output, _code) = run_cli(&[
        "eval",
        "5*sin(x+arctan(4/3)) - 5*cos(x-arctan(3/4))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(
        steps[0]["rule"],
        "Collapse Common-Scale Equivalent Difference"
    );
}
#[test]
fn eval_general_shifted_sine_cosine_shifted_quotient_collapses_to_one() {
    let (output, _code) = run_cli(&[
        "eval",
        "((5*sin(x+arctan(4/3))+a) + 1)/((5*cos(x-arctan(3/4))+a) + 1)",
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
fn eval_general_shifted_sine_cosine_common_denominator_difference_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "((5*sin(x+arctan(4/3))+a)/q) - ((5*cos(x-arctan(3/4))+a)/q)",
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
fn eval_tangent_triple_angle_passthrough_difference_collapses_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "((tan(3*x)) + m) - (((3*tan(x)-tan(x)^3)/(1-3*tan(x)^2)) + m)",
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
        "Cancelar la subexpresión idénticamente nula"
    );
}
#[test]
fn eval_tangent_triple_angle_raw_difference_collapses_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "((3*tan(x)-tan(x)^3)/(1-3*tan(x)^2)) - tan(3*x)",
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
        "Cancelar la subexpresión idénticamente nula"
    );
}
#[test]
fn eval_tangent_triple_angle_shifted_quotient_collapses_to_one_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "((tan(3*x)) + 1)/(((3*tan(x)-tan(x)^3)/(1-3*tan(x)^2)) + 1)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "1");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Cancelar el cociente de expresiones equivalentes"
    );
}
#[test]
fn eval_trig_binomial_square_difference_to_zero_uses_named_identity_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "(sin(x) + cos(x))^2 - (1 + sin(2*x))",
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
        "Cancelar la subexpresión idénticamente nula"
    );
    assert!(steps[0].get("substeps").is_none());
}
#[test]
fn eval_trig_binomial_square_difference_minus_passthrough_collapses_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "((sin(x)-cos(x))^2 + m) - ((1-sin(2*x)) + m)",
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
        "Cancelar la subexpresión idénticamente nula"
    );
}
#[test]
fn eval_trig_binomial_square_difference_minus_shifted_quotient_collapses_to_one_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "(((sin(x)-cos(x))^2) + 1)/((1-sin(2*x)) + 1)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "1");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Cancelar el cociente de expresiones equivalentes"
    );
}
#[test]
fn eval_half_angle_square_difference_to_zero_collapses_in_one_half_angle_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "sin(x)^2 - (1 - cos(2*x))/2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_rule_eq(&steps[0]["rule"], "Aplicar identidad de ángulo mitad");
    assert!(steps[0].get("substeps").is_none());
}
#[test]
fn eval_half_angle_square_passthrough_difference_collapses_in_one_half_angle_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "((sin(x)^2) + m) - (((1-cos(2*x))/2) + m)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_rule_matches_any(
        &steps[0]["rule"],
        &[
            "Aplicar identidad de ángulo mitad",
            "Cancelar la subexpresión idénticamente nula",
        ],
    );
    assert!(steps[0].get("substeps").is_none());
}
#[test]
fn eval_double_angle_cos_one_minus_two_sin_sq_passthrough_difference_collapses_to_zero_in_one_step()
{
    let (output, _code) = run_cli(&[
        "eval",
        "((cos(2*x)) + m) - ((1 - 2*sin(x)^2) + m)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(
        steps[0]["rule"],
        "Cancelar la subexpresión idénticamente nula"
    );
    assert!(steps[0].get("substeps").is_none());
}
#[test]
fn eval_double_angle_cos_one_minus_two_sin_sq_raw_difference_collapses_to_zero_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "cos(2*x) - (1 - 2*sin(x)^2)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_rule_matches_any(
        &steps[0]["rule"],
        &[
            "Cancelar la subexpresión idénticamente nula",
            "Expandir ángulo doble",
        ],
    );
    // Pinned SILENCE before the matcher migration (2026-07-28): the rule sat
    // in the silenced list because its «Usar L = R» was an unverified claim.
    // Now the identity is stated only when the pair provably instantiates the
    // censused row, and the sub-step's own sides are the LOCAL pair the
    // identity applies to — `cos(2·x) ⟹ 1 - 2·sin(x)^2` inside the difference
    // that collapses — so the cited formula is the one on screen.
    let substeps = steps[0]["substeps"].as_array().expect("substeps array");
    assert_eq!(substeps.len(), 1);
    assert_eq!(substeps[0]["title"], "Usar cos(2u) = 1 - 2 · sin(u)^2");
    assert_eq!(substeps[0]["before"], "cos(2·x)");
    assert_eq!(substeps[0]["after"], "1 - 2·sin(x)^2");
}
#[test]
fn eval_double_angle_cos_one_minus_two_sin_sq_scaled_difference_collapses_to_zero_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "k*(cos(2*x)) - k*(1 - 2*sin(x)^2)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_rule_matches_any(
        &steps[0]["rule"],
        &[
            "Collapse Common-Scale Equivalent Difference",
            "Expandir ángulo doble",
        ],
    );
    assert!(steps[0].get("substeps").is_none());
}
#[test]
fn eval_double_angle_cos_two_cos_sq_minus_one_shifted_quotient_collapses_to_one_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "((2*cos(x)^2 - 1) + 1)/((cos(2*x)) + 1)",
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
    assert!(steps[0].get("substeps").is_none());
}
#[test]
fn eval_trig_sine_product_cubic_cosine_difference_to_zero_uses_combined_bridge_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "2*sin(2*x)*sin(x) - (4*cos(x) - 4*cos(x)^3)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    assert_eq!(wire["warnings"], json!([]));
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Cancelar la subexpresión idénticamente nula"
    );
    assert!(steps[0].get("substeps").is_none());
}
#[test]
fn eval_trig_cos_double_angle_product_passthrough_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "((2*cos(2*x)*cos(x)) + m) - ((4*cos(x)^3-2*cos(x)) + m)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(
        steps[0]["rule"],
        "Cancelar la subexpresión idénticamente nula"
    );
}
#[test]
fn eval_morrie_passthrough_difference_collapses_to_zero_with_nonzero_sine_condition() {
    let (output, _code) = run_cli(&[
        "eval",
        "((cos(x)*cos(2*x)*cos(4*x)) + m) - ((sin(8*x)/(8*sin(x))) + m)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(
        steps[0]["rule"],
        "Cancelar la subexpresión idénticamente nula"
    );
    assert_eq!(wire["required_display"][0], "sin(x) ≠ 0");
}
#[test]
fn eval_morrie_scaled_difference_collapses_to_zero_with_nonzero_sine_condition() {
    let (output, _code) = run_cli(&[
        "eval",
        "k*(cos(x)*cos(2*x)*cos(4*x)) - k*(sin(8*x)/(8*sin(x)))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(
        steps[0]["rule"],
        "Collapse Common-Scale Equivalent Difference"
    );
    assert_eq!(wire["required_display"][0], "sin(x) ≠ 0");
}
#[test]
fn eval_scaled_abs_log_product_difference_to_zero_uses_single_didactic_log_cancellation_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "2*ln(abs(x*y)) - 2*ln(abs(x)) - 2*ln(abs(y))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    assert_eq!(wire["steps_count"], 1);
    assert_eq!(wire["required_display"], json!(["x ≠ 0", "y ≠ 0"]));

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Expandir logaritmos y cancelar términos iguales"
    );
    assert_eq!(steps[0]["after"], "0");
    let substeps = steps[0]["substeps"].as_array().expect("substeps array");
    assert_eq!(substeps.len(), 2);
    assert_eq!(
        substeps[0]["title"],
        "Expandir el logaritmo del producto o del cociente"
    );
    assert_eq!(substeps[1]["title"], "Cancelar términos iguales");
}
#[test]
fn eval_solve_periodic_trig_equation_narrates_solve_steps() {
    // The bare periodic trig solve narrates through `solve_steps`: the per-period
    // roots (principal + supplementary) and one periodic-family line per base,
    // in the exact `x = base + k·T` shape the result set displays.
    let (output, code) = run_cli(&[
        "eval",
        "solve(sin(x)=1/2,x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    assert_eq!(code, 0, "output: {output}");
    let wire = parse_wire(&output);
    let steps = wire["solve_steps"].as_array().cloned().unwrap_or_default();
    let descs: Vec<&str> = steps
        .iter()
        .filter_map(|s| s["description"].as_str())
        .collect();
    assert_eq!(
        descs,
        vec![
            "Invierte sin en un periodo",
            "La segunda solución dentro del periodo",
            "Familia periódica de soluciones (k entero cualquiera)",
            "Familia periódica de soluciones (k entero cualquiera)",
        ],
        "expected the four-step periodic trig narration, got {steps:?}"
    );
    assert_eq!(steps[0]["equation"], "x = 1/6·pi");
    assert_eq!(steps[1]["equation"], "x = 5/6·pi");

    // Affine argument: no synthetic-u steps, but the mapped families narrate
    // (zero base folds to `x = pi·k`, not `x = 0 + pi·k`).
    let (output, code) = run_cli(&[
        "eval",
        "solve(cos(2*x)=1,x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    assert_eq!(code, 0, "output: {output}");
    let wire = parse_wire(&output);
    let steps = wire["solve_steps"].as_array().cloned().unwrap_or_default();
    assert_eq!(steps.len(), 1, "expected one family line, got {steps:?}");
    assert_eq!(steps[0]["equation"], "x = pi·k");

    // Coefficient/offset peel narrates the isolation in the CLEAN quotient form
    // (`cos(x) = √3/2`, not the classifier's rationalized `3/2·3^(-1/2)`), then
    // the inner periodic inversion narrates onto the same channel.
    let (output, code) = run_cli(&[
        "eval",
        "solve(2*cos(x)-sqrt(3)=0,x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    assert_eq!(code, 0, "output: {output}");
    let wire = parse_wire(&output);
    let steps = wire["solve_steps"].as_array().cloned().unwrap_or_default();
    let descs: Vec<&str> = steps
        .iter()
        .filter_map(|s| s["description"].as_str())
        .collect();
    assert_eq!(
        descs,
        vec![
            "Aísla el término trigonométrico",
            "Invierte cos en un periodo",
            "La segunda solución dentro del periodo",
            "Familia periódica de soluciones (k entero cualquiera)",
            "Familia periódica de soluciones (k entero cualquiera)",
        ],
        "expected the isolate + periodic narration, got {steps:?}"
    );
    assert_eq!(steps[0]["equation"], "cos(x) = sqrt(3) / 2");
    assert_eq!(steps[1]["equation"], "x = 1/6·pi");

    // Homogeneous reduction narrates its tan line, then the inner periodic
    // solve narrates onto the same channel.
    let (output, code) = run_cli(&[
        "eval",
        "solve(sin(x)=cos(x),x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    assert_eq!(code, 0, "output: {output}");
    let wire = parse_wire(&output);
    let steps = wire["solve_steps"].as_array().cloned().unwrap_or_default();
    let descs: Vec<&str> = steps
        .iter()
        .filter_map(|s| s["description"].as_str())
        .collect();
    assert_eq!(
        descs,
        vec![
            "Reduce a la tangente (divide ambos lados entre cos)",
            "Invierte tan en un periodo",
            "Familia periódica de soluciones (k entero cualquiera)",
        ],
        "expected the tan-reduction narration, got {steps:?}"
    );
    assert_eq!(steps[0]["equation"], "tan(x) = 1");

    // Auxiliary angle narrates its rewrite plus the mapped periodic families.
    let (output, code) = run_cli(&[
        "eval",
        "solve(sin(x)+sqrt(3)*cos(x)=1,x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    assert_eq!(code, 0, "output: {output}");
    let wire = parse_wire(&output);
    let steps = wire["solve_steps"].as_array().cloned().unwrap_or_default();
    let descs: Vec<&str> = steps
        .iter()
        .filter_map(|s| s["description"].as_str())
        .collect();
    assert_eq!(
        descs,
        vec![
            "Ángulo auxiliar: reescribe como R·sin(g + φ) = c/R",
            "Familia periódica de soluciones (k entero cualquiera)",
            "Familia periódica de soluciones (k entero cualquiera)",
        ],
        "expected the auxiliary-angle narration, got {steps:?}"
    );
}
#[test]
fn eval_solve_equal_tangents_narrates_identity_and_family() {
    let (output, code) = run_cli(&[
        "eval",
        "solve(tan(x)=tan(2*x), x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    assert_eq!(code, 0, "output: {output}");
    let wire = parse_wire(&output);
    let steps = wire["solve_steps"].as_array().cloned().unwrap_or_default();
    let descs: Vec<&str> = steps
        .iter()
        .filter_map(|s| s["description"].as_str())
        .collect();
    assert_eq!(
        descs,
        vec![
            "Tangentes iguales: los argumentos difieren en un múltiplo de π",
            "Familia periódica de soluciones (k entero cualquiera)",
        ],
        "expected the equal-tangents narration, got {steps:?}"
    );
    assert_eq!(steps[1]["equation"], "x = pi·k");
}
#[test]
fn eval_abs_of_provably_signed_constants_resolves_both_orientations() {
    // The negativity arm now consults the exact const/surd sign layer
    // (mirroring the positivity arm), and Abs Sub Normalize leaves provably
    // signed CONSTANT arguments to those rules — so both orientations fold
    // and the result no longer depends on steps mode.
    for (input, expected) in [
        ("abs(pi - 3)", "pi - 3"),
        ("abs(3 - pi)", "pi - 3"),
        ("abs(e - 3)", "3 - e"),
        ("abs(3 - e)", "3 - e"),
        ("abs(sqrt(2) - 1)", "sqrt(2) - 1"),
        ("abs(1 - sqrt(2))", "sqrt(2) - 1"),
        ("abs(phi - 1)", "phi - 1"),
        ("abs(1 - phi)", "phi - 1"),
        ("abs(pi - 4)", "4 - pi"),
        // The undecidable/variable cases keep their contract.
        ("abs(x - 1)", "|x - 1|"),
        ("abs(u - 1) - abs(1 - u)", "0"),
    ] {
        let (output, code) = run_cli(&["eval", input, "--format", "json"]);
        assert_eq!(code, 0, "{input}: {output}");
        let wire = parse_wire(&output);
        assert_eq!(wire["result"], expected, "{input}");
    }

    // Steps mode must not change the result (the old divergence: |π−3| folded
    // with steps on but not off).
    let (output, code) = run_cli(&["eval", "abs(pi - 3)", "--steps", "on", "--format", "json"]);
    assert_eq!(code, 0, "output: {output}");
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "pi - 3");
}
