use super::*;

#[test]
fn eval_diff_atanh_rejects_empty_open_interval_domain() {
    let (output, code) = run_cli(&[
        "eval",
        "diff(atanh(x^2+1), x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    assert_eq!(code, 0);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "undefined");
    assert_eq!(wire["required_display"], json!([]));
    assert_optional_empty_domain_blocked_hint(&wire, &output);
    assert!(
        !output.contains("Usar regla de atanh(u)")
            && !output.contains("Identificar u y du")
            && !output.contains("sqrt(1 - (x^2 + 1)^2)"),
        "empty atanh domain should not expose a chain rule or impossible derivative: {output}"
    );
    assert!(
        output.contains("Detectar dominio real vacío de la función inversa"),
        "empty atanh domain should explain the undefined derivative: {output}"
    );
}
#[test]
fn eval_diff_acosh_rejects_empty_derivative_domain_with_specific_trace() {
    let (output, code) = run_cli(&[
        "eval",
        "diff(acosh(1-x^2), x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    assert_eq!(code, 0);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "undefined");
    assert_eq!(wire["required_display"], json!([]));
    assert_optional_empty_domain_blocked_hint(&wire, &output);
    assert!(
        !output.contains("-(x^2) > 0"),
        "empty derivative domain should not expose an impossible condition: {output}"
    );
    assert!(
        output.contains("Detectar dominio real vacío de la derivada de la función inversa"),
        "empty derivative domain should be distinguished from empty source domain: {output}"
    );
}
#[test]
fn eval_hyperbolic_half_angle_square_passthrough_difference_collapses_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "((cosh(x/2)^2) + m) - (((cosh(x)+1)/2) + m)",
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
fn eval_hyperbolic_half_angle_square_shifted_quotient_collapses_to_one_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "((cosh(x/2)^2) + 1)/(((cosh(x)+1)/2) + 1)",
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
    assert!(steps[0].get("substeps").is_some());
}
#[test]
fn eval_hyperbolic_angle_sum_difference_to_zero_uses_expand_then_self_cancel() {
    let (output, _code) = run_cli(&[
        "eval",
        "sinh(x+y) - (sinh(x)*cosh(y) + cosh(x)*sinh(y))",
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
fn eval_hyperbolic_angle_sum_difference_with_passthrough_one_collapses_to_one() {
    let (output, _code) = run_cli(&[
        "eval",
        "sinh(x+y) + 1 - (sinh(x)*cosh(y) + cosh(x)*sinh(y))",
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
        "Cancelar la subexpresión idénticamente nula"
    );
}
#[test]
fn eval_hyperbolic_cosh_angle_difference_shifted_quotient_collapses_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "((cosh(x)*cosh(y)-sinh(x)*sinh(y)) + 1)/((cosh(x-y)) + 1)",
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
fn eval_hyperbolic_sinh_angle_difference_scaled_difference_collapses_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "k*(sinh(x)*cosh(y)-cosh(x)*sinh(y)) - k*(sinh(x-y))",
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
fn eval_hyperbolic_angle_sum_difference_negated_orientation_still_reaches_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "sinh(x)*cosh(y) + sinh(y)*cosh(x) - sinh(x+y)",
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
fn eval_hyperbolic_tanh_angle_sum_difference_passthrough_collapses_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "((tanh(x+y)) + m) - ((((tanh(x)+tanh(y))/(1+tanh(x)*tanh(y))) + m))",
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
fn eval_hyperbolic_tanh_double_angle_passthrough_collapses_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "((2*tanh(x)/(1+tanh(x)^2)) + m) - ((tanh(2*x)) + m)",
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
fn eval_hyperbolic_tanh_exp_definition_passthrough_collapses_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "((tanh(x)) + m) - (((e^x - e^(-x))/(e^x + e^(-x))) + m)",
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
fn eval_hyperbolic_tanh_exp_definition_scaled_difference_collapses_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "k*(tanh(x)) - k*((e^x - e^(-x))/(e^x + e^(-x)))",
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
fn eval_hyperbolic_tanh_exp_definition_common_denominator_collapses_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "((tanh(x))/q) - (((e^x - e^(-x))/(e^x + e^(-x)))/q)",
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
fn eval_hyperbolic_tanh_exp_definition_shifted_quotient_collapses_to_one_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "((tanh(x)) + 1)/(((e^x - e^(-x))/(e^x + e^(-x))) + 1)",
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
fn eval_hyperbolic_sinh_exp_definition_passthrough_collapses_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "((sinh(x)) + m) - (((e^x - e^(-x))/2) + m)",
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
fn eval_hyperbolic_cosh_exp_definition_shifted_quotient_collapses_to_one_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "((cosh(x)) + 1)/(((e^x + e^(-x))/2) + 1)",
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
fn eval_hyperbolic_cosh_triple_angle_passthrough_collapses_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "((4*cosh(x)^3-3*cosh(x)) + m) - ((cosh(3*x)) + m)",
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
fn eval_hyperbolic_cosh_triple_angle_difference_collapses_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "4*cosh(x)^3-3*cosh(x) - cosh(3*x)",
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
fn eval_hyperbolic_sinh_double_angle_passthrough_collapses_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "((2*sinh(x)*cosh(x)) + m) - ((sinh(2*x)) + m)",
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
fn eval_hyperbolic_tanh_double_angle_shifted_quotient_collapses_to_one_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "((2*tanh(x)/(1+tanh(x)^2)) + 1)/((tanh(2*x)) + 1)",
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
fn eval_hyperbolic_tanh_triple_angle_passthrough_collapses_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "((tanh(3*x)) + m) - ((((3*tanh(x)+tanh(x)^3)/(1+3*tanh(x)^2)) + m))",
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
fn eval_hyperbolic_sinh_sum_to_product_passthrough_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "((sinh(x)+sinh(y)) + m) - ((2*sinh((x+y)/2)*cosh((x-y)/2)) + m)",
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
fn eval_hyperbolic_sinh_sum_to_product_scaled_difference_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "k*(sinh(x)+sinh(y)) - k*(2*sinh((x+y)/2)*cosh((x-y)/2))",
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
        "Collapse Common-Scale Equivalent Difference"
    );
}
#[test]
fn eval_hyperbolic_sinh_sum_to_product_common_denominator_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "((sinh(x)+sinh(y))/q) - ((2*sinh((x+y)/2)*cosh((x-y)/2))/q)",
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
        "Collapse Common-Scale Equivalent Difference"
    );
}
#[test]
fn eval_hyperbolic_cosh_sum_to_product_scaled_difference_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "k*(cosh(x)+cosh(y)) - k*(2*cosh((x+y)/2)*cosh((x-y)/2))",
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
        "Collapse Common-Scale Equivalent Difference"
    );
}
#[test]
fn eval_hyperbolic_cosh_difference_shifted_quotient_collapses_to_one() {
    let (output, _code) = run_cli(&[
        "eval",
        "((cosh(x)-cosh(y)) + 1)/((2*sinh((x+y)/2)*sinh((x-y)/2)) + 1)",
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
fn eval_hyperbolic_cosh_difference_passthrough_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "((cosh(x)-cosh(y)+a) + m) - ((2*sinh((x+y)/2)*sinh((x-y)/2)+a) + m)",
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
}
#[test]
fn eval_hyperbolic_cosh_difference_scaled_passthrough_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "k*(cosh(x)-cosh(y)+a) - k*(2*sinh((x+y)/2)*sinh((x-y)/2)+a)",
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
fn eval_recursive_hyperbolic_sinh_passthrough_difference_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "((sinh(6*x)) + m) - ((sinh(5*x)*cosh(x)+cosh(5*x)*sinh(x)) + m)",
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
fn eval_recursive_hyperbolic_cosh_scaled_difference_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "k*(cosh(6*x)) - k*(cosh(5*x)*cosh(x)+sinh(5*x)*sinh(x))",
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
        "Collapse Common-Scale Equivalent Difference"
    );
}
#[test]
fn eval_hyperbolic_product_sum_triple_angle_passthrough_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "((2*sinh(2*x)*cosh(x)) + m) - ((4*sinh(x)+4*sinh(x)^3) + m)",
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
fn eval_hyperbolic_sinh_cubic_scaled_difference_collapses_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "k*(2*sinh(2*x)*cosh(x)) - k*(4*sinh(x)+4*sinh(x)^3)",
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
fn eval_hyperbolic_sinh_cubic_shifted_quotient_collapses_to_one_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "((2*sinh(2*x)*cosh(x)) + 1)/((4*sinh(x)+4*sinh(x)^3) + 1)",
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
fn eval_shifted_hyperbolic_pythagorean_with_passthrough_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "((cosh(x)^2 - 1) + m) - ((sinh(x)^2) + m)",
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
fn eval_shifted_hyperbolic_pythagorean_shifted_quotient_collapses_to_one() {
    let (output, _code) = run_cli(&[
        "eval",
        "((cosh(x)^2 - 1) + 1)/((sinh(x)^2) + 1)",
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
fn eval_shifted_hyperbolic_double_angle_shifted_quotient_collapses_to_one() {
    let (output, _code) = run_cli(&[
        "eval",
        "((cosh(2*x)) + 1)/((2*cosh(x)^2 - 1) + 1)",
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
fn eval_shifted_hyperbolic_double_angle_scaled_difference_collapses_to_zero_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "k*(cosh(2*x)) - k*(2*cosh(x)^2 - 1)",
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
fn eval_shifted_hyperbolic_double_angle_common_denominator_collapses_to_zero_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "((cosh(2*x))/q) - ((2*cosh(x)^2 - 1)/q)",
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
fn eval_hyperbolic_double_angle_sum_difference_collapses_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "cosh(2*x) - (cosh(x)^2 + sinh(x)^2)",
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
}
#[test]
fn eval_hyperbolic_double_angle_sum_passthrough_collapses_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "((cosh(2*x)) + m) - ((cosh(x)^2 + sinh(x)^2) + m)",
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
}
#[test]
fn eval_exp_minus_hyperbolic_sum_with_passthrough_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "((exp(x)) + m) - ((sinh(x) + cosh(x)) + m)",
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
fn eval_sinh_plus_cosh_minus_exp_to_zero_skips_convert_exp_to_power_noop() {
    let (output, _code) = run_cli(&[
        "eval",
        "cosh(x) + sinh(x) - e^x",
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
fn eval_exp_sum_minus_double_cosh_to_zero_uses_direct_hyperbolic_recognition() {
    let (output, _code) = run_cli(&[
        "eval",
        "exp(x) + exp(-x) - 2*cosh(x)",
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
fn eval_exp_difference_minus_double_sinh_to_zero_uses_direct_hyperbolic_recognition() {
    let (output, _code) = run_cli(&[
        "eval",
        "exp(x) - exp(-x) - 2*sinh(x)",
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
fn eval_hyperbolic_cubic_residual_difference_to_zero_uses_pythagorean_bridge_after_double_angle() {
    let (output, _code) = run_cli(&[
        "eval",
        "2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))",
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
    assert_eq!(steps[0]["rule"], "Hyperbolic Pythagorean Residual");
    assert_eq!(
        steps[0]["before"],
        "2 · sinh(x) · sinh(2 · x) - (4 · cosh(x)^3 - 4 · cosh(x))"
    );
    assert_eq!(steps[0]["after"], "0");
    assert!(steps[0].get("substeps").is_none());
}
#[test]
fn eval_hyperbolic_cosh_cubic_passthrough_difference_collapses_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "((2*sinh(2*x)*sinh(x)+a) + m) - ((4*cosh(x)^3 - 4*cosh(x)+a) + m)",
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
    assert_eq!(steps[0]["rule"], "Hyperbolic Pythagorean Residual");
}
#[test]
fn eval_hyperbolic_cosh_cubic_shifted_quotient_collapses_to_one_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "((2*sinh(2*x)*sinh(x)+a) + 1)/((4*cosh(x)^3 - 4*cosh(x)+a) + 1)",
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
fn eval_atanh_square_ratio_log_difference_collapses_to_zero() {
    let (output, code) = run_cli(&[
        "eval",
        "atanh((x^2 - 1)/(x^2 + 1)) - log(x)",
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
fn eval_atanh_square_ratio_log_difference_steps_hide_noop_negation_cleanup() {
    let (output, code) = run_cli(&[
        "eval",
        "atanh((x^2 - 1)/(x^2 + 1)) - log(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    assert_eq!(
        code, 0,
        "expected successful CLI exit, got {code} with output: {output}"
    );

    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "0");
    let steps = wire["steps"].as_array().expect("steps array");
    assert!(
        steps
            .iter()
            .all(|step| step["rule"] != "Quitar paréntesis tras el signo menos"),
        "unexpected noop negation cleanup step(s): {steps:?}"
    );
}
#[test]
fn eval_composed_log_hyperbolic_inverse_trig_zero_mix_collapses_to_zero() {
    let (output, code) = run_cli(&[
        "eval",
        "((exp(y * log(x)) - x^y) + (atanh((x^2 - 1)/(x^2 + 1)) - log(x)) + (asin(x/sqrt(x^2 + 1)) - atan(x)))",
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
fn eval_composed_log_hyperbolic_inverse_trig_zero_mix_keeps_exact_zero_subset_step() {
    let (output, code) = run_cli(&[
        "eval",
        "((exp(y * log(x)) - x^y) + (atanh((x^2 - 1)/(x^2 + 1)) - log(x)) + (asin(x/sqrt(x^2 + 1)) - atan(x)))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    assert_eq!(
        code, 0,
        "expected successful CLI exit, got {code} with output: {output}"
    );

    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "0");
    let steps = wire["steps"].as_array().expect("steps array");
    assert!(
        steps
            .iter()
            .any(|step| step["rule"] == "Cancelar la subexpresión idénticamente nula"),
        "expected composed simplification to retain exact-zero subset step: {steps:?}"
    );
}
#[test]
fn eval_log_square_hyperbolic_cubic_mix_uses_single_targeted_zero_step_and_keeps_domain_guards() {
    let (output, code) = run_cli(&[
        "eval",
        "(ln((x*y)^2) - ln(x^2) - ln(y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    assert_eq!(
        code, 0,
        "expected successful CLI exit, got {code} with output: {output}"
    );

    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "0");
    assert_eq!(wire["steps_count"], 1);
    assert_eq!(wire["required_display"], json!(["x ≠ 0", "y ≠ 0"]));

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Cancelar la subexpresión idénticamente nula"
    );
    assert_eq!(steps[0]["after"], "0");
}
#[test]
fn eval_acosh_log_chain_cancellation_does_not_show_conjugate_substeps() {
    let (output, _code) = run_cli(&[
        "eval",
        "acosh( (log((sin(y))^2) / log((cos(x))^2)) * (log((cosh(z))^2) / log((sin(y))^2)) * (log((sec(x))^(-2)) / log((cosh(z))^2)) )",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");

    let steps = wire["steps"].as_array().expect("steps array");
    let special_value = steps
        .iter()
        .find(|step| step["rule"] == "Evaluar valor hiperbólico especial")
        .expect("hyperbolic special-value step");

    assert_eq!(special_value["before"], "acosh(1)");
    for step in steps {
        let Some(substeps) = step.get("substeps").and_then(|value| value.as_array()) else {
            continue;
        };
        let rendered = format!("{substeps:?}");
        assert!(
            !rendered.contains("Conjugate") && !rendered.contains("conjugate"),
            "acosh log cancellation should not inherit conjugate substeps: {step:?}"
        );
    }
}
