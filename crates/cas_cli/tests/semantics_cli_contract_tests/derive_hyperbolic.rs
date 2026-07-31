use super::*;

#[test]
fn derive_hyperbolic_double_angle_avoids_generic_canonicalize_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*sinh(x)*cosh(x), sinh(2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_ne!(
        steps[0]["rule"],
        "Reescribir el producto",
        "hyperbolic double-angle contraction should not fall back to a generic canonicalization step"
    );
}
#[test]
fn derive_hyperbolic_sum_to_product_prunes_canonicalize_multiplication_tail() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sinh(x)+sinh(y), 2*sinh((x+y)/2)*cosh((x-y)/2)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand");
    assert_eq!(wire["steps_count"], 1);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Product-to-Sum Identity");
}
#[test]
fn derive_hyperbolic_half_angle_backward_avoids_generic_canonicalize_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (cosh(x)-1)/2, sinh(x/2)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Half-Angle Squares");
}
#[test]
fn derive_negative_hyperbolic_cosh_half_angle_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -(cosh(x)+1)/2, -cosh(x/2)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Half-Angle Squares");
}
#[test]
fn derive_negative_hyperbolic_sinh_half_angle_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -(cosh(x)-1)/2, -sinh(x/2)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Half-Angle Squares");
}
#[test]
fn derive_negative_hyperbolic_cosh_half_angle_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -cosh(x/2)^2, -(cosh(x)+1)/2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Half-Angle Squares");
}
#[test]
fn derive_negative_hyperbolic_sinh_half_angle_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -sinh(x/2)^2, -(cosh(x)-1)/2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Half-Angle Squares");
}
#[test]
fn derive_hyperbolic_cosh_double_angle_variant_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cosh(2*x), 2*cosh(x)^2-1",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["ok"], true);
    assert_eq!(wire["strategy"], "rewrite hyperbolics");

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}
#[test]
fn derive_hyperbolic_expansion_from_exp_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive exp(x), sinh(x)+cosh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Exponential Identity");
}
#[test]
fn derive_hyperbolic_expansion_from_negative_exp_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive exp(-x), cosh(x)-sinh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Exponential Identity");
}
#[test]
fn derive_hyperbolic_expansion_from_negated_negative_exp_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -exp(-x), sinh(x)-cosh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Exponential Identity");
}
#[test]
fn derive_hyperbolic_tanh_pythagorean_backward_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 1/cosh(x)^2, 1-tanh(x)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Pythagorean Identity");
}
#[test]
fn derive_negative_hyperbolic_tanh_pythagorean_forward_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive tanh(x)^2-1, -1/cosh(x)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Pythagorean Identity");
}
#[test]
fn derive_negative_hyperbolic_tanh_pythagorean_backward_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -1/cosh(x)^2, tanh(x)^2-1",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Pythagorean Identity");
}
#[test]
fn derive_hyperbolic_sinh_definition_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sinh(x), (e^x-e^(-x))/2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Exponential Identity");
}
#[test]
fn derive_negative_hyperbolic_cosh_definition_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -(e^x+e^(-x))/2, -cosh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Exponential Identity");
}
#[test]
fn derive_negative_hyperbolic_cosh_definition_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -cosh(x), -(e^x+e^(-x))/2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Exponential Identity");
}
#[test]
fn derive_hyperbolic_tanh_definition_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive tanh(x), (e^x-e^(-x))/(e^x+e^(-x))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Exponential Identity");
}
#[test]
fn derive_negative_hyperbolic_tanh_quotient_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -sinh(x)/cosh(x), -tanh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(
        &steps[0]["rule"],
        "Reconocer tangente hiperbólica desde un cociente",
    );
}
#[test]
fn derive_negative_hyperbolic_tanh_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -tanh(x), -sinh(x)/cosh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Quotient Identity");
}
#[test]
fn derive_negative_hyperbolic_sinh_double_angle_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -2*sinh(x)*cosh(x), -sinh(2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}
#[test]
fn derive_negative_hyperbolic_sinh_double_angle_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -sinh(2*x), -2*sinh(x)*cosh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}
#[test]
fn derive_hyperbolic_cosh_double_angle_expansion_to_sinh_mixed_polynomial_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*cosh(2*x)*sinh(x), 4*cosh(x)^2*sinh(x)-2*sinh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}
#[test]
fn derive_hyperbolic_cosh_double_angle_expansion_to_cosh_mixed_polynomial_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*cosh(2*x)*cosh(x), 2*cosh(x)+4*sinh(x)^2*cosh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}
#[test]
fn derive_hyperbolic_cosh_double_angle_contraction_from_sinh_mixed_polynomial_uses_single_named_step(
) {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 4*cosh(x)^2*sinh(x)-2*sinh(x), 2*cosh(2*x)*sinh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}
#[test]
fn derive_hyperbolic_cosh_double_angle_contraction_from_cosh_mixed_polynomial_uses_single_named_step(
) {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*cosh(x)+4*sinh(x)^2*cosh(x), 2*cosh(2*x)*cosh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}
#[test]
fn derive_hyperbolic_product_to_sum_expansion_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*sinh(2*x)*cosh(x), sinh(3*x)+sinh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Product-to-Sum Identity");
}
#[test]
fn derive_hyperbolic_sum_to_product_contraction_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sinh(3*x)-sinh(x), 2*cosh(2*x)*sinh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Product-to-Sum Identity");
}
#[test]
fn derive_hyperbolic_product_to_sum_polynomial_expands_to_polynomial_form() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*sinh(2*x)*sinh(x), 4*cosh(x)^3-4*cosh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    // Proves 2·sinh(2x)·sinh(x) = 4·cosh³(x) − 4·cosh(x) (both = 4·cosh·sinh²).
    // This previously took two visible steps (product-to-sum → cosh(3x)−cosh(x),
    // then triple-angle) because the intermediate cosh(3x)−cosh(x) was a forced
    // checkpoint: simplifying it further hit the bug that collapsed it to 0. Once
    // that wrong-answer is fixed, the intermediate simplifies cleanly in one
    // pass, so the expansion now lands on the polynomial form in a single step.
    assert_eq!(wire["strategy"], "expand");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Product-to-Sum Identity");
    assert_eq!(wire["result"], "4·cosh(x)^3 - 4·cosh(x)");
}
#[test]
fn derive_mixed_hyperbolic_double_angle_product_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 4*sinh(x)^2*cosh(x), 2*sinh(2*x)*sinh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}
#[test]
fn derive_negative_hyperbolic_tanh_double_angle_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -2*tanh(x)/(1+tanh(x)^2), -tanh(2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}
#[test]
fn derive_negative_hyperbolic_tanh_double_angle_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -tanh(2*x), -2*tanh(x)/(1+tanh(x)^2)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}
#[test]
fn derive_negative_hyperbolic_sinh_triple_angle_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -(3*sinh(x)+4*sinh(x)^3), -sinh(3*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Triple-Angle Identity");
}
#[test]
fn derive_negative_hyperbolic_sinh_triple_angle_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -sinh(3*x), -(3*sinh(x)+4*sinh(x)^3)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Triple-Angle Identity");
}
#[test]
fn derive_negative_hyperbolic_cosh_triple_angle_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -(4*cosh(x)^3-3*cosh(x)), -cosh(3*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Triple-Angle Identity");
}
#[test]
fn derive_negative_hyperbolic_cosh_triple_angle_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -cosh(3*x), -(4*cosh(x)^3-3*cosh(x))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Triple-Angle Identity");
}
#[test]
fn derive_hyperbolic_shifted_pythagorean_forward_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cosh(x)^2-1, sinh(x)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Pythagorean Identity");
}
#[test]
fn derive_hyperbolic_shifted_pythagorean_add_backward_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cosh(x)^2, 1+sinh(x)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Pythagorean Identity");
}
#[test]
fn derive_hyperbolic_shifted_double_angle_minus_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cosh(2*x)-1, 2*sinh(x)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}
#[test]
fn derive_hyperbolic_negative_shifted_double_angle_minus_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 1-cosh(2*x), -2*sinh(x)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}
#[test]
fn derive_hyperbolic_negative_shifted_double_angle_minus_backward_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -2*sinh(x)^2, 1-cosh(2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}
#[test]
fn derive_hyperbolic_negative_shifted_double_angle_plus_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -1-cosh(2*x), -2*cosh(x)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}
#[test]
fn derive_hyperbolic_negative_shifted_double_angle_plus_backward_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -2*cosh(x)^2, -1-cosh(2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}
#[test]
fn derive_hyperbolic_negative_double_angle_cosh_sq_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 1-2*cosh(x)^2, -cosh(2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}
#[test]
fn derive_hyperbolic_negative_double_angle_cosh_sq_backward_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -cosh(2*x), 1-2*cosh(x)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}
#[test]
fn derive_hyperbolic_shifted_double_angle_plus_backward_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*cosh(x)^2, cosh(2*x)+1",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}
#[test]
fn derive_hyperbolic_double_angle_two_cosh_sq_minus_one_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*cosh(x)^2-1, cosh(2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}
#[test]
fn derive_hyperbolic_double_angle_two_sinh_sq_plus_one_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*sinh(x)^2+1, cosh(2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}
#[test]
fn derive_hyperbolic_double_angle_with_passthrough_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*sinh(x)*cosh(x)+a, sinh(2*x)+a",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}
#[test]
fn derive_hyperbolic_pythagorean_with_passthrough_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cosh(x)^2-sinh(x)^2+a, 1+a",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Pythagorean Identity");
}
#[test]
fn derive_hyperbolic_angle_sum_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sinh(x)*cosh(y)+cosh(x)*sinh(y), sinh(x+y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(
        &steps[0]["rule"],
        "Hyperbolic Angle Sum/Difference Identity",
    );
}
#[test]
fn derive_hyperbolic_angle_diff_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cosh(x)*cosh(y)-sinh(x)*sinh(y), cosh(x-y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(
        &steps[0]["rule"],
        "Hyperbolic Angle Sum/Difference Identity",
    );
}
#[test]
fn derive_negative_hyperbolic_angle_diff_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -(cosh(x)*cosh(y)-sinh(x)*sinh(y)), -cosh(x-y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(
        &steps[0]["rule"],
        "Hyperbolic Angle Sum/Difference Identity",
    );
}
#[test]
fn derive_recursive_hyperbolic_sinh_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sinh(6*x), sinh(5*x)*cosh(x)+cosh(5*x)*sinh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(
        &steps[0]["rule"],
        "Hyperbolic Angle Sum/Difference Identity",
    );
}
#[test]
fn derive_hyperbolic_sinh_angle_sum_expansion_uses_expand_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sinh(x+y), sinh(x)*cosh(y)+cosh(x)*sinh(y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(
        &steps[0]["rule"],
        "Hyperbolic Angle Sum/Difference Identity",
    );
}
#[test]
fn derive_recursive_hyperbolic_cosh_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cosh(6*x), cosh(5*x)*cosh(x)+sinh(5*x)*sinh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(
        &steps[0]["rule"],
        "Hyperbolic Angle Sum/Difference Identity",
    );
}
#[test]
fn derive_hyperbolic_tanh_angle_sum_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive tanh(x+y), (tanh(x)+tanh(y))/(1+tanh(x)*tanh(y))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(
        &steps[0]["rule"],
        "Hyperbolic Angle Sum/Difference Identity",
    );
}
#[test]
fn derive_hyperbolic_tanh_angle_sum_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (tanh(x)+tanh(y))/(1+tanh(x)*tanh(y)), tanh(x+y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(
        &steps[0]["rule"],
        "Hyperbolic Angle Sum/Difference Identity",
    );
}
#[test]
fn derive_hyperbolic_tanh_triple_angle_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive tanh(3*x), (3*tanh(x)+tanh(x)^3)/(1+3*tanh(x)^2)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Triple-Angle Identity");
}
#[test]
fn derive_hyperbolic_tanh_triple_angle_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (3*tanh(x)+tanh(x)^3)/(1+3*tanh(x)^2), tanh(3*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Triple-Angle Identity");
}
#[test]
fn derive_exponential_scaled_argument_to_scaled_cosh_uses_direct_identity() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive exp(2*x)+exp(-2*x), 2*cosh(2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Exponential Identity");
}
#[test]
fn derive_exponential_to_scaled_cosh_uses_direct_identity() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive exp(x)+exp(-x), 2*cosh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Exponential Identity");
}
#[test]
fn derive_scaled_cosh_to_exponential_sum_uses_direct_identity() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*cosh(2*x), exp(2*x)+exp(-2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Exponential Identity");
}
#[test]
fn derive_scaled_sinh_to_exponential_difference_uses_direct_identity() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*sinh(x), exp(x)-exp(-x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Exponential Identity");
}
#[test]
fn derive_scaled_exponential_ratio_to_tanh_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (e^(2*x)-e^(-2*x))/(e^(2*x)+e^(-2*x)), tanh(2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Exponential Identity");
}
#[test]
fn derive_tanh_exponential_definition_drops_redundant_nonzero_requires() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive tanh(x), (e^x - e^(-x))/(e^x + e^(-x))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    assert_eq!(wire["required_display"], json!([]));
}
#[test]
fn derive_tanh_reciprocal_exponential_definition_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive tanh(x), (exp(x)-1/exp(x))/(exp(x)+1/exp(x))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    assert_eq!(wire["steps_count"], 1);
    assert_eq!(wire["required_display"], json!([]));
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Exponential Identity");
}
#[test]
fn derive_negative_tanh_reciprocal_exponential_definition_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -tanh(x), (1/exp(x)-exp(x))/(exp(x)+1/exp(x))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    assert_eq!(wire["steps_count"], 1);
    assert_eq!(wire["required_display"], json!([]));
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Exponential Identity");
}
#[test]
fn derive_scaled_cosh_reciprocal_exponential_definition_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*cosh(x), exp(x)+1/exp(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    assert_eq!(wire["steps_count"], 1);
    assert_eq!(wire["required_display"], json!([]));
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Exponential Identity");
}
#[test]
fn derive_scaled_sinh_reciprocal_exponential_definition_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*sinh(x), exp(x)-1/exp(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    assert_eq!(wire["steps_count"], 1);
    assert_eq!(wire["required_display"], json!([]));
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Exponential Identity");
}
#[test]
fn derive_half_cosh_reciprocal_exponential_recognition_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (exp(x)+1/exp(x))/2, cosh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    assert_eq!(wire["steps_count"], 1);
    assert_eq!(wire["required_display"], json!([]));
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Exponential Identity");
}
#[test]
fn derive_half_sinh_reciprocal_exponential_recognition_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (exp(x)-1/exp(x))/2, sinh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    assert_eq!(wire["steps_count"], 1);
    assert_eq!(wire["required_display"], json!([]));
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Exponential Identity");
}
#[test]
fn derive_negative_scaled_cosh_reciprocal_exponential_definition_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -2*cosh(x), -(exp(x)+1/exp(x))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    assert_eq!(wire["steps_count"], 1);
    assert_eq!(wire["required_display"], json!([]));
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Exponential Identity");
}
#[test]
fn derive_negative_scaled_sinh_reciprocal_exponential_definition_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -2*sinh(x), 1/exp(x)-exp(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    assert_eq!(wire["steps_count"], 1);
    assert_eq!(wire["required_display"], json!([]));
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Exponential Identity");
}
#[test]
fn derive_negative_half_cosh_reciprocal_exponential_definition_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -cosh(x), -(exp(x)+1/exp(x))/2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    assert_eq!(wire["steps_count"], 1);
    assert_eq!(wire["required_display"], json!([]));
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Exponential Identity");
}
#[test]
fn derive_negative_half_sinh_reciprocal_exponential_definition_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -sinh(x), -(exp(x)-1/exp(x))/2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    assert_eq!(wire["steps_count"], 1);
    assert_eq!(wire["required_display"], json!([]));
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Hyperbolic Exponential Identity");
}
