use super::*;

#[test]
fn eval_exact_phase_shift_pair_common_denominator_difference_collapses_after_shared_denominator() {
    let (output, _code) = run_cli(&[
        "eval",
        "((sin(x)+cos(x)+sin(y)+cos(y))/q) - ((sqrt(2)*sin(x+pi/4)+sqrt(2)*sin(y+pi/4))/q)",
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
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Collapse Common-Scale Equivalent Difference"
    );
}
#[test]
fn eval_exact_phase_shift_pair_shifted_quotient_collapses_to_one() {
    let (output, _code) = run_cli(&[
        "eval",
        "((sin(x)+cos(x)+sin(y)+cos(y)) + 1)/((sqrt(2)*sin(x+pi/4)+sqrt(2)*sin(y+pi/4)) + 1)",
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
fn eval_fraction_difference_to_zero_shows_common_denominator_substeps() {
    let (output, _code) = run_cli(&[
        "eval",
        "1/(x - 1) - 1/(x + 1) - 2/(x^2 - 1)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    assert_eq!(wire["steps_count"], 3);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_rule_eq(&steps[0]["rule"], "Sumar fracciones");
    let substeps = steps[0]["substeps"].as_array().expect("substeps array");
    assert_eq!(substeps.len(), 2);
    assert_eq!(substeps[0]["title"], "Llevar a denominador común");
    assert_eq!(
        substeps[1]["title"],
        "Simplificar el numerador y el denominador"
    );
    assert_rule_matches_any(
        &steps[1]["rule"],
        &[
            "Agrupar términos semejantes",
            "Cancel Exact Additive Pairs",
            "Cancelar términos opuestos",
        ],
    );
    let before_latex = steps[1]["before_latex"]
        .as_str()
        .expect("step 2 before_latex");
    assert!(
        before_latex.contains("\\frac{{\\color{red}{1 + x - (x - 1)}}}")
            || before_latex.contains("\\frac{{\\color{red}{x + 1 - (x - 1)}}}"),
        "step 2 should highlight the full numerator scope, got: {before_latex}"
    );
    assert!(
        !before_latex.contains("{\\color{red}{{x}^{2}}}"),
        "step 2 should not leak the highlight into the unrelated second denominator, got: {before_latex}"
    );
    assert_rule_eq(&steps[2]["rule"], "Cancel Equal Fractions Difference");
}
#[test]
fn eval_dirichlet_common_denominator_difference_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "((1 + 2*cos(x) + 2*cos(2*x))/q) - ((sin(5*x/2)/sin(x/2))/q)",
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
fn eval_dirichlet_shifted_quotient_collapses_to_one() {
    let (output, _code) = run_cli(&[
        "eval",
        "((1 + 2*cos(x) + 2*cos(2*x)) + 1)/((sin(5*x/2)/sin(x/2)) + 1)",
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
fn eval_complete_square_common_denominator_difference_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "((a*x^2 + b*x + c)/q) - ((a*(x + b/(2*a))^2 + c - b^2/(4*a))/q)",
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
    let required = wire["required_display"].as_array().expect("required array");
    assert!(required.iter().any(|item| item == "a ≠ 0"));
    assert!(required.iter().any(|item| item == "q ≠ 0"));
}
#[test]
fn eval_complete_square_shifted_quotient_collapses_to_one() {
    let (output, _code) = run_cli(&[
        "eval",
        "((a*x^2 - b*x + c) + 1)/((a*(x - b/(2*a))^2 + c - b^2/(4*a)) + 1)",
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
    let required = wire["required_display"].as_array().expect("required array");
    assert!(required.iter().any(|item| item == "a ≠ 0"));
}
#[test]
fn eval_complete_square_fractional_symbolic_passthrough_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "(((a/2)*x^2 + b*x + c) + m) - (((a/2)*(x + b/a)^2 + c - b^2/(2*a)) + m)",
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
    let required = wire["required_display"].as_array().expect("required array");
    assert!(required.iter().any(|item| item == "a ≠ 0"));
}
#[test]
fn eval_complete_square_fractional_symbolic_common_denominator_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "(((a/2)*x^2 + b*x + c)/q) - (((a/2)*(x + b/a)^2 + c - b^2/(2*a))/q)",
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
    let required = wire["required_display"].as_array().expect("required array");
    assert!(required.iter().any(|item| item == "a ≠ 0"));
    assert!(required.iter().any(|item| item == "q ≠ 0"));
}
#[test]
fn eval_complete_square_fractional_symbolic_scaled_difference_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "k*((a/2)*x^2 + b*x + c) - k*((a/2)*(x + b/a)^2 + c - b^2/(2*a))",
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
fn eval_complete_square_fractional_symbolic_shifted_quotient_collapses_to_one() {
    let (output, _code) = run_cli(&[
        "eval",
        "(((a/2)*x^2 + b*x + c) + 1)/(((a/2)*(x + b/a)^2 + c - b^2/(2*a)) + 1)",
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
    let required = wire["required_display"].as_array().expect("required array");
    assert!(required.iter().any(|item| item == "a ≠ 0"));
}
#[test]
fn eval_trinomial_square_shifted_quotient_collapses_to_one() {
    let (output, _code) = run_cli(&[
        "eval",
        "((a + b + c)^2 + 1)/(a^2 + b^2 + c^2 + 2*a*b + 2*a*c + 2*b*c + 1)",
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
fn eval_cubes_quotient_zero_uses_factor_then_cancel_substeps() {
    let (output, _code) = run_cli(&[
        "eval",
        "(a^3-b^3)/(a-b) - (a^2 + a*b + b^2)",
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
        "Subtract Expanded Sum/Difference of Cubes Quotient"
    );
    let substeps = steps[0]["substeps"].as_array().expect("substeps array");
    assert_eq!(substeps.len(), 2);
    assert_eq!(
        substeps[0]["title"],
        "Usar a^3 - b^3 = (a - b)(a^2 + ab + b^2)"
    );
    assert_eq!(
        substeps[1]["title"],
        "Cancelar el factor común del numerador y el denominador"
    );
}
#[test]
fn eval_cubes_quotient_passthrough_zero_collapses_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "((a^3-b^3)/(a-b)+c) - (a^2 + a*b + b^2 + c)",
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
fn eval_sum_cubes_quotient_shifted_quotient_collapses_to_one_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "(((a^3+b^3)/(a+b)+c) + 1)/((a^2 - a*b + b^2 + c) + 1)",
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
fn eval_abs_denominator_displays_inner_nonzero_guard() {
    let (output, _code) = run_cli(&["eval", "1/abs(x)", "--format", "json"]);
    let wire = parse_wire(&output);

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert_eq!(required.len(), 1);
    assert_eq!(required[0], "x ≠ 0");
}
#[test]
fn eval_telescoping_fraction_shifted_quadratic_same_denominator_collapses_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "((1/(x+b) - 1/(x+c))/q) - (((c-b)/(x^2+(b+c)*x+b*c))/q)",
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
fn eval_telescoping_fraction_symbolic_difference_squares_passthrough_collapses_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "((1/(2*a)*(1/(x-a) - 1/(x+a))) + m) - ((1/(x^2-a^2)) + m)",
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
fn eval_telescoping_fraction_symbolic_difference_squares_shifted_quotient_collapses_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "(((1/(2*a)*(1/(x-a) - 1/(x+a))) + 1))/(((1/(x^2-a^2)) + 1))",
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
fn eval_telescoping_fraction_symbolic_difference_squares_same_denominator_collapses_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "((1/(2*a)*(1/(x-a) - 1/(x+a)))/q) - ((1/(x^2-a^2))/q)",
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
fn eval_telescoping_fraction_symbolic_difference_squares_same_denominator_reverse_collapses_in_one_step(
) {
    let (output, _code) = run_cli(&[
        "eval",
        "((1/(x^2-a^2))/q) - ((1/(2*a)*(1/(x-a) - 1/(x+a)))/q)",
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
fn eval_finite_telescoping_product_same_denominator_collapses_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "((product(1 - 1/(k+a)^2, k, m, n))/q) - ((((m+a-1)*(n+a+1))/((m+a)*(n+a)))/q)",
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
fn eval_solve_reciprocal_sign_inequality_narrates_denominator_reduction() {
    let (output, code) = run_cli(&[
        "eval",
        "solve(1/(x-sqrt(2))>0,x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    assert_eq!(code, 0, "output: {output}");
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "(sqrt(2), infinity)");
    let steps = wire["solve_steps"].as_array().cloned().unwrap_or_default();
    let descs: Vec<&str> = steps
        .iter()
        .filter_map(|s| s["description"].as_str())
        .collect();
    assert_eq!(
        descs,
        vec!["Signo de un recíproco: c/g se compara con cero igual que su denominador"],
        "expected the reciprocal-sign reduction line, got {steps:?}"
    );
    assert_eq!(steps[0]["equation"], "x - sqrt(2) > 0");
}
