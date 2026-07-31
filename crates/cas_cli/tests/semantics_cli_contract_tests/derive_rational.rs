use super::*;

#[test]
fn derive_nested_fraction_one_over_sum_uses_named_strategy() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 1/(1/a + 1/b), (a*b)/(a+b)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "nested fraction");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Cancelar factores en una fracción");
}
#[test]
fn derive_nested_fraction_one_over_sum_uses_common_denominator_substep() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 1/(1/x + 1/y), (x*y)/(x+y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "nested fraction");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    let substeps = steps[0]["substeps"].as_array().expect("substeps array");
    assert_eq!(substeps.len(), 2);
    assert_eq!(
        substeps[0]["title"],
        "Llevar a denominador común dentro del denominador"
    );
    let after = substeps[0]["after_latex"].as_str().expect("after_latex");
    assert!(
        after.contains("x\\cdot y")
            || after.contains("y\\cdot x")
            || after.contains("x \\cdot y")
            || after.contains("y \\cdot x"),
        "expected common denominator product in substep, got: {after}"
    );
    assert!(
        !after.contains("\\frac{1}{y} \\cdot x") && !after.contains("\\frac{1}{x} \\cdot y"),
        "expected to avoid partially simplified reciprocal product in substep, got: {after}"
    );
    assert_eq!(substeps[1]["title"], "Invertir la fracción del denominador");
}
#[test]
fn derive_nested_fraction_one_over_sum_drops_redundant_reciprocal_sum_guard() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 1/(1/a + 1/b), (a*b)/(a+b)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    let required: Vec<&str> = required.iter().filter_map(|item| item.as_str()).collect();

    assert!(
        !required.contains(&"1 / a + 1 / b ≠ 0"),
        "expected reciprocal-sum guard to be dominated by atomic guards: {required:?}"
    );
    assert!(required.contains(&"a + b ≠ 0"));
    assert!(required.contains(&"a ≠ 0"));
    assert!(required.contains(&"b ≠ 0"));
}
#[test]
fn derive_nested_fraction_structural_uses_named_strategy() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive a/(b + c/d), a*d/(b*d+c)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "nested fraction");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Cancelar factores en una fracción");
}
#[test]
fn derive_nested_fraction_reciprocal_sum_difference_shows_common_denominator_substeps() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (1/x + 1/y)/(1/x - 1/y), (x+y)/(y-x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "nested fraction");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Cancelar factores en una fracción");

    let substeps = steps[0]["substeps"].as_array().expect("substeps array");
    assert_eq!(substeps.len(), 2);
    assert_eq!(
        substeps[0]["title"],
        "Llevar el numerador y el denominador a común denominador"
    );
    assert_eq!(
        substeps[1]["title"],
        "Cancelar el denominador común de numerador y denominador"
    );
}
#[test]
fn derive_difference_of_squares_fraction_uses_named_fraction_cancel_and_keeps_guard() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (a^2-b^2)/(a-b), a+b",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "cancel fraction");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Factorizar una diferencia de cuadrados y cancelar"
    );
    assert_eq!(steps[0]["before"], "(a^2 - b^2)/(a - b)");

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert!(
        required.iter().any(|item| item == "a - b ≠ 0"),
        "expected denominator nonzero guard in required_display: {required:?}"
    );
}
#[test]
fn derive_difference_of_squares_fraction_with_passthrough_uses_named_fraction_cancel_and_keeps_guard(
) {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (a^2-b^2)/(a-b)+c, a+b+c",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "cancel fraction");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Factorizar una diferencia de cuadrados y cancelar"
    );
    assert_eq!(steps[0]["before"], "(a^2 - b^2)/(a - b) + c");

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert!(
        required.iter().any(|item| item == "a - b ≠ 0"),
        "expected denominator nonzero guard in required_display: {required:?}"
    );
}
#[test]
fn derive_difference_of_cubes_fraction_uses_named_fraction_cancel_and_keeps_guard() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (a^3-b^3)/(a-b), a^2+a*b+b^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "cancel fraction");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Factorizar cubos y cancelar");

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert!(
        required.iter().any(|item| item == "a - b ≠ 0"),
        "expected denominator nonzero guard in required_display: {required:?}"
    );
}
#[test]
fn derive_difference_of_cubes_fraction_with_passthrough_uses_named_fraction_cancel_and_keeps_guard()
{
    let (output, _code) = run_cli(&[
        "eval",
        "derive (a^3-b^3)/(a-b)+c, a^2+a*b+b^2+c",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "cancel fraction");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Factorizar cubos y cancelar");

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert!(
        required.iter().any(|item| item == "a - b ≠ 0"),
        "expected denominator nonzero guard in required_display: {required:?}"
    );
}
#[test]
fn derive_common_factor_fraction_uses_named_fraction_cancel_and_keeps_guards() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (a*x^2)/(b*x), (a*x)/b",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "cancel fraction");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Cancelar un factor común");

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert!(
        required.iter().any(|item| item == "x ≠ 0"),
        "expected cancelled-factor guard in required_display: {required:?}"
    );
}
#[test]
fn derive_difference_of_cubes_fraction_keeps_requested_target_text_in_final_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (x^3-1)/(x-1), x^2+x+1",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(wire["strategy"], "cancel fraction");
    assert_rule_eq(&steps[0]["rule"], "Factorizar cubos y cancelar");
    assert_eq!(steps[0]["after"], "x^2 + x + 1");
}
#[test]
fn derive_geometric_difference_fraction_uses_direct_cancel_fraction() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (x^4-1)/(x-1), x^3+x^2+x+1",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "cancel fraction");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["before"], "(x^4 - 1)/(x - 1)");
    assert_eq!(steps[0]["after"], "x^3 + x^2 + x + 1");
}
#[test]
fn derive_geometric_difference_even_quotient_uses_direct_cancel_fraction() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (x^4-1)/(x^2-1), x^2+1",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "cancel fraction");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["before"], "(x^4 - 1)/(x^2 - 1)");
    assert_eq!(steps[0]["after"], "x^2 + 1");
}
#[test]
fn derive_rationalize_then_cancel_to_zero_uses_rationalize_strategy() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 1 / (sqrt(x) - 1) - (sqrt(x) + 1) / (x - 1), 0",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rationalize");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 2);
    assert_rule_eq(&steps[0]["rule"], "Racionalizar el denominador");
    assert_rule_eq(&steps[1]["rule"], "Restar dos expresiones iguales");
}
#[test]
fn derive_finite_telescoping_sum_uses_concrete_partial_fraction_substeps() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sum(1/(k*(k+1)), k, 1, n), 1 - 1/(n+1)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "finite sums/products");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Evaluar suma telescópica finita");
    let substeps = steps[0]["substeps"].as_array().expect("substeps array");
    assert_eq!(substeps.len(), 2);
    assert_eq!(
        substeps[0]["title"],
        "Usar 1 / (u · (u + 1)) = 1 / u - 1 / (u + 1)"
    );
    assert_eq!(
        substeps[0]["before_latex"],
        json!("\\frac{1}{k\\cdot (k + 1)}")
    );
    assert_eq!(
        substeps[0]["after_latex"],
        json!("\\frac{1}{k} - \\frac{1}{k + 1}")
    );
    assert_eq!(
        substeps[1]["title"],
        "La suma telescópica cancela los términos intermedios"
    );
    assert_eq!(substeps[1]["after_latex"], json!("1 - \\frac{1}{n + 1}"));
}
#[test]
fn derive_affine_finite_telescoping_sum_uses_concrete_partial_fraction_substeps() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sum(1/((a*k+b)*(a*k+b+a)), k, m, n), 1/a*(1/(a*m+b) - 1/(a*n+a+b))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "finite sums/products");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Evaluar suma telescópica finita");
    let substeps = steps[0]["substeps"].as_array().expect("substeps array");
    assert_eq!(substeps.len(), 2);
    assert_eq!(
        substeps[0]["title"],
        "Usar 1 / (u · (u + g)) = 1 / g · (1 / u - 1 / (u + g))"
    );
    assert_eq!(
        substeps[0]["before_latex"],
        json!("\\frac{1}{(a\\cdot k + b)\\cdot (a\\cdot k + a + b)}")
    );
    assert_eq!(
        substeps[0]["after_latex"],
        json!("\\frac{1}{a}\\cdot \\left(\\frac{1}{a\\cdot k + b} - \\frac{1}{a\\cdot k + a + b}\\right)")
    );
    assert_eq!(
        substeps[1]["title"],
        "La suma telescópica cancela los términos intermedios"
    );
    assert_eq!(
        substeps[1]["after_latex"],
        json!("\\frac{1}{a}\\cdot (\\frac{1}{a\\cdot m + b} - \\frac{1}{a\\cdot n + a + b})")
    );
}
