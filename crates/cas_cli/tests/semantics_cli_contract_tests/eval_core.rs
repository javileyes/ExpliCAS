use super::*;

#[test]
fn eval_solve_inequality_input_latex_shows_the_real_relational_operator() {
    // The `input_latex` echo of `solve(<inequality>, x)` hard-coded ` = ` between the two sides, so an
    // inequality (`x²−2x−3 > 0`) was rendered as an EQUATION. It must show the real relation.
    for (input, needle, forbidden) in [
        ("solve(x^2-2*x-3>0,x)", "> 0", "= 0"),
        ("solve(x^2-4<=0,x)", "\\leq 0", "= 0"),
        ("solve(1/(x-2)>=1,x)", "\\geq 1", "= 1"),
        ("solve((x-1)/(x-2)<0,x)", "< 0", "= 0"),
    ] {
        let (output, _code) = run_cli(&["eval", input, "--format", "json"]);
        let wire = parse_wire(&output);
        let input_latex = wire["input_latex"].as_str().expect("input_latex");
        assert!(
            input_latex.contains(needle),
            "expected `{needle}` in input_latex for `{input}`, got: {input_latex}"
        );
        assert!(
            !input_latex.contains(forbidden),
            "did not expect `{forbidden}` in input_latex for `{input}`, got: {input_latex}"
        );
    }
    // Control: a genuine equation still renders `=`.
    let (output, _code) = run_cli(&["eval", "solve(x^2-5*x+6=0,x)", "--format", "json"]);
    let input_latex = parse_wire(&output)["input_latex"]
        .as_str()
        .expect("input_latex")
        .to_string();
    assert!(input_latex.contains("= 0"), "got: {input_latex}");
}
#[test]
fn eval_default_variable_integration_keeps_integral_latex_in_wire() {
    let (output, _code) = run_cli(&[
        "eval",
        "integrate(x^2)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], json!("1/3·x^3"));
    let input_latex = wire["input_latex"].as_str().expect("input_latex");
    assert!(
        input_latex.contains("\\int") && input_latex.contains("\\, dx"),
        "expected input_latex to show an integral with default dx, got: {input_latex}"
    );
    assert!(
        !input_latex.contains("\\text{integrate}"),
        "expected input_latex to avoid function-style integrate(), got: {input_latex}"
    );

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    let before_latex = steps[0]["before_latex"].as_str().expect("before_latex");
    assert!(
        before_latex.contains("\\int") && before_latex.contains("\\, dx"),
        "expected before_latex to show an integral with default dx, got: {before_latex}"
    );
    assert!(
        !before_latex.contains("\\text{integrate}"),
        "expected before_latex to avoid function-style integrate(), got: {before_latex}"
    );
}
#[test]
fn eval_unresolved_integration_residual_uses_integral_text_display() {
    let (output, _code) = run_cli(&[
        "eval",
        "integrate(sin(x^2))",
        "--format",
        "json",
        "--steps",
        "off",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], json!("int sin(x^2) , dx"));
    let result_latex = wire["result_latex"].as_str().expect("result_latex");
    assert!(
        result_latex.contains("\\int") && result_latex.contains("\\, dx"),
        "expected result_latex to show an unresolved integral, got: {result_latex}"
    );
    assert!(
        !result_latex.contains("\\text{integrate}"),
        "expected result_latex to avoid function-style integrate(), got: {result_latex}"
    );

    let text = wire["wire"]["messages"][0]["text"]
        .as_str()
        .expect("wire output text");
    assert!(
        text.starts_with("Result: int sin(x^2) , dx "),
        "expected wire text to use integral display, got: {text}"
    );
}
#[test]
fn eval_web_examples_keep_partial_additive_highlights_off_residual_terms() {
    for (expr, step_index, unchanged_residual) in [
        (
            "(ln(x*sqrt(x)) + ln(sqrt(x)/x^2)) + (sqrt(y)/(sqrt(y)-1) - sqrt(y)/(sqrt(y)+1) - (2*sqrt(y))/(y-1)) + (((1/x) - (1/y))/((y-x)/(x*y)) - 1)",
            10usize,
            "\\frac{2\\cdot \\sqrt{y}}{y - 1}",
        ),
        (
            "1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)",
            4usize,
            "\\frac{2 + 3\\cdot x}{1 + 2\\cdot x}",
        ),
    ] {
        let (output, _code) = run_cli(&["eval", expr, "--format", "json", "--steps", "on"]);
        let wire = parse_wire(&output);
        let steps = wire["steps"].as_array().expect("steps array");
        let step = &steps[step_index - 1];
        let before_latex = step["before_latex"].as_str().expect("before_latex");

        assert!(
            before_latex.contains("\\color{red}"),
            "step {step_index} should keep a red focus, got: {before_latex}"
        );
        assert!(
            before_latex.contains(unchanged_residual),
            "step {step_index} should keep the residual visible, got: {before_latex}"
        );
        assert!(
            !before_latex.contains(&format!("{{\\color{{red}}{{{unchanged_residual}}}}}")),
            "step {step_index} should not highlight the unchanged residual, got: {before_latex}"
        );
    }
}
#[test]
fn eval_web_example_erased_multiplicative_one_does_not_highlight_neighbor_factor() {
    let (output, _code) = run_cli(&[
        "eval",
        "acos(log( (sqrt(x^2 + 2)^(2/log(x^2 + 2)))^((log((cosh(x*y) + sinh(x*y))^3)/(3*x*y)) * ((x^3 + y^3)/((x+y)*(x^2 - x*y + y^2)))) ))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);
    let steps = wire["steps"].as_array().expect("steps array");
    let cancel_step = &steps[7];

    assert_eq!(
        cancel_step["rule"],
        "Cancelar numerador y denominador iguales"
    );
    let before_latex = cancel_step["before_latex"].as_str().expect("before_latex");
    let after_latex = cancel_step["after_latex"].as_str().expect("after_latex");
    assert!(
        before_latex.contains("{\\color{red}{\\frac{3\\cdot x\\cdot y}{3\\cdot x\\cdot y}}}"),
        "expected the canceling quotient highlighted before, got: {before_latex}"
    );
    assert!(
        !after_latex.contains("{\\color{green}{{x}^{3} + {y}^{3}}}"),
        "after_latex should not highlight the neighboring quotient numerator when the factor 1 disappears: {after_latex}"
    );
}
#[test]
fn eval_equiv_input_uses_latex_relation_in_wire() {
    let (output, _code) = run_cli(&[
        "eval",
        "equiv(sin(x+y), sin(x)*cos(y) + cos(x)*sin(y))",
        "--format",
        "json",
    ]);
    let wire = parse_wire(&output);

    let input_latex = wire["input_latex"].as_str().expect("input_latex");
    assert!(
        input_latex.contains("\\leftrightarrow"),
        "expected input_latex to show an equivalence relation, got: {input_latex}"
    );
    assert!(
        !input_latex.contains("\\operatorname{equiv}") && !input_latex.contains("\\text{equiv}"),
        "expected input_latex to avoid function-style equiv(), got: {input_latex}"
    );
    assert_eq!(wire["result"], "true");
}
#[test]
fn eval_general_phase_shift_with_passthrough_scaled_difference_collapses_after_common_factor() {
    let (output, _code) = run_cli(&[
        "eval",
        "k*(3*sin(x)+4*cos(x)+a) - k*(5*sin(x+arctan(4/3))+a)",
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
fn eval_scaled_exact_phase_shift_pair_difference_to_zero_collapses_after_common_factor() {
    let (output, _code) = run_cli(&[
        "eval",
        "k*(sin(x)+cos(x)) - k*(sqrt(2)*sin(x+pi/4))",
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
fn eval_dirichlet_passthrough_difference_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "((1 + 2*cos(x) + 2*cos(2*x)) + m) - ((sin(5*x/2)/sin(x/2)) + m)",
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
fn eval_dirichlet_scaled_difference_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "k*(1 + 2*cos(x) + 2*cos(2*x)) - k*(sin(5*x/2)/sin(x/2))",
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
fn eval_reverse_dirichlet_raw_difference_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "sin(5*x/2)/sin(x/2) - (1 + 2*cos(x) + 2*cos(2*x))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Trig Summation Identity");
    assert_eq!(wire["required_display"][0], "sin(x / 2) ≠ 0");
}
#[test]
fn eval_reverse_dirichlet_passthrough_difference_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "((sin(5*x/2)/sin(x/2)) + m) - ((1 + 2*cos(x) + 2*cos(2*x)) + m)",
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
    assert_eq!(wire["required_display"][0], "sin(x / 2) ≠ 0");
}
#[test]
fn eval_complete_square_passthrough_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "((a*x^2 - b*x + c) + m) - ((a*(x - b/(2*a))^2 + c - b^2/(4*a)) + m)",
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
fn eval_complete_square_symbolic_leading_passthrough_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "((a*x^2 + b*x + c) + m) - ((a*(x + b/(2*a))^2 + c - b^2/(4*a)) + m)",
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
fn eval_trinomial_square_passthrough_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "((a + b + c)^2 + m) - ((a^2 + b^2 + c^2 + 2*a*b + 2*a*c + 2*b*c) + m)",
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
fn eval_complete_square_scaled_difference_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "k*(a*x^2 - b*x + c) - k*(a*(x - b/(2*a))^2 + c - b^2/(4*a))",
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
fn eval_complete_square_symbolic_leading_scaled_difference_collapses_to_zero() {
    let (output, _code) = run_cli(&[
        "eval",
        "k*(a*x^2 + b*x + c) - k*(a*(x + b/(2*a))^2 + c - b^2/(4*a))",
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
/// A system trace must let the reader CHECK it, not just be told a method ran.
///
/// The pre-2026-07-28 narration emitted three steps that all carried
/// `exprs.first()` as a filler snapshot: «Identificar sistema de 2
/// ecuaciones», «Resolver por eliminación exacta (Cramer/Gauss)» and
/// «Solución única: cada valor sustituye exacto en todas las ecuaciones» all
/// displayed `x + y - 3 = 0`. Equation 2 never appeared, the elimination step
/// showed something it was not eliminating, and the verification asserted an
/// arithmetic the reader could not see.
///
/// This pins the three properties that fix required: every equation of the
/// system appears, each step carries its OWN equation, and the verification
/// shows the substituted arithmetic.
#[test]
fn eval_linear_system_trace_shows_every_equation_and_the_verification_arithmetic() {
    let (output, _code) = run_cli(&[
        "eval",
        "solve([x+y=3, x-y=1], [x, y])",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "{ x = 2, y = 1 }");

    let steps = wire["solve_steps"].as_array().expect("solve_steps array");
    let described: Vec<(&str, &str)> = steps
        .iter()
        .map(|s| {
            (
                s["description"].as_str().unwrap_or_default(),
                s["equation"].as_str().unwrap_or_default(),
            )
        })
        .collect();

    // Both equations of the system are stated, each on its own step.
    assert_eq!(described[0].1, "x + y - 3 = 0");
    assert_eq!(described[1].1, "x - y - 1 = 0");
    assert!(described[0].0.contains("Ecuación 1 de 2"));
    assert!(described[1].0.contains("Ecuación 2 de 2"));

    // Back-substitution publishes the value it produced, one per unknown.
    assert!(described[2].0.contains("Eliminación gaussiana exacta"));
    assert_eq!(described[2].1, "x = 2");
    assert_eq!(described[3].1, "y = 1");

    // The honesty clause is now CHECKABLE: the originals with the values in
    // place. The raw rebuild is what keeps the coefficients readable — a
    // canonical re-add folds and reorders (`2·x + 3·y` came out `2·3 + 2·3`).
    assert!(described[4].0.contains("Verificación exacta"));
    assert_eq!(described[4].1, "1 + 2 - 3 = 0");
    assert_eq!(described[5].1, "2 - 1 - 1 = 0");

    // No step may repeat another's snapshot: that WAS the defect.
    let equations: Vec<&str> = described.iter().map(|(_, eq)| *eq).collect();
    let mut unique = equations.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(
        unique.len(),
        equations.len(),
        "every step must carry its own equation, got {equations:?}"
    );
}
#[test]
fn eval_diff_bounded_inverse_trig_rejects_empty_open_interval_domain() {
    let (output, code) = run_cli(&[
        "eval",
        "diff(arcsin(sqrt(x^2+1)), x)",
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
        !output.contains("Usar regla de arcsin(u)")
            && !output.contains("Identificar u y du")
            && !output.contains("sqrt(-(x^2))"),
        "empty inverse-trig domain should not expose a chain rule or impossible derivative: {output}"
    );
    assert!(
        output.contains("Detectar dominio real vacío de la derivada de la función inversa"),
        "empty inverse-trig domain should explain the undefined derivative: {output}"
    );
}
#[test]
fn eval_diff_bounded_inverse_trig_rejects_shifted_empty_open_interval_domain() {
    let (output, code) = run_cli(&[
        "eval",
        "diff(arcsin((x+1)^2+1), x)",
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
        !output.contains("Usar regla de arcsin(u)")
            && !output.contains("Identificar u y du")
            && !output.contains("1 - (x^2 + 2·x + 2)^2 > 0"),
        "shifted empty inverse-trig domain should not expose a chain rule or impossible condition: {output}"
    );
    assert!(
        output.contains("Detectar dominio real vacío de la derivada de la función inversa"),
        "shifted empty inverse-trig domain should explain the undefined derivative: {output}"
    );
}
#[test]
fn eval_diff_bounded_inverse_trig_keeps_finite_boundary_constants() {
    for expr in ["diff(arcsin(-1), x)", "diff(arccos(-1), x)"] {
        let (output, code) = run_cli(&["eval", expr, "--format", "json", "--steps", "on"]);
        assert_eq!(code, 0, "unexpected command failure for {expr}: {output}");
        let wire = parse_wire(&output);

        assert_eq!(wire["result"], "0", "unexpected result for {expr}");
        assert_eq!(wire["required_display"], json!([]));
        assert!(
            wire["blocked_hints"].is_null(),
            "finite inverse-trig boundary should not be reported as an empty domain: {output}"
        );
    }
}
#[test]
fn eval_arcsin_arctan_expansion_difference_collapses_to_zero() {
    let (output, code) = run_cli(&[
        "eval",
        "asin(x/sqrt(x^2 + 1)) - atan(x)",
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
fn eval_complex_nested_fraction_pipeline_keeps_before_after_highlights_in_wire_latex() {
    let (output, _code) = run_cli(&[
        "eval",
        "1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 5);

    for (idx, step) in steps.iter().enumerate() {
        let before_latex = step["before_latex"].as_str().expect("before_latex");
        let after_latex = step["after_latex"].as_str().expect("after_latex");
        assert!(
            before_latex.contains("\\color{red}"),
            "expected red highlight in before_latex for step {}: {}",
            idx + 1,
            before_latex
        );
        assert!(
            after_latex.contains("\\color{green}"),
            "expected green highlight in after_latex for step {}: {}",
            idx + 1,
            after_latex
        );
    }

    let step4_before = steps[3]["before_latex"]
        .as_str()
        .expect("step4 before_latex");
    assert!(
        step4_before.contains("{\\color{red}{1}}")
            && step4_before.contains("{\\color{red}{\\frac{1 + x}{1 + x + x}}}"),
        "expected step 4 before_latex to highlight only changed additive terms, got: {step4_before}"
    );
    assert!(
        step4_before.contains("\\frac{2 + 3\\cdot x}{1 + 2\\cdot x}")
            && !step4_before.contains("{\\color{red}{\\frac{2 + 3\\cdot x}{1 + 2\\cdot x}}}"),
        "expected step 4 before_latex to leave the unchanged residual term uncolored, got: {step4_before}"
    );

    let step4_after = steps[3]["after_latex"].as_str().expect("step4 after_latex");
    assert!(
        step4_after.contains("\\color{green}")
            && step4_after.contains("\\frac{2 + 3\\cdot x}{1 + 2\\cdot x}")
            && !step4_after.contains("{\\color{green}{\\frac{2 + 3\\cdot x}{1 + 2\\cdot x}}}"),
        "expected step 4 after_latex to keep the unchanged residual term uncolored, got: {step4_after}"
    );
}
#[test]
fn eval_complex_nested_fraction_pipeline_shows_denominator_common_denominator_then_inversion() {
    let (output, _code) = run_cli(&[
        "eval",
        "1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 5);
    assert_rule_eq(&steps[2]["rule"], "Simplificar fracción anidada");
    let substeps = steps[2]["substeps"].as_array().expect("substeps array");
    assert_eq!(substeps.len(), 2);
    assert_eq!(
        substeps[0]["title"],
        "Llevar a denominador común dentro del denominador"
    );
    assert_eq!(substeps[1]["title"], "Invertir la fracción del denominador");
}
