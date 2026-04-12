use cas_solver::runtime::{Engine, EvalAction, EvalOptions, EvalRequest};
use cas_solver_core::engine_events::EngineEvent;

fn eval_output_for(expr: &str) -> (Engine, cas_solver::runtime::EvalOutput) {
    let mut engine = Engine::new();
    let parsed = cas_parser::parse(expr, &mut engine.simplifier.context).expect("parse");
    let output = engine
        .eval_stateless(
            EvalOptions::default(),
            EvalRequest {
                raw_input: expr.to_string(),
                parsed,
                action: EvalAction::Simplify,
                auto_store: false,
            },
        )
        .expect("eval");
    (engine, output)
}

#[test]
fn step_wire_off_mode_is_empty() {
    let (engine, output) = eval_output_for("x + x");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "off");
    assert!(steps.is_empty());
}

#[test]
fn step_wire_on_mode_matches_deterministically() {
    let (engine, output) = eval_output_for("(x + 2) + (x + 3)");
    let first =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");
    let second =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");
    assert_eq!(first.len(), second.len());
    if let (Some(a), Some(b)) = (first.first(), second.first()) {
        assert_eq!(a.rule, b.rule);
        assert_eq!(a.before, b.before);
        assert_eq!(a.after, b.after);
    }
}

#[test]
fn step_wire_events_fallback_is_used_when_steps_are_missing() {
    let mut ctx = cas_ast::Context::new();
    let x = ctx.var("x");
    let zero = ctx.num(0);
    let before = ctx.add(cas_ast::Expr::Add(x, zero));

    let steps = cas_didactic::collect_step_payloads_with_events(
        &[],
        &[EngineEvent::RuleApplied {
            rule_name: "Additive Identity".to_string(),
            before: x,
            after: x,
            global_before: Some(before),
            global_after: Some(x),
            is_chained: false,
        }],
        &ctx,
        "on",
    );

    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].rule, "Additive Identity");
    assert_eq!(steps[0].before, "x + 0");
    assert_eq!(steps[0].after, "x");
}

#[test]
fn step_wire_events_fallback_respects_off_mode() {
    let mut ctx = cas_ast::Context::new();
    let x = ctx.var("x");
    let steps = cas_didactic::collect_step_payloads_with_events(
        &[],
        &[EngineEvent::RuleApplied {
            rule_name: "test".to_string(),
            before: x,
            after: x,
            global_before: None,
            global_after: None,
            is_chained: false,
        }],
        &ctx,
        "off",
    );
    assert!(steps.is_empty());
}

#[test]
fn step_wire_substeps_preserve_math_latex_for_rationalization_example() {
    let (engine, output) = eval_output_for("1 / (sqrt(x) - 1) - (sqrt(x) + 1) / (x - 1)");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let all_substep_math: Vec<&str> = steps
        .iter()
        .flat_map(|step| step.substeps.iter())
        .flat_map(|substep| {
            [
                substep.before_latex.as_deref(),
                substep.after_latex.as_deref(),
            ]
        })
        .flatten()
        .collect();

    assert!(
        !all_substep_math.is_empty(),
        "expected didactic substeps for rationalization example"
    );
    assert!(
        all_substep_math
            .iter()
            .any(|latex| latex.contains("\\frac") || latex.contains("\\sqrt")),
        "expected math-like didactic substep content, got: {all_substep_math:?}"
    );
    assert!(
        all_substep_math
            .iter()
            .all(|latex| !latex.starts_with("\\text{\\frac") && !latex.starts_with("\\text{\\sqrt")),
        "didactic wire payload should not wrap math latex in \\\\text{{...}}: {all_substep_math:?}"
    );
}

#[test]
fn step_wire_humanizes_root_notation_in_before_after_text() {
    let (engine, output) = eval_output_for("1 / (sqrt(x) - 1) - (sqrt(x) + 1) / (x - 1)");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let rationalize_step = steps
        .iter()
        .find(|step| step.rule == "Racionalizar el denominador")
        .expect("expected rationalization step");
    assert!(
        rationalize_step.before.contains("sqrt(x)"),
        "expected wire before text to humanize roots, got: {}",
        rationalize_step.before
    );
    assert!(
        !rationalize_step.before.contains("x^(1/2)"),
        "expected wire before text to avoid fractional-exponent root notation, got: {}",
        rationalize_step.before
    );
}

#[test]
fn step_wire_combine_like_terms_explains_coefficient_sum_without_repeating_step_title() {
    let (engine, output) = eval_output_for("x + x");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let step = steps
        .iter()
        .find(|step| step.rule == "Agrupar términos semejantes")
        .expect("expected combine like terms step");

    assert_eq!(
        step.substeps.len(),
        1,
        "expected a single focused didactic substep, got: {:?}",
        step.substeps
            .iter()
            .map(|substep| &substep.title)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        step.substeps[0].title,
        "Sumar los coeficientes que acompañan a x"
    );
    assert!(
        step.substeps[0]
            .before_latex
            .as_deref()
            .is_some_and(|latex| latex == "1 + 1"),
        "expected the hidden substep to show only the coefficient sum, got: {:?}",
        step.substeps[0].before_latex
    );
    assert!(
        step.substeps[0]
            .after_latex
            .as_deref()
            .is_some_and(|latex| latex == "2"),
        "expected the hidden substep to end in the summed coefficient, got: {:?}",
        step.substeps[0].after_latex
    );
}

#[test]
fn step_wire_phase_shift_substeps_are_structured_without_narrative_lines() {
    let (engine, output) = eval_output_for("3*sin(x) + 4*cos(x) - 5*sin(x + arctan(4/3))");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let step = steps
        .iter()
        .find(|step| step.rule == "Aplicar identidad de desfase")
        .expect("expected phase shift step");

    assert_eq!(step.substeps.len(), 1);
    assert!(step.substeps.iter().all(|substep| substep.lines.is_empty()));
    assert!(
        step.substeps
            .iter()
            .all(|substep| substep.before_latex.is_some() && substep.after_latex.is_some()),
        "phase-shift substeps should always expose before/after expressions, got: {:?}",
        step.substeps
            .iter()
            .map(|substep| (
                substep.title.as_str(),
                substep.before_latex.as_deref(),
                substep.after_latex.as_deref()
            ))
            .collect::<Vec<_>>()
    );
    assert_eq!(step.substeps[0].title, "Cancelar términos iguales");
    assert!(
        step.substeps[0]
            .before_latex
            .as_deref()
            .is_some_and(
                |latex| latex.contains("5\\cdot \\sin") && latex.contains("- 5\\cdot \\sin")
            ),
        "phase-shift substep should show the concrete cancellation, got: {:?}",
        step.substeps[0].before_latex
    );
    assert_eq!(step.substeps[0].after_latex.as_deref(), Some("0"));
}

#[test]
fn step_wire_path_latex_renders_human_subtractive_products_for_factor_example() {
    let (engine, output) = eval_output_for("factor(a^3*(b - c) + b^3*(c - a) + c^3*(a - b))");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let first_distribute = steps
        .iter()
        .find(|step| step.rule == "Expandir la expresión")
        .expect("expected distributive step");
    let factor_step = steps
        .iter()
        .find(|step| step.rule == "Factor Polynomial")
        .expect("expected factor polynomial step");

    let first_after_latex = first_distribute.after_latex.as_str();
    assert!(
        first_after_latex.contains("{a}^{3}\\cdot b - {a}^{3}\\cdot c"),
        "expected human subtraction in highlighted latex, got: {first_after_latex}"
    );
    assert!(
        !first_after_latex.contains("\\cdot -c"),
        "highlighted latex should not keep the negative factor inside the product: {first_after_latex}"
    );

    let factor_before_latex = factor_step.before_latex.as_str();
    assert!(
        !factor_before_latex.contains("- (a\\cdot {b}^{3})")
            && !factor_before_latex.contains("- ({c}^{3}\\cdot b)"),
        "simple subtracted products should not be wrapped in parentheses: {factor_before_latex}"
    );
}

#[test]
fn step_wire_sophie_germain_after_text_and_latex_stay_human() {
    let (engine, output) = eval_output_for("factor(x^4 + 4*y^4)");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let step = steps
        .iter()
        .find(|step| step.rule == "Factor Polynomial")
        .expect("expected factor step");

    assert!(
        !step.after.contains("  "),
        "expected no repeated spaces in human after text, got: {}",
        step.after
    );
    assert!(
        step.after == "(x^2 + 2 · y^2 + 2 · x · y) · (x^2 + 2 · y^2 - 2 · x · y)",
        "expected tightened human spacing, got: {}",
        step.after
    );
    assert!(
        !step.after_latex.contains("- (2\\cdot x\\cdot y)"),
        "resulting latex should not parenthesize the final simple product: {}",
        step.after_latex
    );
}

#[test]
fn step_wire_small_polynomial_product_emits_hidden_didactic_expansion_and_cancellation() {
    let (engine, output) = eval_output_for("(x - 1)*(x^5 + x^4 + x^3 + x^2 + x + 1)");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let step = steps
        .iter()
        .find(|step| step.rule == "Expandir y reagrupar un producto polinómico")
        .expect("expected polynomial product normalize step");

    let titles: Vec<&str> = step
        .substeps
        .iter()
        .map(|substep| substep.title.as_str())
        .collect();
    assert!(
        titles.contains(&"Distribuir cada término del producto")
            && titles.contains(&"Agrupar los términos del mismo grado")
            && titles.contains(&"Los términos intermedios se cancelan por parejas"),
        "expected full didactic substep narrative, got {titles:?}"
    );

    let expansion_substep = step
        .substeps
        .iter()
        .find(|substep| substep.title == "Distribuir cada término del producto")
        .expect("missing expansion substep");
    let joined_sides = format!(
        "{} | {}",
        expansion_substep
            .before_latex
            .as_deref()
            .unwrap_or_default(),
        expansion_substep.after_latex.as_deref().unwrap_or_default()
    );
    assert!(
        joined_sides.contains("{x}^{6} + {x}^{5} + {x}^{4} + {x}^{3} + {x}^{2} + x - {x}^{5} - {x}^{4} - {x}^{3} - {x}^{2} - x - 1")
            || joined_sides
                .contains("{x}^{6} + {x}^{5} + {x}^{4} + {x}^{3} + {x}^{2} + x - 1 - {x}^{5} - {x}^{4} - {x}^{3} - {x}^{2} - x")
            || joined_sides
                .contains("x^6 + x^5 + x^4 + x^3 + x^2 + x - x^5 - x^4 - x^3 - x^2 - x - 1")
            || joined_sides
                .contains("x^6 + x^5 + x^4 + x^3 + x^2 + x - 1 - x^5 - x^4 - x^3 - x^2 - x"),
        "expected a visible full expansion in the hidden web substep, got: {joined_sides}"
    );
}

#[test]
fn step_wire_uses_human_visible_rule_titles() {
    let (engine, output) = eval_output_for("1 / (sqrt(x) - 1) - (sqrt(x) + 1) / (x - 1)");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let rule_titles: Vec<&str> = steps.iter().map(|step| step.rule.as_str()).collect();
    assert!(
        rule_titles.contains(&"Racionalizar el denominador"),
        "expected humanized visible rule title, got {:?}",
        rule_titles
    );
    assert!(
        rule_titles.contains(&"Restar dos expresiones iguales")
            || rule_titles.contains(&"Collapse Common-Scale Equivalent Difference"),
        "expected humanized visible rule title, got {:?}",
        rule_titles
    );
}

#[test]
fn step_wire_perfect_square_root_uses_human_visible_rule_title() {
    let (engine, output) = eval_output_for("sqrt(x^2 + 2*x + 1)");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let rule_titles: Vec<&str> = steps.iter().map(|step| step.rule.as_str()).collect();
    assert!(
        rule_titles.contains(&"Reconocer un cuadrado perfecto bajo la raíz"),
        "expected humanized visible rule title, got {:?}",
        rule_titles
    );
}

#[test]
fn step_wire_common_factor_cancel_stays_direct_when_the_only_specific_substep_would_duplicate_the_parent_step(
) {
    let (engine, output) = eval_output_for("(2*x)/(4*x)");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let step = steps
        .iter()
        .find(|step| step.rule == "Cancelar un factor común")
        .expect("expected common factor cancel step");

    assert!(
        step.substeps.is_empty(),
        "when the only possible substep would duplicate the parent step, it should stay direct, got {:?}",
        step.substeps
            .iter()
            .map(|substep| substep.title.as_str())
            .collect::<Vec<_>>()
    );
}

#[test]
fn step_wire_log_cancellation_uses_concrete_substep_expressions() {
    let (engine, output) = eval_output_for("ln(x^3) + ln(y^2) - ln(x^3 * y^2)");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let step = steps
        .iter()
        .find(|step| {
            step.rule == "Expandir logaritmos y cancelar términos iguales" && step.after == "0"
        })
        .expect("expected final log cancellation step");
    let titles = step
        .substeps
        .iter()
        .map(|substep| substep.title.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        titles,
        vec![
            "Expandir el logaritmo del producto o del cociente",
            "Sacar exponentes fuera del logaritmo cuando sea necesario",
            "Cancelar términos iguales",
        ],
        "expected the new explicit three-phase log cancellation story, got {:?}",
        titles
    );
    assert_eq!(
        step.substeps[0].before_latex.as_deref(),
        Some("\\ln({x}^{3}\\cdot {y}^{2})")
    );
    assert_eq!(
        step.substeps[0].after_latex.as_deref(),
        Some("\\ln({x}^{3}) + \\ln({y}^{2})")
    );
    assert_eq!(
        step.substeps[1].before_latex.as_deref(),
        Some("\\ln({x}^{3}) + \\ln({y}^{2}) - \\ln({x}^{3}) - \\ln({y}^{2})")
    );
    let extracted_latex = step.substeps[1]
        .after_latex
        .as_deref()
        .expect("concrete extracted latex");
    assert!(
        extracted_latex.contains("3\\cdot \\ln(x)")
            && extracted_latex.contains("2\\cdot \\ln(|y|)")
            && extracted_latex.matches("3\\cdot \\ln(x)").count() == 2
            && extracted_latex.matches("2\\cdot \\ln(|y|)").count() == 2,
        "expected concrete extracted terms regardless of canonical order, got {extracted_latex}"
    );
    assert_eq!(
        step.substeps[2].before_latex.as_deref(),
        Some(extracted_latex)
    );
    assert_eq!(step.substeps[2].after_latex.as_deref(), Some("0"));
}

#[test]
fn step_wire_cos_diff_sin_diff_quotient_uses_concrete_substep_expressions() {
    let (engine, output) = eval_output_for("(cos(a)-cos(b))/(sin(b)-sin(a))");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    assert_eq!(
        steps.len(),
        3,
        "expected the three-step quotient contraction"
    );
    assert!(
        steps[0].substeps.is_empty() && steps[1].substeps.is_empty(),
        "the first two steps should stay direct because the parent step already shows the concrete rewrite"
    );

    assert_eq!(
        steps[2].substeps[0].before_latex.as_deref(),
        Some(
            "\\frac{2\\cdot \\sin(\\frac{a + b}{2})\\cdot \\sin(\\frac{b - a}{2})}{2\\cdot \\cos(\\frac{a + b}{2})\\cdot \\sin(\\frac{b - a}{2})}"
        )
    );
    assert_eq!(
        steps[2].substeps[0].after_latex.as_deref(),
        Some("\\frac{\\sin(\\frac{a + b}{2})}{\\cos(\\frac{a + b}{2})}")
    );
    assert_eq!(
        steps[2].substeps[1].before_latex.as_deref(),
        Some("\\frac{\\sin(\\frac{a + b}{2})}{\\cos(\\frac{a + b}{2})}")
    );
    assert_eq!(
        steps[2].substeps[1].after_latex.as_deref(),
        Some("\\tan(\\frac{a + b}{2})")
    );
}

#[test]
fn step_wire_log_exponent_rules_stay_direct_when_substeps_only_repeat_the_rule() {
    let (engine, output) = eval_output_for("ln(x^3) + ln(y^2) - ln(x^3 * y^2)");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let redundant_titles: Vec<_> = steps
        .iter()
        .filter(|step| step.rule == "Sacar un exponente fuera del logaritmo")
        .map(|step| {
            (
                step.before.clone(),
                step.substeps
                    .iter()
                    .map(|substep| substep.title.clone())
                    .collect::<Vec<_>>(),
            )
        })
        .collect();

    assert!(
        redundant_titles.iter().all(|(_, titles)| titles.is_empty()),
        "log exponent steps should stay direct without formula-template substeps, got: {:?}",
        redundant_titles
    );
}

#[test]
fn step_wire_difference_of_squares_cancel_recaps_factorization() {
    let (engine, output) = eval_output_for("(x^2 - 1)/(x - 1)");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let step = steps
        .iter()
        .find(|step| step.rule == "Factorizar una diferencia de cuadrados y cancelar")
        .expect("expected difference-of-squares cancel step");

    assert!(
        step.substeps.len() >= 2,
        "expected a two-phase didactic narrative, got: {:?}",
        step.substeps
            .iter()
            .map(|substep| &substep.title)
            .collect::<Vec<_>>()
    );
    assert!(
        step.substeps
            .iter()
            .any(|substep| substep.title.contains("diferencia de cuadrados")),
        "expected a substep that recaps the factorization, got: {:?}",
        step.substeps
            .iter()
            .map(|substep| &substep.title)
            .collect::<Vec<_>>()
    );
    assert!(
        step.substeps
            .iter()
            .any(|substep| substep.title.contains("Ahora se cancela el factor")),
        "expected one substep to show the exact cancellation after factoring, got: {:?}",
        step.substeps
            .iter()
            .map(|substep| &substep.title)
            .collect::<Vec<_>>()
    );
}

#[test]
fn step_wire_inverse_tan_relation_isolates_pair_before_applying_identity() {
    let (engine, output) = eval_output_for("atan(3) + (atan(1/3) - pi/2)");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let step = steps
        .iter()
        .find(|step| step.rule == "Aplicar identidad de arctangentes")
        .expect("expected inverse tan relation step");

    assert!(
        step.substeps.len() == 1,
        "expected a single preparatory substep, got: {:?}",
        step.substeps
            .iter()
            .map(|substep| &substep.title)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        step.substeps[0].title,
        "Juntar la pareja que encaja con la identidad"
    );
    assert_eq!(
        step.substeps[0].before_latex.as_deref(),
        Some("\\arctan(\\frac{1}{3}) + \\arctan(3) - \\frac{1}{2}\\cdot \\pi")
    );
    assert_eq!(
        step.substeps[0].after_latex.as_deref(),
        Some("\\arctan(\\frac{1}{3}) + \\arctan(3)")
    );
}

#[test]
fn step_wire_cancel_reciprocal_exponents_uses_concrete_square_root_block() {
    let (engine, output) = eval_output_for("(sqrt(x^3) - 1)/(sqrt(x) - 1)");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let step = steps
        .iter()
        .find(|step| step.rule == "Deshacer raíz y potencia")
        .expect("expected reciprocal exponent cancellation step");

    assert!(
        !step.substeps.is_empty(),
        "expected didactic substeps for reciprocal exponent cancellation"
    );
    assert_eq!(
        step.substeps[0].title,
        "Reemplazar ese bloque en la expresión"
    );
    assert_eq!(
        step.substeps[0].before_latex.as_deref(),
        Some("\\sqrt{x} + {\\sqrt{x}}^{2} + 1")
    );
    assert_eq!(
        step.substeps[0].after_latex.as_deref(),
        Some("\\sqrt{x} + x + 1")
    );
}

#[test]
fn step_wire_perfect_square_root_shows_square_then_absolute_value() {
    let (engine, output) = eval_output_for("sqrt(x^2 + 2*x + 1)");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let step = steps
        .iter()
        .find(|step| step.rule == "Reconocer un cuadrado perfecto bajo la raíz")
        .expect("expected perfect-square sqrt step");

    assert!(
        step.substeps.len() >= 2,
        "expected a two-phase didactic narrative, got: {:?}",
        step.substeps
            .iter()
            .map(|substep| &substep.title)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        step.substeps[0].title,
        "Reescribir el radicando como un cuadrado perfecto"
    );
    assert_eq!(
        step.substeps[1].title,
        "La raíz de un cuadrado da un valor absoluto"
    );
    assert!(
        step.substeps[0]
            .after_latex
            .as_deref()
            .is_some_and(|latex| latex.contains("^{2}")),
        "expected the first substep to show a square, got: {:?}",
        step.substeps[0].after_latex
    );
    assert!(
        step.substeps[1]
            .after_latex
            .as_deref()
            .is_some_and(|latex| latex.contains("|")),
        "expected the second substep to end in an absolute value, got: {:?}",
        step.substeps[1].after_latex
    );
}

#[test]
fn step_wire_rationalization_explains_conjugate_then_multiply_then_simplify() {
    let (engine, output) = eval_output_for("1 / (sqrt(x) - 1) - (sqrt(x) + 1) / (x - 1)");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let step = steps
        .iter()
        .find(|step| step.rule == "Racionalizar el denominador")
        .expect("expected rationalization step");

    assert!(
        step.substeps.len() >= 3,
        "expected a three-phase didactic narrative, got: {:?}",
        step.substeps
            .iter()
            .map(|substep| &substep.title)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        step.substeps[0].title,
        "Cambiar el signo para formar el conjugado"
    );
    assert_eq!(
        step.substeps[1].title,
        "Multiplicar numerador y denominador por ese conjugado"
    );
    assert_eq!(
        step.substeps[2].title,
        "En el denominador aparece una diferencia de cuadrados"
    );
    assert!(
        step.substeps[2]
            .after_latex
            .as_deref()
            .is_some_and(|latex| latex.contains("x - 1^{2}") && !latex.contains("{(-1)}^{2}")),
        "expected the didactic difference-of-squares substep to humanize the square as 1^2, got: {:?}",
        step.substeps[2].after_latex
    );
}

#[test]
fn step_wire_rationalization_self_cancel_stays_direct_without_tautological_substeps() {
    let (engine, output) = eval_output_for("1 / (sqrt(x) - 1) - (sqrt(x) + 1) / (x - 1)");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let step = steps
        .iter()
        .find(|step| {
            step.rule == "Restar dos expresiones iguales"
                || step.rule == "Collapse Common-Scale Equivalent Difference"
        })
        .expect("expected subtraction self-cancel step");

    assert!(
        step.substeps.is_empty(),
        "self-cancel step should stay direct without tautological substeps, got: {:?}",
        step.substeps
            .iter()
            .map(|substep| &substep.title)
            .collect::<Vec<_>>()
    );
}

#[test]
fn step_wire_cube_quotient_recaps_factor_then_exact_cancellation() {
    let (engine, output) = eval_output_for("((sqrt(x))^3 - 1)/(sqrt(x) - 1)");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let step = steps
        .iter()
        .find(|step| step.rule == "Factorizar cubos y cancelar")
        .expect("expected sum/difference of cubes cancellation step");

    assert!(
        step.substeps.len() >= 3,
        "expected a factor/cancel/replace narrative, got: {:?}",
        step.substeps
            .iter()
            .map(|substep| &substep.title)
            .collect::<Vec<_>>()
    );
    assert!(
        step.substeps
            .iter()
            .any(|substep| substep.title
                == "Factorizar el numerador como suma o diferencia de cubos"),
        "expected the step to factor the numerator first, got: {:?}",
        step.substeps
            .iter()
            .map(|substep| &substep.title)
            .collect::<Vec<_>>()
    );
    assert!(
        step.substeps
            .iter()
            .any(|substep| substep.title.contains("Ahora se cancela el factor")),
        "expected the step to name the common factor being cancelled, got: {:?}",
        step.substeps
            .iter()
            .map(|substep| &substep.title)
            .collect::<Vec<_>>()
    );
    assert!(
        step.substeps
            .iter()
            .any(|substep| substep.title == "Reemplazar ese bloque en la expresión"),
        "expected the step to close with an explicit replacement substep, got: {:?}",
        step.substeps
            .iter()
            .map(|substep| &substep.title)
            .collect::<Vec<_>>()
    );
}

#[test]
fn step_wire_root_denesting_keeps_didactic_substeps() {
    let expr =
        "(sqrt(5 + 2*sqrt(6))) + (1/(u*(u+2))) - ((sqrt(2) + sqrt(3)) + (1/(2*u) - 1/(2*(u+2))))";
    let (engine, output) = eval_output_for(expr);
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let step = steps
        .iter()
        .find(|step| step.rule == "Root Denesting")
        .expect("expected root denesting step");

    assert!(
        step.substeps.len() >= 2,
        "expected didactic root-denesting substeps, got: {:?}",
        step.substeps
            .iter()
            .map(|substep| &substep.title)
            .collect::<Vec<_>>()
    );
    assert!(
        step.substeps
            .iter()
            .any(|substep| substep.title.contains("Identificar la forma")),
        "expected root denesting narrative to identify the sqrt(a ± c·sqrt(d)) pattern, got: {:?}",
        step.substeps
            .iter()
            .map(|substep| &substep.title)
            .collect::<Vec<_>>()
    );
}
