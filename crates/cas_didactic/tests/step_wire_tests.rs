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
fn step_wire_path_latex_renders_human_subtractive_products_for_factor_example() {
    let (engine, output) = eval_output_for("factor(a^3*(b - c) + b^3*(c - a) + c^3*(a - b))");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let first_distribute = steps
        .iter()
        .find(|step| step.rule == "Distributive Property")
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
        rule_titles.contains(&"Restar dos expresiones iguales"),
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
fn step_wire_common_factor_cancel_stays_direct_when_the_only_substep_matches_the_parent_step() {
    let (engine, output) = eval_output_for("(2*x)/(4*x)");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let step = steps
        .iter()
        .find(|step| step.rule == "Cancelar un factor común")
        .expect("expected common factor cancel step");

    assert!(
        step.substeps.is_empty(),
        "single substep identical to the parent step should be pruned, got: {:?}",
        step.substeps
            .iter()
            .map(|substep| &substep.title)
            .collect::<Vec<_>>()
    );
}

#[test]
fn step_wire_log_cancellation_stays_direct_without_single_redundant_substep() {
    let (engine, output) = eval_output_for("ln(x^3) + ln(y^2) - ln(x^3 * y^2)");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let step = steps
        .iter()
        .find(|step| step.rule == "Agrupar términos semejantes" && step.after == "0")
        .expect("expected final log cancellation step");

    assert!(
        step.substeps.is_empty(),
        "single substep identical to the log-cancellation parent step should be pruned, got: {:?}",
        step.substeps
            .iter()
            .map(|substep| &substep.title)
            .collect::<Vec<_>>()
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
        step.substeps.iter().any(|substep| {
            substep
                .before_latex
                .as_deref()
                .is_some_and(|latex| latex.contains("{x}^{2}") || latex.contains("x^2"))
        }),
        "expected one substep to show the original numerator as a difference of squares"
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
        step.substeps.len() >= 3,
        "expected a three-phase didactic narrative, got: {:?}",
        step.substeps
            .iter()
            .map(|substep| &substep.title)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        step.substeps[0].title,
        "Usar arctan(u) + arctan(1/u) = pi/2"
    );
    assert_eq!(
        step.substeps[1].title,
        "Juntar la pareja que encaja con la identidad"
    );
    assert_eq!(step.substeps[2].title, "Esa pareja vale pi/2");
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
        .find(|step| step.rule == "Restar dos expresiones iguales")
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

    let factor_step = steps
        .iter()
        .find(|step| step.rule == "Factorizar suma o diferencia de cubos")
        .expect("expected sum/difference of cubes factor step");
    let cancel_step = steps
        .iter()
        .find(|step| step.rule == "Cancelar factor tras factorizar cubos")
        .expect("expected sum/difference of cubes cancel step");

    assert!(
        factor_step.substeps.len() >= 2,
        "expected a two-phase factor narrative, got: {:?}",
        factor_step
            .substeps
            .iter()
            .map(|substep| &substep.title)
            .collect::<Vec<_>>()
    );
    assert!(
        factor_step
            .substeps
            .iter()
            .any(|substep| substep.title.contains("a^3 - b^3")),
        "expected the factor step to mention the cube identity, got: {:?}",
        factor_step
            .substeps
            .iter()
            .map(|substep| &substep.title)
            .collect::<Vec<_>>()
    );

    assert!(
        cancel_step.substeps.len() >= 2,
        "expected an exact-cancellation narrative, got: {:?}",
        cancel_step
            .substeps
            .iter()
            .map(|substep| &substep.title)
            .collect::<Vec<_>>()
    );
    assert!(
        cancel_step
            .substeps
            .iter()
            .any(|substep| substep.title.contains("sqrt(x) - 1")),
        "expected the cancel step to name the common factor, got: {:?}",
        cancel_step
            .substeps
            .iter()
            .map(|substep| &substep.title)
            .collect::<Vec<_>>()
    );
}
