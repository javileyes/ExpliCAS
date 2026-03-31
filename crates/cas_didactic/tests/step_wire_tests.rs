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
fn step_wire_common_factor_cancel_names_the_factor_and_the_result() {
    let (engine, output) = eval_output_for("(2*x)/(4*x)");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let step = steps
        .iter()
        .find(|step| step.rule == "Cancelar un factor común")
        .expect("expected common factor cancel step");

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
            .any(|substep| substep.title.contains("factor comun x")),
        "expected a substep that names the canceled factor, got: {:?}",
        step.substeps
            .iter()
            .map(|substep| &substep.title)
            .collect::<Vec<_>>()
    );
    assert!(
        step.substeps
            .iter()
            .any(|substep| substep.title.contains("Al cancelar x")),
        "expected a substep that explains the remaining fraction, got: {:?}",
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
        step.substeps.len() >= 2,
        "expected a two-phase didactic narrative, got: {:?}",
        step.substeps
            .iter()
            .map(|substep| &substep.title)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        step.substeps[0].title,
        "Juntar la pareja que encaja con la identidad"
    );
    assert_eq!(step.substeps[1].title, "Esa pareja vale pi/2");
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
}

#[test]
fn step_wire_rationalization_self_cancel_explains_why_the_difference_is_zero() {
    let (engine, output) = eval_output_for("1 / (sqrt(x) - 1) - (sqrt(x) + 1) / (x - 1)");
    let steps =
        cas_didactic::collect_step_payloads(&output.steps, &engine.simplifier.context, "on");

    let step = steps
        .iter()
        .find(|step| step.rule == "Restar dos expresiones iguales")
        .expect("expected subtraction self-cancel step");

    assert!(
        step.substeps.len() >= 2,
        "expected a two-phase self-cancel narrative, got: {:?}",
        step.substeps
            .iter()
            .map(|substep| &substep.title)
            .collect::<Vec<_>>()
    );
    assert_eq!(step.substeps[0].title, "Los dos términos ya son el mismo");
    assert_eq!(step.substeps[1].title, "Restar algo consigo mismo da 0");
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
