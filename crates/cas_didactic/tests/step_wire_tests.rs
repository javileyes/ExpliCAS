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
