use cas_solver::{Engine, EvalAction, EvalOptions, EvalRequest};

fn eval_output_for(expr: &str) -> (Engine, cas_solver::EvalOutput) {
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
fn eval_json_steps_off_mode_is_empty() {
    let (engine, output) = eval_output_for("x + x");
    let steps =
        cas_didactic::collect_eval_json_steps(&output.steps, &engine.simplifier.context, "off");
    assert!(steps.is_empty());
}

#[test]
fn eval_json_steps_on_mode_matches_deterministically() {
    let (engine, output) = eval_output_for("(x + 2) + (x + 3)");
    let first =
        cas_didactic::collect_eval_json_steps(&output.steps, &engine.simplifier.context, "on");
    let second =
        cas_didactic::collect_eval_json_steps(&output.steps, &engine.simplifier.context, "on");
    assert_eq!(first.len(), second.len());
    if let (Some(a), Some(b)) = (first.first(), second.first()) {
        assert_eq!(a.rule, b.rule);
        assert_eq!(a.before, b.before);
        assert_eq!(a.after, b.after);
    }
}
