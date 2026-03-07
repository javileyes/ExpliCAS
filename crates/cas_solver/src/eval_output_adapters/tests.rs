use super::{eval_output_view, steps_from_eval_output};

fn sample_output() -> crate::EvalOutput {
    let mut ctx = cas_ast::Context::new();
    let one = ctx.num(1);
    let step = crate::Step::new_compact("demo", "DemoRule", one, one);

    crate::EvalOutput {
        stored_id: Some(7),
        parsed: one,
        resolved: one,
        result: crate::EvalResult::Expr(one),
        domain_warnings: Vec::new(),
        steps: cas_solver_core::display_steps::DisplaySteps(vec![step]),
        solve_steps: Vec::new(),
        solver_assumptions: Vec::new(),
        output_scopes: Vec::new(),
        required_conditions: Vec::new(),
        blocked_hints: Vec::new(),
        diagnostics: crate::Diagnostics::default(),
    }
}

#[test]
fn steps_adapter_returns_solver_owned_wrapper() {
    let output = sample_output();
    let steps = steps_from_eval_output(&output);
    assert_eq!(steps.len(), 1);
    assert_eq!(steps.as_slice()[0].rule_name, "DemoRule");
}

#[test]
fn eval_output_view_converts_steps_into_solver_wrapper() {
    let output = sample_output();
    let view = eval_output_view(&output);
    assert_eq!(view.steps.len(), 1);
    assert_eq!(view.steps.as_slice()[0].description, "demo");
}
