use cas_api_models::EngineJsonStep;
use cas_ast::hold::strip_all_holds;

pub(crate) fn render_eval_result(
    context: &mut cas_ast::Context,
    result: &cas_solver::EvalResult,
) -> String {
    match result {
        cas_solver::EvalResult::Expr(expr_id) => render_expr(context, *expr_id),
        cas_solver::EvalResult::Set(values) if !values.is_empty() => {
            render_expr(context, values[0])
        }
        cas_solver::EvalResult::Bool(flag) => flag.to_string(),
        _ => "(no result)".to_string(),
    }
}

pub(crate) fn render_eval_steps(
    context: &mut cas_ast::Context,
    steps: &[cas_solver::Step],
) -> Vec<EngineJsonStep> {
    steps
        .iter()
        .map(|step| {
            let before = step.global_before.map(|id| render_expr(context, id));
            let after = step.global_after.map(|id| render_expr(context, id));
            EngineJsonStep {
                phase: "Simplify".to_string(),
                rule: step.rule_name.clone(),
                before: before.unwrap_or_default(),
                after: after.unwrap_or_default(),
                substeps: vec![],
            }
        })
        .collect()
}

fn render_expr(context: &mut cas_ast::Context, expr_id: cas_ast::ExprId) -> String {
    let clean = strip_all_holds(context, expr_id);
    format!("{}", cas_formatter::DisplayExpr { context, id: clean })
}
