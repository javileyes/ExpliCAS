use crate::EvalResult;
use cas_ast::hold::strip_all_holds;

pub(super) fn render_eval_result(ctx: &mut cas_ast::Context, result: &EvalResult) -> String {
    match result {
        EvalResult::Expr(e) => render_expr(ctx, *e),
        EvalResult::Set(v) if !v.is_empty() => render_expr(ctx, v[0]),
        EvalResult::SolutionSet(solution_set) => crate::display_solution_set(ctx, solution_set),
        EvalResult::Bool(b) => b.to_string(),
        _ => "(no result)".to_string(),
    }
}

fn render_expr(ctx: &mut cas_ast::Context, id: cas_ast::ExprId) -> String {
    let clean = strip_all_holds(ctx, id);
    format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: clean
        }
    )
}
