use cas_ast::Context;
use cas_solver::Step;

pub(super) struct RenderedStepJsonExprs {
    pub(super) before: String,
    pub(super) after: String,
}

pub(super) fn render_step_json_exprs(context: &Context, step: &Step) -> RenderedStepJsonExprs {
    let before_expr = step.global_before.unwrap_or(step.before);
    let after_expr = step.global_after.unwrap_or(step.after);

    RenderedStepJsonExprs {
        before: format!(
            "{}",
            cas_formatter::DisplayExpr {
                context,
                id: before_expr
            }
        ),
        after: format!(
            "{}",
            cas_formatter::DisplayExpr {
                context,
                id: after_expr
            }
        ),
    }
}
