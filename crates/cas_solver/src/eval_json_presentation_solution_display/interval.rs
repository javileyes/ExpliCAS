use cas_ast::Context;
use cas_formatter::DisplayExpr;

pub(super) fn format_interval_eval_json(
    ctx: &Context,
    min: cas_ast::ExprId,
    max: cas_ast::ExprId,
) -> String {
    format!(
        "[{}, {}]",
        DisplayExpr {
            context: ctx,
            id: min
        },
        DisplayExpr {
            context: ctx,
            id: max
        }
    )
}
