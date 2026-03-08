use cas_ast::{Context, ExprId};
use cas_formatter::DisplayExpr;

pub(super) fn display_expr(ctx: &Context, expr: ExprId) -> String {
    format!(
        "{}",
        DisplayExpr {
            context: ctx,
            id: expr
        }
    )
}
