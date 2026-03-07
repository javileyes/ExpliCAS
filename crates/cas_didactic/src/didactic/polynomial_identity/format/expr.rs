use cas_ast::Context;

pub(super) fn display_expr(ctx: &Context, expr_id: cas_ast::ExprId) -> String {
    format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: expr_id
        }
    )
}

pub(super) fn latex_expr(ctx: &Context, expr_id: cas_ast::ExprId) -> String {
    cas_formatter::LaTeXExpr {
        context: ctx,
        id: expr_id,
    }
    .to_latex()
}
