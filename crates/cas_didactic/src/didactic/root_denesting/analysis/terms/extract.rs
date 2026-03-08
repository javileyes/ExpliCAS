use cas_ast::{Context, Expr, ExprId};

pub(super) fn extract_denesting_terms(
    ctx: &Context,
    inner_expr: ExprId,
) -> Option<(ExprId, ExprId, bool)> {
    match ctx.get(inner_expr) {
        Expr::Add(left, right) => Some((*left, *right, true)),
        Expr::Sub(left, right) => Some((*left, *right, false)),
        _ => None,
    }
}
