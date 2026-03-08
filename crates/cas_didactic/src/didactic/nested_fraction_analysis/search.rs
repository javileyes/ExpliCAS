use cas_ast::{Context, Expr, ExprId};

/// Find and return the first `Div` node within an expression.
pub(crate) fn find_div_in_expr(ctx: &Context, id: ExprId) -> Option<ExprId> {
    match ctx.get(id) {
        Expr::Div(_, _) => Some(id),
        Expr::Add(l, r) | Expr::Sub(l, r) => {
            find_div_in_expr(ctx, *l).or_else(|| find_div_in_expr(ctx, *r))
        }
        Expr::Mul(l, r) => find_div_in_expr(ctx, *l).or_else(|| find_div_in_expr(ctx, *r)),
        Expr::Neg(inner) | Expr::Hold(inner) => find_div_in_expr(ctx, *inner),
        Expr::Pow(b, e) => find_div_in_expr(ctx, *b).or_else(|| find_div_in_expr(ctx, *e)),
        Expr::Function(_, args) => args.iter().find_map(|a| find_div_in_expr(ctx, *a)),
        Expr::Matrix { data, .. } => data.iter().find_map(|e| find_div_in_expr(ctx, *e)),
        Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => None,
    }
}
