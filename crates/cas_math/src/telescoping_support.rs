use crate::expr_nary::AddView;
use cas_ast::{Context, Expr, ExprId};

/// Find a suitable multiplier to clear denominators in telescoping-style sums.
pub fn find_denominator_for_clearing(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let view = AddView::from_expr(ctx, expr);

    for &(term, _sign) in &view.terms {
        if let Some(denom) = extract_denominator(ctx, term) {
            return Some(denom);
        }
    }

    None
}

// nary-lint: allow-binary (structural, not n-ary sum traversal)
fn extract_denominator(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Div(_, denom) => Some(*denom),
        Expr::Neg(inner) => extract_denominator(ctx, *inner),
        Expr::Mul(l, r) => extract_denominator(ctx, *l).or_else(|| extract_denominator(ctx, *r)),
        _ => None,
    }
}
