use super::sqrt::get_sqrt_inner;
use cas_ast::{Context, Expr, ExprId};
use num_bigint::BigInt;
use num_rational::BigRational;

pub(super) fn analyze_surd(ctx: &Context, expr: ExprId) -> Option<(BigRational, ExprId)> {
    if let Some(inner) = get_sqrt_inner(ctx, expr) {
        return Some((BigRational::from_integer(BigInt::from(1)), inner));
    }

    match ctx.get(expr) {
        Expr::Mul(left, right) => analyze_scaled_surd(ctx, *left, *right),
        _ => None,
    }
}

fn analyze_scaled_surd(
    ctx: &Context,
    left: ExprId,
    right: ExprId,
) -> Option<(BigRational, ExprId)> {
    match (ctx.get(left), ctx.get(right)) {
        (Expr::Number(coeff), _) => get_sqrt_inner(ctx, right).map(|inner| (coeff.clone(), inner)),
        (_, Expr::Number(coeff)) => get_sqrt_inner(ctx, left).map(|inner| (coeff.clone(), inner)),
        _ => None,
    }
}
