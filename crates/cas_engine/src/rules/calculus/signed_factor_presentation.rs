//! Small signed factor utilities shared by calculus presentation routes.

use cas_ast::{Context, Expr, ExprId};

pub(super) fn signed_mul_leaves_for_calculus_presentation(
    ctx: &mut Context,
    root: ExprId,
) -> Vec<ExprId> {
    let mut negative = false;
    let root = match ctx.get(root).clone() {
        Expr::Neg(inner) => {
            negative = !negative;
            inner
        }
        _ => root,
    };
    let mut factors = Vec::new();
    for factor in cas_math::expr_nary::mul_leaves(ctx, root) {
        match ctx.get(factor).clone() {
            Expr::Neg(inner) => {
                negative = !negative;
                factors.extend(cas_math::expr_nary::mul_leaves(ctx, inner));
            }
            _ => factors.push(factor),
        }
    }
    if negative {
        factors.insert(0, ctx.num(-1));
    }
    factors
}
