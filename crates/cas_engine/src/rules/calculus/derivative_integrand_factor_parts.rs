use cas_ast::{Context, Expr, ExprId};

pub(super) fn split_mul_div_factor_parts(
    ctx: &Context,
    expr: ExprId,
) -> Option<(Vec<ExprId>, Vec<ExprId>)> {
    let factors = cas_math::expr_nary::mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    let mut numerator_factors = Vec::new();
    let mut denominator_factors = Vec::new();
    for factor in factors {
        match ctx.get(factor) {
            Expr::Div(num, den) => {
                numerator_factors.extend(cas_math::expr_nary::mul_leaves(ctx, *num));
                denominator_factors.extend(cas_math::expr_nary::mul_leaves(ctx, *den));
            }
            _ => numerator_factors.push(factor),
        }
    }

    (!denominator_factors.is_empty()).then_some((numerator_factors, denominator_factors))
}
