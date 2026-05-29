use cas_ast::{Context, Expr, ExprId};
use cas_math::multipoly::{multipoly_from_expr, multipoly_to_expr, PolyBudget};
use num_rational::BigRational;
use num_traits::{One, Signed};

pub(super) fn primitive_positive_gap(ctx: &mut Context, gap: ExprId) -> (ExprId, BigRational) {
    let budget = PolyBudget {
        max_terms: 50,
        max_total_degree: 20,
        max_pow_exp: 10,
    };

    let Ok(poly) = multipoly_from_expr(ctx, gap, &budget) else {
        return (gap, BigRational::one());
    };
    let (content, primitive) = poly.primitive_part();
    if !content.is_positive() {
        return (gap, BigRational::one());
    }
    let primitive_expr = multipoly_to_expr(&primitive, ctx);
    if content.is_one() {
        return (primitive_expr, BigRational::one());
    }

    (primitive_expr, content)
}

pub(super) fn reciprocal_positive_rational(value: &BigRational) -> BigRational {
    BigRational::new(value.denom().clone(), value.numer().clone())
}

pub(super) fn squared_expr_for_compact_gap_presentation(ctx: &mut Context, expr: ExprId) -> ExprId {
    let Expr::Pow(base, exp) = ctx.get(expr).clone() else {
        return square_expr(ctx, expr);
    };
    let Some(exp_value) = cas_ast::views::as_rational_const(ctx, exp, 8) else {
        return square_expr(ctx, expr);
    };
    let doubled_exp = exp_value * BigRational::from_integer(2.into());
    let doubled_exp = ctx.add(Expr::Number(doubled_exp));
    ctx.add(Expr::Pow(base, doubled_exp))
}

fn square_expr(ctx: &mut Context, expr: ExprId) -> ExprId {
    let two = ctx.num(2);
    ctx.add(Expr::Pow(expr, two))
}
