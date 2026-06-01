use cas_ast::{Context, Expr, ExprId};
use cas_math::multipoly::{multipoly_from_expr, multipoly_to_expr, PolyBudget};
use cas_math::polynomial::Polynomial;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

use super::scalar_presentation::exact_positive_rational_sqrt_for_calculus_presentation;

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

pub(super) fn compact_squared_affine_gap_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> ExprId {
    let Expr::Sub(left, right) = ctx.get(expr).clone() else {
        return expr;
    };
    let Expr::Pow(base, exp) = ctx.get(right).clone() else {
        return expr;
    };
    if cas_ast::views::as_rational_const(ctx, exp, 8) != Some(BigRational::from_integer(2.into())) {
        return expr;
    }
    if let Expr::Pow(_, inner_exp) = ctx.get(base).clone() {
        if cas_ast::views::as_rational_const(ctx, inner_exp, 8)
            == Some(BigRational::from_integer(2.into()))
        {
            return expr;
        }
    }

    let Some(affine) = affine_square_root_for_calculus_presentation(ctx, base, var_name) else {
        return expr;
    };
    let four = ctx.num(4);
    let compact_power = ctx.add(Expr::Pow(affine, four));
    ctx.add(Expr::Sub(left, compact_power))
}

fn affine_square_root_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let poly = Polynomial::from_expr(ctx, expr, var_name).ok()?;
    if poly.degree() != 2 {
        return None;
    }

    let a = poly
        .coeffs
        .get(2)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let b = poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let c = poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);

    let linear_coeff = exact_positive_rational_sqrt_for_calculus_presentation(&a)?;
    let constant_abs = if c.is_zero() {
        BigRational::zero()
    } else {
        exact_positive_rational_sqrt_for_calculus_presentation(&c)?
    };
    let expected_cross =
        BigRational::from_integer(2.into()) * linear_coeff.clone() * constant_abs.clone();
    let constant = if b == expected_cross {
        constant_abs
    } else if b == -expected_cross {
        -constant_abs
    } else {
        return None;
    };

    let affine = Polynomial::new(vec![constant, linear_coeff], var_name.to_string());
    Some(affine.to_expr(ctx))
}

fn square_expr(ctx: &mut Context, expr: ExprId) -> ExprId {
    let two = ctx.num(2);
    ctx.add(Expr::Pow(expr, two))
}
