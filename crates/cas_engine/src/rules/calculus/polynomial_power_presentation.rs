//! Polynomial-power helpers for post-calculus presentation routes.
//!
//! These helpers are shared by denominator/root presentation families that need
//! the same bounded policy for powered polynomial bases and expanded affine
//! squares. They are presentation helpers, not a general polynomial API.

use super::polynomial_support::polynomial_radicand_for_calculus_presentation;
use super::scalar_presentation::exact_positive_rational_sqrt_for_calculus_presentation;
use cas_ast::{Context, Expr, ExprId};
use cas_math::polynomial::Polynomial;
use num_rational::BigRational;
use num_traits::{One, ToPrimitive, Zero};

pub(super) fn positive_integer_polynomial_power_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(ExprId, usize, Polynomial)> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let Expr::Pow(base, exp) = ctx.get(expr).clone() else {
        return None;
    };
    let exponent = cas_ast::views::as_rational_const(ctx, exp, 8)?;
    if !exponent.is_integer() || exponent <= BigRational::one() {
        return None;
    }
    let exponent = exponent.to_integer().to_usize()?;
    if let Some((compact_base, compact_exponent, compact_base_poly)) =
        expanded_affine_square_for_calculus_presentation(ctx, base, var_name)
    {
        return Some((compact_base, exponent * compact_exponent, compact_base_poly));
    }
    let base_poly = polynomial_radicand_for_calculus_presentation(ctx, base, var_name)?;
    Some((base, exponent, base_poly))
}

pub(super) fn expanded_affine_square_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(ExprId, usize, Polynomial)> {
    let poly = polynomial_radicand_for_calculus_presentation(ctx, expr, var_name)?;
    if poly.degree() != 2 {
        return None;
    }

    let leading = poly
        .coeffs
        .get(2)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let slope = exact_positive_rational_sqrt_for_calculus_presentation(&leading)?;

    let linear = poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let constant = poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let shift = linear / (BigRational::from_integer(2.into()) * slope.clone());
    if constant != shift.clone() * shift.clone() {
        return None;
    }

    let base_poly = Polynomial::new(vec![shift, slope], var_name.to_string());
    let base = base_poly.to_expr(ctx);
    Some((base, 2, base_poly))
}

pub(super) fn polynomial_power_for_calculus_presentation(
    poly: &Polynomial,
    exponent: usize,
) -> Polynomial {
    let mut result = Polynomial::one(poly.var.clone());
    for _ in 0..exponent {
        result = result.mul(poly);
    }
    result
}
