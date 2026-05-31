use super::gap_presentation::squared_expr_for_compact_gap_presentation;
use super::polynomial_support::{
    polynomial_radicand_for_calculus_presentation,
    rational_polynomial_content_for_calculus_presentation,
};
use super::scalar_presentation::{
    rational_const_for_calculus_presentation, scale_expr_for_calculus_presentation,
    signed_numerator_for_calculus_presentation,
};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{Signed, Zero};

pub(super) fn arctan_rational_affine_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    if !matches!(
        ctx.builtin_of(*fn_id),
        Some(BuiltinFn::Arctan | BuiltinFn::Atan)
    ) || args.len() != 1
    {
        return None;
    }

    let arg_poly = polynomial_radicand_for_calculus_presentation(ctx, args[0], var_name)?;
    if arg_poly.degree() != 1 {
        return None;
    }
    let arg_content = rational_polynomial_content_for_calculus_presentation(&arg_poly);
    if arg_content.is_zero() || arg_content.is_integer() {
        return None;
    }

    let primitive_arg_poly = arg_poly.div_scalar(&arg_content);
    let derivative_poly = primitive_arg_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    if derivative_poly.degree() != 0 {
        return None;
    }

    let derivative_coeff = derivative_poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let content_num = BigRational::from_integer(arg_content.numer().clone());
    let content_den = BigRational::from_integer(arg_content.denom().clone());
    let numerator_coeff = derivative_coeff * content_num.clone() * content_den.clone();
    let one = ctx.num(1);
    let numerator = signed_numerator_for_calculus_presentation(ctx, numerator_coeff, one);

    let square_arg_poly = if primitive_arg_poly.leading_coeff().is_negative() {
        primitive_arg_poly.neg()
    } else {
        primitive_arg_poly.clone()
    };
    let square_arg = square_arg_poly.to_expr(ctx);
    let primitive_arg_sq = squared_expr_for_compact_gap_presentation(ctx, square_arg);
    let content_num_sq =
        BigRational::from_integer(arg_content.numer().clone() * arg_content.numer().clone());
    let content_den_sq =
        BigRational::from_integer(arg_content.denom().clone() * arg_content.denom().clone());
    let scaled_arg_sq = scale_expr_for_calculus_presentation(ctx, content_num_sq, primitive_arg_sq);
    let denominator_constant = rational_const_for_calculus_presentation(ctx, content_den_sq);
    let denominator = ctx.add(Expr::Add(scaled_arg_sq, denominator_constant));

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(super) fn atanh_rational_affine_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    if args.len() != 1 || !ctx.is_builtin(*fn_id, BuiltinFn::Atanh) {
        return None;
    }

    let arg_poly = polynomial_radicand_for_calculus_presentation(ctx, args[0], var_name)?;
    if arg_poly.degree() != 1 {
        return None;
    }
    let arg_content = rational_polynomial_content_for_calculus_presentation(&arg_poly);
    if arg_content.is_zero() || arg_content.is_integer() {
        return None;
    }

    let primitive_arg_poly = arg_poly.div_scalar(&arg_content);
    let derivative_poly = primitive_arg_poly.derivative();
    if derivative_poly.is_zero() || derivative_poly.degree() != 0 {
        return None;
    }

    let derivative_coeff = derivative_poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let content_num = BigRational::from_integer(arg_content.numer().clone());
    let content_den = BigRational::from_integer(arg_content.denom().clone());
    let numerator_coeff = derivative_coeff * content_num.clone() * content_den.clone();
    let one = ctx.num(1);
    let numerator = signed_numerator_for_calculus_presentation(ctx, numerator_coeff, one);

    let square_arg_poly = if primitive_arg_poly.leading_coeff().is_negative() {
        primitive_arg_poly.neg()
    } else {
        primitive_arg_poly.clone()
    };
    let square_arg = square_arg_poly.to_expr(ctx);
    let primitive_arg_sq = squared_expr_for_compact_gap_presentation(ctx, square_arg);
    let content_num_sq =
        BigRational::from_integer(arg_content.numer().clone() * arg_content.numer().clone());
    let scaled_arg_sq = scale_expr_for_calculus_presentation(ctx, content_num_sq, primitive_arg_sq);
    let content_den_sq =
        BigRational::from_integer(arg_content.denom().clone() * arg_content.denom().clone());
    let denominator_constant = rational_const_for_calculus_presentation(ctx, content_den_sq);
    let denominator = ctx.add(Expr::Sub(denominator_constant, scaled_arg_sq));

    Some(ctx.add(Expr::Div(numerator, denominator)))
}
