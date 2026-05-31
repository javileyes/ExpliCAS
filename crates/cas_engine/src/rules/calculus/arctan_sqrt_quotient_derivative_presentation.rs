use super::arctan_sqrt_radicand_arg;
use super::polynomial_support::{
    polynomial_is_strictly_positive_everywhere, split_polynomial_content_for_calculus_presentation,
};
use super::presentation_utils::unwrap_internal_hold_for_calculus;
use super::scalar_presentation::{
    nonzero_rational_parts, rational_const_for_calculus_presentation,
    signed_numerator_for_calculus_presentation,
};
use crate::symbolic_calculus_call_support::try_extract_diff_call;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::polynomial::Polynomial;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

pub(super) fn arctan_sqrt_affine_partition_quotient_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
    derivative_sign: BigRational,
) -> Option<ExprId> {
    let radicand = arctan_sqrt_radicand_arg(ctx, target)?;
    let Expr::Div(num, den) = ctx.get(radicand).clone() else {
        return None;
    };

    let numerator_poly = Polynomial::from_expr(ctx, num, var_name).ok()?;
    let denominator_poly = Polynomial::from_expr(ctx, den, var_name).ok()?;
    if numerator_poly.degree() > 1 || denominator_poly.degree() > 1 {
        return None;
    }

    let partition_sum = numerator_poly.add(&denominator_poly);
    if partition_sum.degree() != 0 {
        return None;
    }
    let partition_total = partition_sum
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if !partition_total.is_positive() {
        return None;
    }

    let wronskian = numerator_poly
        .derivative()
        .mul(&denominator_poly)
        .sub(&numerator_poly.mul(&denominator_poly.derivative()));
    if wronskian.degree() != 0 {
        return None;
    }
    let wronskian_value = wronskian
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if wronskian_value.is_zero() {
        return Some(ctx.num(0));
    }

    let coefficient =
        derivative_sign * wronskian_value / (BigRational::from_integer(2.into()) * partition_total);
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator = rational_const_for_calculus_presentation(ctx, numerator_coeff);
    let sqrt_num = ctx.call_builtin(BuiltinFn::Sqrt, vec![num]);
    let sqrt_den = ctx.call_builtin(BuiltinFn::Sqrt, vec![den]);
    let core_denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_num, sqrt_den]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };
    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(super) fn arctan_sqrt_polynomial_quotient_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
    derivative_sign: BigRational,
) -> Option<ExprId> {
    let radicand = arctan_sqrt_radicand_arg(ctx, target)?;
    let Expr::Div(num, den) = ctx.get(radicand).clone() else {
        return None;
    };

    let numerator_poly = Polynomial::from_expr(ctx, num, var_name).ok()?;
    let denominator_poly = Polynomial::from_expr(ctx, den, var_name).ok()?;
    if numerator_poly.degree() == 0
        || denominator_poly.degree() == 0
        || numerator_poly.degree() > 2
        || denominator_poly.degree() > 2
    {
        return None;
    }

    let sum_poly = numerator_poly.add(&denominator_poly);
    if sum_poly.degree() == 0 || sum_poly.degree() > 2 {
        return None;
    }

    let wronskian = numerator_poly
        .derivative()
        .mul(&denominator_poly)
        .sub(&numerator_poly.mul(&denominator_poly.derivative()));
    if wronskian.degree() > 1 {
        return None;
    }
    if wronskian.is_zero() {
        return Some(ctx.num(0));
    }

    let sum_expr = sum_poly.to_expr(ctx);
    let (sum_core, sum_content) = split_polynomial_content_for_calculus_presentation(ctx, sum_expr);
    if sum_content.is_zero() {
        return None;
    }

    let wronskian_expr = wronskian.to_expr(ctx);
    let (wronskian_core, wronskian_content) =
        split_polynomial_content_for_calculus_presentation(ctx, wronskian_expr);
    if wronskian_content.is_zero() {
        return None;
    }

    let coefficient =
        derivative_sign * wronskian_content / (BigRational::from_integer(2.into()) * sum_content);
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, wronskian_core);
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let denominator = if denominator_coeff == BigRational::one() {
        cas_math::expr_nary::build_balanced_mul(ctx, &[sum_core, den, sqrt_radicand])
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(
            ctx,
            &[denominator_scale, sum_core, den, sqrt_radicand],
        )
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(super) fn arctan_sqrt_positive_polynomial_quotient_derivative_shortcut(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let radicand = arctan_sqrt_radicand_arg(ctx, target)?;
    let Expr::Div(num, den) = ctx.get(radicand).clone() else {
        return None;
    };

    let numerator_poly = Polynomial::from_expr(ctx, num, var_name).ok()?;
    let denominator_poly = Polynomial::from_expr(ctx, den, var_name).ok()?;
    let sum_poly = numerator_poly.add(&denominator_poly);

    if !polynomial_is_strictly_positive_everywhere(&numerator_poly)
        || !polynomial_is_strictly_positive_everywhere(&denominator_poly)
        || !polynomial_is_strictly_positive_everywhere(&sum_poly)
    {
        return None;
    }

    let compact = arctan_sqrt_polynomial_quotient_derivative_presentation(
        ctx,
        target,
        var_name,
        BigRational::one(),
    )?;
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

pub(crate) fn arctan_sqrt_positive_polynomial_quotient_derivative_for_diff_call(
    ctx: &mut Context,
    source: ExprId,
) -> Option<ExprId> {
    let call = try_extract_diff_call(ctx, source)?;
    let target = unwrap_internal_hold_for_calculus(ctx, call.target);
    arctan_sqrt_positive_polynomial_quotient_derivative_shortcut(ctx, target, &call.var_name)
}
