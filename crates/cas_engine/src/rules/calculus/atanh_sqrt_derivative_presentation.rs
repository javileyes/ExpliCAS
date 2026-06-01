use super::gap_presentation::reciprocal_positive_rational;
use super::polynomial_support::{
    polynomial_radicand_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation,
};
use super::result_presentation::cancel_denominator_content_with_numerator_for_calculus_presentation;
use super::scalar_presentation::{
    nonzero_rational_parts, rational_const_for_calculus_presentation,
    signed_numerator_for_calculus_presentation, subtract_from_rational_for_calculus_presentation,
};
use super::scaled_sqrt_args::scaled_sqrt_polynomial_arg_for_calculus_presentation;
use super::sqrt_polynomial_scale_presentation::constant_scaled_sqrt_polynomial_derivative_presentation;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::polynomial::Polynomial;
use cas_math::root_forms::extract_square_root_base;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

fn preserve_atanh_sqrt_open_interval_gap_orientation(poly: &Polynomial) -> bool {
    poly.degree() == 1
        && poly
            .coeffs
            .first()
            .is_some_and(|constant| constant.is_positive())
        && poly.leading_coeff().is_negative()
}

pub(super) fn atanh_sqrt_polynomial_derivative_presentation(
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

    let radicand = extract_square_root_base(ctx, args[0])?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    if radicand_poly.degree() != 1 {
        return None;
    }

    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }

    let mut numerator_sign = BigRational::one();
    let one = Polynomial::one(radicand_poly.var.clone());
    let mut gap_poly = one.sub(&radicand_poly);
    if gap_poly.leading_coeff().is_negative()
        && !preserve_atanh_sqrt_open_interval_gap_orientation(&gap_poly)
    {
        gap_poly = gap_poly.neg();
        numerator_sign = -numerator_sign;
    }

    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = numerator_sign * derivative_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;

    let gap = gap_poly.to_expr(ctx);
    let (numerator_coeff, denominator_coeff, gap) =
        cancel_denominator_content_with_numerator_for_calculus_presentation(
            ctx,
            numerator_coeff,
            denominator_coeff,
            gap,
        );
    let (sqrt_radicand, numerator_coeff, denominator_coeff) =
        if let Some((compact_sqrt, compact_numerator_coeff, compact_denominator_coeff)) =
            compact_rational_monomial_sqrt_denominator_for_calculus_presentation(
                ctx,
                &radicand_poly,
                numerator_coeff.clone(),
                denominator_coeff.clone(),
            )
        {
            (
                compact_sqrt,
                compact_numerator_coeff,
                compact_denominator_coeff,
            )
        } else {
            (
                ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]),
                numerator_coeff,
                denominator_coeff,
            )
        };
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, derivative_core);
    let core_denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, gap]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn compact_rational_monomial_sqrt_denominator_for_calculus_presentation(
    ctx: &mut Context,
    radicand_poly: &Polynomial,
    numerator_coeff: BigRational,
    denominator_coeff: BigRational,
) -> Option<(ExprId, BigRational, BigRational)> {
    if !denominator_coeff.is_integer()
        || !denominator_coeff.is_positive()
        || denominator_coeff.is_one()
    {
        return None;
    }
    if radicand_poly.degree() != 1
        || radicand_poly
            .coeffs
            .first()
            .is_some_and(|constant| !constant.is_zero())
    {
        return None;
    }

    let slope = radicand_poly.coeffs.get(1)?;
    let radicand_denominator = slope.denom().clone();
    if radicand_denominator <= BigInt::one() {
        return None;
    }

    let radicand_denominator_rational = BigRational::from_integer(radicand_denominator.clone());
    let compact_slope = slope * &radicand_denominator_rational * &radicand_denominator_rational;
    if !compact_slope.is_integer() {
        return None;
    }

    let compact_poly = Polynomial::new(
        vec![BigRational::zero(), compact_slope],
        radicand_poly.var.clone(),
    );
    let compact_radicand = compact_poly.to_expr(ctx);
    let compact_sqrt = ctx.call_builtin(BuiltinFn::Sqrt, vec![compact_radicand]);
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(
        &(numerator_coeff * radicand_denominator_rational / denominator_coeff),
    )?;
    Some((compact_sqrt, numerator_coeff, denominator_coeff))
}

pub(super) fn scaled_atanh_sqrt_polynomial_derivative_presentation(
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

    let (radicand, radicand_poly, sqrt_scale) =
        scaled_sqrt_polynomial_arg_for_calculus_presentation(ctx, args[0], var_name)?;
    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);

    let coefficient =
        derivative_content / (BigRational::from_integer(2.into()) * sqrt_scale.clone());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;

    let inverse_scale_square = reciprocal_positive_rational(&(sqrt_scale.clone() * sqrt_scale));
    let gap = subtract_from_rational_for_calculus_presentation(ctx, inverse_scale_square, radicand);
    let (numerator_coeff, denominator_coeff, gap) =
        cancel_denominator_content_with_numerator_for_calculus_presentation(
            ctx,
            numerator_coeff,
            denominator_coeff,
            gap,
        );

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, derivative_core);
    let core_denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, gap]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(super) fn constant_scaled_atanh_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    constant_scaled_sqrt_polynomial_derivative_presentation(
        ctx,
        target,
        var_name,
        scaled_atanh_sqrt_polynomial_derivative_presentation,
        atanh_sqrt_polynomial_derivative_presentation,
    )
}

pub(super) fn atanh_sqrt_family_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if let Some(compact) =
        scaled_atanh_sqrt_polynomial_derivative_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = atanh_sqrt_polynomial_derivative_presentation(ctx, target, var_name) {
        return Some(compact);
    }

    constant_scaled_atanh_sqrt_polynomial_derivative_presentation(ctx, target, var_name)
}
