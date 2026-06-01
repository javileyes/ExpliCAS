use super::gap_presentation::reciprocal_positive_rational;
use super::polynomial_support::{
    polynomial_radicand_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation,
};
use super::scalar_presentation::{
    nonzero_rational_parts, rational_const_for_calculus_presentation,
    signed_numerator_for_calculus_presentation,
};
use super::scaled_sqrt_args::scaled_sqrt_polynomial_arg_for_calculus_presentation;
use super::sqrt_polynomial_scale_presentation::constant_scaled_sqrt_polynomial_derivative_presentation;
use super::{
    add_one_for_calculus_presentation, add_rational_for_calculus_presentation,
    polynomial_is_strictly_positive_everywhere,
    shared_positive_content_sqrt_product_for_calculus_presentation,
};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::polynomial::Polynomial;
use cas_math::root_forms::{extract_square_root_base, try_rewrite_simplify_square_root_expr};
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

pub(super) fn asinh_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    if args.len() != 1 || !ctx.is_builtin(*fn_id, BuiltinFn::Asinh) {
        return None;
    }

    let radicand = extract_square_root_base(ctx, args[0])?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    if !asinh_sqrt_presentation_safe_radicand(&radicand_poly) {
        return None;
    }
    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);

    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = derivative_content * BigRational::new(1.into(), 2.into());
    let (mut numerator_coeff, mut denominator_coeff) = nonzero_rational_parts(&coefficient)?;

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let radicand_plus_one = add_one_for_calculus_presentation(ctx, radicand);
    let sqrt_gap = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand_plus_one]);
    let core_denominator = if let Some((primitive_product, shared_content)) =
        shared_positive_content_sqrt_product_for_calculus_presentation(
            ctx,
            radicand,
            radicand_plus_one,
        ) {
        (numerator_coeff, denominator_coeff) =
            nonzero_rational_parts(&(numerator_coeff / (denominator_coeff * shared_content)))?;
        primitive_product
    } else {
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, sqrt_gap])
    };
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, derivative_core);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(super) fn constant_scaled_asinh_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    constant_scaled_sqrt_polynomial_derivative_presentation(
        ctx,
        target,
        var_name,
        scaled_asinh_sqrt_polynomial_derivative_presentation,
        asinh_sqrt_polynomial_derivative_presentation,
    )
}

pub(super) fn scaled_asinh_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    if args.len() != 1 || !ctx.is_builtin(*fn_id, BuiltinFn::Asinh) {
        return None;
    }

    let (radicand, radicand_poly, sqrt_scale) =
        scaled_sqrt_polynomial_arg_for_calculus_presentation(ctx, args[0], var_name)?;
    if !asinh_sqrt_presentation_safe_radicand(&radicand_poly) {
        return None;
    }

    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);

    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let scale_orientation = if sqrt_scale.is_negative() {
        -BigRational::one()
    } else {
        BigRational::one()
    };
    let coefficient = scale_orientation * derivative_content * BigRational::new(1.into(), 2.into());
    let (mut numerator_coeff, mut denominator_coeff) = nonzero_rational_parts(&coefficient)?;

    let inverse_scale_square =
        reciprocal_positive_rational(&(sqrt_scale.clone() * sqrt_scale.clone()));
    let gap = add_rational_for_calculus_presentation(ctx, radicand, inverse_scale_square);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let raw_sqrt_gap = ctx.call_builtin(BuiltinFn::Sqrt, vec![gap]);
    let sqrt_gap = try_rewrite_simplify_square_root_expr(ctx, raw_sqrt_gap)
        .map(|rewrite| rewrite.rewritten)
        .unwrap_or(raw_sqrt_gap);
    let core_denominator = if let Some((primitive_product, shared_content)) =
        shared_positive_content_sqrt_product_for_calculus_presentation(ctx, radicand, gap)
    {
        (numerator_coeff, denominator_coeff) =
            nonzero_rational_parts(&(numerator_coeff / (denominator_coeff * shared_content)))?;
        primitive_product
    } else {
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, sqrt_gap])
    };
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, derivative_core);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(super) fn asinh_sqrt_family_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if let Some(compact) = asinh_sqrt_polynomial_derivative_presentation(ctx, target, var_name) {
        return Some(compact);
    }
    if let Some(compact) =
        scaled_asinh_sqrt_polynomial_derivative_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }

    constant_scaled_asinh_sqrt_polynomial_derivative_presentation(ctx, target, var_name)
}

fn asinh_sqrt_presentation_safe_radicand(poly: &Polynomial) -> bool {
    match poly.degree() {
        1 => poly.coeffs.get(1).is_some_and(|linear| !linear.is_zero()),
        2 => polynomial_is_strictly_positive_everywhere(poly),
        _ => false,
    }
}
