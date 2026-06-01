use super::gap_presentation::{primitive_positive_gap, reciprocal_positive_rational};
use super::polynomial_support::{
    polynomial_radicand_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation,
};
use super::result_presentation::scale_compact_derivative_by_rational;
use super::scalar_presentation::{
    add_rational_for_calculus_presentation, exact_positive_rational_sqrt_for_calculus_presentation,
    nonzero_rational_parts, rational_const_for_calculus_presentation,
    rational_scaled_single_factor, scale_expr_by_sqrt_positive_rational_for_calculus_presentation,
    signed_numerator_for_calculus_presentation, subtract_from_rational_for_calculus_presentation,
};
use super::scaled_sqrt_args::{
    reciprocal_sqrt_like_arg_for_calculus_presentation,
    scaled_sqrt_polynomial_arg_for_calculus_presentation,
};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::polynomial::Polynomial;
use cas_math::root_forms::extract_square_root_base;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

pub(super) fn bounded_inverse_trig_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    let derivative_sign = match ctx.builtin_of(*fn_id) {
        Some(BuiltinFn::Arcsin | BuiltinFn::Asin) => BigRational::one(),
        Some(BuiltinFn::Arccos | BuiltinFn::Acos) => -BigRational::one(),
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }

    let radicand = extract_square_root_base(ctx, args[0])?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);

    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = derivative_sign * derivative_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, derivative_core);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let raw_gap = Polynomial::one(radicand_poly.var.clone())
        .sub(&radicand_poly)
        .to_expr(ctx);
    let (gap, gap_content) = primitive_positive_gap(ctx, raw_gap);
    let (gap, numerator) = if gap_content.is_one()
        || exact_positive_rational_sqrt_for_calculus_presentation(&gap_content).is_some()
    {
        let numerator = scale_expr_by_sqrt_positive_rational_for_calculus_presentation(
            ctx,
            reciprocal_positive_rational(&gap_content),
            numerator,
        );
        (gap, numerator)
    } else {
        (raw_gap, numerator)
    };
    let sqrt_gap = ctx.call_builtin(BuiltinFn::Sqrt, vec![gap]);
    let core_denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, sqrt_gap]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(super) fn scaled_bounded_inverse_trig_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    let derivative_sign = match ctx.builtin_of(*fn_id) {
        Some(BuiltinFn::Arcsin | BuiltinFn::Asin) => BigRational::one(),
        Some(BuiltinFn::Arccos | BuiltinFn::Acos) => -BigRational::one(),
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }

    let (radicand, radicand_poly, sqrt_scale) =
        scaled_sqrt_polynomial_arg_for_calculus_presentation(ctx, args[0], var_name)?;
    if sqrt_scale.is_zero() || sqrt_scale.abs().is_one() {
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
    let coefficient = derivative_sign
        * scale_orientation
        * derivative_content
        * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, derivative_core);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let inverse_scale_square = reciprocal_positive_rational(&(sqrt_scale.clone() * sqrt_scale));
    let gap = subtract_from_rational_for_calculus_presentation(ctx, inverse_scale_square, radicand);
    let sqrt_gap = ctx.call_builtin(BuiltinFn::Sqrt, vec![gap]);
    let core_denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, sqrt_gap]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(super) fn bounded_inverse_trig_sqrt_affine_quotient_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target).clone() else {
        return None;
    };
    let derivative_sign = match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Arcsin | BuiltinFn::Asin) => BigRational::one(),
        Some(BuiltinFn::Arccos | BuiltinFn::Acos) => -BigRational::one(),
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }

    let radicand = extract_square_root_base(ctx, args[0])?;
    let Expr::Div(num, den) = ctx.get(radicand).clone() else {
        return None;
    };

    let numerator_poly = Polynomial::from_expr(ctx, num, var_name).ok()?;
    let denominator_poly = Polynomial::from_expr(ctx, den, var_name).ok()?;
    if numerator_poly.degree() != 1 || denominator_poly.degree() != 1 {
        return None;
    }

    let numerator_slope = numerator_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let denominator_slope = denominator_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if numerator_slope.is_zero() || numerator_slope != denominator_slope {
        return None;
    }

    let numerator_constant = numerator_poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let denominator_constant = denominator_poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let quotient_gap = denominator_constant - numerator_constant;
    if !quotient_gap.is_positive() {
        return None;
    }

    let coefficient = derivative_sign * numerator_slope * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let one = ctx.num(1);
    let numerator = signed_numerator_for_calculus_presentation(ctx, numerator_coeff, one);
    let numerator = scale_expr_by_sqrt_positive_rational_for_calculus_presentation(
        ctx,
        quotient_gap,
        numerator,
    );

    let sqrt_numerator = ctx.call_builtin(BuiltinFn::Sqrt, vec![num]);
    let core_denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[den, sqrt_numerator]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(super) fn bounded_inverse_trig_reciprocal_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target).clone() else {
        return None;
    };
    let derivative_sign = match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Arcsin | BuiltinFn::Asin) => -BigRational::one(),
        Some(BuiltinFn::Arccos | BuiltinFn::Acos) => BigRational::one(),
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }

    let (radicand, argument_scale) =
        reciprocal_sqrt_like_arg_for_calculus_presentation(ctx, args[0])?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);

    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = derivative_sign
        * argument_scale.clone()
        * derivative_content
        * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, derivative_core);

    let gap = add_rational_for_calculus_presentation(
        ctx,
        radicand,
        -(argument_scale.clone() * argument_scale),
    );
    let sqrt_gap = ctx.call_builtin(BuiltinFn::Sqrt, vec![gap]);
    let core_denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[radicand, sqrt_gap]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(super) fn constant_scaled_bounded_inverse_trig_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (scale, inner) = rational_scaled_single_factor(ctx, target)?;
    let derivative = if let Some(derivative) =
        scaled_bounded_inverse_trig_sqrt_polynomial_derivative_presentation(ctx, inner, var_name)
    {
        derivative
    } else {
        bounded_inverse_trig_sqrt_polynomial_derivative_presentation(ctx, inner, var_name)?
    };

    Some(scale_compact_derivative_by_rational(ctx, derivative, scale))
}

pub(super) fn bounded_inverse_trig_sqrt_family_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if let Some(compact) =
        bounded_inverse_trig_sqrt_polynomial_derivative_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        scaled_bounded_inverse_trig_sqrt_polynomial_derivative_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        bounded_inverse_trig_sqrt_affine_quotient_derivative_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = bounded_inverse_trig_reciprocal_sqrt_polynomial_derivative_presentation(
        ctx, target, var_name,
    ) {
        return Some(compact);
    }
    constant_scaled_bounded_inverse_trig_sqrt_polynomial_derivative_presentation(
        ctx, target, var_name,
    )
}
