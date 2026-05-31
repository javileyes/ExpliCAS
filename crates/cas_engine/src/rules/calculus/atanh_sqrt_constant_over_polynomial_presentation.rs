use super::polynomial_support::{
    polynomial_radicand_for_calculus_presentation, scale_polynomial_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation,
};
use super::result_presentation::{
    cancel_denominator_content_with_numerator_for_calculus_presentation,
    cancel_positive_denominator_content_with_numerator_for_calculus_presentation,
};
use super::scalar_presentation::{
    add_rational_for_calculus_presentation, nonzero_rational_parts,
    positive_constant_over_inverse_sqrt_arg_for_calculus_presentation,
    positive_rational_sqrt_denominator_factor_for_calculus_presentation,
    rational_const_for_calculus_presentation, scale_expr_for_calculus_presentation,
    subtract_from_rational_for_calculus_presentation,
};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Zero};

pub(super) fn atanh_sqrt_constant_over_polynomial_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, crate::ImplicitCondition)> {
    let Expr::Function(fn_id, args) = ctx.get(target).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let builtin = ctx.builtin_of(fn_id)?;
    if builtin != BuiltinFn::Atanh {
        return None;
    }
    let (numerator_value, den) =
        positive_constant_over_inverse_sqrt_arg_for_calculus_presentation(ctx, args[0])?;
    let (displayed_numerator_value, sqrt_numerator_denominator) =
        positive_rational_sqrt_denominator_factor_for_calculus_presentation(ctx, &numerator_value)?;

    let denominator_poly = polynomial_radicand_for_calculus_presentation(ctx, den, var_name)?;
    let derivative_poly = denominator_poly.derivative();
    if derivative_poly.is_zero() {
        return Some((ctx.num(0), crate::ImplicitCondition::Positive(den)));
    }
    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let denominator_constant = denominator_poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let orient_gap_to_numerator = numerator_value > denominator_constant;
    let coefficient_sign = if orient_gap_to_numerator {
        BigRational::one()
    } else {
        -BigRational::one()
    };
    let coefficient = coefficient_sign
        * displayed_numerator_value
        * derivative_content
        * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;

    let combine_constant_with_sqrt_denominator =
        sqrt_numerator_denominator.is_some() && numerator_value.is_integer();
    let sqrt_denominator = if combine_constant_with_sqrt_denominator {
        let scaled_denominator_poly =
            scale_polynomial_for_calculus_presentation(&denominator_poly, &numerator_value);
        let scaled_denominator = scaled_denominator_poly.to_expr(ctx);
        ctx.call_builtin(BuiltinFn::Sqrt, vec![scaled_denominator])
    } else {
        ctx.call_builtin(BuiltinFn::Sqrt, vec![den])
    };
    let denominator_gap = if orient_gap_to_numerator {
        subtract_from_rational_for_calculus_presentation(ctx, numerator_value, den)
    } else {
        add_rational_for_calculus_presentation(ctx, den, -numerator_value)
    };
    let (numerator_coeff, denominator_coeff, denominator_gap) = if orient_gap_to_numerator {
        cancel_positive_denominator_content_with_numerator_for_calculus_presentation(
            ctx,
            numerator_coeff,
            denominator_coeff,
            denominator_gap,
        )
    } else {
        cancel_denominator_content_with_numerator_for_calculus_presentation(
            ctx,
            numerator_coeff,
            denominator_coeff,
            denominator_gap,
        )
    };
    let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, derivative_core);
    let core_denominator = if combine_constant_with_sqrt_denominator {
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_denominator, denominator_gap])
    } else if let Some(sqrt_numerator_denominator) = sqrt_numerator_denominator {
        cas_math::expr_nary::build_balanced_mul(
            ctx,
            &[
                sqrt_numerator_denominator,
                sqrt_denominator,
                denominator_gap,
            ],
        )
    } else {
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_denominator, denominator_gap])
    };
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    let compact = ctx.add(Expr::Div(numerator, denominator));

    Some((
        cas_ast::hold::wrap_hold(ctx, compact),
        crate::ImplicitCondition::Positive(den),
    ))
}
