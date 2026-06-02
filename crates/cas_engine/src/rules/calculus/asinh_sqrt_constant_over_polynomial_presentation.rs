use super::derivative_result_scaling_presentation::cancel_denominator_content_with_numerator_for_calculus_presentation;
use super::polynomial_support::{
    polynomial_radicand_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation,
};
use super::presentation_utils::unwrap_internal_hold_for_calculus;
use super::scalar_presentation::{
    add_rational_for_calculus_presentation, nonzero_rational_parts,
    positive_constant_over_inverse_sqrt_arg_for_calculus_presentation,
    rational_const_for_calculus_presentation,
    scale_expr_by_sqrt_positive_rational_for_calculus_presentation,
    scale_expr_for_calculus_presentation,
    split_square_factor_positive_rational_for_calculus_presentation,
};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::expr_predicates::contains_named_var;
use num_rational::BigRational;
use num_traits::One;

pub(super) fn asinh_sqrt_constant_over_polynomial_presentation(
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
    if builtin != BuiltinFn::Asinh {
        return None;
    }
    let (numerator_value, den) =
        positive_constant_over_inverse_sqrt_arg_for_calculus_presentation(ctx, args[0])?;
    let (sqrt_numerator_outside, sqrt_numerator_inside) =
        split_square_factor_positive_rational_for_calculus_presentation(&numerator_value);

    let denominator_poly = polynomial_radicand_for_calculus_presentation(ctx, den, var_name)?;
    let derivative_poly = denominator_poly.derivative();
    if derivative_poly.is_zero() {
        return Some((ctx.num(0), crate::ImplicitCondition::Positive(den)));
    }
    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient =
        -sqrt_numerator_outside * derivative_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;

    let denominator_plus_numerator =
        add_rational_for_calculus_presentation(ctx, den, numerator_value);
    let (numerator_coeff, denominator_coeff, den) =
        cancel_denominator_content_with_numerator_for_calculus_presentation(
            ctx,
            numerator_coeff,
            denominator_coeff,
            den,
        );
    let scaled_derivative =
        scale_expr_for_calculus_presentation(ctx, numerator_coeff, derivative_core);
    let numerator = if sqrt_numerator_inside.is_one() {
        scaled_derivative
    } else {
        scale_expr_by_sqrt_positive_rational_for_calculus_presentation(
            ctx,
            sqrt_numerator_inside,
            scaled_derivative,
        )
    };
    let sqrt_gap = ctx.call_builtin(BuiltinFn::Sqrt, vec![denominator_plus_numerator]);
    let core_denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[den, sqrt_gap]);
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

pub(super) fn scaled_asinh_sqrt_constant_over_polynomial_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, crate::ImplicitCondition)> {
    let factors = cas_math::expr_nary::mul_leaves(ctx, target);
    if factors.len() < 2 {
        return None;
    }

    let mut scale_factors = Vec::new();
    let mut asinh_target = None;
    for factor in factors {
        if asinh_target.is_none()
            && matches!(
                ctx.get(factor),
                Expr::Function(fn_id, args)
                    if args.len() == 1 && ctx.is_builtin(*fn_id, BuiltinFn::Asinh)
            )
        {
            asinh_target = Some(factor);
        } else if !contains_named_var(ctx, factor, var_name) {
            scale_factors.push(factor);
        } else {
            return None;
        }
    }

    let asinh_target = asinh_target?;
    if scale_factors.is_empty() {
        return None;
    }

    let (compact, required_condition) =
        asinh_sqrt_constant_over_polynomial_presentation(ctx, asinh_target, var_name)?;
    let compact = unwrap_internal_hold_for_calculus(ctx, compact);
    scale_factors.push(compact);
    let scaled = cas_math::expr_nary::build_balanced_mul(ctx, &scale_factors);

    Some((cas_ast::hold::wrap_hold(ctx, scaled), required_condition))
}
