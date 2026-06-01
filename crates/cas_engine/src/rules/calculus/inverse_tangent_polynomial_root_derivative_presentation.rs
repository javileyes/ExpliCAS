use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::One;

use super::domain_checks::positive_polynomial_radicand_required_conditions;
use super::inverse_tangent_reciprocal_sqrt_derivative_presentation::arctan_sqrt_reciprocal_content_presentation;
use super::inverse_tangent_root_args::{arccot_sqrt_radicand_arg, arctan_sqrt_radicand_arg};
use super::polynomial_support::{
    polynomial_radicand_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation,
};
use super::presentation_utils::scaled_sqrt_argument_for_calculus_presentation;
use super::result_presentation::{
    cancel_denominator_content_with_numerator_for_calculus_presentation,
    scale_compact_derivative_by_rational,
};
use super::scalar_presentation::{
    add_one_for_calculus_presentation, nonzero_rational_parts,
    rational_const_for_calculus_presentation, rational_scaled_single_factor,
    scale_expr_for_calculus_presentation, signed_numerator_for_calculus_presentation,
};

pub(super) fn arctan_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
    derivative_sign: BigRational,
) -> Option<ExprId> {
    let radicand = arctan_sqrt_radicand_arg(ctx, target)?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    if let Some(compact) = arctan_sqrt_reciprocal_content_presentation(
        ctx,
        radicand,
        &radicand_poly,
        derivative_sign.clone(),
    ) {
        return Some(compact);
    }
    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);

    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = derivative_sign * derivative_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let radicand_plus_one = add_one_for_calculus_presentation(ctx, radicand);
    let (numerator_coeff, denominator_coeff, radicand_plus_one) =
        cancel_denominator_content_with_numerator_for_calculus_presentation(
            ctx,
            numerator_coeff,
            denominator_coeff,
            radicand_plus_one,
        );
    let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, derivative_core);
    let core_denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, radicand_plus_one]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(super) fn constant_scaled_arctan_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (scale, inner) = rational_scaled_single_factor(ctx, target)?;
    let derivative =
        arctan_sqrt_polynomial_derivative_presentation(ctx, inner, var_name, BigRational::one())
            .or_else(|| arccot_sqrt_polynomial_derivative_presentation(ctx, inner, var_name))?;
    Some(scale_compact_derivative_by_rational(ctx, derivative, scale))
}

pub(super) fn arccot_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let radicand = arccot_sqrt_radicand_arg(ctx, target)?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    if let Some(compact) = arctan_sqrt_reciprocal_content_presentation(
        ctx,
        radicand,
        &radicand_poly,
        -BigRational::one(),
    ) {
        return Some(compact);
    }
    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);

    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = -derivative_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let radicand_plus_one = add_one_for_calculus_presentation(ctx, radicand);
    let (numerator_coeff, denominator_coeff, radicand_plus_one) =
        cancel_denominator_content_with_numerator_for_calculus_presentation(
            ctx,
            numerator_coeff,
            denominator_coeff,
            radicand_plus_one,
        );
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, derivative_core);
    let core_denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, radicand_plus_one]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(super) fn negative_arccot_sqrt_polynomial_derivative_shortcut(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let Expr::Function(fn_id, args) = ctx.get(target).clone() else {
        return None;
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(fn_id),
            Some(BuiltinFn::Acot | BuiltinFn::Arccot)
        )
    {
        return None;
    }

    let (radicand, argument_scale) = scaled_sqrt_argument_for_calculus_presentation(ctx, args[0])?;
    if argument_scale != -BigRational::one() {
        return None;
    }

    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some((ctx.num(0), Vec::new()));
    }
    let derivative = derivative_poly.to_expr(ctx);

    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = derivative_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let radicand_plus_one = add_one_for_calculus_presentation(ctx, radicand);
    let (numerator_coeff, denominator_coeff, radicand_plus_one) =
        cancel_denominator_content_with_numerator_for_calculus_presentation(
            ctx,
            numerator_coeff,
            denominator_coeff,
            radicand_plus_one,
        );
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, derivative_core);
    let core_denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, radicand_plus_one]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    let required_conditions =
        positive_polynomial_radicand_required_conditions(radicand, &radicand_poly);
    Some((
        ctx.add(Expr::Div(numerator, denominator)),
        required_conditions,
    ))
}
