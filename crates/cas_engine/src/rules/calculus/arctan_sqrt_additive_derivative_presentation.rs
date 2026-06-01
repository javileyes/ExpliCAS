use cas_ast::{Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::One;

use super::domain_checks::positive_required_conditions_first;
use super::inverse_tangent_root_args::arctan_sqrt_radicand_arg;
use super::presentation_utils::{
    sqrt_raw_for_calculus_presentation, unwrap_internal_hold_for_calculus,
};
use super::scalar_presentation::{
    add_one_for_calculus_presentation,
    add_rational_combining_additive_constant_for_calculus_presentation,
};
use super::sqrt_small_additive_derivative_presentation::small_additive_elementary_radicand_derivative_for_calculus_presentation;
use super::{
    compact_numeric_mul_factors_for_calculus_presentation,
    sqrt_additive_tan_polynomial_derivative_inline_presentation,
    sqrt_additive_tan_polynomial_derivative_presentation,
    sqrt_additive_trig_polynomial_derivative_presentation,
};

pub(super) fn arctan_sqrt_additive_tan_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId, Vec<crate::ImplicitCondition>)> {
    let radicand = arctan_sqrt_radicand_arg(ctx, target)?;
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let (sqrt_derivative, required_positive, required_conditions) =
        sqrt_additive_tan_polynomial_derivative_presentation(ctx, sqrt_radicand, var_name)?;
    if required_positive != radicand {
        return None;
    }

    let sqrt_derivative = unwrap_internal_hold_for_calculus(ctx, sqrt_derivative);
    let radicand_plus_one = add_one_for_calculus_presentation(ctx, radicand);
    let result = match ctx.get(sqrt_derivative).clone() {
        Expr::Div(numerator, denominator) => {
            let denominator =
                cas_math::expr_nary::build_balanced_mul(ctx, &[denominator, radicand_plus_one]);
            ctx.add_raw(Expr::Div(numerator, denominator))
        }
        _ => ctx.add_raw(Expr::Div(sqrt_derivative, radicand_plus_one)),
    };

    Some((
        cas_ast::hold::wrap_hold(ctx, result),
        radicand,
        required_conditions,
    ))
}

pub(super) fn arctan_sqrt_additive_tan_polynomial_derivative_inline_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId, Vec<crate::ImplicitCondition>)> {
    let radicand = arctan_sqrt_radicand_arg(ctx, target)?;
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let (sqrt_derivative, required_positive, required_conditions) =
        sqrt_additive_tan_polynomial_derivative_inline_presentation(ctx, sqrt_radicand, var_name)?;
    if required_positive != radicand {
        return None;
    }

    let sqrt_derivative = unwrap_internal_hold_for_calculus(ctx, sqrt_derivative);
    let radicand_plus_one = add_one_for_calculus_presentation(ctx, radicand);
    let result = match ctx.get(sqrt_derivative).clone() {
        Expr::Div(numerator, denominator) => {
            let denominator =
                cas_math::expr_nary::build_balanced_mul(ctx, &[denominator, radicand_plus_one]);
            ctx.add_raw(Expr::Div(numerator, denominator))
        }
        _ => ctx.add_raw(Expr::Div(sqrt_derivative, radicand_plus_one)),
    };

    Some((
        cas_ast::hold::wrap_hold(ctx, result),
        radicand,
        required_conditions,
    ))
}

pub(super) fn arctan_sqrt_additive_trig_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId, Vec<crate::ImplicitCondition>)> {
    let radicand = arctan_sqrt_radicand_arg(ctx, target)?;
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let (sqrt_derivative, required_positive, required_conditions) =
        sqrt_additive_trig_polynomial_derivative_presentation(ctx, sqrt_radicand, var_name)?;
    if required_positive != radicand {
        return None;
    }

    let sqrt_derivative = unwrap_internal_hold_for_calculus(ctx, sqrt_derivative);
    let radicand_plus_one = add_rational_combining_additive_constant_for_calculus_presentation(
        ctx,
        radicand,
        BigRational::one(),
    );
    let result = match ctx.get(sqrt_derivative).clone() {
        Expr::Div(numerator, denominator) => {
            let denominator =
                cas_math::expr_nary::build_balanced_mul(ctx, &[denominator, radicand_plus_one]);
            ctx.add_raw(Expr::Div(numerator, denominator))
        }
        _ => ctx.add_raw(Expr::Div(sqrt_derivative, radicand_plus_one)),
    };

    Some((
        cas_ast::hold::wrap_hold(ctx, result),
        radicand,
        required_conditions,
    ))
}

pub(super) fn arctan_sqrt_small_additive_elementary_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId, Vec<crate::ImplicitCondition>)> {
    let radicand = arctan_sqrt_radicand_arg(ctx, target)?;
    let (radicand_derivative, derivative_denominator, required_conditions) =
        small_additive_elementary_radicand_derivative_for_calculus_presentation(
            ctx, radicand, var_name,
        )?;

    let two = ctx.num(2);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let radicand_plus_one = add_one_for_calculus_presentation(ctx, radicand);
    let mut denominator_factors = vec![two];
    if let Some(derivative_denominator) = derivative_denominator {
        denominator_factors.push(derivative_denominator);
    }
    denominator_factors.push(sqrt_radicand);
    denominator_factors.push(radicand_plus_one);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_factors);
    let denominator = compact_numeric_mul_factors_for_calculus_presentation(ctx, denominator);
    let compact = ctx.add_raw(Expr::Div(radicand_derivative, denominator));
    Some((
        cas_ast::hold::wrap_hold(ctx, compact),
        radicand,
        required_conditions,
    ))
}

pub(crate) fn arctan_sqrt_additive_tan_polynomial_derivative_presentation_with_domain(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (result, required_positive, required_conditions) =
        arctan_sqrt_additive_tan_polynomial_derivative_presentation(ctx, target, var_name)?;
    let required_conditions =
        positive_required_conditions_first(required_positive, required_conditions);
    Some((
        unwrap_internal_hold_for_calculus(ctx, result),
        required_conditions,
    ))
}

pub(crate) fn arctan_sqrt_additive_tan_polynomial_derivative_inline_presentation_with_domain(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (result, required_positive, required_conditions) =
        arctan_sqrt_additive_tan_polynomial_derivative_inline_presentation(ctx, target, var_name)?;
    let required_conditions =
        positive_required_conditions_first(required_positive, required_conditions);
    Some((
        unwrap_internal_hold_for_calculus(ctx, result),
        required_conditions,
    ))
}

pub(crate) fn arctan_sqrt_additive_trig_polynomial_derivative_presentation_with_domain(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (result, required_positive, required_conditions) =
        arctan_sqrt_additive_trig_polynomial_derivative_presentation(ctx, target, var_name)?;
    let required_conditions =
        positive_required_conditions_first(required_positive, required_conditions);
    Some((
        unwrap_internal_hold_for_calculus(ctx, result),
        required_conditions,
    ))
}

pub(crate) fn arctan_sqrt_small_additive_elementary_derivative_presentation_with_domain(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (result, required_positive, required_conditions) =
        arctan_sqrt_small_additive_elementary_derivative_presentation(ctx, target, var_name)?;
    let required_conditions =
        positive_required_conditions_first(required_positive, required_conditions);
    Some((
        unwrap_internal_hold_for_calculus(ctx, result),
        required_conditions,
    ))
}
