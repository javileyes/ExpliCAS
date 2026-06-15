use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::root_forms::extract_square_root_base;
use num_rational::BigRational;
use num_traits::One;

use super::polynomial_support::{
    polynomial_radicand_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation,
};
use super::polynomial_times_sqrt_polynomial_derivative_presentation::polynomial_times_sqrt_polynomial_derivative_presentation;
use super::presentation_utils::unwrap_internal_hold_for_calculus;
use super::scalar_presentation::{
    nonzero_rational_parts, rational_const_for_calculus_presentation,
    scale_expr_for_calculus_presentation,
};
use super::sqrt_elementary_function_derivative_presentation::sqrt_elementary_function_derivative_presentation;
use super::sqrt_reciprocal_trig_function_derivative_presentation::sqrt_reciprocal_trig_function_derivative_presentation;
use super::sqrt_small_additive_derivative_presentation::sqrt_small_additive_elementary_derivative_presentation;
use super::{
    log_over_sqrt_polynomial_derivative_presentation,
    polynomial_over_sqrt_polynomial_derivative_presentation,
    reciprocal_positive_shifted_sqrt_derivative,
    reciprocal_sqrt_polynomial_product_derivative_presentation,
    signed_elementary_sqrt_polynomial_derivative_presentation,
    sqrt_of_polynomial_quotient_derivative_presentation,
    sqrt_over_log_polynomial_derivative_presentation, sqrt_over_positive_shifted_sqrt_derivative,
    sqrt_polynomial_quotient_derivative_presentation, sqrt_shifted_exp_derivative_presentation,
    sqrt_shifted_ln_derivative_presentation,
};

pub(super) fn sqrt_derivative_post_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if let Some(compact) = reciprocal_positive_shifted_sqrt_derivative(ctx, target, var_name) {
        return Some(compact);
    }
    if let Some((compact, _)) = sqrt_over_positive_shifted_sqrt_derivative(ctx, target, var_name) {
        return Some(compact);
    }
    if let Some(compact) =
        reciprocal_sqrt_polynomial_product_derivative_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = log_over_sqrt_polynomial_derivative_presentation(ctx, target, var_name) {
        return Some(compact);
    }
    if let Some(compact) = sqrt_over_log_polynomial_derivative_presentation(ctx, target, var_name) {
        return Some(compact);
    }
    if let Some(compact) =
        polynomial_over_sqrt_polynomial_derivative_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = sqrt_polynomial_quotient_derivative_presentation(ctx, target, var_name) {
        return Some(compact);
    }
    if let Some(compact) =
        sqrt_of_polynomial_quotient_derivative_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = sqrt_shifted_exp_derivative_presentation(ctx, target, var_name) {
        return Some(compact);
    }
    if let Some(compact) = sqrt_shifted_ln_derivative_presentation(ctx, target, var_name) {
        return Some(compact);
    }
    if let Some((compact, _, _)) =
        sqrt_small_additive_elementary_derivative_presentation(ctx, target, var_name)
    {
        return Some(unwrap_internal_hold_for_calculus(ctx, compact));
    }
    if let Some(compact) = sqrt_elementary_function_derivative_presentation(ctx, target, var_name) {
        return Some(compact);
    }
    if let Some(compact) =
        sqrt_reciprocal_trig_function_derivative_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        polynomial_times_sqrt_polynomial_derivative_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = sqrt_polynomial_derivative_presentation(ctx, target, var_name) {
        return Some(compact);
    }
    signed_elementary_sqrt_polynomial_derivative_presentation(ctx, target, var_name)
}

/// Whether `radicand` is structurally `base^2`, so `sqrt(radicand) = |base|`.
fn radicand_is_structural_square(ctx: &Context, radicand: ExprId) -> bool {
    match ctx.get(radicand) {
        Expr::Pow(_, exp) => matches!(
            ctx.get(*exp),
            Expr::Number(n) if n.is_integer() && *n.numer() == 2.into()
        ),
        _ => false,
    }
}

pub(super) fn sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let radicand = extract_square_root_base(ctx, target)?;
    // A perfect-square radicand makes sqrt(g) = |base|; presenting the derivative
    // as g'/(2 sqrt(g)) would regress the clean sign(base) form (and leave the
    // radicand un-reduced, e.g. d/dx sqrt((-x)^2) -> x/sqrt((-x)^2)). Decline and
    // let the DiffRule sign route own it.
    if radicand_is_structural_square(ctx, radicand) {
        return None;
    }
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);

    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = derivative_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, derivative_core);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let denominator = if denominator_coeff == BigRational::one() {
        sqrt_radicand
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, sqrt_radicand])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}
