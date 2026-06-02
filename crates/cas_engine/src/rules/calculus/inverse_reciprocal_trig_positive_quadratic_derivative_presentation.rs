use super::inverse_reciprocal_trig_affine_abs_derivative_presentation::inverse_reciprocal_trig_affine_abs_presentation;
use super::inverse_reciprocal_trig_positive_quadratic_square_presentation::inverse_reciprocal_trig_positive_quadratic_square_presentation;
use super::inverse_reciprocal_trig_positive_quadratic_surd_quotient_presentation::inverse_reciprocal_trig_positive_quadratic_surd_quotient_presentation;
use super::inverse_reciprocal_trig_sqrt_derivative_presentation::{
    inverse_reciprocal_trig_sqrt_affine_derivative_presentation,
    inverse_reciprocal_trig_sqrt_quadratic_derivative_presentation,
};
use super::polynomial_support::{
    expanded_polynomial_expr_for_calculus_presentation,
    polynomial_radicand_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation, strictly_positive_quadratic_on_reals,
};
use super::presentation_utils::squared_expr;
use super::scalar_presentation::signed_numerator_for_calculus_presentation;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::One;

pub(super) fn inverse_reciprocal_trig_positive_quadratic_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    let sign = match ctx.builtin_of(*fn_id) {
        Some(BuiltinFn::Arcsec | BuiltinFn::Asec) => BigRational::one(),
        Some(BuiltinFn::Arccsc | BuiltinFn::Acsc) => -BigRational::one(),
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }

    let arg = args[0];
    let arg_poly = polynomial_radicand_for_calculus_presentation(ctx, arg, var_name)?;
    if !strictly_positive_quadratic_on_reals(&arg_poly) {
        return None;
    }

    let derivative_poly = arg_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, sign * derivative_content, derivative_core);

    let arg_sq = squared_expr(ctx, arg);
    let one = ctx.num(1);
    let raw_gap = ctx.add(Expr::Sub(arg_sq, one));
    let gap = expanded_polynomial_expr_for_calculus_presentation(ctx, raw_gap, 4);
    let sqrt_gap = ctx.call_builtin(BuiltinFn::Sqrt, vec![gap]);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[arg, sqrt_gap]);

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(super) fn inverse_reciprocal_trig_post_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if let Some(compact) =
        inverse_reciprocal_trig_sqrt_affine_derivative_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        inverse_reciprocal_trig_sqrt_quadratic_derivative_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = inverse_reciprocal_trig_affine_abs_presentation(ctx, target, var_name) {
        return Some(compact);
    }
    if let Some(compact) =
        inverse_reciprocal_trig_positive_quadratic_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        inverse_reciprocal_trig_positive_quadratic_surd_quotient_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }

    inverse_reciprocal_trig_positive_quadratic_square_presentation(ctx, target, var_name)
}
