use super::gap_presentation::squared_expr_for_compact_gap_presentation;
use super::polynomial_support::{
    expanded_polynomial_expr_for_calculus_presentation,
    polynomial_radicand_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation, strictly_positive_quadratic_on_reals,
};
use super::presentation_utils::{
    multiply_by_sqrt_factor_for_calculus_presentation, squared_expr,
    unwrap_internal_hold_for_calculus,
};
use super::scalar_presentation::{
    rational_const_for_calculus_presentation, scale_expr_for_calculus_presentation,
    signed_numerator_for_calculus_presentation,
};
use super::surd_quotient_args::{
    atanh_arg_over_sqrt_parts, sqrt_scaled_arg_over_sqrt_parts_for_calculus_presentation,
};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

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

pub(super) fn inverse_reciprocal_trig_positive_quadratic_surd_quotient_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target).clone() else {
        return None;
    };
    let sign = match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Arcsec | BuiltinFn::Asec) => BigRational::one(),
        Some(BuiltinFn::Arccsc | BuiltinFn::Acsc) => -BigRational::one(),
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }

    let (num, radicand) = atanh_arg_over_sqrt_parts(ctx, args[0])
        .or_else(|| sqrt_scaled_arg_over_sqrt_parts_for_calculus_presentation(ctx, args[0]))?;
    let radicand_value = cas_ast::views::as_rational_const(ctx, radicand, 8)?;
    if !radicand_value.is_positive() {
        return None;
    }

    let num_poly = polynomial_radicand_for_calculus_presentation(ctx, num, var_name)?;
    if !strictly_positive_quadratic_on_reals(&num_poly) {
        return None;
    }

    let d_num = num_poly.derivative().to_expr(ctx);
    if cas_ast::views::as_rational_const(ctx, d_num, 8).is_some_and(|value| value.is_zero()) {
        return Some(ctx.num(0));
    }
    let (d_num_core, d_num_content) =
        split_polynomial_content_for_calculus_presentation(ctx, d_num);
    let num_square = squared_expr_for_compact_gap_presentation(ctx, num);
    let derivative_factor =
        signed_numerator_for_calculus_presentation(ctx, sign * d_num_content, d_num_core);
    let radicand_numer = BigRational::from_integer(radicand_value.numer().clone());
    let radicand_denom = BigRational::from_integer(radicand_value.denom().clone());
    let numerator = if radicand_numer.is_one() {
        derivative_factor
    } else {
        let compact_numer = rational_const_for_calculus_presentation(ctx, radicand_numer.clone());
        multiply_by_sqrt_factor_for_calculus_presentation(ctx, derivative_factor, compact_numer)
    };
    let scaled_num_square = if radicand_denom.is_one() {
        num_square
    } else {
        scale_expr_for_calculus_presentation(ctx, radicand_denom, num_square)
    };
    let compact_numer = rational_const_for_calculus_presentation(ctx, radicand_numer);
    let raw_gap = ctx.add(Expr::Sub(scaled_num_square, compact_numer));
    let sqrt_gap = ctx.call_builtin(BuiltinFn::Sqrt, vec![raw_gap]);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[num, sqrt_gap]);
    let compact = ctx.add(Expr::Div(numerator, denominator));

    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

pub(crate) fn inverse_reciprocal_trig_positive_quadratic_surd_quotient_presentation_with_domain(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let compact = inverse_reciprocal_trig_positive_quadratic_surd_quotient_presentation(
        ctx, target, var_name,
    )?;
    Some((unwrap_internal_hold_for_calculus(ctx, compact), Vec::new()))
}
