use super::polynomial_support::{
    polynomial_radicand_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation, strictly_positive_quadratic_on_reals,
};
use super::presentation_utils::unwrap_internal_hold_for_calculus;
use super::scalar_presentation::rational_const_for_calculus_presentation;
use cas_ast::{Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::Zero;

pub(super) fn compact_positive_quadratic_square_derivative_result(
    ctx: &mut Context,
    result: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let result = unwrap_internal_hold_for_calculus(ctx, result);
    let Expr::Div(numerator, denominator) = ctx.get(result).clone() else {
        return None;
    };
    let numerator_value = cas_ast::views::as_rational_const(ctx, numerator, 8)?;
    if numerator_value.is_zero() {
        return None;
    }

    let Expr::Pow(base, exp) = ctx.get(denominator).clone() else {
        return None;
    };
    let two = BigRational::from_integer(2.into());
    if cas_ast::views::as_rational_const(ctx, exp, 8).as_ref() != Some(&two) {
        return None;
    }

    let base_poly = polynomial_radicand_for_calculus_presentation(ctx, base, var_name)?;
    if !strictly_positive_quadratic_on_reals(&base_poly) {
        return None;
    }

    let (base_core, base_content) = split_polynomial_content_for_calculus_presentation(ctx, base);
    let compact_numerator_value = numerator_value / (&base_content * &base_content);
    let numerator = rational_const_for_calculus_presentation(ctx, compact_numerator_value);
    let two_expr = ctx.num(2);
    let denominator = ctx.add(Expr::Pow(base_core, two_expr));
    let compact = ctx.add(Expr::Div(numerator, denominator));

    Some(cas_ast::hold::wrap_hold(ctx, compact))
}
