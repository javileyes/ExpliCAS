use cas_ast::{Context, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed};

use super::polynomial_support::{
    polynomial_radicand_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation,
};
use super::scalar_presentation::signed_numerator_for_calculus_presentation;

pub(super) fn compact_surd_quotient_polynomial_presentation_parts(
    ctx: &mut Context,
    num: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId)> {
    let num_poly = polynomial_radicand_for_calculus_presentation(ctx, num, var_name)?;
    let mut derivative_poly = num_poly.derivative();
    if derivative_poly.is_zero() {
        return Some((ctx.num(0), num));
    }

    let square_poly = if num_poly.leading_coeff().is_negative() {
        num_poly.neg()
    } else {
        num_poly
    };
    let square_base = square_poly.to_expr(ctx);

    let mut derivative_sign = BigRational::one();
    if derivative_poly.leading_coeff().is_negative() {
        derivative_poly = derivative_poly.neg();
        derivative_sign = -derivative_sign;
    }
    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let derivative = signed_numerator_for_calculus_presentation(
        ctx,
        derivative_sign * derivative_content,
        derivative_core,
    );

    Some((derivative, square_base))
}
