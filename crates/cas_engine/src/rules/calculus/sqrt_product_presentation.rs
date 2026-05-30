use cas_ast::{BuiltinFn, Context, ExprId};
use cas_math::multipoly::{multipoly_from_expr, multipoly_to_expr, PolyBudget};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed};

use super::polynomial_support::{
    multipoly_denominator_lcm_for_calculus_presentation,
    multipoly_has_integer_coefficients_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation,
};

pub(super) fn shared_positive_content_sqrt_product_for_calculus_presentation(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<(ExprId, BigRational)> {
    let (left_primitive, left_content) =
        split_polynomial_content_for_calculus_presentation(ctx, left);
    let (right_primitive, right_content) =
        split_polynomial_content_for_calculus_presentation(ctx, right);

    if !left_content.is_one() && left_content.is_positive() && right_content == left_content {
        let left_sqrt = ctx.call_builtin(BuiltinFn::Sqrt, vec![left_primitive]);
        let right_sqrt = ctx.call_builtin(BuiltinFn::Sqrt, vec![right_primitive]);
        let product = cas_math::expr_nary::build_balanced_mul(ctx, &[left_sqrt, right_sqrt]);
        return Some((product, left_content));
    }

    shared_positive_denominator_sqrt_product_for_calculus_presentation(ctx, left, right)
}

fn shared_positive_denominator_sqrt_product_for_calculus_presentation(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<(ExprId, BigRational)> {
    let budget = PolyBudget {
        max_terms: 8,
        max_total_degree: 4,
        max_pow_exp: 4,
    };
    let left_poly = multipoly_from_expr(ctx, left, &budget).ok()?;
    let right_poly = multipoly_from_expr(ctx, right, &budget).ok()?;
    if left_poly.vars != right_poly.vars {
        return None;
    }

    let left_denominator = multipoly_denominator_lcm_for_calculus_presentation(&left_poly);
    let right_denominator = multipoly_denominator_lcm_for_calculus_presentation(&right_poly);
    if left_denominator != right_denominator || left_denominator <= BigInt::one() {
        return None;
    }

    let denominator = BigRational::from_integer(left_denominator);
    let left_primitive = left_poly.mul_scalar(&denominator);
    let right_primitive = right_poly.mul_scalar(&denominator);
    if !multipoly_has_integer_coefficients_for_calculus_presentation(&left_primitive)
        || !multipoly_has_integer_coefficients_for_calculus_presentation(&right_primitive)
    {
        return None;
    }

    let left_expr = multipoly_to_expr(&left_primitive, ctx);
    let right_expr = multipoly_to_expr(&right_primitive, ctx);
    let left_sqrt = ctx.call_builtin(BuiltinFn::Sqrt, vec![left_expr]);
    let right_sqrt = ctx.call_builtin(BuiltinFn::Sqrt, vec![right_expr]);
    let product = cas_math::expr_nary::build_balanced_mul(ctx, &[left_sqrt, right_sqrt]);
    Some((product, BigRational::one() / denominator))
}
