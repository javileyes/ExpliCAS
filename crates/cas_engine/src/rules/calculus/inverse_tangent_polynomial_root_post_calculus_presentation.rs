use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::One;

use super::arctan_sqrt_constant_over_polynomial_presentation::{
    arccot_sqrt_constant_over_polynomial_presentation,
    arctan_sqrt_constant_over_polynomial_presentation,
};
use super::arctan_sqrt_quotient_derivative_presentation::arctan_sqrt_scaled_variable_arg;
use super::inverse_tangent_polynomial_root_derivative_presentation::{
    arccot_sqrt_polynomial_derivative_presentation, arctan_sqrt_polynomial_derivative_presentation,
};
use super::inverse_tangent_reciprocal_sqrt_derivative_presentation::inverse_tangent_reciprocal_sqrt_polynomial_derivative_presentation;
use super::scalar_presentation::{
    add_one_for_calculus_presentation, nonzero_rational_parts,
    rational_const_for_calculus_presentation,
};

pub(super) fn inverse_tangent_polynomial_root_post_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if let Some(compact) =
        arctan_sqrt_constant_over_polynomial_presentation(ctx, target, var_name, BigRational::one())
    {
        return Some(compact);
    }
    if let Some(compact) = arccot_sqrt_constant_over_polynomial_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        inverse_tangent_reciprocal_sqrt_polynomial_derivative_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = arccot_sqrt_polynomial_derivative_presentation(ctx, target, var_name) {
        return Some(compact);
    }
    if let Some(compact) =
        arctan_sqrt_polynomial_derivative_presentation(ctx, target, var_name, BigRational::one())
    {
        return Some(compact);
    }

    let (radicand, derivative_scale) = arctan_sqrt_scaled_variable_arg(ctx, target, var_name)?;
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let half = BigRational::new(1.into(), 2.into());
    let coefficient = derivative_scale * half;
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator = rational_const_for_calculus_presentation(ctx, numerator_coeff);
    let denominator_head = if denominator_coeff == BigRational::one() {
        sqrt_radicand
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        ctx.add(Expr::Mul(denominator_scale, sqrt_radicand))
    };
    let radicand_plus_one = add_one_for_calculus_presentation(ctx, radicand);
    let denominator = ctx.add(Expr::Mul(denominator_head, radicand_plus_one));
    let compact = ctx.add(Expr::Div(numerator, denominator));
    Some(compact)
}
