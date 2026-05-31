use super::result_presentation::{
    divide_compact_derivative_by_constant_factor,
    reciprocal_constant_denominator_for_calculus_presentation,
    remove_unit_mul_factors_for_calculus_presentation, scale_compact_derivative_by_rational,
};
use super::scalar_presentation::{negate_calculus_presentation, rational_scaled_single_factor};
use cas_ast::{Context, Expr, ExprId};
use cas_math::expr_predicates::contains_named_var;
use num_rational::BigRational;
use num_traits::{One, Zero};

pub(super) fn constant_scaled_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
    scaled_derivative: fn(&mut Context, ExprId, &str) -> Option<ExprId>,
    unscaled_derivative: fn(&mut Context, ExprId, &str) -> Option<ExprId>,
) -> Option<ExprId> {
    if let Expr::Div(inner, outer_den) = ctx.get(target).clone() {
        if contains_named_var(ctx, outer_den, var_name) {
            return None;
        }
        let inner = remove_unit_mul_factors_for_calculus_presentation(ctx, inner);
        let derivative = scaled_sqrt_polynomial_derivative_for_constant_scale(
            ctx,
            inner,
            var_name,
            scaled_derivative,
            unscaled_derivative,
        )?;
        if let Some(denominator_scale) = cas_ast::views::as_rational_const(ctx, outer_den, 8) {
            if denominator_scale.is_zero() {
                return None;
            }
            return Some(scale_compact_derivative_by_rational(
                ctx,
                derivative,
                BigRational::one() / denominator_scale,
            ));
        }
        return Some(divide_compact_derivative_by_constant_factor(
            ctx, derivative, outer_den,
        ));
    }

    if let Some((scale, inner)) = rational_scaled_single_factor(ctx, target) {
        let derivative = scaled_sqrt_polynomial_derivative_for_constant_scale(
            ctx,
            inner,
            var_name,
            scaled_derivative,
            unscaled_derivative,
        )?;
        return Some(scale_compact_derivative_by_rational(ctx, derivative, scale));
    }

    let Expr::Mul(_, _) = ctx.get(target).clone() else {
        return None;
    };

    let factors = cas_math::expr_nary::mul_leaves(ctx, target);
    for idx in 0..factors.len() {
        let inner = factors[idx];
        let mut constant_factors = factors.clone();
        constant_factors.remove(idx);
        let [constant_factor] = constant_factors.as_slice() else {
            continue;
        };
        let Some(outer_den) = reciprocal_constant_denominator_for_calculus_presentation(
            ctx,
            *constant_factor,
            var_name,
        ) else {
            continue;
        };
        let Some(derivative) = scaled_sqrt_polynomial_derivative_for_constant_scale(
            ctx,
            inner,
            var_name,
            scaled_derivative,
            unscaled_derivative,
        ) else {
            continue;
        };
        return Some(divide_compact_derivative_by_constant_factor(
            ctx, derivative, outer_den,
        ));
    }

    None
}

fn scaled_sqrt_polynomial_derivative_for_constant_scale(
    ctx: &mut Context,
    inner: ExprId,
    var_name: &str,
    scaled_derivative: fn(&mut Context, ExprId, &str) -> Option<ExprId>,
    unscaled_derivative: fn(&mut Context, ExprId, &str) -> Option<ExprId>,
) -> Option<ExprId> {
    if let Expr::Neg(inner) = ctx.get(inner).clone() {
        let derivative = scaled_sqrt_polynomial_derivative_for_constant_scale(
            ctx,
            inner,
            var_name,
            scaled_derivative,
            unscaled_derivative,
        )?;
        return Some(negate_calculus_presentation(ctx, derivative));
    }

    scaled_derivative(ctx, inner, var_name).or_else(|| unscaled_derivative(ctx, inner, var_name))
}
