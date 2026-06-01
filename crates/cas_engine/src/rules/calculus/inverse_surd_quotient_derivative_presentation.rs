use cas_ast::{Context, Expr, ExprId};
use cas_math::expr_predicates::contains_named_var;

use super::arctan_surd_derivative_presentation::arctan_surd_quotient_compact_derivative;
use super::asinh_surd_derivative_presentation::asinh_surd_quotient_compact_derivative;
use super::atanh_surd_derivative_presentation::atanh_surd_quotient_compact_derivative;
use super::inverse_trig_derivative_presentation::bounded_inverse_trig_surd_quotient_compact_derivative;
use super::result_presentation::{
    divide_compact_derivative_by_constant_factor,
    reciprocal_constant_denominator_for_calculus_presentation,
    remove_unit_mul_factors_for_calculus_presentation,
};

pub(super) fn inverse_surd_quotient_post_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if let Some(compact) = constant_divisor_bounded_inverse_trig_surd_quotient_compact_derivative(
        ctx, target, var_name,
    ) {
        return Some(compact);
    }
    if let Some(compact) =
        bounded_inverse_trig_surd_quotient_compact_derivative(ctx, target, var_name)
    {
        return Some(compact);
    }

    asinh_surd_quotient_compact_derivative(ctx, target, var_name)
}

pub(super) fn constant_divisor_bounded_inverse_trig_surd_quotient_compact_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Div(inner, outer_den) = ctx.get(target).clone() else {
        return None;
    };
    if contains_named_var(ctx, outer_den, var_name) {
        return None;
    }

    let inner = remove_unit_mul_factors_for_calculus_presentation(ctx, inner);
    let inner_derivative =
        bounded_inverse_trig_surd_quotient_compact_derivative(ctx, inner, var_name)
            .or_else(|| arctan_surd_quotient_compact_derivative(ctx, inner, var_name))
            .or_else(|| asinh_surd_quotient_compact_derivative(ctx, inner, var_name))
            .or_else(|| atanh_surd_quotient_compact_derivative(ctx, inner, var_name))?;
    Some(divide_compact_derivative_by_constant_factor(
        ctx,
        inner_derivative,
        outer_den,
    ))
}

pub(super) fn reciprocal_constant_scaled_bounded_inverse_trig_surd_quotient_compact_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
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
        let Some(inner_derivative) =
            bounded_inverse_trig_surd_quotient_compact_derivative(ctx, inner, var_name)
                .or_else(|| asinh_surd_quotient_compact_derivative(ctx, inner, var_name))
                .or_else(|| atanh_surd_quotient_compact_derivative(ctx, inner, var_name))
        else {
            continue;
        };
        return Some(divide_compact_derivative_by_constant_factor(
            ctx,
            inner_derivative,
            outer_den,
        ));
    }

    None
}
