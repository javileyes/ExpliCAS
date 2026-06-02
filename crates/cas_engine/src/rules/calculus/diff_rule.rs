use crate::define_rule;
use crate::symbolic_calculus_call_support::try_extract_diff_call;

use super::acosh_derivative_routes::{
    acosh_direct_derivative_route, constant_scaled_acosh_derivative_route,
};
use super::arctan_by_parts_result_presentation::arctan_affine_by_parts_compact_derivative;
use super::arctan_sqrt_positive_affine_derivative_routes::arctan_sqrt_positive_affine_derivative_route;
use super::arctan_sqrt_positive_shift_derivative_presentation::arctan_sqrt_plus_sqrt_over_x_plus_one_derivative_presentation;
use super::asinh_conditioned_sqrt_derivative_routes::conditioned_asinh_sqrt_constant_over_polynomial_derivative_route;
use super::inverse_hyperbolic_scaled_sqrt_derivative_routes::constant_scaled_inverse_hyperbolic_sqrt_polynomial_derivative_route;
use super::inverse_surd_quotient_derivative_presentation::constant_divisor_bounded_inverse_trig_surd_quotient_compact_derivative;
use super::inverse_tangent_reciprocal_sqrt_derivative_routes::inverse_tangent_reciprocal_sqrt_derivative_route;
use super::inverse_tangent_scaled_root_derivative_routes::inverse_tangent_scaled_root_derivative_rewrite;
use super::ln_sqrt_derivative_routes::ln_sqrt_derivative_route;
use super::log_sqrt_trig_derivative_routes::log_sqrt_trig_derivative_rewrite;
use super::primitive_derivative_routes::primitive_derivative_route;
use super::shifted_sqrt_derivative_routes::shifted_sqrt_derivative_route;
use super::sqrt_trig_log_antiderivative_presentation::sqrt_trig_log_antiderivative_derivative_presentation;
use super::surd_quotient_derivative_routes::{
    constant_scaled_surd_quotient_derivative_route, surd_quotient_derivative_route,
};
use super::*;

define_rule!(DiffRule, "Symbolic Differentiation", |ctx, expr| {
    let call = try_extract_diff_call(ctx, expr)?;
    let target = unwrap_internal_hold_for_calculus(ctx, call.target);
    if diff_target_known_undefined_or_empty_domain_over_reals(ctx, target, &call.var_name) {
        return Some(undefined_diff_rewrite(ctx, &call));
    }
    if let Some(rewrite) = sign_polynomial_diff_rewrite(ctx, &call, target) {
        return Some(rewrite);
    }
    if let Some(rewrite) = reciprocal_trig_shifted_sqrt_derivative_rewrite(ctx, &call, target) {
        return Some(rewrite);
    }
    if let Some(rewrite) = arctan_sqrt_additive_derivative_rewrite(ctx, &call, target) {
        return Some(rewrite);
    }
    if let Some(rewrite) = log_sqrt_trig_derivative_rewrite(ctx, &call, target) {
        return Some(rewrite);
    }
    if let Some(rewrite) = supported_integral_diff_shortcut_rewrite(ctx, &call, target) {
        return Some(rewrite);
    }
    if let Some(rewrite) = inverse_tangent_scaled_root_derivative_rewrite(ctx, &call, target) {
        return Some(rewrite);
    }
    let mut shortcut_required_conditions = Vec::new();
    let result =
        arctan_sqrt_plus_sqrt_over_x_plus_one_derivative_presentation(ctx, target, &call.var_name)
            .map(|(result, required_conditions)| {
                shortcut_required_conditions.extend(required_conditions);
                result
            })
            .or_else(|| {
                let (result, required_conditions) =
                    arctan_sqrt_positive_affine_derivative_route(ctx, target, &call.var_name)?;
                shortcut_required_conditions.extend(required_conditions);
                Some(result)
            })
            .or_else(|| {
                ln_sum_of_equal_derivative_roots_derivative_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )
            })
            .or_else(|| {
                let (result, required_conditions) =
                    shifted_sqrt_derivative_route(ctx, target, &call.var_name)?;
                shortcut_required_conditions.extend(required_conditions);
                Some(result)
            })
            .or_else(|| {
                let (result, required_positive, required_conditions) =
                    sqrt_additive_trig_polynomial_derivative_presentation(
                        ctx,
                        target,
                        &call.var_name,
                    )?;
                append_positive_required_conditions(
                    &mut shortcut_required_conditions,
                    required_positive,
                    required_conditions,
                );
                Some(result)
            })
            .or_else(|| {
                reciprocal_sqrt_polynomial_product_derivative_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )
            })
            .or_else(|| {
                let (result, required_conditions) =
                    sqrt_of_polynomial_quotient_derivative_presentation_with_domain(
                        ctx,
                        target,
                        &call.var_name,
                    )?;
                shortcut_required_conditions.extend(required_conditions);
                Some(result)
            })
            .or_else(|| constant_scaled_surd_quotient_derivative_route(ctx, target, &call.var_name))
            .or_else(|| {
                let (result, required_conditions) =
                constant_scaled_inverse_tangent_reciprocal_sqrt_product_derivative_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )?;
                shortcut_required_conditions.extend(required_conditions);
                Some(result)
            })
            .or_else(|| {
                constant_divisor_bounded_inverse_trig_surd_quotient_compact_derivative(
                    ctx,
                    target,
                    &call.var_name,
                )
            })
            .or_else(|| {
                let (result, required_conditions) =
                    constant_scaled_acosh_derivative_route(ctx, target, &call.var_name)?;
                shortcut_required_conditions.extend(required_conditions);
                Some(result)
            })
            .or_else(|| {
                let (result, required_conditions) =
                    constant_scaled_inverse_reciprocal_trig_affine_abs_presentation(
                        ctx,
                        target,
                        &call.var_name,
                    )?;
                shortcut_required_conditions.extend(required_conditions);
                Some(result)
            })
            .or_else(|| bounded_inverse_trig_derivative_route(ctx, target, &call.var_name))
            .or_else(|| asinh_surd_quotient_compact_derivative(ctx, target, &call.var_name))
            .or_else(|| {
                let (result, required_condition) =
                    conditioned_asinh_sqrt_constant_over_polynomial_derivative_route(
                        ctx,
                        target,
                        &call.var_name,
                    )?;
                shortcut_required_conditions.push(required_condition);
                Some(result)
            })
            .or_else(|| {
                let (result, required_conditions) =
                    surd_quotient_derivative_route(ctx, target, &call.var_name)?;
                shortcut_required_conditions.extend(required_conditions);
                Some(result)
            })
            .or_else(|| arctan_affine_by_parts_compact_derivative(ctx, target, &call.var_name))
            .or_else(|| atanh_surd_quotient_compact_derivative(ctx, target, &call.var_name))
            .or_else(|| {
                polynomial_times_sqrt_polynomial_derivative_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )
            })
            .or_else(|| {
                let (result, required_condition) =
                    atanh_sqrt_constant_over_polynomial_presentation(ctx, target, &call.var_name)?;
                shortcut_required_conditions.push(required_condition);
                Some(result)
            })
            .or_else(|| {
                constant_scaled_inverse_hyperbolic_sqrt_polynomial_derivative_route(
                    ctx,
                    target,
                    &call.var_name,
                )
            })
            .or_else(|| {
                let (result, required_conditions) =
                    inverse_tangent_reciprocal_sqrt_derivative_route(ctx, target, &call.var_name)?;
                shortcut_required_conditions.extend(required_conditions);
                Some(result)
            })
            .or_else(|| {
                let (result, required_conditions) =
                    sqrt_trig_log_antiderivative_derivative_presentation(
                        ctx,
                        target,
                        &call.var_name,
                    )?;
                shortcut_required_conditions.extend(required_conditions);
                Some(result)
            })
            .or_else(|| {
                sqrt_bounded_trig_positive_shift_derivative_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )
            })
            .or_else(|| {
                sqrt_additive_derivative_shortcut(
                    ctx,
                    target,
                    &call.var_name,
                    &mut shortcut_required_conditions,
                )
            })
            .or_else(|| {
                let (result, required_conditions) =
                    sqrt_elementary_function_derivative_route(ctx, target, &call.var_name)?;
                shortcut_required_conditions.extend(required_conditions);
                Some(result)
            })
            .or_else(|| {
                scaled_reciprocal_trig_power_derivative_presentation(ctx, target, &call.var_name)
            })
            .or_else(|| {
                inverse_tangent_direct_trig_affine_derivative_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )
            })
            .or_else(|| {
                arctan_sqrt_constant_over_polynomial_presentation(
                    ctx,
                    target,
                    &call.var_name,
                    BigRational::one(),
                )
            })
            .or_else(|| {
                arctan_sqrt_positive_polynomial_quotient_derivative_shortcut(
                    ctx,
                    target,
                    &call.var_name,
                )
            })
            .or_else(|| {
                let (result, required_conditions) =
                    acosh_direct_derivative_route(ctx, target, &call.var_name)?;
                shortcut_required_conditions.extend(required_conditions);
                Some(result)
            })
            .or_else(|| positive_quadratic_derivative_route(ctx, target, &call.var_name))
            .or_else(|| {
                let (result, required_conditions) =
                    ln_sqrt_derivative_route(ctx, target, &call.var_name)?;
                shortcut_required_conditions.extend(required_conditions);
                Some(result)
            })
            .or_else(|| primitive_derivative_route(ctx, target, &call.var_name))
            .or_else(|| log_sqrt_quotient_derivative_route(ctx, target, &call.var_name))
            .or_else(|| differentiate(ctx, target, &call.var_name))?;
    Some(finalize_diff_rewrite_with_conditions(
        ctx,
        &call,
        target,
        result,
        shortcut_required_conditions,
    ))
});
