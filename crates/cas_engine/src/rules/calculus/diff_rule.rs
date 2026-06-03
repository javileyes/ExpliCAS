use crate::define_rule;
use crate::symbolic_calculus_call_support::try_extract_diff_call;

use super::acosh_derivative_routes::{
    acosh_direct_derivative_route, constant_scaled_acosh_derivative_rewrite,
};
use super::arctan_by_parts_result_presentation::arctan_affine_by_parts_compact_derivative;
use super::arctan_sqrt_positive_affine_derivative_routes::arctan_sqrt_positive_affine_derivative_rewrite;
use super::arctan_sqrt_positive_shift_derivative_presentation::arctan_sqrt_plus_sqrt_over_x_plus_one_derivative_rewrite;
use super::asinh_conditioned_sqrt_derivative_routes::conditioned_asinh_sqrt_constant_over_polynomial_derivative_rewrite;
use super::atanh_sqrt_constant_over_polynomial_presentation::atanh_sqrt_constant_over_polynomial_derivative_rewrite;
use super::inverse_hyperbolic_scaled_sqrt_derivative_routes::constant_scaled_inverse_hyperbolic_sqrt_polynomial_derivative_route;
use super::inverse_reciprocal_trig_affine_abs_derivative_presentation::constant_scaled_inverse_reciprocal_trig_affine_abs_rewrite;
use super::inverse_surd_quotient_derivative_presentation::constant_divisor_bounded_inverse_trig_surd_quotient_compact_derivative;
use super::inverse_tangent_reciprocal_sqrt_derivative_routes::inverse_tangent_reciprocal_sqrt_derivative_route;
use super::inverse_tangent_scaled_root_derivative_routes::inverse_tangent_scaled_root_derivative_rewrite;
use super::ln_sqrt_derivative_routes::ln_sqrt_derivative_route;
use super::log_sqrt_trig_derivative_routes::log_sqrt_trig_derivative_rewrite;
use super::primitive_derivative_routes::primitive_derivative_route;
use super::reciprocal_sqrt_product_derivative_presentation::constant_scaled_inverse_tangent_reciprocal_sqrt_product_derivative_rewrite;
use super::shifted_sqrt_derivative_routes::shifted_sqrt_derivative_rewrite;
use super::sqrt_polynomial_quotient_derivative_presentation::sqrt_of_polynomial_quotient_derivative_rewrite;
use super::sqrt_trig_log_antiderivative_presentation::sqrt_trig_log_antiderivative_derivative_presentation;
use super::surd_quotient_derivative_routes::{
    constant_scaled_surd_quotient_derivative_route, surd_quotient_derivative_rewrite,
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
    if let Some(rewrite) =
        arctan_sqrt_plus_sqrt_over_x_plus_one_derivative_rewrite(ctx, &call, target)
    {
        return Some(rewrite);
    }
    if let Some(rewrite) = arctan_sqrt_positive_affine_derivative_rewrite(ctx, &call, target) {
        return Some(rewrite);
    }
    if let Some(result) =
        ln_sum_of_equal_derivative_roots_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(finalize_diff_rewrite_with_conditions(
            ctx,
            &call,
            target,
            result,
            Vec::new(),
        ));
    }
    if let Some(rewrite) = shifted_sqrt_derivative_rewrite(ctx, &call, target) {
        return Some(rewrite);
    }
    let mut shortcut_required_conditions = Vec::new();
    let result = sqrt_additive_trig_polynomial_derivative_presentation(ctx, target, &call.var_name)
        .map(|(result, required_positive, required_conditions)| {
            append_positive_required_conditions(
                &mut shortcut_required_conditions,
                required_positive,
                required_conditions,
            );
            result
        })
        .or_else(|| {
            reciprocal_sqrt_polynomial_product_derivative_presentation(ctx, target, &call.var_name)
        });
    if let Some(result) = result {
        return Some(finalize_diff_rewrite_with_conditions(
            ctx,
            &call,
            target,
            result,
            shortcut_required_conditions,
        ));
    }
    if let Some(rewrite) = sqrt_of_polynomial_quotient_derivative_rewrite(ctx, &call, target) {
        return Some(rewrite);
    }

    let result = constant_scaled_surd_quotient_derivative_route(ctx, target, &call.var_name);
    if let Some(result) = result {
        return Some(finalize_diff_rewrite_with_conditions(
            ctx,
            &call,
            target,
            result,
            Vec::new(),
        ));
    }
    if let Some(rewrite) =
        constant_scaled_inverse_tangent_reciprocal_sqrt_product_derivative_rewrite(
            ctx, &call, target,
        )
    {
        return Some(rewrite);
    }

    let result = {
        constant_divisor_bounded_inverse_trig_surd_quotient_compact_derivative(
            ctx,
            target,
            &call.var_name,
        )
    };
    if let Some(result) = result {
        return Some(finalize_diff_rewrite_with_conditions(
            ctx,
            &call,
            target,
            result,
            Vec::new(),
        ));
    }
    if let Some(rewrite) = constant_scaled_acosh_derivative_rewrite(ctx, &call, target) {
        return Some(rewrite);
    }
    if let Some(rewrite) =
        constant_scaled_inverse_reciprocal_trig_affine_abs_rewrite(ctx, &call, target)
    {
        return Some(rewrite);
    }

    let result = bounded_inverse_trig_derivative_route(ctx, target, &call.var_name)
        .or_else(|| asinh_surd_quotient_compact_derivative(ctx, target, &call.var_name));
    if let Some(result) = result {
        return Some(finalize_diff_rewrite_with_conditions(
            ctx,
            &call,
            target,
            result,
            Vec::new(),
        ));
    }
    if let Some(rewrite) =
        conditioned_asinh_sqrt_constant_over_polynomial_derivative_rewrite(ctx, &call, target)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) = surd_quotient_derivative_rewrite(ctx, &call, target) {
        return Some(rewrite);
    }

    let result = arctan_affine_by_parts_compact_derivative(ctx, target, &call.var_name)
        .map(|result| (result, Vec::new()))
        .or_else(|| {
            atanh_surd_quotient_compact_derivative(ctx, target, &call.var_name)
                .map(|result| (result, Vec::new()))
        })
        .or_else(|| {
            polynomial_times_sqrt_polynomial_derivative_presentation(ctx, target, &call.var_name)
                .map(|result| (result, Vec::new()))
        });
    if let Some((result, shortcut_required_conditions)) = result {
        return Some(finalize_diff_rewrite_with_conditions(
            ctx,
            &call,
            target,
            result,
            shortcut_required_conditions,
        ));
    }
    if let Some(rewrite) =
        atanh_sqrt_constant_over_polynomial_derivative_rewrite(ctx, &call, target)
    {
        return Some(rewrite);
    }

    let result = constant_scaled_inverse_hyperbolic_sqrt_polynomial_derivative_route(
        ctx,
        target,
        &call.var_name,
    )
    .map(|result| (result, Vec::new()))
    .or_else(|| inverse_tangent_reciprocal_sqrt_derivative_route(ctx, target, &call.var_name))
    .or_else(|| sqrt_trig_log_antiderivative_derivative_presentation(ctx, target, &call.var_name))
    .or_else(|| {
        sqrt_bounded_trig_positive_shift_derivative_presentation(ctx, target, &call.var_name)
            .map(|result| (result, Vec::new()))
    })
    .or_else(|| {
        let mut required_conditions = Vec::new();
        sqrt_additive_derivative_shortcut(ctx, target, &call.var_name, &mut required_conditions)
            .map(|result| (result, required_conditions))
    })
    .or_else(|| sqrt_elementary_function_derivative_route(ctx, target, &call.var_name))
    .or_else(|| {
        scaled_reciprocal_trig_power_derivative_presentation(ctx, target, &call.var_name)
            .map(|result| (result, Vec::new()))
    })
    .or_else(|| {
        inverse_tangent_direct_trig_affine_derivative_presentation(ctx, target, &call.var_name)
            .map(|result| (result, Vec::new()))
    })
    .or_else(|| {
        arctan_sqrt_constant_over_polynomial_presentation(
            ctx,
            target,
            &call.var_name,
            BigRational::one(),
        )
        .map(|result| (result, Vec::new()))
    })
    .or_else(|| {
        arctan_sqrt_positive_polynomial_quotient_derivative_shortcut(ctx, target, &call.var_name)
            .map(|result| (result, Vec::new()))
    })
    .or_else(|| acosh_direct_derivative_route(ctx, target, &call.var_name))
    .or_else(|| {
        positive_quadratic_derivative_route(ctx, target, &call.var_name)
            .map(|result| (result, Vec::new()))
    })
    .or_else(|| ln_sqrt_derivative_route(ctx, target, &call.var_name))
    .or_else(|| {
        primitive_derivative_route(ctx, target, &call.var_name).map(|result| (result, Vec::new()))
    })
    .or_else(|| {
        log_sqrt_quotient_derivative_route(ctx, target, &call.var_name)
            .map(|result| (result, Vec::new()))
    })
    .or_else(|| differentiate(ctx, target, &call.var_name).map(|result| (result, Vec::new())))?;
    let (result, shortcut_required_conditions) = result;
    Some(finalize_diff_rewrite_with_conditions(
        ctx,
        &call,
        target,
        result,
        shortcut_required_conditions,
    ))
});
