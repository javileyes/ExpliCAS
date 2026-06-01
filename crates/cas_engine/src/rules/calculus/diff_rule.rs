use crate::define_rule;
use crate::symbolic_calculus_call_support::try_extract_diff_call;

use super::*;

define_rule!(DiffRule, "Symbolic Differentiation", |ctx, expr| {
    let call = try_extract_diff_call(ctx, expr)?;
    let target = unwrap_internal_hold_for_calculus(ctx, call.target);
    if diff_target_known_undefined_or_empty_domain_over_reals(ctx, target, &call.var_name) {
        return Some(undefined_diff_rewrite(ctx, &call));
    }
    if let Some((result, required_conditions)) =
        sign_polynomial_diff_result(ctx, target, &call.var_name)
    {
        return Some(diff_rewrite_with_conditions(
            ctx,
            &call,
            result,
            required_conditions,
        ));
    }
    if let Some((result, required_conditions)) =
        reciprocal_trig_shifted_sqrt_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(diff_rewrite_with_conditions(
            ctx,
            &call,
            result,
            required_conditions,
        ));
    }
    if let Some(rewrite) = arctan_sqrt_additive_derivative_rewrite(ctx, &call, target) {
        return Some(rewrite);
    }
    let mut shortcut_required_conditions = Vec::new();
    let mut result = ln_reciprocal_trig_sqrt_derivative_presentation(ctx, target, &call.var_name)
        .map(|(result, required_conditions)| {
            shortcut_required_conditions.extend(required_conditions);
            result
        })
        .or_else(|| {
            ln_constant_shifted_tan_sqrt_derivative_presentation(ctx, target, &call.var_name).map(
                |(result, required_conditions)| {
                    shortcut_required_conditions.extend(required_conditions);
                    result
                },
            )
        })
        .or_else(|| {
            supported_integral_diff_shortcut_presentation(ctx, target, &call.var_name).map(
                |(result, required_conditions)| {
                    shortcut_required_conditions.extend(required_conditions);
                    result
                },
            )
        })
        .or_else(|| {
            let (result, required_conditions) =
                inverse_tangent_scaled_sqrt_polynomial_derivative_shortcut(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                inverse_tangent_sqrt_over_symbolic_constant_derivative_shortcut(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                atanh_sqrt_over_symbolic_constant_derivative_shortcut(ctx, target, &call.var_name)?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                constant_scaled_inverse_tangent_sqrt_over_symbolic_constant_derivative_shortcut(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                arctan_sqrt_plus_sqrt_over_x_plus_one_derivative_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                constant_scaled_arctan_sqrt_variable_over_positive_affine_derivative_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                arctan_sqrt_variable_over_positive_affine_derivative_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            ln_sum_of_equal_derivative_roots_derivative_presentation(ctx, target, &call.var_name)
        })
        .or_else(|| reciprocal_positive_shifted_sqrt_derivative(ctx, target, &call.var_name))
        .or_else(|| {
            let (result, required_conditions) =
                reciprocal_sqrt_times_nonzero_shifted_sqrt_derivative(ctx, target, &call.var_name)?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_positive) =
                sqrt_over_positive_shifted_sqrt_derivative(ctx, target, &call.var_name)?;
            shortcut_required_conditions
                .push(crate::ImplicitCondition::Positive(required_positive));
            Some(result)
        })
        .or_else(|| {
            let (result, required_positive, required_conditions) =
                sqrt_additive_trig_polynomial_derivative_presentation(ctx, target, &call.var_name)?;
            append_positive_required_conditions(
                &mut shortcut_required_conditions,
                required_positive,
                required_conditions,
            );
            Some(result)
        })
        .or_else(|| {
            reciprocal_sqrt_polynomial_product_derivative_presentation(ctx, target, &call.var_name)
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
        .or_else(|| {
            reciprocal_constant_scaled_bounded_inverse_trig_surd_quotient_compact_derivative(
                ctx,
                target,
                &call.var_name,
            )
        })
        .or_else(|| {
            constant_scaled_arctan_surd_quotient_scaled_compact_derivative(
                ctx,
                target,
                &call.var_name,
            )
        })
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
                constant_scaled_acosh_affine_derivative_presentation(ctx, target, &call.var_name)?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                constant_scaled_acosh_polynomial_over_sqrt_derivative_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )?;
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
        .or_else(|| {
            bounded_inverse_trig_surd_quotient_compact_derivative(ctx, target, &call.var_name)
        })
        .or_else(|| {
            unit_interval_bounded_inverse_trig_derivative_presentation(ctx, target, &call.var_name)
                .map(|compact| cas_ast::hold::wrap_hold(ctx, compact))
        })
        .or_else(|| {
            bounded_inverse_trig_self_normalized_projection_derivative_presentation(
                ctx,
                target,
                &call.var_name,
            )
        })
        .or_else(|| asinh_surd_quotient_compact_derivative(ctx, target, &call.var_name))
        .or_else(|| {
            let (result, required_condition) =
                asinh_sqrt_constant_over_polynomial_presentation(ctx, target, &call.var_name)?;
            shortcut_required_conditions.push(required_condition);
            Some(result)
        })
        .or_else(|| {
            let (result, required_condition) =
                scaled_asinh_sqrt_constant_over_polynomial_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.push(required_condition);
            Some(result)
        })
        .or_else(|| arctan_surd_quotient_scaled_compact_derivative(ctx, target, &call.var_name))
        .or_else(|| arctan_surd_quotient_compact_derivative(ctx, target, &call.var_name))
        .or_else(|| {
            direct_self_normalized_surd_quotient_post_calculus_presentation(
                ctx,
                target,
                &call.var_name,
            )
        })
        .or_else(|| {
            let (result, required_condition) =
                arctan_self_normalized_surd_reciprocal_compact_derivative(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.push(required_condition);
            Some(result)
        })
        .or_else(|| arctan_affine_by_parts_compact_derivative(ctx, target, &call.var_name))
        .or_else(|| atanh_surd_quotient_compact_derivative(ctx, target, &call.var_name))
        .or_else(|| {
            polynomial_times_sqrt_polynomial_derivative_presentation(ctx, target, &call.var_name)
        })
        .or_else(|| {
            let (result, required_condition) =
                atanh_sqrt_constant_over_polynomial_presentation(ctx, target, &call.var_name)?;
            shortcut_required_conditions.push(required_condition);
            Some(result)
        })
        .or_else(|| {
            constant_scaled_asinh_sqrt_polynomial_derivative_presentation(
                ctx,
                target,
                &call.var_name,
            )
        })
        .or_else(|| {
            constant_scaled_atanh_sqrt_polynomial_derivative_presentation(
                ctx,
                target,
                &call.var_name,
            )
        })
        .or_else(|| {
            constant_scaled_acosh_sqrt_polynomial_derivative_presentation(
                ctx,
                target,
                &call.var_name,
            )
        })
        .or_else(|| {
            let (result, required_conditions) =
                arctan_reciprocal_abs_inverse_sqrt_polynomial_derivative_shortcut(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                negative_arccot_sqrt_polynomial_derivative_shortcut(ctx, target, &call.var_name)?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                inverse_tangent_reciprocal_sqrt_polynomial_product_derivative_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                inverse_tangent_reciprocal_sqrt_polynomial_derivative_shortcut(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                sqrt_trig_log_antiderivative_derivative_presentation(ctx, target, &call.var_name)?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            sqrt_bounded_trig_positive_shift_derivative_presentation(ctx, target, &call.var_name)
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
            let result =
                sqrt_elementary_function_derivative_presentation(ctx, target, &call.var_name)?;
            if let Some(radicand) = extract_square_root_base(ctx, target) {
                shortcut_required_conditions.push(crate::ImplicitCondition::Positive(radicand));
            }
            Some(result)
        })
        .or_else(|| {
            scaled_reciprocal_trig_power_derivative_presentation(ctx, target, &call.var_name)
        })
        .or_else(|| {
            inverse_tangent_direct_trig_affine_derivative_presentation(ctx, target, &call.var_name)
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
            let result =
                acosh_sqrt_shifted_quadratic_derivative_presentation(ctx, target, &call.var_name)?;
            shortcut_required_conditions.extend(acosh_sqrt_diff_required_conditions(ctx, target));
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                acosh_affine_derivative_presentation(ctx, target, &call.var_name)?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                acosh_strictly_positive_polynomial_derivative_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                acosh_polynomial_over_sqrt_derivative_presentation(ctx, target, &call.var_name)?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            positive_quadratic_square_derivative_result_presentation(ctx, target, &call.var_name)
        })
        .or_else(|| {
            positive_quadratic_quotient_derivative_presentation(ctx, target, &call.var_name)
        })
        .or_else(|| {
            let (result, required_conditions) =
                ln_sqrt_positive_shift_nonpolynomial_derivative_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| ln_sqrt_polynomial_gap_derivative_presentation(ctx, target, &call.var_name))
        .or_else(|| {
            let (result, required_conditions) =
                ln_sqrt_plus_polynomial_direct_derivative_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            exp_trig_by_parts_primitive_derivative_presentation(ctx, target, &call.var_name)
        })
        .or_else(|| affine_tanh_even_primitive_derivative_presentation(ctx, target, &call.var_name))
        .or_else(|| {
            affine_hyperbolic_odd_primitive_derivative_presentation(ctx, target, &call.var_name)
        })
        .or_else(|| log_over_sqrt_polynomial_derivative_presentation(ctx, target, &call.var_name))
        .or_else(|| sqrt_over_log_polynomial_derivative_presentation(ctx, target, &call.var_name))
        .or_else(|| differentiate(ctx, target, &call.var_name))?;
    if let Some(compact) =
        compact_positive_quadratic_square_derivative_result(ctx, result, &call.var_name)
    {
        result = compact;
    }
    if let Some(compact) = compact_sqrt_var_over_var_times_positive_shift_square_diff_result(
        ctx,
        result,
        &call.var_name,
    ) {
        result = compact;
    }
    let required_conditions = diff_required_conditions_for_target(ctx, target, &call.var_name)
        .into_iter()
        .chain(shortcut_required_conditions);
    Some(diff_rewrite_with_conditions(
        ctx,
        &call,
        result,
        required_conditions,
    ))
});
