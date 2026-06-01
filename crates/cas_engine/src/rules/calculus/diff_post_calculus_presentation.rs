use cas_ast::{Context, ExprId};

use crate::symbolic_calculus_call_support::try_extract_diff_call;

use super::arctan_sqrt_additive_post_calculus_presentation::arctan_sqrt_additive_post_calculus_presentation;
use super::domain_checks::bounded_inverse_trig_known_empty_open_interval_gap;
use super::elementary_inverse_function_post_calculus_presentation::elementary_inverse_function_post_calculus_presentation;
use super::integral_derivative_shortcut_presentation::supported_integral_derivative_presentation;
use super::inverse_tangent_polynomial_root_post_calculus_presentation::inverse_tangent_polynomial_root_post_calculus_presentation;
use super::inverse_tangent_trig_affine_derivative_presentation::inverse_tangent_direct_trig_affine_derivative_presentation;
use super::log_derivative_presentation::{
    ln_power_derivative_numeric_presentation, variable_base_constant_argument_log_presentation,
};
use super::log_root_derivative_presentation::log_root_post_calculus_presentation;
use super::log_shifted_tan_sqrt_derivative_presentation::ln_constant_shifted_tan_sqrt_derivative_presentation;
use super::presentation_utils::unwrap_internal_hold_for_calculus;
use super::reciprocal_trig_derivative_presentation::direct_reciprocal_trig_post_calculus_presentation;
use super::result_sensitive_post_calculus_presentation::result_sensitive_post_calculus_presentation;
use super::root_and_inverse_family_post_calculus_presentation::root_and_inverse_family_post_calculus_presentation;

pub(super) fn try_diff_post_calculus_presentation(
    ctx: &mut Context,
    source: ExprId,
    result: ExprId,
) -> Option<ExprId> {
    let call = try_extract_diff_call(ctx, source)?;
    let target = unwrap_internal_hold_for_calculus(ctx, call.target);
    if bounded_inverse_trig_known_empty_open_interval_gap(ctx, target, &call.var_name).is_some() {
        return None;
    }
    if let Some(compact) =
        arctan_sqrt_additive_post_calculus_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some((compact, _)) =
        ln_constant_shifted_tan_sqrt_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        result_sensitive_post_calculus_presentation(ctx, target, result, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = supported_integral_derivative_presentation(ctx, target, &call.var_name) {
        return Some(compact);
    }
    if let Some(compact) =
        direct_reciprocal_trig_post_calculus_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        root_and_inverse_family_post_calculus_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = log_root_post_calculus_presentation(ctx, target, &call.var_name) {
        return Some(compact);
    }
    if let Some(compact) =
        variable_base_constant_argument_log_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        inverse_tangent_direct_trig_affine_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(unwrap_internal_hold_for_calculus(ctx, compact));
    }
    if let Some(compact) = ln_power_derivative_numeric_presentation(ctx, target, result) {
        return Some(compact);
    }
    if let Some(compact) =
        elementary_inverse_function_post_calculus_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    inverse_tangent_polynomial_root_post_calculus_presentation(ctx, target, &call.var_name)
}
