use crate::define_rule;
use crate::symbolic_calculus_call_support::try_extract_integrate_call;

use super::arctan_polynomial_integrand_presentation::polynomial_times_arctan_affine_integrand_for_diff_shortcut;
use super::integration::{
    integrate, integrate_rewrite_with_conditions, IntegrationRequiredConditions,
};
use super::result_presentation::{
    apply_integration_final_presentation, compact_arctan_additive_terms_for_calculus_presentation,
};
use super::result_preservation::{
    apply_integration_result_preservation, integration_source_preservation_gates,
};

define_rule!(IntegrateRule, "Symbolic Integration", |ctx, expr| {
    let call = try_extract_integrate_call(ctx, expr)?;
    if let Some((mut result, required_nonzero)) =
        cas_math::symbolic_integration_support::integrate_symbolic_polynomial_trig_reciprocal_derivative_root_gate(
            ctx,
            call.target,
            &call.var_name,
        )
    {
        if polynomial_times_arctan_affine_integrand_for_diff_shortcut(
            ctx,
            call.target,
            &call.var_name,
        ) {
            if let Some(compact) =
                compact_arctan_additive_terms_for_calculus_presentation(ctx, result, &call.var_name)
            {
                result = compact;
            }
        }
        return Some(integrate_rewrite_with_conditions(
            ctx,
            &call,
            result,
            std::iter::once(crate::ImplicitCondition::NonZero(required_nonzero)),
        ));
    }

    let mut required_conditions =
        IntegrationRequiredConditions::from_target(ctx, call.target, &call.var_name);
    let source_preservation =
        integration_source_preservation_gates(ctx, call.target, &call.var_name);
    let mut result = integrate(ctx, call.target, &call.var_name)?;
    let compact_polynomial_arctan_by_parts_result =
        compact_arctan_additive_terms_for_calculus_presentation(ctx, result, &call.var_name);
    required_conditions.extend_atanh_result_conditions_if_source_positive_absent(ctx, result);
    result = apply_integration_result_preservation(
        ctx,
        result,
        &call.var_name,
        required_conditions.has_positive(),
        &source_preservation,
        compact_polynomial_arctan_by_parts_result,
    );
    result = apply_integration_final_presentation(ctx, result, &call.var_name);
    Some(integrate_rewrite_with_conditions(
        ctx,
        &call,
        result,
        required_conditions.into_implicit_conditions(),
    ))
});
