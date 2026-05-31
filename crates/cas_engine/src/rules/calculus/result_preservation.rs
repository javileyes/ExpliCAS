use super::arctan_integrand_preservation::arctan_integrand_for_calculus_presentation;
use super::by_parts_integrand_preservation::by_parts_integrand_preservation_gates;
use super::fractional_denominator_power_integrand_preservation::fractional_denominator_power_substitution_integrand_for_calculus_presentation;
use super::hyperbolic_power_integrand_presentation::hyperbolic_power_integrand_for_calculus_presentation;
use super::inverse_hyperbolic_affine_integrand_preservation::inverse_hyperbolic_affine_integrand_for_calculus_presentation;
use super::log_product_integrand_preservation::log_product_integrand_for_calculus_presentation;
use super::power_result_presentation::{
    compact_half_power_sum_root_product_for_integration_presentation,
    compact_negative_half_power_result_for_integration_presentation,
    compact_negative_three_half_power_result_for_integration_presentation,
    compact_positive_half_power_result_for_integration_presentation,
};
use super::rational_partial_fraction_integrand_preservation::rational_linear_partial_fraction_integrand_for_calculus_presentation;
use super::result_presentation::{
    compact_arctan_additive_terms_for_calculus_presentation,
    compact_positive_cosh_log_abs_for_integration_presentation,
    compact_sqrt_hyperbolic_reciprocal_for_integration_presentation,
    compact_sqrt_trig_log_abs_for_integration_presentation,
    flatten_subtracting_additive_group_for_calculus_presentation, has_compactable_ln_abs_trig_sqrt,
    has_compactable_sqrt_hyperbolic_reciprocal_result,
};
use super::scalar_presentation::{
    fold_numeric_mul_constants_for_hold, fold_numeric_mul_constants_for_hold_additive_terms,
};
use super::sqrt_chain_integrand_preservation::sqrt_chain_integrand_preservation_gates;
use super::sqrt_denominator_result_presentation::{
    has_sqrt_denominator_result, inverse_sqrt_quotient_arg_result,
};
use super::sqrt_reciprocal_trig_antiderivative_presentation::sqrt_reciprocal_trig_antiderivative_result;
use super::trig_power_integrand_presentation::affine_trig_power_integrand_for_calculus_presentation;
use cas_ast::{Context, Expr, ExprId};

pub(super) struct IntegrationSourcePreservation {
    active: bool,
    pub(super) preserve_compact_fractional_denominator_power: bool,
    pub(super) preserve_compact_sqrt_trig_log: bool,
    pub(super) preserve_compact_sqrt_hyperbolic_log: bool,
    pub(super) preserve_compact_sqrt_hyperbolic_reciprocal_derivative: bool,
    pub(super) preserve_compact_rational_linear_partial_fraction: bool,
    pub(super) preserve_compact_log_by_parts: bool,
}

impl IntegrationSourcePreservation {
    pub(super) fn should_preserve_compact_result(&self) -> bool {
        self.active
    }
}

pub(super) fn integration_source_preservation_gates(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> IntegrationSourcePreservation {
    let preserve_compact_reciprocal = cas_math::symbolic_integration_support::integrate_symbolic_is_reciprocal_negative_power_denominator_quotient_target(
        ctx,
        target,
        var_name,
    );
    let preserve_compact_fractional_denominator_power =
        fractional_denominator_power_substitution_integrand_for_calculus_presentation(
            ctx, target, var_name,
        );
    let preserve_compact_arctan_integrand =
        arctan_integrand_for_calculus_presentation(ctx, target, var_name);
    let preserve_compact_atanh_polynomial = cas_math::symbolic_integration_support::integrate_symbolic_is_atanh_polynomial_substitution_target(
        ctx,
        target,
        var_name,
    );
    let preserve_compact_inverse_hyperbolic_affine =
        inverse_hyperbolic_affine_integrand_for_calculus_presentation(ctx, target, var_name);
    let preserve_compact_bounded_inverse_trig = cas_math::symbolic_integration_support::integrate_symbolic_is_bounded_inverse_trig_variable_target(
        ctx,
        target,
        var_name,
    );
    let preserve_compact_trig_polynomial = cas_math::symbolic_integration_support::integrate_symbolic_is_trig_polynomial_substitution_target(
        ctx,
        target,
        var_name,
    );
    let preserve_compact_affine_trig_power =
        affine_trig_power_integrand_for_calculus_presentation(ctx, target, var_name);
    let sqrt_chain_preservation = sqrt_chain_integrand_preservation_gates(ctx, target, var_name);
    let preserve_compact_inverse_hyperbolic_sqrt_reciprocal =
        cas_math::symbolic_integration_support::integrate_symbolic_is_inverse_hyperbolic_sqrt_reciprocal_target(
            ctx,
            target,
            var_name,
        );
    let preserve_compact_affine_sqrt_product_derivative =
        cas_math::symbolic_integration_support::integrate_symbolic_is_affine_sqrt_product_derivative_target(
            ctx,
            target,
            var_name,
        );
    let preserve_compact_arcsin_inverse_sqrt_product =
        cas_math::symbolic_integration_support::integrate_symbolic_is_arcsin_inverse_sqrt_product_target(
            ctx,
            target,
            var_name,
        );
    let preserve_compact_log_product_integrand =
        log_product_integrand_for_calculus_presentation(ctx, target, var_name);
    let preserve_compact_rational_linear_partial_fraction =
        rational_linear_partial_fraction_integrand_for_calculus_presentation(ctx, target, var_name);
    let preserve_compact_hyperbolic_power =
        hyperbolic_power_integrand_for_calculus_presentation(ctx, target, var_name);
    let by_parts_preservation = by_parts_integrand_preservation_gates(ctx, target, var_name);

    let active = preserve_compact_reciprocal
        || preserve_compact_fractional_denominator_power
        || preserve_compact_arctan_integrand
        || preserve_compact_atanh_polynomial
        || preserve_compact_inverse_hyperbolic_affine
        || preserve_compact_bounded_inverse_trig
        || preserve_compact_trig_polynomial
        || preserve_compact_affine_trig_power
        || sqrt_chain_preservation.should_preserve_compact_result()
        || preserve_compact_inverse_hyperbolic_sqrt_reciprocal
        || preserve_compact_affine_sqrt_product_derivative
        || preserve_compact_arcsin_inverse_sqrt_product
        || preserve_compact_log_product_integrand
        || preserve_compact_rational_linear_partial_fraction
        || preserve_compact_hyperbolic_power
        || by_parts_preservation.should_preserve_compact_result();

    IntegrationSourcePreservation {
        active,
        preserve_compact_fractional_denominator_power,
        preserve_compact_sqrt_trig_log: sqrt_chain_preservation.preserve_compact_sqrt_trig_log,
        preserve_compact_sqrt_hyperbolic_log: sqrt_chain_preservation
            .preserve_compact_sqrt_hyperbolic_log,
        preserve_compact_sqrt_hyperbolic_reciprocal_derivative: sqrt_chain_preservation
            .preserve_compact_sqrt_hyperbolic_reciprocal_derivative,
        preserve_compact_rational_linear_partial_fraction,
        preserve_compact_log_by_parts: by_parts_preservation.preserve_compact_log_by_parts,
    }
}

pub(super) fn apply_integration_result_preservation(
    ctx: &mut Context,
    mut result: ExprId,
    var_name: &str,
    has_required_positive_conditions: bool,
    source_preservation: &IntegrationSourcePreservation,
    compact_polynomial_arctan_by_parts_result: Option<ExprId>,
) -> ExprId {
    let preserve_compact_polynomial_arctan_by_parts_result =
        compact_polynomial_arctan_by_parts_result.is_some();
    let preserve_compact_inverse_sqrt_arg = inverse_sqrt_quotient_arg_result(ctx, result);
    let preserve_compact_sqrt_denominator_result = has_sqrt_denominator_result(ctx, result);
    let compact_negative_half_power_result =
        compact_negative_half_power_result_for_integration_presentation(ctx, result);
    let preserve_compact_negative_half_power_result = compact_negative_half_power_result.is_some();
    let compact_negative_three_half_power_result =
        compact_negative_three_half_power_result_for_integration_presentation(
            ctx,
            result,
            var_name,
            has_required_positive_conditions,
        );
    let preserve_compact_negative_three_half_power_result =
        compact_negative_three_half_power_result.is_some();
    let compact_positive_half_power_result =
        compact_positive_half_power_result_for_integration_presentation(ctx, result);
    let preserve_compact_positive_half_power_result = compact_positive_half_power_result.is_some();
    let preserve_compact_sqrt_reciprocal_trig_result =
        sqrt_reciprocal_trig_antiderivative_result(ctx, result, var_name);
    let preserve_compact_sqrt_trig_log_result =
        has_compactable_ln_abs_trig_sqrt(ctx, result, var_name);
    let preserve_compact_sqrt_hyperbolic_reciprocal_result =
        has_compactable_sqrt_hyperbolic_reciprocal_result(ctx, result, var_name);
    let compact_half_power_sum_root_product_result =
        compact_half_power_sum_root_product_for_integration_presentation(ctx, result, var_name);
    let preserve_compact_half_power_sum_root_product_result =
        compact_half_power_sum_root_product_result.is_some();
    if source_preservation.should_preserve_compact_result()
        || preserve_compact_polynomial_arctan_by_parts_result
        || preserve_compact_inverse_sqrt_arg
        || preserve_compact_sqrt_denominator_result
        || preserve_compact_negative_half_power_result
        || preserve_compact_negative_three_half_power_result
        || preserve_compact_positive_half_power_result
        || preserve_compact_sqrt_reciprocal_trig_result
        || preserve_compact_sqrt_trig_log_result
        || preserve_compact_sqrt_hyperbolic_reciprocal_result
        || preserve_compact_half_power_sum_root_product_result
    {
        if let Some(compact) = compact_half_power_sum_root_product_result {
            result = compact;
        }
        if let Some(compact) = compact_negative_half_power_result {
            result = compact;
        }
        if let Some(compact) = compact_negative_three_half_power_result {
            result = compact;
        }
        if let Some(compact) = compact_positive_half_power_result {
            result = compact;
        }
        if source_preservation.preserve_compact_sqrt_hyperbolic_log {
            result =
                compact_positive_cosh_log_abs_for_integration_presentation(ctx, result, var_name);
        }
        if source_preservation.preserve_compact_sqrt_trig_log
            || preserve_compact_sqrt_trig_log_result
        {
            result = compact_sqrt_trig_log_abs_for_integration_presentation(ctx, result, var_name);
        }
        if source_preservation.preserve_compact_sqrt_hyperbolic_reciprocal_derivative
            || preserve_compact_sqrt_hyperbolic_reciprocal_result
        {
            result = compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(
                ctx, result, var_name,
            );
        }
        if let Some(compact) = compact_polynomial_arctan_by_parts_result {
            result = compact;
        }
        result = if source_preservation.preserve_compact_rational_linear_partial_fraction {
            fold_numeric_mul_constants_for_hold_additive_terms(ctx, result)
        } else {
            fold_numeric_mul_constants_for_hold(ctx, result)
        };
        if preserve_compact_polynomial_arctan_by_parts_result {
            if let Some(compact) =
                compact_arctan_additive_terms_for_calculus_presentation(ctx, result, var_name)
            {
                result = compact;
            }
        }
        if source_preservation.preserve_compact_log_by_parts {
            if let Some(compact) =
                flatten_subtracting_additive_group_for_calculus_presentation(ctx, result, var_name)
            {
                result = compact;
            }
        }
        result = if source_preservation.preserve_compact_fractional_denominator_power {
            ctx.add(Expr::Hold(result))
        } else {
            cas_ast::hold::wrap_hold(ctx, result)
        };
    }
    result
}
