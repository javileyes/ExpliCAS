//! Source-side presentation shortcuts for derivatives of supported integrals.
//!
//! This module owns the bounded `diff(integrate(...), x)` shortcut gate. It
//! preserves the route order from `calculus/mod.rs`: each accepted integrand is
//! either verified by the symbolic integrator before being returned, or is an
//! explicitly recognized source-side presentation case whose required
//! conditions are still collected by the caller.

use super::arctan_polynomial_integrand_presentation::polynomial_times_arctan_affine_integrand_for_diff_shortcut;
use super::direct_trig_affine_integrand_presentation::expr_contains_direct_trig_with_affine_arg;
use super::fractional_denominator_power_integrand_preservation::fractional_denominator_power_substitution_integrand_for_calculus_presentation;
use super::integration::{integrate, IntegrationRequiredConditions};
use super::inverse_sqrt_product_integrand_presentation::compact_inverse_sqrt_product_integrand_for_calculus_presentation;
use super::sqrt_reciprocal_trig_product_integrand_presentation::sqrt_reciprocal_trig_product_integrand_target;
use super::sqrt_trig_log_integrand_presentation::{
    compact_direct_sqrt_trig_log_derivative_integrand, compact_sqrt_trig_log_derivative_integrand,
};
use crate::symbolic_calculus_call_support::try_extract_integrate_call;
use cas_ast::{Context, ExprId};

fn verify_source_integrand_with_integrator(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<()> {
    integrate(ctx, target, var_name)?;
    Some(())
}

fn verified_source_integrand_target(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    verify_source_integrand_with_integrator(ctx, target, var_name)?;
    Some(target)
}

pub(super) fn supported_integral_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let integrate_call = try_extract_integrate_call(ctx, target)?;
    if integrate_call.var_name != var_name {
        return None;
    }
    if cas_math::symbolic_integration_support::integrate_symbolic_is_arctan_sqrt_var_reciprocal_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_arctan_sqrt_var_unit_shift_square_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        return verified_source_integrand_target(ctx, integrate_call.target, &integrate_call.var_name);
    }

    let supported_sqrt_chain_log_target =
        cas_math::symbolic_integration_support::integrate_symbolic_is_sqrt_hyperbolic_log_derivative_target(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
    );
    if supported_sqrt_chain_log_target {
        return verified_source_integrand_target(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        );
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_sqrt_hyperbolic_reciprocal_square_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        return verified_source_integrand_target(ctx, integrate_call.target, &integrate_call.var_name);
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_sqrt_hyperbolic_reciprocal_derivative_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        return verified_source_integrand_target(ctx, integrate_call.target, &integrate_call.var_name);
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_quadratic_times_affine_ln_by_parts_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        return verified_source_integrand_target(ctx, integrate_call.target, &integrate_call.var_name);
    }

    if polynomial_times_arctan_affine_integrand_for_diff_shortcut(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        return Some(integrate_call.target);
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_rational_linear_partial_fraction_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        return verified_source_integrand_target(ctx, integrate_call.target, &integrate_call.var_name);
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_rational_linear_positive_quadratic_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        return verified_source_integrand_target(ctx, integrate_call.target, &integrate_call.var_name);
    }

    if fractional_denominator_power_substitution_integrand_for_calculus_presentation(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        return verified_source_integrand_target(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        );
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_sqrt_derivative_substitution_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        return verified_source_integrand_target(ctx, integrate_call.target, &integrate_call.var_name);
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_arcsin_inverse_sqrt_product_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        if let Some(compact) =
            compact_inverse_sqrt_product_integrand_for_calculus_presentation(
                ctx,
                integrate_call.target,
            )
        {
            return Some(cas_ast::hold::wrap_hold(ctx, compact));
        }
        return Some(cas_ast::hold::wrap_hold(ctx, integrate_call.target));
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_affine_sqrt_product_derivative_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_acosh_polynomial_substitution_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        verify_source_integrand_with_integrator(ctx, integrate_call.target, &integrate_call.var_name)?;
        if let Some(compact) =
            compact_inverse_sqrt_product_integrand_for_calculus_presentation(
                ctx,
                integrate_call.target,
            )
        {
            return Some(cas_ast::hold::wrap_hold(ctx, compact));
        }
        return Some(cas_ast::hold::wrap_hold(ctx, integrate_call.target));
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_arcsin_polynomial_substitution_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_asinh_polynomial_substitution_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        verify_source_integrand_with_integrator(ctx, integrate_call.target, &integrate_call.var_name)?;
        return Some(cas_ast::hold::wrap_hold(ctx, integrate_call.target));
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_positive_quadratic_square_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_positive_quadratic_cube_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        return verified_source_integrand_target(ctx, integrate_call.target, &integrate_call.var_name);
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_inverse_hyperbolic_sqrt_reciprocal_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        return verified_source_integrand_target(ctx, integrate_call.target, &integrate_call.var_name);
    }

    if let Some(compact) = compact_direct_sqrt_trig_log_derivative_integrand(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        verify_source_integrand_with_integrator(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        )?;
        return Some(compact);
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_sqrt_trig_log_derivative_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        verify_source_integrand_with_integrator(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        )?;
        if let Some(compact) = compact_sqrt_trig_log_derivative_integrand(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        ) {
            return Some(compact);
        }
    }

    let supported_compact_target = sqrt_reciprocal_trig_product_integrand_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    );
    if supported_compact_target {
        return Some(integrate_call.target);
    }

    if expr_contains_direct_trig_with_affine_arg(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        verify_source_integrand_with_integrator(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        )?;
        return Some(integrate_call.target);
    }

    None
}

pub(super) fn supported_integral_diff_shortcut_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let integrate_call = try_extract_integrate_call(ctx, target)?;
    if integrate_call.var_name != var_name {
        return None;
    }

    let compact = supported_integral_derivative_presentation(ctx, target, var_name)?;
    let required_conditions = IntegrationRequiredConditions::from_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    )
    .into_implicit_conditions()
    .collect();

    Some((cas_ast::hold::wrap_hold(ctx, compact), required_conditions))
}
