//! Initial verified source-return routes for derivative-of-integral shortcuts.
//!
//! This module owns routes 1-5 from `integral_derivative_shortcut_presentation`.
//! It preserves their source order and returns only integrands that pass the
//! bounded antiderivative verification policy.

use super::integration_antiderivative_verification::verified_integrand_target;
use super::sqrt_chain_integrand_preservation::{
    sqrt_hyperbolic_log_integrand_for_calculus_presentation,
    sqrt_hyperbolic_reciprocal_derivative_integrand_for_calculus_presentation,
    sqrt_hyperbolic_reciprocal_square_integrand_for_calculus_presentation,
};
use cas_ast::{Context, ExprId};

pub(super) fn initial_verified_source_integral_derivative_shortcut(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if cas_math::symbolic_integration_support::integrate_symbolic_is_arctan_sqrt_var_reciprocal_target(
        ctx, target, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_arctan_sqrt_var_unit_shift_square_target(
        ctx, target, var_name,
    ) {
        return verified_integrand_target(ctx, target, var_name);
    }

    if sqrt_hyperbolic_log_integrand_for_calculus_presentation(ctx, target, var_name) {
        return verified_integrand_target(ctx, target, var_name);
    }

    if sqrt_hyperbolic_reciprocal_square_integrand_for_calculus_presentation(ctx, target, var_name)
    {
        return verified_integrand_target(ctx, target, var_name);
    }

    if sqrt_hyperbolic_reciprocal_derivative_integrand_for_calculus_presentation(
        ctx, target, var_name,
    ) {
        return verified_integrand_target(ctx, target, var_name);
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_quadratic_times_affine_ln_by_parts_target(
        ctx, target, var_name,
    ) {
        return verified_integrand_target(ctx, target, var_name);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn accepts_verified_sqrt_hyperbolic_log_source() {
        let mut ctx = Context::new();
        let target = parse("tanh(sqrt(x))/(2*sqrt(x))", &mut ctx).unwrap();

        assert_eq!(
            initial_verified_source_integral_derivative_shortcut(&mut ctx, target, "x"),
            Some(target)
        );
    }

    #[test]
    fn rejects_source_outside_initial_verified_group() {
        let mut ctx = Context::new();
        let target = parse("sin(x)", &mut ctx).unwrap();

        assert_eq!(
            initial_verified_source_integral_derivative_shortcut(&mut ctx, target, "x"),
            None
        );
    }
}
