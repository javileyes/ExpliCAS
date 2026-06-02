//! Late verified source-return routes for integration-backed derivative shortcuts.
//!
//! This module owns routes 14-15 from `integral_derivative_shortcut_presentation`.
//! They stay after held-presentation routes and before compact-only presentation
//! routes, preserving the source-order priority of the shortcut gate.

use super::integration_antiderivative_verification::verified_integrand_target;
use cas_ast::{Context, ExprId};

pub(super) fn power_inverse_verified_integral_derivative_shortcut(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if cas_math::symbolic_integration_support::integrate_symbolic_is_positive_quadratic_square_target(
        ctx, target, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_positive_quadratic_cube_target(
        ctx, target, var_name,
    ) {
        return verified_integrand_target(ctx, target, var_name);
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_inverse_hyperbolic_sqrt_reciprocal_target(
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
    fn accepts_positive_quadratic_square_source() {
        let mut ctx = Context::new();
        let target = parse("1/(x^2+1)^2", &mut ctx).unwrap();

        assert_eq!(
            power_inverse_verified_integral_derivative_shortcut(&mut ctx, target, "x"),
            Some(target)
        );
    }

    #[test]
    fn accepts_positive_quadratic_cube_source() {
        let mut ctx = Context::new();
        let target = parse("1/(x^2+1)^3", &mut ctx).unwrap();

        assert_eq!(
            power_inverse_verified_integral_derivative_shortcut(&mut ctx, target, "x"),
            Some(target)
        );
    }

    #[test]
    fn accepts_inverse_hyperbolic_sqrt_reciprocal_source() {
        let mut ctx = Context::new();
        let target = parse("-1/(2*x*sqrt(x+1))", &mut ctx).unwrap();

        assert_eq!(
            power_inverse_verified_integral_derivative_shortcut(&mut ctx, target, "x"),
            Some(target)
        );
    }

    #[test]
    fn rejects_source_outside_power_inverse_group() {
        let mut ctx = Context::new();
        let target = parse("sin(x)", &mut ctx).unwrap();

        assert_eq!(
            power_inverse_verified_integral_derivative_shortcut(&mut ctx, target, "x"),
            None
        );
    }
}
