//! Verified rational and substitution source-return routes.
//!
//! This module owns routes 7-10 from `integral_derivative_shortcut_presentation`.
//! Route 6 is owned by `integral_derivative_arctan_polynomial_routes`; this
//! module stays after it to preserve source-order priority.

use super::fractional_denominator_power_integrand_preservation::fractional_denominator_power_substitution_integrand_for_calculus_presentation;
use super::integration_antiderivative_verification::verified_integrand_target;
use cas_ast::{Context, ExprId};

pub(super) fn rational_substitution_verified_integral_derivative_shortcut(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if cas_math::symbolic_integration_support::integrate_symbolic_is_rational_linear_partial_fraction_target(
        ctx, target, var_name,
    ) {
        return verified_integrand_target(ctx, target, var_name);
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_rational_linear_positive_quadratic_target(
        ctx, target, var_name,
    ) {
        return verified_integrand_target(ctx, target, var_name);
    }

    if fractional_denominator_power_substitution_integrand_for_calculus_presentation(
        ctx, target, var_name,
    ) {
        return verified_integrand_target(ctx, target, var_name);
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_sqrt_derivative_substitution_target(
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
    fn accepts_verified_rational_partial_fraction_source() {
        let mut ctx = Context::new();
        let target = parse("1/((x+1)*(x+2))", &mut ctx).unwrap();

        assert_eq!(
            rational_substitution_verified_integral_derivative_shortcut(&mut ctx, target, "x"),
            Some(target)
        );
    }

    #[test]
    fn accepts_verified_fractional_denominator_power_source() {
        let mut ctx = Context::new();
        let target = parse("(2*x+1)/(x^2+x+1)^(3/2)", &mut ctx).unwrap();

        assert_eq!(
            rational_substitution_verified_integral_derivative_shortcut(&mut ctx, target, "x"),
            Some(target)
        );
    }

    #[test]
    fn rejects_source_outside_rational_substitution_group() {
        let mut ctx = Context::new();
        let target = parse("sin(x)", &mut ctx).unwrap();

        assert_eq!(
            rational_substitution_verified_integral_derivative_shortcut(&mut ctx, target, "x"),
            None
        );
    }
}
