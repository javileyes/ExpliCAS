//! Verified held polynomial-substitution routes for integration-backed derivative shortcuts.
//!
//! This module owns route 13 from `integral_derivative_shortcut_presentation`.
//! Matched polynomial-substitution sources are verified by the integrator before
//! returning the original source integrand held.

use super::integral_derivative_shortcut_return_policy::verified_held_source_integrand_target;
use cas_ast::{Context, ExprId};

pub(super) fn verified_held_polynomial_substitution_integral_derivative_shortcut(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if !(cas_math::symbolic_integration_support::integrate_symbolic_is_arcsin_polynomial_substitution_target(
        ctx, target, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_asinh_polynomial_substitution_target(
        ctx, target, var_name,
    )) {
        return None;
    }

    verified_held_source_integrand_target(ctx, target, var_name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::hold;
    use cas_parser::parse;

    #[test]
    fn accepts_arcsin_polynomial_substitution_as_verified_held_source_integrand() {
        let mut ctx = Context::new();
        let target = parse("2*x/sqrt(1-x^4)", &mut ctx).unwrap();

        let held = verified_held_polynomial_substitution_integral_derivative_shortcut(
            &mut ctx, target, "x",
        )
        .unwrap();

        assert!(hold::is_hold(&ctx, held));
        assert_eq!(hold::unwrap_internal_hold(&ctx, held), target);
    }

    #[test]
    fn accepts_asinh_polynomial_substitution_as_verified_held_source_integrand() {
        let mut ctx = Context::new();
        let target = parse("2*x/sqrt(1+x^4)", &mut ctx).unwrap();

        let held = verified_held_polynomial_substitution_integral_derivative_shortcut(
            &mut ctx, target, "x",
        )
        .unwrap();

        assert!(hold::is_hold(&ctx, held));
        assert_eq!(hold::unwrap_internal_hold(&ctx, held), target);
    }

    #[test]
    fn rejects_source_outside_verified_held_polynomial_substitution_group() {
        let mut ctx = Context::new();
        let target = parse("sin(x)", &mut ctx).unwrap();

        assert_eq!(
            verified_held_polynomial_substitution_integral_derivative_shortcut(
                &mut ctx, target, "x",
            ),
            None
        );
    }
}
