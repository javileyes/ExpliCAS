//! Verified held inverse-sqrt routes for integration-backed derivative shortcuts.
//!
//! This module owns route 12 from `integral_derivative_shortcut_presentation`.
//! Matched sources are verified by the integrator before returning a held
//! compact/source integrand.

use super::integral_derivative_shortcut_return_policy::verified_held_inverse_sqrt_compact_or_source_integrand_target;
use super::inverse_sqrt_product_integrand_preservation::affine_sqrt_product_derivative_integrand_for_calculus_presentation;
use cas_ast::{Context, ExprId};

pub(super) fn verified_held_inverse_sqrt_integral_derivative_shortcut(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if !(affine_sqrt_product_derivative_integrand_for_calculus_presentation(ctx, target, var_name)
        || cas_math::symbolic_integration_support::integrate_symbolic_is_acosh_polynomial_substitution_target(
            ctx, target, var_name,
        ))
    {
        return None;
    }

    verified_held_inverse_sqrt_compact_or_source_integrand_target(ctx, target, var_name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::hold;
    use cas_parser::parse;

    #[test]
    fn accepts_affine_sqrt_product_derivative_as_verified_held_integrand() {
        let mut ctx = Context::new();
        let target = parse("1/sqrt(x)", &mut ctx).unwrap();

        let held =
            verified_held_inverse_sqrt_integral_derivative_shortcut(&mut ctx, target, "x").unwrap();

        assert!(hold::is_hold(&ctx, held));
    }

    #[test]
    fn rejects_source_outside_verified_held_inverse_sqrt_group() {
        let mut ctx = Context::new();
        let target = parse("sin(x)", &mut ctx).unwrap();

        assert_eq!(
            verified_held_inverse_sqrt_integral_derivative_shortcut(&mut ctx, target, "x"),
            None
        );
    }
}
