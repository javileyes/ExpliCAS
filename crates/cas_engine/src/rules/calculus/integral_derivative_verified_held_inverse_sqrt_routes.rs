//! Verified held inverse-sqrt routes for integration-backed derivative shortcuts.
//!
//! This module owns route 12 from `integral_derivative_shortcut_presentation`.
//! Matched sources are verified by the integrator before returning a held
//! compact/source integrand.

use super::integral_derivative_shortcut_return_policy::verified_held_inverse_sqrt_compact_or_source_route;
use super::inverse_sqrt_product_integrand_preservation::affine_sqrt_product_derivative_integrand_for_calculus_presentation;
use cas_ast::{Context, ExprId};

pub(super) enum VerifiedHeldInverseSqrtRoute {
    NoMatch,
    VerificationFailed,
    VerifiedHeldCompactOrSource(ExprId),
}

#[derive(Debug, PartialEq, Eq)]
enum VerifiedHeldInverseSqrtSourceRoute {
    NoMatch,
    AffineSqrtProductDerivative,
    AcoshPolynomialSubstitution,
}

impl VerifiedHeldInverseSqrtSourceRoute {
    fn matches(self) -> bool {
        !matches!(self, VerifiedHeldInverseSqrtSourceRoute::NoMatch)
    }
}

pub(super) fn verified_held_inverse_sqrt_integral_derivative_route(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> VerifiedHeldInverseSqrtRoute {
    if !verified_held_inverse_sqrt_route_matches(ctx, target, var_name) {
        return VerifiedHeldInverseSqrtRoute::NoMatch;
    }

    verified_held_inverse_sqrt_compact_or_source_route(
        ctx,
        target,
        var_name,
        VerifiedHeldInverseSqrtRoute::VerifiedHeldCompactOrSource,
        VerifiedHeldInverseSqrtRoute::VerificationFailed,
    )
}

fn verified_held_inverse_sqrt_route_matches(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    verified_held_inverse_sqrt_source_route(ctx, target, var_name).matches()
}

fn verified_held_inverse_sqrt_source_route(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> VerifiedHeldInverseSqrtSourceRoute {
    if affine_sqrt_product_derivative_integrand_for_calculus_presentation(ctx, target, var_name) {
        return VerifiedHeldInverseSqrtSourceRoute::AffineSqrtProductDerivative;
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_acosh_polynomial_substitution_target(
        ctx, target, var_name,
    ) {
        return VerifiedHeldInverseSqrtSourceRoute::AcoshPolynomialSubstitution;
    }

    VerifiedHeldInverseSqrtSourceRoute::NoMatch
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

        let route = verified_held_inverse_sqrt_integral_derivative_route(&mut ctx, target, "x");

        assert!(matches!(
            route,
            VerifiedHeldInverseSqrtRoute::VerifiedHeldCompactOrSource(held)
                if hold::is_hold(&ctx, held)
        ));
    }

    #[test]
    fn route_matcher_separates_detection_from_verification() {
        let mut ctx = Context::new();
        let target = parse("1/sqrt(x)", &mut ctx).unwrap();

        assert!(verified_held_inverse_sqrt_route_matches(
            &mut ctx, target, "x"
        ));
    }

    #[test]
    fn route_classifier_preserves_verified_held_inverse_sqrt_order() {
        let mut ctx = Context::new();

        let affine_target = parse("1/sqrt(x)", &mut ctx).unwrap();
        assert_eq!(
            verified_held_inverse_sqrt_source_route(&mut ctx, affine_target, "x"),
            VerifiedHeldInverseSqrtSourceRoute::AffineSqrtProductDerivative
        );

        let acosh_target = parse("2*x/sqrt(x^4-4)", &mut ctx).unwrap();
        assert_eq!(
            verified_held_inverse_sqrt_source_route(&mut ctx, acosh_target, "x"),
            VerifiedHeldInverseSqrtSourceRoute::AcoshPolynomialSubstitution
        );
    }

    #[test]
    fn route_reports_no_match_outside_verified_held_inverse_sqrt_group() {
        let mut ctx = Context::new();
        let target = parse("sin(x)", &mut ctx).unwrap();

        assert!(matches!(
            verified_held_inverse_sqrt_integral_derivative_route(&mut ctx, target, "x"),
            VerifiedHeldInverseSqrtRoute::NoMatch
        ));
    }
}
