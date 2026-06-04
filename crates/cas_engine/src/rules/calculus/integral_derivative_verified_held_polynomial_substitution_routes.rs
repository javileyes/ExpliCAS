//! Verified held polynomial-substitution routes for integration-backed derivative shortcuts.
//!
//! This module owns route 13 from `integral_derivative_shortcut_presentation`.
//! Matched polynomial-substitution sources are verified by the integrator before
//! returning the original source integrand held.

use super::integral_derivative_shortcut_return_policy::verified_held_source_route;
use cas_ast::{Context, ExprId};

pub(super) enum VerifiedHeldPolynomialSubstitutionRoute {
    NoMatch,
    VerificationFailed,
    VerifiedHeldSource(ExprId),
}

#[derive(Debug, PartialEq, Eq)]
enum VerifiedHeldPolynomialSubstitutionSourceRoute {
    NoMatch,
    ArcsinPolynomialSubstitution,
    AsinhPolynomialSubstitution,
}

impl VerifiedHeldPolynomialSubstitutionSourceRoute {
    fn matches(self) -> bool {
        !matches!(self, VerifiedHeldPolynomialSubstitutionSourceRoute::NoMatch)
    }
}

pub(super) fn verified_held_polynomial_substitution_integral_derivative_route(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> VerifiedHeldPolynomialSubstitutionRoute {
    if !verified_held_polynomial_substitution_route_matches(ctx, target, var_name) {
        return VerifiedHeldPolynomialSubstitutionRoute::NoMatch;
    }

    verified_held_source_route(
        ctx,
        target,
        var_name,
        VerifiedHeldPolynomialSubstitutionRoute::VerifiedHeldSource,
        VerifiedHeldPolynomialSubstitutionRoute::VerificationFailed,
    )
}

fn verified_held_polynomial_substitution_route_matches(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    verified_held_polynomial_substitution_source_route(ctx, target, var_name).matches()
}

fn verified_held_polynomial_substitution_source_route(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> VerifiedHeldPolynomialSubstitutionSourceRoute {
    if cas_math::symbolic_integration_support::integrate_symbolic_is_arcsin_polynomial_substitution_target(
        ctx, target, var_name,
    ) {
        return VerifiedHeldPolynomialSubstitutionSourceRoute::ArcsinPolynomialSubstitution;
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_asinh_polynomial_substitution_target(
        ctx, target, var_name,
    ) {
        return VerifiedHeldPolynomialSubstitutionSourceRoute::AsinhPolynomialSubstitution;
    }

    VerifiedHeldPolynomialSubstitutionSourceRoute::NoMatch
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

        let route =
            verified_held_polynomial_substitution_integral_derivative_route(&mut ctx, target, "x");

        assert!(matches!(
            route,
            VerifiedHeldPolynomialSubstitutionRoute::VerifiedHeldSource(held)
                if hold::is_hold(&ctx, held)
                    && hold::unwrap_internal_hold(&ctx, held) == target
        ));
    }

    #[test]
    fn route_matcher_separates_detection_from_verification() {
        let mut ctx = Context::new();
        let target = parse("2*x/sqrt(1-x^4)", &mut ctx).unwrap();

        assert!(verified_held_polynomial_substitution_route_matches(
            &mut ctx, target, "x"
        ));
    }

    #[test]
    fn accepts_asinh_polynomial_substitution_as_verified_held_source_integrand() {
        let mut ctx = Context::new();
        let target = parse("2*x/sqrt(1+x^4)", &mut ctx).unwrap();

        let route =
            verified_held_polynomial_substitution_integral_derivative_route(&mut ctx, target, "x");

        assert!(matches!(
            route,
            VerifiedHeldPolynomialSubstitutionRoute::VerifiedHeldSource(held)
                if hold::is_hold(&ctx, held)
                    && hold::unwrap_internal_hold(&ctx, held) == target
        ));
    }

    #[test]
    fn route_classifier_preserves_verified_held_polynomial_substitution_order() {
        let mut ctx = Context::new();

        let arcsin_target = parse("2*x/sqrt(1-x^4)", &mut ctx).unwrap();
        assert_eq!(
            verified_held_polynomial_substitution_source_route(&mut ctx, arcsin_target, "x"),
            VerifiedHeldPolynomialSubstitutionSourceRoute::ArcsinPolynomialSubstitution
        );

        let asinh_target = parse("2*x/sqrt(1+x^4)", &mut ctx).unwrap();
        assert_eq!(
            verified_held_polynomial_substitution_source_route(&mut ctx, asinh_target, "x"),
            VerifiedHeldPolynomialSubstitutionSourceRoute::AsinhPolynomialSubstitution
        );
    }

    #[test]
    fn route_reports_no_match_outside_verified_held_polynomial_substitution_group() {
        let mut ctx = Context::new();
        let target = parse("sin(x)", &mut ctx).unwrap();

        assert!(matches!(
            verified_held_polynomial_substitution_integral_derivative_route(&mut ctx, target, "x",),
            VerifiedHeldPolynomialSubstitutionRoute::NoMatch
        ));
    }
}
