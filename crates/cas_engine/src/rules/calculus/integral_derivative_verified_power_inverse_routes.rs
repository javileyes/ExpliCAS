//! Late verified source-return routes for integration-backed derivative shortcuts.
//!
//! This module owns routes 14-15 from `integral_derivative_shortcut_presentation`.
//! They stay after held-presentation routes and before compact-only presentation
//! routes, preserving the source-order priority of the shortcut gate.

use super::integration_antiderivative_verification::verified_source_route;
use cas_ast::{Context, ExprId};

pub(super) enum PowerInverseIntegralDerivativeRoute {
    NoMatch,
    VerificationFailed,
    VerifiedSource(ExprId),
}

#[cfg(test)]
impl PowerInverseIntegralDerivativeRoute {
    fn into_source_target(self) -> Option<ExprId> {
        match self {
            PowerInverseIntegralDerivativeRoute::VerifiedSource(source) => Some(source),
            PowerInverseIntegralDerivativeRoute::NoMatch
            | PowerInverseIntegralDerivativeRoute::VerificationFailed => None,
        }
    }
}

enum PowerInverseSourceRoute {
    NoMatch,
    PositiveQuadraticSquare,
    PositiveQuadraticCube,
    InverseHyperbolicSqrtReciprocal,
}

impl PowerInverseSourceRoute {
    fn matches(self) -> bool {
        !matches!(self, PowerInverseSourceRoute::NoMatch)
    }
}

#[cfg(test)]
fn power_inverse_verified_integral_derivative_shortcut(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    power_inverse_verified_integral_derivative_route(ctx, target, var_name).into_source_target()
}

pub(super) fn power_inverse_verified_integral_derivative_route(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> PowerInverseIntegralDerivativeRoute {
    if !power_inverse_source_route_matches(ctx, target, var_name) {
        return PowerInverseIntegralDerivativeRoute::NoMatch;
    }

    verified_source_route(
        ctx,
        target,
        var_name,
        PowerInverseIntegralDerivativeRoute::VerifiedSource,
        PowerInverseIntegralDerivativeRoute::VerificationFailed,
    )
}

fn power_inverse_source_route_matches(ctx: &mut Context, target: ExprId, var_name: &str) -> bool {
    power_inverse_source_route(ctx, target, var_name).matches()
}

fn power_inverse_source_route(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> PowerInverseSourceRoute {
    if cas_math::symbolic_integration_support::integrate_symbolic_is_positive_quadratic_square_target(
        ctx, target, var_name,
    ) {
        return PowerInverseSourceRoute::PositiveQuadraticSquare;
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_positive_quadratic_cube_target(
        ctx, target, var_name,
    ) {
        return PowerInverseSourceRoute::PositiveQuadraticCube;
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_inverse_hyperbolic_sqrt_reciprocal_target(
        ctx, target, var_name,
    ) {
        return PowerInverseSourceRoute::InverseHyperbolicSqrtReciprocal;
    }

    PowerInverseSourceRoute::NoMatch
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
    fn route_matcher_separates_detection_from_verification() {
        let mut ctx = Context::new();
        let target = parse("1/(x^2+1)^2", &mut ctx).unwrap();

        assert!(power_inverse_source_route_matches(&mut ctx, target, "x"));
        assert!(matches!(
            power_inverse_verified_integral_derivative_route(&mut ctx, target, "x"),
            PowerInverseIntegralDerivativeRoute::VerifiedSource(source) if source == target
        ));
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
        assert!(matches!(
            power_inverse_verified_integral_derivative_route(&mut ctx, target, "x"),
            PowerInverseIntegralDerivativeRoute::NoMatch
        ));
    }

    #[test]
    fn route_classifier_preserves_power_inverse_family_order() {
        let mut ctx = Context::new();
        let square = parse("1/(x^2+1)^2", &mut ctx).unwrap();
        let cube = parse("1/(x^2+1)^3", &mut ctx).unwrap();
        let inverse_hyperbolic = parse("-1/(2*x*sqrt(x+1))", &mut ctx).unwrap();

        assert!(matches!(
            power_inverse_source_route(&mut ctx, square, "x"),
            PowerInverseSourceRoute::PositiveQuadraticSquare
        ));
        assert!(matches!(
            power_inverse_source_route(&mut ctx, cube, "x"),
            PowerInverseSourceRoute::PositiveQuadraticCube
        ));
        assert!(matches!(
            power_inverse_source_route(&mut ctx, inverse_hyperbolic, "x"),
            PowerInverseSourceRoute::InverseHyperbolicSqrtReciprocal
        ));
    }
}
