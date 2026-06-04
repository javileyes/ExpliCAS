//! Verified rational and substitution source-return routes.
//!
//! This module owns routes 7-10 from `integral_derivative_shortcut_presentation`.
//! Route 6 is owned by `integral_derivative_arctan_polynomial_routes`; this
//! module stays after it to preserve source-order priority.

use super::fractional_denominator_power_integrand_preservation::fractional_denominator_power_substitution_integrand_for_calculus_presentation;
use super::integration_antiderivative_verification::verified_source_route;
use cas_ast::{Context, ExprId};

pub(super) enum RationalSubstitutionIntegralDerivativeRoute {
    NoMatch,
    VerificationFailed,
    VerifiedSource(ExprId),
}

#[cfg(test)]
impl RationalSubstitutionIntegralDerivativeRoute {
    fn into_source_target(self) -> Option<ExprId> {
        match self {
            RationalSubstitutionIntegralDerivativeRoute::VerifiedSource(source) => Some(source),
            RationalSubstitutionIntegralDerivativeRoute::NoMatch
            | RationalSubstitutionIntegralDerivativeRoute::VerificationFailed => None,
        }
    }
}

enum RationalSubstitutionSourceRoute {
    NoMatch,
    RationalLinearPartialFraction,
    RationalLinearPositiveQuadratic,
    FractionalDenominatorPowerSubstitution,
    SqrtDerivativeSubstitution,
}

impl RationalSubstitutionSourceRoute {
    fn matches(self) -> bool {
        !matches!(self, RationalSubstitutionSourceRoute::NoMatch)
    }
}

#[cfg(test)]
fn rational_substitution_verified_integral_derivative_shortcut(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    rational_substitution_verified_integral_derivative_route(ctx, target, var_name)
        .into_source_target()
}

pub(super) fn rational_substitution_verified_integral_derivative_route(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> RationalSubstitutionIntegralDerivativeRoute {
    if !rational_substitution_source_route_matches(ctx, target, var_name) {
        return RationalSubstitutionIntegralDerivativeRoute::NoMatch;
    }

    verified_source_route(
        ctx,
        target,
        var_name,
        RationalSubstitutionIntegralDerivativeRoute::VerifiedSource,
        RationalSubstitutionIntegralDerivativeRoute::VerificationFailed,
    )
}

fn rational_substitution_source_route_matches(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    rational_substitution_source_route(ctx, target, var_name).matches()
}

fn rational_substitution_source_route(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> RationalSubstitutionSourceRoute {
    if cas_math::symbolic_integration_support::integrate_symbolic_is_rational_linear_partial_fraction_target(
        ctx, target, var_name,
    ) {
        return RationalSubstitutionSourceRoute::RationalLinearPartialFraction;
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_rational_linear_positive_quadratic_target(
        ctx, target, var_name,
    ) {
        return RationalSubstitutionSourceRoute::RationalLinearPositiveQuadratic;
    }

    if fractional_denominator_power_substitution_integrand_for_calculus_presentation(
        ctx, target, var_name,
    ) {
        return RationalSubstitutionSourceRoute::FractionalDenominatorPowerSubstitution;
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_sqrt_derivative_substitution_target(
        ctx, target, var_name,
    ) {
        return RationalSubstitutionSourceRoute::SqrtDerivativeSubstitution;
    }

    RationalSubstitutionSourceRoute::NoMatch
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
        assert!(matches!(
            rational_substitution_verified_integral_derivative_route(&mut ctx, target, "x"),
            RationalSubstitutionIntegralDerivativeRoute::NoMatch
        ));
    }

    #[test]
    fn route_matcher_separates_detection_from_verification() {
        let mut ctx = Context::new();
        let target = parse("(2*x+1)/(x^2+x+1)^(3/2)", &mut ctx).unwrap();

        assert!(rational_substitution_source_route_matches(
            &mut ctx, target, "x"
        ));
        assert!(matches!(
            rational_substitution_verified_integral_derivative_route(&mut ctx, target, "x"),
            RationalSubstitutionIntegralDerivativeRoute::VerifiedSource(source) if source == target
        ));
    }

    #[test]
    fn route_classifier_preserves_rational_substitution_family_order() {
        let mut ctx = Context::new();
        let partial_fraction = parse("1/((x+1)*(x+2))", &mut ctx).unwrap();
        let positive_quadratic = parse("1/(x^4-1)", &mut ctx).unwrap();
        let fractional_power = parse("(2*x+1)/(x^2+x+1)^(3/2)", &mut ctx).unwrap();
        let sqrt_derivative = parse("x/sqrt(x^2+1)", &mut ctx).unwrap();

        assert!(matches!(
            rational_substitution_source_route(&mut ctx, partial_fraction, "x"),
            RationalSubstitutionSourceRoute::RationalLinearPartialFraction
        ));
        assert!(matches!(
            rational_substitution_source_route(&mut ctx, positive_quadratic, "x"),
            RationalSubstitutionSourceRoute::RationalLinearPositiveQuadratic
        ));
        assert!(matches!(
            rational_substitution_source_route(&mut ctx, fractional_power, "x"),
            RationalSubstitutionSourceRoute::FractionalDenominatorPowerSubstitution
        ));
        assert!(matches!(
            rational_substitution_source_route(&mut ctx, sqrt_derivative, "x"),
            RationalSubstitutionSourceRoute::SqrtDerivativeSubstitution
        ));
    }
}
