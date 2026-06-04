//! Initial verified source-return routes for derivative-of-integral shortcuts.
//!
//! This module owns routes 1-5 from `integral_derivative_shortcut_presentation`.
//! It preserves their source order and returns only integrands that pass the
//! bounded antiderivative verification policy.

use super::integration_antiderivative_verification::verified_source_route;
use super::sqrt_chain_integrand_preservation::{
    sqrt_hyperbolic_log_integrand_for_calculus_presentation,
    sqrt_hyperbolic_reciprocal_derivative_integrand_for_calculus_presentation,
    sqrt_hyperbolic_reciprocal_square_integrand_for_calculus_presentation,
};
use cas_ast::{Context, ExprId};

pub(super) enum InitialVerifiedSourceIntegralDerivativeRoute {
    NoMatch,
    VerificationFailed,
    VerifiedSource(ExprId),
}

#[cfg(test)]
impl InitialVerifiedSourceIntegralDerivativeRoute {
    fn into_source_target(self) -> Option<ExprId> {
        match self {
            InitialVerifiedSourceIntegralDerivativeRoute::VerifiedSource(source) => Some(source),
            InitialVerifiedSourceIntegralDerivativeRoute::NoMatch
            | InitialVerifiedSourceIntegralDerivativeRoute::VerificationFailed => None,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum InitialVerifiedSourceRoute {
    NoMatch,
    ArctanSqrtReciprocal,
    ArctanSqrtUnitShiftSquare,
    SqrtHyperbolicLog,
    SqrtHyperbolicReciprocalSquare,
    SqrtHyperbolicReciprocalDerivative,
    QuadraticTimesAffineLnByParts,
}

impl InitialVerifiedSourceRoute {
    fn matches(self) -> bool {
        !matches!(self, InitialVerifiedSourceRoute::NoMatch)
    }
}

#[cfg(test)]
fn initial_verified_source_integral_derivative_shortcut(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    initial_verified_source_integral_derivative_route(ctx, target, var_name).into_source_target()
}

pub(super) fn initial_verified_source_integral_derivative_route(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> InitialVerifiedSourceIntegralDerivativeRoute {
    if !initial_verified_source_route_matches(ctx, target, var_name) {
        return InitialVerifiedSourceIntegralDerivativeRoute::NoMatch;
    }

    verified_source_route(
        ctx,
        target,
        var_name,
        InitialVerifiedSourceIntegralDerivativeRoute::VerifiedSource,
        InitialVerifiedSourceIntegralDerivativeRoute::VerificationFailed,
    )
}

fn initial_verified_source_route_matches(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    initial_verified_source_route(ctx, target, var_name).matches()
}

fn initial_verified_source_route(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> InitialVerifiedSourceRoute {
    if cas_math::symbolic_integration_support::integrate_symbolic_is_arctan_sqrt_var_reciprocal_target(
        ctx, target, var_name,
    ) {
        return InitialVerifiedSourceRoute::ArctanSqrtReciprocal;
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_arctan_sqrt_var_unit_shift_square_target(
        ctx, target, var_name,
    ) {
        return InitialVerifiedSourceRoute::ArctanSqrtUnitShiftSquare;
    }

    if sqrt_hyperbolic_log_integrand_for_calculus_presentation(ctx, target, var_name) {
        return InitialVerifiedSourceRoute::SqrtHyperbolicLog;
    }

    if sqrt_hyperbolic_reciprocal_square_integrand_for_calculus_presentation(ctx, target, var_name)
    {
        return InitialVerifiedSourceRoute::SqrtHyperbolicReciprocalSquare;
    }

    if sqrt_hyperbolic_reciprocal_derivative_integrand_for_calculus_presentation(
        ctx, target, var_name,
    ) {
        return InitialVerifiedSourceRoute::SqrtHyperbolicReciprocalDerivative;
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_quadratic_times_affine_ln_by_parts_target(
        ctx, target, var_name,
    ) {
        return InitialVerifiedSourceRoute::QuadraticTimesAffineLnByParts;
    }

    InitialVerifiedSourceRoute::NoMatch
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
        assert!(matches!(
            initial_verified_source_integral_derivative_route(&mut ctx, target, "x"),
            InitialVerifiedSourceIntegralDerivativeRoute::VerifiedSource(source) if source == target
        ));
    }

    #[test]
    fn route_matcher_separates_detection_from_verification() {
        let mut ctx = Context::new();
        let target = parse("tanh(sqrt(x))/(2*sqrt(x))", &mut ctx).unwrap();

        assert!(initial_verified_source_route_matches(&mut ctx, target, "x"));
    }

    #[test]
    fn rejects_source_outside_initial_verified_group() {
        let mut ctx = Context::new();
        let target = parse("sin(x)", &mut ctx).unwrap();

        assert_eq!(
            initial_verified_source_integral_derivative_shortcut(&mut ctx, target, "x"),
            None
        );
        assert!(matches!(
            initial_verified_source_integral_derivative_route(&mut ctx, target, "x"),
            InitialVerifiedSourceIntegralDerivativeRoute::NoMatch
        ));
    }

    #[test]
    fn route_classifier_preserves_initial_verified_source_order() {
        let mut ctx = Context::new();

        let cases = [
            (
                "1/(2*sqrt(x)*(x+1))",
                InitialVerifiedSourceRoute::ArctanSqrtReciprocal,
            ),
            (
                "1/(sqrt(x)*(x+1)^2)",
                InitialVerifiedSourceRoute::ArctanSqrtUnitShiftSquare,
            ),
            (
                "tanh(sqrt(x))/(2*sqrt(x))",
                InitialVerifiedSourceRoute::SqrtHyperbolicLog,
            ),
            (
                "1/(2*sqrt(x)*cosh(sqrt(x))^2)",
                InitialVerifiedSourceRoute::SqrtHyperbolicReciprocalSquare,
            ),
            (
                "sinh(sqrt(x))/(2*sqrt(x)*cosh(sqrt(x))^2)",
                InitialVerifiedSourceRoute::SqrtHyperbolicReciprocalDerivative,
            ),
            (
                "x^2*ln(2*x+1)",
                InitialVerifiedSourceRoute::QuadraticTimesAffineLnByParts,
            ),
        ];

        for (source, expected_route) in cases {
            let target = parse(source, &mut ctx).unwrap();

            assert_eq!(
                initial_verified_source_route(&mut ctx, target, "x"),
                expected_route,
                "unexpected route for {source}"
            );
        }
    }
}
