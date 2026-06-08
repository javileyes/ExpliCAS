//! Direct trig-affine route for integration-backed derivative shortcuts.
//!
//! This route is the final verified fallback in the `diff(integrate(...), x)`
//! shortcut gate. It accepts only source integrands containing direct
//! sine/cosine calls with affine arguments and verifies the source integrand
//! through the symbolic integrator before returning it.

use super::direct_trig_affine_integrand_presentation::expr_contains_direct_trig_with_affine_arg;
use super::integration_antiderivative_verification::verified_source_route;
use cas_ast::{Context, ExprId};

pub(super) enum DirectTrigAffineIntegralDerivativeRoute {
    NoMatch,
    VerificationFailed,
    VerifiedSource(ExprId),
}

pub(super) fn direct_trig_affine_integral_derivative_route(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> DirectTrigAffineIntegralDerivativeRoute {
    if !direct_trig_affine_route_matches(ctx, target, var_name) {
        return DirectTrigAffineIntegralDerivativeRoute::NoMatch;
    }

    verified_source_route(
        ctx,
        target,
        var_name,
        DirectTrigAffineIntegralDerivativeRoute::VerifiedSource,
        DirectTrigAffineIntegralDerivativeRoute::VerificationFailed,
    )
}

fn direct_trig_affine_route_matches(ctx: &mut Context, target: ExprId, var_name: &str) -> bool {
    expr_contains_direct_trig_with_affine_arg(ctx, target, var_name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn accepts_verified_direct_trig_affine_source() {
        let mut ctx = Context::new();
        let target = parse("sin(2*x+1)", &mut ctx).unwrap();

        assert!(matches!(
            direct_trig_affine_integral_derivative_route(&mut ctx, target, "x"),
            DirectTrigAffineIntegralDerivativeRoute::VerifiedSource(source) if source == target
        ));
    }

    #[test]
    fn accepts_verified_reciprocal_trig_affine_source() {
        let mut ctx = Context::new();
        let target = parse("2/3*sec(2*x+1)", &mut ctx).unwrap();

        assert!(matches!(
            direct_trig_affine_integral_derivative_route(&mut ctx, target, "x"),
            DirectTrigAffineIntegralDerivativeRoute::VerifiedSource(source) if source == target
        ));
    }

    #[test]
    fn route_matcher_separates_detection_from_verification() {
        let mut ctx = Context::new();
        let target = parse("2/3*csc(2*x+1)", &mut ctx).unwrap();

        assert!(direct_trig_affine_route_matches(&mut ctx, target, "x"));
    }

    #[test]
    fn route_reports_no_match_for_direct_trig_nonlinear_source() {
        let mut ctx = Context::new();
        let target = parse("sin(x^2)", &mut ctx).unwrap();

        assert!(matches!(
            direct_trig_affine_integral_derivative_route(&mut ctx, target, "x"),
            DirectTrigAffineIntegralDerivativeRoute::NoMatch
        ));
    }

    #[test]
    fn route_reports_no_match_for_reciprocal_trig_nonlinear_source() {
        let mut ctx = Context::new();
        let target = parse("sec(x^2)", &mut ctx).unwrap();

        assert!(matches!(
            direct_trig_affine_integral_derivative_route(&mut ctx, target, "x"),
            DirectTrigAffineIntegralDerivativeRoute::NoMatch
        ));
    }

    #[test]
    fn route_reports_verification_failed_for_matched_unsupported_source() {
        let mut ctx = Context::new();
        let target = parse("sin(2*x+1)+exp(x^2)", &mut ctx).unwrap();

        assert!(matches!(
            direct_trig_affine_integral_derivative_route(&mut ctx, target, "x"),
            DirectTrigAffineIntegralDerivativeRoute::VerificationFailed
        ));
    }

    #[test]
    fn route_reports_verification_failed_for_reciprocal_trig_mixed_unsupported_source() {
        let mut ctx = Context::new();
        let target = parse("sec(2*x+1)+exp(x^2)", &mut ctx).unwrap();

        assert!(matches!(
            direct_trig_affine_integral_derivative_route(&mut ctx, target, "x"),
            DirectTrigAffineIntegralDerivativeRoute::VerificationFailed
        ));
    }
}
