//! Final source/fallback routes for integration-backed derivative shortcuts.
//!
//! This module owns routes 18-19 from
//! `integral_derivative_shortcut_presentation`. It stays after the compact-only
//! sqrt-trig-log routes and preserves their source-order priority.

use super::integral_derivative_direct_trig_affine_routes::{
    direct_trig_affine_integral_derivative_route, DirectTrigAffineIntegralDerivativeRoute,
};
use super::integral_derivative_sqrt_reciprocal_trig_routes::sqrt_reciprocal_trig_product_integral_derivative_route;
use cas_ast::{Context, ExprId};

pub(super) enum FinalSourceIntegralDerivativeRoute {
    NoMatch,
    SqrtReciprocalTrigProduct(ExprId),
    DirectTrigAffine(ExprId),
    DirectTrigAffineVerificationFailed,
}

#[cfg(test)]
impl FinalSourceIntegralDerivativeRoute {
    fn into_source_target(self) -> Option<ExprId> {
        match self {
            FinalSourceIntegralDerivativeRoute::SqrtReciprocalTrigProduct(source_target)
            | FinalSourceIntegralDerivativeRoute::DirectTrigAffine(source_target) => {
                Some(source_target)
            }
            FinalSourceIntegralDerivativeRoute::NoMatch
            | FinalSourceIntegralDerivativeRoute::DirectTrigAffineVerificationFailed => None,
        }
    }
}

pub(super) fn final_source_integral_derivative_route(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> FinalSourceIntegralDerivativeRoute {
    if let Some(source_target) =
        sqrt_reciprocal_trig_product_integral_derivative_route(ctx, target, var_name)
            .into_source_target()
    {
        return FinalSourceIntegralDerivativeRoute::SqrtReciprocalTrigProduct(source_target);
    }

    match direct_trig_affine_integral_derivative_route(ctx, target, var_name) {
        DirectTrigAffineIntegralDerivativeRoute::VerifiedSource(source_target) => {
            FinalSourceIntegralDerivativeRoute::DirectTrigAffine(source_target)
        }
        DirectTrigAffineIntegralDerivativeRoute::VerificationFailed => {
            FinalSourceIntegralDerivativeRoute::DirectTrigAffineVerificationFailed
        }
        DirectTrigAffineIntegralDerivativeRoute::NoMatch => {
            FinalSourceIntegralDerivativeRoute::NoMatch
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn returns_source_for_sqrt_reciprocal_trig_product_route() {
        let mut ctx = Context::new();
        let target = parse("sec(sqrt(x))*tan(sqrt(x))/(2*sqrt(x))", &mut ctx).unwrap();

        assert!(matches!(
            final_source_integral_derivative_route(&mut ctx, target, "x"),
            FinalSourceIntegralDerivativeRoute::SqrtReciprocalTrigProduct(source)
                if source == target
        ));
        assert_eq!(
            final_source_integral_derivative_route(&mut ctx, target, "x").into_source_target(),
            Some(target)
        );
    }

    #[test]
    fn returns_source_for_direct_trig_affine_fallback() {
        let mut ctx = Context::new();
        let target = parse("sin(2*x+1)", &mut ctx).unwrap();

        assert!(matches!(
            final_source_integral_derivative_route(&mut ctx, target, "x"),
            FinalSourceIntegralDerivativeRoute::DirectTrigAffine(source) if source == target
        ));
        assert_eq!(
            final_source_integral_derivative_route(&mut ctx, target, "x").into_source_target(),
            Some(target)
        );
    }

    #[test]
    fn rejects_expression_outside_final_source_routes() {
        let mut ctx = Context::new();
        let target = parse("exp(x^2)", &mut ctx).unwrap();

        assert_eq!(
            final_source_integral_derivative_route(&mut ctx, target, "x").into_source_target(),
            None
        );
    }

    #[test]
    fn rejects_matched_but_unverified_direct_trig_affine_fallback() {
        let mut ctx = Context::new();
        let target = parse("sin(2*x+1)+exp(x^2)", &mut ctx).unwrap();

        assert!(matches!(
            final_source_integral_derivative_route(&mut ctx, target, "x"),
            FinalSourceIntegralDerivativeRoute::DirectTrigAffineVerificationFailed
        ));
        assert_eq!(
            final_source_integral_derivative_route(&mut ctx, target, "x").into_source_target(),
            None
        );
    }
}
