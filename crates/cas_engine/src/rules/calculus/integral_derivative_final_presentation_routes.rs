//! Final presentation routes for integration-backed derivative shortcuts.
//!
//! This module owns routes 16-17 from `integral_derivative_shortcut_presentation`
//! and delegates routes 18-19 to `integral_derivative_final_source_routes`. It
//! preserves the parent gate's final route order, including the early abort on
//! sqrt-trig-log compact verifier failure.

use super::integral_derivative_final_source_routes::{
    final_source_integral_derivative_route, FinalSourceIntegralDerivativeRoute,
};
use super::integral_derivative_sqrt_trig_log_routes::{
    sqrt_trig_log_integral_derivative_shortcut, SqrtTrigLogPresentationDecision,
};
use cas_ast::{Context, ExprId};

pub(super) enum FinalPresentationIntegralDerivativeRoute {
    NoMatch,
    SqrtTrigLogCompact(ExprId),
    SqrtTrigLogAbortFallback,
    FinalSourceSqrtReciprocalTrigProduct(ExprId),
    FinalSourceDirectTrigAffine(ExprId),
    FinalSourceDirectTrigAffineVerificationFailed,
}

impl FinalPresentationIntegralDerivativeRoute {
    fn from_final_source_route(
        route: FinalSourceIntegralDerivativeRoute,
    ) -> FinalPresentationIntegralDerivativeRoute {
        match route {
            FinalSourceIntegralDerivativeRoute::SqrtReciprocalTrigProduct(source_target) => {
                FinalPresentationIntegralDerivativeRoute::FinalSourceSqrtReciprocalTrigProduct(
                    source_target,
                )
            }
            FinalSourceIntegralDerivativeRoute::DirectTrigAffine(source_target) => {
                FinalPresentationIntegralDerivativeRoute::FinalSourceDirectTrigAffine(source_target)
            }
            FinalSourceIntegralDerivativeRoute::DirectTrigAffineVerificationFailed => {
                FinalPresentationIntegralDerivativeRoute::FinalSourceDirectTrigAffineVerificationFailed
            }
            FinalSourceIntegralDerivativeRoute::NoMatch => {
                FinalPresentationIntegralDerivativeRoute::NoMatch
            }
        }
    }
}

#[cfg(test)]
impl FinalPresentationIntegralDerivativeRoute {
    fn into_presentation_target(self) -> Option<ExprId> {
        match self {
            FinalPresentationIntegralDerivativeRoute::SqrtTrigLogCompact(compact)
            | FinalPresentationIntegralDerivativeRoute::FinalSourceSqrtReciprocalTrigProduct(
                compact,
            )
            | FinalPresentationIntegralDerivativeRoute::FinalSourceDirectTrigAffine(compact) => {
                Some(compact)
            }
            FinalPresentationIntegralDerivativeRoute::NoMatch
            | FinalPresentationIntegralDerivativeRoute::SqrtTrigLogAbortFallback
            | FinalPresentationIntegralDerivativeRoute::FinalSourceDirectTrigAffineVerificationFailed => None,
        }
    }
}

#[cfg(test)]
fn final_presentation_integral_derivative_shortcut(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    final_presentation_integral_derivative_route(ctx, target, var_name).into_presentation_target()
}

pub(super) fn final_presentation_integral_derivative_route(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> FinalPresentationIntegralDerivativeRoute {
    match sqrt_trig_log_integral_derivative_shortcut(ctx, target, var_name)
        .into_final_presentation_decision()
    {
        SqrtTrigLogPresentationDecision::AbortFallback => {
            return FinalPresentationIntegralDerivativeRoute::SqrtTrigLogAbortFallback;
        }
        SqrtTrigLogPresentationDecision::Compact(compact) => {
            return FinalPresentationIntegralDerivativeRoute::SqrtTrigLogCompact(compact);
        }
        SqrtTrigLogPresentationDecision::Continue => {}
    }

    let final_source_route = FinalPresentationIntegralDerivativeRoute::from_final_source_route(
        final_source_integral_derivative_route(ctx, target, var_name),
    );
    if !matches!(
        final_source_route,
        FinalPresentationIntegralDerivativeRoute::NoMatch
    ) {
        return final_source_route;
    }

    FinalPresentationIntegralDerivativeRoute::NoMatch
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn returns_compact_sqrt_trig_log_route() {
        let mut ctx = Context::new();
        let target = parse("tan(sqrt(x))/(2*sqrt(x))", &mut ctx).unwrap();

        let route = final_presentation_integral_derivative_route(&mut ctx, target, "x");

        let FinalPresentationIntegralDerivativeRoute::SqrtTrigLogCompact(compact) = route else {
            panic!("expected sqrt-trig-log compact route");
        };

        assert_eq!(rendered(&ctx, compact), "tan(sqrt(x)) / (2 * sqrt(x))");
        assert!(final_presentation_integral_derivative_shortcut(&mut ctx, target, "x").is_some());
    }

    #[test]
    fn returns_final_source_route_after_compact_routes_continue() {
        let mut ctx = Context::new();
        let target = parse("sin(2*x+1)", &mut ctx).unwrap();

        assert!(matches!(
            final_presentation_integral_derivative_route(&mut ctx, target, "x"),
            FinalPresentationIntegralDerivativeRoute::FinalSourceDirectTrigAffine(source)
                if source == target
        ));
        assert_eq!(
            final_presentation_integral_derivative_shortcut(&mut ctx, target, "x"),
            Some(target)
        );
    }

    #[test]
    fn converts_final_source_route_signals_locally() {
        let mut ctx = Context::new();
        let target = ctx.var("x");

        assert!(matches!(
            FinalPresentationIntegralDerivativeRoute::from_final_source_route(
                FinalSourceIntegralDerivativeRoute::SqrtReciprocalTrigProduct(target)
            ),
            FinalPresentationIntegralDerivativeRoute::FinalSourceSqrtReciprocalTrigProduct(
                source
            ) if source == target
        ));
        assert!(matches!(
            FinalPresentationIntegralDerivativeRoute::from_final_source_route(
                FinalSourceIntegralDerivativeRoute::DirectTrigAffineVerificationFailed
            ),
            FinalPresentationIntegralDerivativeRoute::FinalSourceDirectTrigAffineVerificationFailed
        ));
        assert!(matches!(
            FinalPresentationIntegralDerivativeRoute::from_final_source_route(
                FinalSourceIntegralDerivativeRoute::NoMatch
            ),
            FinalPresentationIntegralDerivativeRoute::NoMatch
        ));
    }

    #[test]
    fn rejects_expression_outside_final_routes() {
        let mut ctx = Context::new();
        let target = parse("exp(x^2)", &mut ctx).unwrap();

        assert!(matches!(
            final_presentation_integral_derivative_route(&mut ctx, target, "x"),
            FinalPresentationIntegralDerivativeRoute::NoMatch
        ));
        assert_eq!(
            final_presentation_integral_derivative_shortcut(&mut ctx, target, "x"),
            None
        );
    }

    #[test]
    fn final_source_verification_failure_remains_absent_from_shortcut_output() {
        let mut ctx = Context::new();
        let target = parse("sin(2*x+1)+exp(x^2)", &mut ctx).unwrap();

        assert!(matches!(
            final_presentation_integral_derivative_route(&mut ctx, target, "x"),
            FinalPresentationIntegralDerivativeRoute::FinalSourceDirectTrigAffineVerificationFailed
        ));
        assert_eq!(
            final_presentation_integral_derivative_shortcut(&mut ctx, target, "x"),
            None
        );
    }
}
