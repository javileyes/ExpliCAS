//! Sqrt-trig log routes for integration-backed derivative shortcuts.
//!
//! This module owns only the source-side route policy for the sqrt-trig log
//! family. It preserves the parent shortcut gate's route order and return
//! semantics.

use super::integral_derivative_shortcut_return_policy::{
    verified_compact_integrand_target, verified_optional_compact_integrand_target_from,
};
use super::sqrt_chain_integrand_preservation::sqrt_trig_log_integrand_for_calculus_presentation;
use super::sqrt_trig_log_integrand_presentation::{
    compact_direct_sqrt_trig_log_derivative_integrand, compact_sqrt_trig_log_derivative_integrand,
};
use cas_ast::{Context, ExprId};

/// Result for the sqrt-trig-log route group.
///
/// `VerificationFailed` means a source-side route matched, but bounded
/// integrator verification rejected it, so later fallback routes must not claim
/// the same source.
pub(super) enum SqrtTrigLogDerivativeRoute {
    NoMatch,
    VerificationFailed,
    VerifiedNoCompact,
    VerifiedCompact(ExprId),
}

pub(super) enum SqrtTrigLogPresentationDecision {
    AbortFallback,
    Compact(ExprId),
    Continue,
}

#[derive(Debug, PartialEq, Eq)]
enum SqrtTrigLogSourceRoute {
    NoMatch,
    DirectCompactCandidate(ExprId),
    PreservedSourceCandidate,
}

impl SqrtTrigLogDerivativeRoute {
    pub(super) fn into_final_presentation_decision(self) -> SqrtTrigLogPresentationDecision {
        match self {
            SqrtTrigLogDerivativeRoute::VerificationFailed => {
                SqrtTrigLogPresentationDecision::AbortFallback
            }
            SqrtTrigLogDerivativeRoute::VerifiedCompact(compact) => {
                SqrtTrigLogPresentationDecision::Compact(compact)
            }
            SqrtTrigLogDerivativeRoute::NoMatch | SqrtTrigLogDerivativeRoute::VerifiedNoCompact => {
                SqrtTrigLogPresentationDecision::Continue
            }
        }
    }
}

pub(super) fn sqrt_trig_log_integral_derivative_shortcut(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> SqrtTrigLogDerivativeRoute {
    match sqrt_trig_log_source_route(ctx, target, var_name) {
        SqrtTrigLogSourceRoute::DirectCompactCandidate(compact) => {
            verified_direct_sqrt_trig_log_compact_route(ctx, target, var_name, compact)
        }
        SqrtTrigLogSourceRoute::PreservedSourceCandidate => {
            verified_preserved_sqrt_trig_log_route(ctx, target, var_name)
        }
        SqrtTrigLogSourceRoute::NoMatch => SqrtTrigLogDerivativeRoute::NoMatch,
    }
}

fn sqrt_trig_log_source_route(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> SqrtTrigLogSourceRoute {
    if let Some(compact) = compact_direct_sqrt_trig_log_derivative_integrand(ctx, target, var_name)
    {
        return SqrtTrigLogSourceRoute::DirectCompactCandidate(compact);
    }

    if sqrt_trig_log_integrand_for_calculus_presentation(ctx, target, var_name) {
        return SqrtTrigLogSourceRoute::PreservedSourceCandidate;
    }

    SqrtTrigLogSourceRoute::NoMatch
}

fn verified_direct_sqrt_trig_log_compact_route(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
    compact: ExprId,
) -> SqrtTrigLogDerivativeRoute {
    let Some(verified) = verified_compact_integrand_target(ctx, target, var_name, compact) else {
        return SqrtTrigLogDerivativeRoute::VerificationFailed;
    };
    SqrtTrigLogDerivativeRoute::VerifiedCompact(verified)
}

fn verified_preserved_sqrt_trig_log_route(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> SqrtTrigLogDerivativeRoute {
    let Some(compact) =
        verified_optional_compact_integrand_target_from(ctx, target, var_name, |ctx| {
            compact_sqrt_trig_log_derivative_integrand(ctx, target, var_name)
        })
    else {
        return SqrtTrigLogDerivativeRoute::VerificationFailed;
    };
    if let Some(compact) = compact {
        return SqrtTrigLogDerivativeRoute::VerifiedCompact(compact);
    }

    SqrtTrigLogDerivativeRoute::VerifiedNoCompact
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
    fn direct_sqrt_trig_log_route_returns_verified_compact_target() {
        let mut ctx = Context::new();
        let target = parse("tan(sqrt(x))/(2*sqrt(x))", &mut ctx).unwrap();

        let route = sqrt_trig_log_integral_derivative_shortcut(&mut ctx, target, "x");

        let SqrtTrigLogDerivativeRoute::VerifiedCompact(compact) = route else {
            panic!("expected verified compact route");
        };
        assert_eq!(rendered(&ctx, compact), "tan(sqrt(x)) / (2 * sqrt(x))");
    }

    #[test]
    fn sqrt_trig_log_route_reports_no_match_without_aborting() {
        let mut ctx = Context::new();
        let target = parse("sin(x)", &mut ctx).unwrap();

        let route = sqrt_trig_log_integral_derivative_shortcut(&mut ctx, target, "x");

        assert!(matches!(route, SqrtTrigLogDerivativeRoute::NoMatch));
    }

    #[test]
    fn route_classifier_separates_direct_compact_from_preserved_source() {
        let mut ctx = Context::new();

        let direct = parse("tan(sqrt(x))/(2*sqrt(x))", &mut ctx).unwrap();
        assert!(matches!(
            sqrt_trig_log_source_route(&mut ctx, direct, "x"),
            SqrtTrigLogSourceRoute::DirectCompactCandidate(_)
        ));

        let preserved = parse("sin(sqrt(x))*x^(-1/2)/(2*cos(sqrt(x)))", &mut ctx).unwrap();
        assert_eq!(
            sqrt_trig_log_source_route(&mut ctx, preserved, "x"),
            SqrtTrigLogSourceRoute::PreservedSourceCandidate
        );
    }
}
