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

/// Result for a matched sqrt-trig-log route.
///
/// The outer `Option` returned by the route is reserved for verification aborts:
/// `None` means a candidate matched but failed bounded integrator verification.
pub(super) enum SqrtTrigLogDerivativeRoute {
    NoMatch,
    VerifiedNoCompact,
    VerifiedCompact(ExprId),
}

pub(super) fn sqrt_trig_log_integral_derivative_shortcut(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<SqrtTrigLogDerivativeRoute> {
    if let Some(compact) = compact_direct_sqrt_trig_log_derivative_integrand(ctx, target, var_name)
    {
        let verified = verified_compact_integrand_target(ctx, target, var_name, compact)?;
        return Some(SqrtTrigLogDerivativeRoute::VerifiedCompact(verified));
    }

    if sqrt_trig_log_integrand_for_calculus_presentation(ctx, target, var_name) {
        if let Some(compact) =
            verified_optional_compact_integrand_target_from(ctx, target, var_name, |ctx| {
                compact_sqrt_trig_log_derivative_integrand(ctx, target, var_name)
            })?
        {
            return Some(SqrtTrigLogDerivativeRoute::VerifiedCompact(compact));
        }

        return Some(SqrtTrigLogDerivativeRoute::VerifiedNoCompact);
    }

    Some(SqrtTrigLogDerivativeRoute::NoMatch)
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

        let route = sqrt_trig_log_integral_derivative_shortcut(&mut ctx, target, "x").unwrap();

        let SqrtTrigLogDerivativeRoute::VerifiedCompact(compact) = route else {
            panic!("expected verified compact route");
        };
        assert_eq!(rendered(&ctx, compact), "tan(sqrt(x)) / (2 * sqrt(x))");
    }

    #[test]
    fn sqrt_trig_log_route_reports_no_match_without_aborting() {
        let mut ctx = Context::new();
        let target = parse("sin(x)", &mut ctx).unwrap();

        let route = sqrt_trig_log_integral_derivative_shortcut(&mut ctx, target, "x").unwrap();

        assert!(matches!(route, SqrtTrigLogDerivativeRoute::NoMatch));
    }
}
