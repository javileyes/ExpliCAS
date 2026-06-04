//! Return policy helpers for integration-backed derivative shortcuts.
//!
//! These helpers own only the presentation return shape after a route has
//! matched. They do not decide which calculus route applies.

use super::integration_antiderivative_verification::verified_integrand_result;
use super::inverse_sqrt_product_integrand_presentation::compact_inverse_sqrt_product_integrand_for_calculus_presentation;
use cas_ast::{Context, ExprId};

pub(super) fn held_source_integrand_target(ctx: &mut Context, target: ExprId) -> ExprId {
    cas_ast::hold::wrap_hold(ctx, target)
}

pub(super) fn held_inverse_sqrt_compact_or_source_integrand_target(
    ctx: &mut Context,
    target: ExprId,
) -> ExprId {
    if let Some(compact) =
        compact_inverse_sqrt_product_integrand_for_calculus_presentation(ctx, target)
    {
        return held_source_integrand_target(ctx, compact);
    }

    held_source_integrand_target(ctx, target)
}

pub(super) fn verified_held_inverse_sqrt_compact_or_source_integrand_target(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    verified_integrand_result(ctx, target, var_name, |ctx, target| {
        held_inverse_sqrt_compact_or_source_integrand_target(ctx, target)
    })
}

pub(super) fn verified_held_source_integrand_target(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    verified_integrand_result(ctx, target, var_name, held_source_integrand_target)
}

pub(super) fn verified_held_inverse_sqrt_compact_or_source_route<R>(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
    verified_held: impl FnOnce(ExprId) -> R,
    verification_failed: R,
) -> R {
    match verified_held_inverse_sqrt_compact_or_source_integrand_target(ctx, target, var_name) {
        Some(held_target) => verified_held(held_target),
        None => verification_failed,
    }
}

pub(super) fn verified_held_source_route<R>(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
    verified_held: impl FnOnce(ExprId) -> R,
    verification_failed: R,
) -> R {
    match verified_held_source_integrand_target(ctx, target, var_name) {
        Some(held_target) => verified_held(held_target),
        None => verification_failed,
    }
}

pub(super) fn verified_compact_integrand_target(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
    compact: ExprId,
) -> Option<ExprId> {
    verified_integrand_result(ctx, target, var_name, |_, _| compact)
}

pub(super) fn verified_optional_compact_integrand_target_from(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
    compact_target: impl FnOnce(&mut Context) -> Option<ExprId>,
) -> Option<Option<ExprId>> {
    verified_integrand_result(ctx, target, var_name, |ctx, _| compact_target(ctx))
}

#[cfg(test)]
mod tests {
    use cas_ast::{hold, Context, ExprId};
    use cas_parser::parse;

    use super::{
        held_source_integrand_target, verified_compact_integrand_target,
        verified_held_inverse_sqrt_compact_or_source_route, verified_held_source_integrand_target,
        verified_held_source_route, verified_optional_compact_integrand_target_from,
    };

    #[derive(Debug, PartialEq, Eq)]
    enum TestVerifiedHeldRoute {
        VerificationFailed,
        VerifiedHeld(ExprId),
    }

    #[test]
    fn held_source_return_wraps_without_rewriting() {
        let mut ctx = Context::new();
        let target = parse("1/(x+1)", &mut ctx).unwrap();

        let held = held_source_integrand_target(&mut ctx, target);

        assert_ne!(held, target);
        assert!(hold::is_hold(&ctx, held));
        assert_eq!(hold::unwrap_internal_hold(&ctx, held), target);
    }

    #[test]
    fn verified_held_source_return_rejects_unsupported_integrand() {
        let mut ctx = Context::new();
        let target = parse("exp(x^2)", &mut ctx).unwrap();

        assert!(verified_held_source_integrand_target(&mut ctx, target, "x").is_none());
    }

    #[test]
    fn verified_compact_return_preserves_supplied_compact_target() {
        let mut ctx = Context::new();
        let target = parse("1/(x+1)", &mut ctx).unwrap();
        let compact = parse("1/(x+1)", &mut ctx).unwrap();

        let verified = verified_compact_integrand_target(&mut ctx, target, "x", compact).unwrap();

        assert_eq!(verified, compact);
    }

    #[test]
    fn verified_optional_compact_return_does_not_compact_unsupported_integrand() {
        let mut ctx = Context::new();
        let target = parse("exp(x^2)", &mut ctx).unwrap();
        let mut compact_called = false;

        let verified =
            verified_optional_compact_integrand_target_from(&mut ctx, target, "x", |_| {
                compact_called = true;
                Some(target)
            });

        assert!(verified.is_none());
        assert!(!compact_called);
    }

    #[test]
    fn verified_held_source_route_maps_supported_and_unsupported_integrands() {
        let mut ctx = Context::new();
        let supported = parse("1/(x+1)", &mut ctx).unwrap();
        let unsupported = parse("exp(x^2)", &mut ctx).unwrap();

        match verified_held_source_route(
            &mut ctx,
            supported,
            "x",
            TestVerifiedHeldRoute::VerifiedHeld,
            TestVerifiedHeldRoute::VerificationFailed,
        ) {
            TestVerifiedHeldRoute::VerifiedHeld(held) => {
                assert!(hold::is_hold(&ctx, held));
                assert_eq!(hold::unwrap_internal_hold(&ctx, held), supported);
            }
            TestVerifiedHeldRoute::VerificationFailed => panic!("expected verified held source"),
        }

        assert_eq!(
            verified_held_source_route(
                &mut ctx,
                unsupported,
                "x",
                TestVerifiedHeldRoute::VerifiedHeld,
                TestVerifiedHeldRoute::VerificationFailed,
            ),
            TestVerifiedHeldRoute::VerificationFailed
        );
    }

    #[test]
    fn verified_held_inverse_sqrt_route_maps_compact_or_source_integrand() {
        let mut ctx = Context::new();
        let target = parse("1/sqrt(x)", &mut ctx).unwrap();

        match verified_held_inverse_sqrt_compact_or_source_route(
            &mut ctx,
            target,
            "x",
            TestVerifiedHeldRoute::VerifiedHeld,
            TestVerifiedHeldRoute::VerificationFailed,
        ) {
            TestVerifiedHeldRoute::VerifiedHeld(held) => assert!(hold::is_hold(&ctx, held)),
            TestVerifiedHeldRoute::VerificationFailed => {
                panic!("expected verified held inverse-sqrt target")
            }
        }
    }
}
