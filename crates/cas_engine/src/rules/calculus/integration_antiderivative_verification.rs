//! Bounded verification helpers for integration-backed calculus presentation.
//!
//! These helpers prove only that the existing symbolic integrator accepts the
//! source integrand. They do not introduce a broader antiderivative search.

use cas_ast::{Context, ExprId};

use super::integration::integrate;

fn verify_integrand_is_supported_by_integrator(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<()> {
    integrate(ctx, target, var_name)?;
    Some(())
}

pub(super) fn verified_integrand_result<T>(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
    result: impl FnOnce(&mut Context, ExprId) -> T,
) -> Option<T> {
    verify_integrand_is_supported_by_integrator(ctx, target, var_name)?;
    Some(result(ctx, target))
}

pub(super) fn verified_integrand_target(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    verified_integrand_result(ctx, target, var_name, |_, target| target)
}

pub(super) fn verified_source_route<R>(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
    verified_source: impl FnOnce(ExprId) -> R,
    verification_failed: R,
) -> R {
    match verified_integrand_target(ctx, target, var_name) {
        Some(source) => verified_source(source),
        None => verification_failed,
    }
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::{verified_integrand_result, verified_integrand_target, verified_source_route};

    #[derive(Debug, PartialEq, Eq)]
    enum TestVerifiedSourceRoute {
        VerificationFailed,
        VerifiedSource(ExprId),
    }

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn verification_accepts_supported_integrand_without_rewriting_target() {
        let mut ctx = Context::new();
        let target = parse("1/(x+1)", &mut ctx).unwrap();

        let verified = verified_integrand_target(&mut ctx, target, "x").unwrap();

        assert_eq!(verified, target);
        assert_eq!(rendered(&ctx, verified), "1 / (x + 1)");
    }

    #[test]
    fn verification_rejects_unsupported_integrand() {
        let mut ctx = Context::new();
        let target = parse("exp(x^2)", &mut ctx).unwrap();

        assert!(verified_integrand_target(&mut ctx, target, "x").is_none());
    }

    #[test]
    fn verified_result_projects_only_after_supported_integrand() {
        let mut ctx = Context::new();
        let target = parse("1/(x+1)", &mut ctx).unwrap();
        let projected = parse("ln(abs(x+1))", &mut ctx).unwrap();

        let verified = verified_integrand_result(&mut ctx, target, "x", |_, _| projected).unwrap();

        assert_eq!(verified, projected);
    }

    #[test]
    fn verified_result_does_not_project_unsupported_integrand() {
        let mut ctx = Context::new();
        let target = parse("exp(x^2)", &mut ctx).unwrap();
        let mut projected = false;

        let verified = verified_integrand_result(&mut ctx, target, "x", |_, target| {
            projected = true;
            target
        });

        assert!(verified.is_none());
        assert!(!projected);
    }

    #[test]
    fn verified_source_route_maps_supported_and_unsupported_integrands() {
        let mut ctx = Context::new();
        let supported = parse("1/(x+1)", &mut ctx).unwrap();
        let unsupported = parse("exp(x^2)", &mut ctx).unwrap();

        assert_eq!(
            verified_source_route(
                &mut ctx,
                supported,
                "x",
                TestVerifiedSourceRoute::VerifiedSource,
                TestVerifiedSourceRoute::VerificationFailed,
            ),
            TestVerifiedSourceRoute::VerifiedSource(supported)
        );
        assert_eq!(
            verified_source_route(
                &mut ctx,
                unsupported,
                "x",
                TestVerifiedSourceRoute::VerifiedSource,
                TestVerifiedSourceRoute::VerificationFailed,
            ),
            TestVerifiedSourceRoute::VerificationFailed
        );
    }
}
