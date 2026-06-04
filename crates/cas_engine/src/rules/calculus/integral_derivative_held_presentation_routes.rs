//! Held-presentation routes for integration-backed derivative shortcuts.
//!
//! This module preserves the route order for routes 11-13 from
//! `integral_derivative_shortcut_presentation`. They intentionally return held
//! source or compact/source integrands so the caller can collect required
//! integration conditions from the original source.

use super::integral_derivative_held_source_routes::source_held_integral_derivative_shortcut;
use super::integral_derivative_verified_held_inverse_sqrt_routes::{
    verified_held_inverse_sqrt_integral_derivative_route, VerifiedHeldInverseSqrtRoute,
};
use super::integral_derivative_verified_held_polynomial_substitution_routes::{
    verified_held_polynomial_substitution_integral_derivative_route,
    VerifiedHeldPolynomialSubstitutionRoute,
};
use cas_ast::{Context, ExprId};

pub(super) enum HeldPresentationIntegralDerivativeRoute {
    NoMatch,
    VerifiedHeldInverseSqrtVerificationFailed,
    VerifiedHeldPolynomialSubstitutionVerificationFailed,
    SourceHeld(ExprId),
    VerifiedHeldInverseSqrt(ExprId),
    VerifiedHeldPolynomialSubstitution(ExprId),
}

#[cfg(test)]
impl HeldPresentationIntegralDerivativeRoute {
    fn into_held_target(self) -> Option<ExprId> {
        match self {
            HeldPresentationIntegralDerivativeRoute::SourceHeld(held_target)
            | HeldPresentationIntegralDerivativeRoute::VerifiedHeldInverseSqrt(held_target)
            | HeldPresentationIntegralDerivativeRoute::VerifiedHeldPolynomialSubstitution(
                held_target,
            ) => Some(held_target),
            HeldPresentationIntegralDerivativeRoute::NoMatch
            | HeldPresentationIntegralDerivativeRoute::VerifiedHeldInverseSqrtVerificationFailed
            | HeldPresentationIntegralDerivativeRoute::VerifiedHeldPolynomialSubstitutionVerificationFailed => None,
        }
    }
}

#[cfg(test)]
fn held_presentation_integral_derivative_shortcut(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    held_presentation_integral_derivative_route(ctx, target, var_name).into_held_target()
}

pub(super) fn held_presentation_integral_derivative_route(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> HeldPresentationIntegralDerivativeRoute {
    if let Some(held_target) = source_held_integral_derivative_shortcut(ctx, target, var_name) {
        return HeldPresentationIntegralDerivativeRoute::SourceHeld(held_target);
    }

    let mut inverse_sqrt_verification_failed = false;
    match verified_held_inverse_sqrt_integral_derivative_route(ctx, target, var_name) {
        VerifiedHeldInverseSqrtRoute::VerifiedHeldCompactOrSource(held_target) => {
            return HeldPresentationIntegralDerivativeRoute::VerifiedHeldInverseSqrt(held_target);
        }
        VerifiedHeldInverseSqrtRoute::VerificationFailed => {
            inverse_sqrt_verification_failed = true;
        }
        VerifiedHeldInverseSqrtRoute::NoMatch => {}
    }

    match verified_held_polynomial_substitution_integral_derivative_route(ctx, target, var_name) {
        VerifiedHeldPolynomialSubstitutionRoute::VerifiedHeldSource(held_target) => {
            return HeldPresentationIntegralDerivativeRoute::VerifiedHeldPolynomialSubstitution(
                held_target,
            );
        }
        VerifiedHeldPolynomialSubstitutionRoute::VerificationFailed => {
            return HeldPresentationIntegralDerivativeRoute::VerifiedHeldPolynomialSubstitutionVerificationFailed;
        }
        VerifiedHeldPolynomialSubstitutionRoute::NoMatch => {}
    }

    if inverse_sqrt_verification_failed {
        return HeldPresentationIntegralDerivativeRoute::VerifiedHeldInverseSqrtVerificationFailed;
    }

    HeldPresentationIntegralDerivativeRoute::NoMatch
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::hold;
    use cas_parser::parse;

    #[test]
    fn accepts_arcsin_inverse_sqrt_product_as_held_integrand() {
        let mut ctx = Context::new();
        let target = parse("1/(sqrt(x)*sqrt(1-x))", &mut ctx).unwrap();

        assert!(matches!(
            held_presentation_integral_derivative_route(&mut ctx, target, "x"),
            HeldPresentationIntegralDerivativeRoute::SourceHeld(held) if hold::is_hold(&ctx, held)
        ));
    }

    #[test]
    fn accepts_affine_sqrt_product_derivative_as_verified_held_integrand() {
        let mut ctx = Context::new();
        let target = parse("1/sqrt(x)", &mut ctx).unwrap();

        assert!(matches!(
            held_presentation_integral_derivative_route(&mut ctx, target, "x"),
            HeldPresentationIntegralDerivativeRoute::VerifiedHeldInverseSqrt(held)
                if hold::is_hold(&ctx, held)
        ));
    }

    #[test]
    fn accepts_polynomial_substitution_as_verified_held_source_integrand() {
        let mut ctx = Context::new();
        let target = parse("2*x/sqrt(1+x^4)", &mut ctx).unwrap();

        assert!(matches!(
            held_presentation_integral_derivative_route(&mut ctx, target, "x"),
            HeldPresentationIntegralDerivativeRoute::VerifiedHeldPolynomialSubstitution(held)
                if hold::is_hold(&ctx, held)
                    && hold::unwrap_internal_hold(&ctx, held) == target
        ));
    }

    #[test]
    fn rejects_source_outside_held_presentation_group() {
        let mut ctx = Context::new();
        let target = parse("sin(x)", &mut ctx).unwrap();

        assert!(matches!(
            held_presentation_integral_derivative_route(&mut ctx, target, "x"),
            HeldPresentationIntegralDerivativeRoute::NoMatch
        ));
        assert_eq!(
            held_presentation_integral_derivative_shortcut(&mut ctx, target, "x"),
            None
        );
    }

    #[test]
    fn verification_failure_routes_remain_absent_from_shortcut_output() {
        let mut ctx = Context::new();
        let target = parse("sin(x)", &mut ctx).unwrap();

        assert_eq!(
            HeldPresentationIntegralDerivativeRoute::VerifiedHeldInverseSqrtVerificationFailed
                .into_held_target(),
            None
        );
        assert_eq!(
            HeldPresentationIntegralDerivativeRoute::VerifiedHeldPolynomialSubstitutionVerificationFailed
                .into_held_target(),
            None
        );
        assert_eq!(
            held_presentation_integral_derivative_shortcut(&mut ctx, target, "x"),
            None
        );
    }
}
