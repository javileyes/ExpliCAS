//! Aggregated route signal for derivative-of-integral presentation shortcuts.
//!
//! This module owns only the top-level signal semantics for the
//! `diff(integrate(...), x)` gate: which routes produce a public presentation
//! target, which routes are internal non-success signals, and how the parent
//! gate preserves the first verification failure while still allowing later
//! successes to win.

use super::integral_derivative_final_presentation_routes::FinalPresentationIntegralDerivativeRoute;
use super::integral_derivative_held_presentation_routes::HeldPresentationIntegralDerivativeRoute;
use super::integral_derivative_verified_power_inverse_routes::PowerInverseIntegralDerivativeRoute;
use super::integral_derivative_verified_rational_substitution_routes::RationalSubstitutionIntegralDerivativeRoute;
use super::integral_derivative_verified_source_routes::InitialVerifiedSourceIntegralDerivativeRoute;
use cas_ast::ExprId;

pub(super) enum SupportedIntegralDerivativePresentationRoute {
    NoMatch,
    InitialVerifiedSource(ExprId),
    InitialVerifiedSourceVerificationFailed,
    ArctanPolynomialSource(ExprId),
    RationalSubstitutionVerifiedSource(ExprId),
    RationalSubstitutionVerifiedSourceVerificationFailed,
    HeldPresentationSourceHeld(ExprId),
    HeldPresentationVerifiedInverseSqrt(ExprId),
    HeldPresentationVerifiedPolynomialSubstitution(ExprId),
    HeldPresentationVerifiedInverseSqrtVerificationFailed,
    HeldPresentationVerifiedPolynomialSubstitutionVerificationFailed,
    PowerInverseVerifiedSource(ExprId),
    PowerInverseVerifiedSourceVerificationFailed,
    FinalPresentationSqrtTrigLogCompact(ExprId),
    FinalPresentationSqrtTrigLogAbortFallback,
    FinalPresentationSourceSqrtReciprocalTrigProduct(ExprId),
    FinalPresentationSourceDirectTrigAffine(ExprId),
    FinalPresentationSourceDirectTrigAffineVerificationFailed,
}

impl SupportedIntegralDerivativePresentationRoute {
    pub(super) fn into_presentation_target(self) -> Option<ExprId> {
        match self {
            SupportedIntegralDerivativePresentationRoute::InitialVerifiedSource(target)
            | SupportedIntegralDerivativePresentationRoute::ArctanPolynomialSource(target)
            | SupportedIntegralDerivativePresentationRoute::RationalSubstitutionVerifiedSource(
                target,
            )
            | SupportedIntegralDerivativePresentationRoute::HeldPresentationSourceHeld(target)
            | SupportedIntegralDerivativePresentationRoute::HeldPresentationVerifiedInverseSqrt(
                target,
            )
            | SupportedIntegralDerivativePresentationRoute::HeldPresentationVerifiedPolynomialSubstitution(
                target,
            )
            | SupportedIntegralDerivativePresentationRoute::PowerInverseVerifiedSource(target)
            | SupportedIntegralDerivativePresentationRoute::FinalPresentationSqrtTrigLogCompact(
                target,
            )
            | SupportedIntegralDerivativePresentationRoute::FinalPresentationSourceSqrtReciprocalTrigProduct(
                target,
            )
            | SupportedIntegralDerivativePresentationRoute::FinalPresentationSourceDirectTrigAffine(
                target,
            ) => Some(target),
            SupportedIntegralDerivativePresentationRoute::NoMatch
            | SupportedIntegralDerivativePresentationRoute::InitialVerifiedSourceVerificationFailed
            | SupportedIntegralDerivativePresentationRoute::RationalSubstitutionVerifiedSourceVerificationFailed
            | SupportedIntegralDerivativePresentationRoute::HeldPresentationVerifiedInverseSqrtVerificationFailed
            | SupportedIntegralDerivativePresentationRoute::HeldPresentationVerifiedPolynomialSubstitutionVerificationFailed
            | SupportedIntegralDerivativePresentationRoute::PowerInverseVerifiedSourceVerificationFailed
            | SupportedIntegralDerivativePresentationRoute::FinalPresentationSqrtTrigLogAbortFallback
            | SupportedIntegralDerivativePresentationRoute::FinalPresentationSourceDirectTrigAffineVerificationFailed => None,
        }
    }

    pub(super) fn remember_first_non_success(
        pending: &mut SupportedIntegralDerivativePresentationRoute,
        route: SupportedIntegralDerivativePresentationRoute,
    ) {
        if matches!(
            pending,
            SupportedIntegralDerivativePresentationRoute::NoMatch
        ) {
            *pending = route;
        }
    }

    pub(super) fn observe_initial_verified_source_route(
        pending_non_success: &mut SupportedIntegralDerivativePresentationRoute,
        route: InitialVerifiedSourceIntegralDerivativeRoute,
    ) -> Option<SupportedIntegralDerivativePresentationRoute> {
        match route {
            InitialVerifiedSourceIntegralDerivativeRoute::VerifiedSource(source_target) => Some(
                SupportedIntegralDerivativePresentationRoute::InitialVerifiedSource(source_target),
            ),
            InitialVerifiedSourceIntegralDerivativeRoute::VerificationFailed => {
                SupportedIntegralDerivativePresentationRoute::remember_first_non_success(
                    pending_non_success,
                    SupportedIntegralDerivativePresentationRoute::InitialVerifiedSourceVerificationFailed,
                );
                None
            }
            InitialVerifiedSourceIntegralDerivativeRoute::NoMatch => None,
        }
    }

    pub(super) fn observe_arctan_polynomial_source_target(
        source_target: Option<ExprId>,
    ) -> Option<SupportedIntegralDerivativePresentationRoute> {
        source_target.map(SupportedIntegralDerivativePresentationRoute::ArctanPolynomialSource)
    }

    pub(super) fn observe_rational_substitution_route(
        pending_non_success: &mut SupportedIntegralDerivativePresentationRoute,
        route: RationalSubstitutionIntegralDerivativeRoute,
    ) -> Option<SupportedIntegralDerivativePresentationRoute> {
        match route {
            RationalSubstitutionIntegralDerivativeRoute::VerifiedSource(source_target) => Some(
                SupportedIntegralDerivativePresentationRoute::RationalSubstitutionVerifiedSource(
                    source_target,
                ),
            ),
            RationalSubstitutionIntegralDerivativeRoute::VerificationFailed => {
                SupportedIntegralDerivativePresentationRoute::remember_first_non_success(
                    pending_non_success,
                    SupportedIntegralDerivativePresentationRoute::RationalSubstitutionVerifiedSourceVerificationFailed,
                );
                None
            }
            RationalSubstitutionIntegralDerivativeRoute::NoMatch => None,
        }
    }

    pub(super) fn observe_held_presentation_route(
        pending_non_success: &mut SupportedIntegralDerivativePresentationRoute,
        route: HeldPresentationIntegralDerivativeRoute,
    ) -> Option<SupportedIntegralDerivativePresentationRoute> {
        match route {
            HeldPresentationIntegralDerivativeRoute::SourceHeld(held_target) => Some(
                SupportedIntegralDerivativePresentationRoute::HeldPresentationSourceHeld(
                    held_target,
                ),
            ),
            HeldPresentationIntegralDerivativeRoute::VerifiedHeldInverseSqrt(held_target) => {
                Some(
                    SupportedIntegralDerivativePresentationRoute::HeldPresentationVerifiedInverseSqrt(
                        held_target,
                    ),
                )
            }
            HeldPresentationIntegralDerivativeRoute::VerifiedHeldPolynomialSubstitution(
                held_target,
            ) => Some(
                SupportedIntegralDerivativePresentationRoute::HeldPresentationVerifiedPolynomialSubstitution(
                    held_target,
                ),
            ),
            HeldPresentationIntegralDerivativeRoute::VerifiedHeldInverseSqrtVerificationFailed => {
                SupportedIntegralDerivativePresentationRoute::remember_first_non_success(
                    pending_non_success,
                    SupportedIntegralDerivativePresentationRoute::HeldPresentationVerifiedInverseSqrtVerificationFailed,
                );
                None
            }
            HeldPresentationIntegralDerivativeRoute::VerifiedHeldPolynomialSubstitutionVerificationFailed => {
                SupportedIntegralDerivativePresentationRoute::remember_first_non_success(
                    pending_non_success,
                    SupportedIntegralDerivativePresentationRoute::HeldPresentationVerifiedPolynomialSubstitutionVerificationFailed,
                );
                None
            }
            HeldPresentationIntegralDerivativeRoute::NoMatch => None,
        }
    }

    pub(super) fn observe_power_inverse_route(
        pending_non_success: &mut SupportedIntegralDerivativePresentationRoute,
        route: PowerInverseIntegralDerivativeRoute,
    ) -> Option<SupportedIntegralDerivativePresentationRoute> {
        match route {
            PowerInverseIntegralDerivativeRoute::VerifiedSource(source_target) => Some(
                SupportedIntegralDerivativePresentationRoute::PowerInverseVerifiedSource(
                    source_target,
                ),
            ),
            PowerInverseIntegralDerivativeRoute::VerificationFailed => {
                SupportedIntegralDerivativePresentationRoute::remember_first_non_success(
                    pending_non_success,
                    SupportedIntegralDerivativePresentationRoute::PowerInverseVerifiedSourceVerificationFailed,
                );
                None
            }
            PowerInverseIntegralDerivativeRoute::NoMatch => None,
        }
    }

    pub(super) fn complete_with_final_presentation_route(
        pending_non_success: SupportedIntegralDerivativePresentationRoute,
        final_route: FinalPresentationIntegralDerivativeRoute,
    ) -> SupportedIntegralDerivativePresentationRoute {
        match final_route {
            FinalPresentationIntegralDerivativeRoute::SqrtTrigLogCompact(final_target) => {
                SupportedIntegralDerivativePresentationRoute::FinalPresentationSqrtTrigLogCompact(
                    final_target,
                )
            }
            FinalPresentationIntegralDerivativeRoute::SqrtTrigLogAbortFallback => {
                pending_non_success.or_else_no_match(
                    SupportedIntegralDerivativePresentationRoute::FinalPresentationSqrtTrigLogAbortFallback,
                )
            }
            FinalPresentationIntegralDerivativeRoute::FinalSourceSqrtReciprocalTrigProduct(
                final_target,
            ) => SupportedIntegralDerivativePresentationRoute::FinalPresentationSourceSqrtReciprocalTrigProduct(
                final_target,
            ),
            FinalPresentationIntegralDerivativeRoute::FinalSourceDirectTrigAffine(final_target) => {
                SupportedIntegralDerivativePresentationRoute::FinalPresentationSourceDirectTrigAffine(
                    final_target,
                )
            }
            FinalPresentationIntegralDerivativeRoute::FinalSourceDirectTrigAffineVerificationFailed => {
                pending_non_success.or_else_no_match(
                    SupportedIntegralDerivativePresentationRoute::FinalPresentationSourceDirectTrigAffineVerificationFailed,
                )
            }
            FinalPresentationIntegralDerivativeRoute::NoMatch => pending_non_success,
        }
    }

    fn or_else_no_match(
        self,
        fallback: SupportedIntegralDerivativePresentationRoute,
    ) -> SupportedIntegralDerivativePresentationRoute {
        if matches!(self, SupportedIntegralDerivativePresentationRoute::NoMatch) {
            fallback
        } else {
            self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Context;

    #[test]
    fn success_routes_return_presentation_targets() {
        let mut ctx = Context::new();
        let target = ctx.var("x");

        assert_eq!(
            SupportedIntegralDerivativePresentationRoute::InitialVerifiedSource(target)
                .into_presentation_target(),
            Some(target)
        );
        assert_eq!(
            SupportedIntegralDerivativePresentationRoute::FinalPresentationSourceDirectTrigAffine(
                target
            )
            .into_presentation_target(),
            Some(target)
        );
    }

    #[test]
    fn non_success_routes_do_not_return_presentation_targets() {
        assert_eq!(
            SupportedIntegralDerivativePresentationRoute::InitialVerifiedSourceVerificationFailed
                .into_presentation_target(),
            None
        );
        assert_eq!(
            SupportedIntegralDerivativePresentationRoute::FinalPresentationSqrtTrigLogAbortFallback
                .into_presentation_target(),
            None
        );
    }

    #[test]
    fn remembers_first_non_success_signal() {
        let mut pending = SupportedIntegralDerivativePresentationRoute::NoMatch;

        SupportedIntegralDerivativePresentationRoute::remember_first_non_success(
            &mut pending,
            SupportedIntegralDerivativePresentationRoute::InitialVerifiedSourceVerificationFailed,
        );
        SupportedIntegralDerivativePresentationRoute::remember_first_non_success(
            &mut pending,
            SupportedIntegralDerivativePresentationRoute::PowerInverseVerifiedSourceVerificationFailed,
        );

        assert!(matches!(
            pending,
            SupportedIntegralDerivativePresentationRoute::InitialVerifiedSourceVerificationFailed
        ));
    }

    #[test]
    fn final_success_overrides_pending_non_success_signal() {
        let mut ctx = Context::new();
        let target = ctx.var("x");

        let route = SupportedIntegralDerivativePresentationRoute::complete_with_final_presentation_route(
            SupportedIntegralDerivativePresentationRoute::InitialVerifiedSourceVerificationFailed,
            FinalPresentationIntegralDerivativeRoute::FinalSourceDirectTrigAffine(target),
        );

        assert!(matches!(
            route,
            SupportedIntegralDerivativePresentationRoute::FinalPresentationSourceDirectTrigAffine(
                route_target
            ) if route_target == target
        ));
    }

    #[test]
    fn pending_non_success_precedes_final_failure_signal() {
        let route = SupportedIntegralDerivativePresentationRoute::complete_with_final_presentation_route(
            SupportedIntegralDerivativePresentationRoute::InitialVerifiedSourceVerificationFailed,
            FinalPresentationIntegralDerivativeRoute::FinalSourceDirectTrigAffineVerificationFailed,
        );

        assert!(matches!(
            route,
            SupportedIntegralDerivativePresentationRoute::InitialVerifiedSourceVerificationFailed
        ));
    }

    #[test]
    fn final_failure_is_preserved_without_pending_non_success() {
        let route = SupportedIntegralDerivativePresentationRoute::complete_with_final_presentation_route(
            SupportedIntegralDerivativePresentationRoute::NoMatch,
            FinalPresentationIntegralDerivativeRoute::FinalSourceDirectTrigAffineVerificationFailed,
        );

        assert!(matches!(
            route,
            SupportedIntegralDerivativePresentationRoute::FinalPresentationSourceDirectTrigAffineVerificationFailed
        ));
    }

    #[test]
    fn observed_group_success_returns_aggregate_signal() {
        let mut ctx = Context::new();
        let target = ctx.var("x");
        let mut pending = SupportedIntegralDerivativePresentationRoute::NoMatch;

        let route =
            SupportedIntegralDerivativePresentationRoute::observe_initial_verified_source_route(
                &mut pending,
                InitialVerifiedSourceIntegralDerivativeRoute::VerifiedSource(target),
            );

        assert!(matches!(
            route,
            Some(SupportedIntegralDerivativePresentationRoute::InitialVerifiedSource(
                route_target
            )) if route_target == target
        ));
        assert!(matches!(
            pending,
            SupportedIntegralDerivativePresentationRoute::NoMatch
        ));
    }

    #[test]
    fn observed_arctan_polynomial_source_target_returns_aggregate_signal() {
        let mut ctx = Context::new();
        let target = ctx.var("x");

        let route =
            SupportedIntegralDerivativePresentationRoute::observe_arctan_polynomial_source_target(
                Some(target),
            );

        assert!(matches!(
            route,
            Some(SupportedIntegralDerivativePresentationRoute::ArctanPolynomialSource(
                route_target
            )) if route_target == target
        ));
        assert!(
            SupportedIntegralDerivativePresentationRoute::observe_arctan_polynomial_source_target(
                None
            )
            .is_none()
        );
    }

    #[test]
    fn observed_group_failure_remembers_first_non_success_signal() {
        let mut pending = SupportedIntegralDerivativePresentationRoute::NoMatch;

        assert!(
            SupportedIntegralDerivativePresentationRoute::observe_rational_substitution_route(
                &mut pending,
                RationalSubstitutionIntegralDerivativeRoute::VerificationFailed,
            )
            .is_none()
        );
        assert!(
            SupportedIntegralDerivativePresentationRoute::observe_power_inverse_route(
                &mut pending,
                PowerInverseIntegralDerivativeRoute::VerificationFailed,
            )
            .is_none()
        );

        assert!(matches!(
            pending,
            SupportedIntegralDerivativePresentationRoute::RationalSubstitutionVerifiedSourceVerificationFailed
        ));
    }
}
