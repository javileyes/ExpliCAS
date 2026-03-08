//! Solver strategy ordering policy shared across engine frontends.
//!
//! Keeps the default strategy sequence in `cas_solver_core` so runtime crates
//! only map these kinds to concrete strategy implementations.

/// Stable strategy identifiers for default solve sequencing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SolveStrategyKind {
    RationalExponent,
    Substitution,
    Unwrap,
    Quadratic,
    RationalRoots,
    CollectTerms,
    Isolation,
}

/// Default solve strategy order.
///
/// Ordering matters for correctness and loop avoidance:
/// - `RationalExponent` must run before `Unwrap` to avoid fractional-power loops.
/// - `CollectTerms` must run before `Isolation` to normalize linear forms first.
pub fn default_solve_strategy_order() -> &'static [SolveStrategyKind] {
    &[
        SolveStrategyKind::RationalExponent,
        SolveStrategyKind::Substitution,
        SolveStrategyKind::Unwrap,
        SolveStrategyKind::Quadratic,
        SolveStrategyKind::RationalRoots,
        SolveStrategyKind::CollectTerms,
        SolveStrategyKind::Isolation,
    ]
}

/// Post-verification policy for each strategy kind.
///
/// `Quadratic` returns analytically derived roots (with didactic substeps),
/// so we skip substitution verification there to avoid expensive and noisy
/// symbolic checks; all other strategies are verified.
pub fn strategy_should_verify(kind: SolveStrategyKind) -> bool {
    !matches!(kind, SolveStrategyKind::Quadratic)
}

/// Dispatch one [`SolveStrategyKind`] to caller-provided handlers.
///
/// `state` is passed to the selected handler only, which avoids creating
/// multiple closures that capture the same mutable runtime reference.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_solve_strategy_kind_with_state<
    S,
    R,
    FRe,
    FSubst,
    FUnwrap,
    FQuad,
    FRoots,
    FCollect,
    FIso,
>(
    state: &mut S,
    kind: SolveStrategyKind,
    on_rational_exponent: FRe,
    on_substitution: FSubst,
    on_unwrap: FUnwrap,
    on_quadratic: FQuad,
    on_rational_roots: FRoots,
    on_collect_terms: FCollect,
    on_isolation: FIso,
) -> R
where
    FRe: FnOnce(&mut S) -> R,
    FSubst: FnOnce(&mut S) -> R,
    FUnwrap: FnOnce(&mut S) -> R,
    FQuad: FnOnce(&mut S) -> R,
    FRoots: FnOnce(&mut S) -> R,
    FCollect: FnOnce(&mut S) -> R,
    FIso: FnOnce(&mut S) -> R,
{
    match kind {
        SolveStrategyKind::RationalExponent => on_rational_exponent(state),
        SolveStrategyKind::Substitution => on_substitution(state),
        SolveStrategyKind::Unwrap => on_unwrap(state),
        SolveStrategyKind::Quadratic => on_quadratic(state),
        SolveStrategyKind::RationalRoots => on_rational_roots(state),
        SolveStrategyKind::CollectTerms => on_collect_terms(state),
        SolveStrategyKind::Isolation => on_isolation(state),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        default_solve_strategy_order, dispatch_solve_strategy_kind_with_state,
        strategy_should_verify, SolveStrategyKind,
    };

    #[test]
    fn default_order_has_expected_length_and_key_positions() {
        let order = default_solve_strategy_order();
        assert_eq!(order.len(), 7);
        assert_eq!(order[0], SolveStrategyKind::RationalExponent);
        assert_eq!(order[2], SolveStrategyKind::Unwrap);
        assert_eq!(order[5], SolveStrategyKind::CollectTerms);
        assert_eq!(order[6], SolveStrategyKind::Isolation);
    }

    #[test]
    fn quadratic_is_not_verified_others_are_verified() {
        assert!(!strategy_should_verify(SolveStrategyKind::Quadratic));
        assert!(strategy_should_verify(SolveStrategyKind::Isolation));
        assert!(strategy_should_verify(SolveStrategyKind::Substitution));
    }

    #[test]
    fn dispatch_solve_strategy_kind_with_state_routes_to_matching_handler() {
        let mut state = ();
        let routed = dispatch_solve_strategy_kind_with_state(
            &mut state,
            SolveStrategyKind::CollectTerms,
            |_state| "re",
            |_state| "subst",
            |_state| "unwrap",
            |_state| "quad",
            |_state| "roots",
            |_state| "collect",
            |_state| "iso",
        );
        assert_eq!(routed, "collect");
    }
}
