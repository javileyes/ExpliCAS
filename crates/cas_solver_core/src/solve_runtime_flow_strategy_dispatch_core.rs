#[allow(clippy::too_many_arguments)]
pub fn dispatch_solve_strategy_kind_with_runtime_handlers_with_state<
    T,
    R,
    FRationalExponent,
    FSubstitution,
    FUnwrap,
    FQuadratic,
    FRationalRoots,
    FCollectTerms,
    FIsolation,
>(
    state: &mut T,
    kind: crate::strategy_order::SolveStrategyKind,
    rational_exponent: FRationalExponent,
    substitution: FSubstitution,
    unwrap: FUnwrap,
    quadratic: FQuadratic,
    rational_roots: FRationalRoots,
    collect_terms: FCollectTerms,
    isolation: FIsolation,
) -> R
where
    FRationalExponent: FnOnce(&mut T) -> R,
    FSubstitution: FnOnce(&mut T) -> R,
    FUnwrap: FnOnce(&mut T) -> R,
    FQuadratic: FnOnce(&mut T) -> R,
    FRationalRoots: FnOnce(&mut T) -> R,
    FCollectTerms: FnOnce(&mut T) -> R,
    FIsolation: FnOnce(&mut T) -> R,
{
    crate::strategy_order::dispatch_solve_strategy_kind_with_state(
        state,
        kind,
        rational_exponent,
        substitution,
        unwrap,
        quadratic,
        rational_roots,
        collect_terms,
        isolation,
    )
}
