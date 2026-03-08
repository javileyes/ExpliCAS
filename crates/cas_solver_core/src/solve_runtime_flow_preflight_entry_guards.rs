use cas_ast::{Equation, SolutionSet};

/// Enforce the default isolation recursion limit.
pub fn ensure_default_isolation_recursion_depth_or_error<E, FMapError>(
    current_depth: usize,
    map_error: FMapError,
) -> Result<(), E>
where
    FMapError: FnOnce() -> E,
{
    crate::solve_analysis::ensure_recursion_depth_within_limit_or_error(
        current_depth,
        crate::solve_budget::MAX_SOLVE_RECURSION_DEPTH,
        map_error,
    )
}

/// Execute isolation dispatch after enforcing the default recursion guard.
pub fn execute_isolation_with_default_depth_guard_and_dispatch_with_state<
    SState,
    S,
    E,
    FMapDepthError,
    FDispatch,
>(
    state: &mut SState,
    current_depth: usize,
    map_depth_error: FMapDepthError,
    dispatch: FDispatch,
) -> Result<(SolutionSet, Vec<S>), E>
where
    FMapDepthError: FnOnce() -> E,
    FDispatch: FnOnce(&mut SState) -> Result<(SolutionSet, Vec<S>), E>,
{
    ensure_default_isolation_recursion_depth_or_error(current_depth, map_depth_error)?;
    dispatch(state)
}

/// Enforce default solve-entry guards:
/// - recursion depth within solver budget
/// - solve variable present in equation
pub fn ensure_default_solve_entry_or_error<E, FDepthError, FMissingVarError>(
    ctx: &cas_ast::Context,
    equation: &Equation,
    var: &str,
    current_depth: usize,
    map_depth_error: FDepthError,
    map_missing_var_error: FMissingVarError,
) -> Result<(), E>
where
    FDepthError: FnOnce() -> E,
    FMissingVarError: FnOnce() -> E,
{
    crate::solve_analysis::ensure_solve_entry_for_equation_or_error(
        ctx,
        equation,
        var,
        current_depth,
        crate::solve_budget::MAX_SOLVE_RECURSION_DEPTH,
        map_depth_error,
        map_missing_var_error,
    )
}

/// Enter the default equation-fingerprint cycle guard using a caller-provided
/// context accessor, returning a mapped error on cycle re-entry.
pub fn try_enter_default_equation_cycle_guard_with_context_ref_and_error_with_state<
    SState,
    E,
    FContextRef,
    FMapError,
>(
    state: &mut SState,
    equation: &Equation,
    var: &str,
    mut context_ref: FContextRef,
    map_error: FMapError,
) -> Result<crate::cycle_guard::CycleGuard, E>
where
    FContextRef: FnMut(&mut SState) -> &cas_ast::Context,
    FMapError: FnOnce() -> E,
{
    crate::solve_analysis::try_enter_equation_cycle_guard_with_error(
        context_ref(state),
        equation,
        var,
        map_error,
    )
}
