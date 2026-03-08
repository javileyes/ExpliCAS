//! Shared entry wrapper for recursive isolation execution.

use cas_ast::SolutionSet;

/// Execute isolation dispatch with the standard depth error used by runtime facades.
pub fn isolate_with_default_depth_guard_and_error_with_state<T, S, FDispatch>(
    state: &mut T,
    current_depth: usize,
    dispatch: FDispatch,
) -> Result<(SolutionSet, Vec<S>), crate::error_model::CasError>
where
    FDispatch: FnMut(&mut T) -> Result<(SolutionSet, Vec<S>), crate::error_model::CasError>,
{
    crate::solve_runtime_flow::execute_isolation_with_default_depth_guard_and_dispatch_with_state(
        state,
        current_depth,
        || {
            crate::error_model::CasError::SolverError(
                "Maximum solver recursion depth exceeded in isolation.".to_string(),
            )
        },
        dispatch,
    )
}
