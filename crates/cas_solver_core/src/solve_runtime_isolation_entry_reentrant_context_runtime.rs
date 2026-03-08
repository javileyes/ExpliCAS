//! Shared isolation-entry wrapper that binds a direct dispatch entrypoint to
//! the current solver options/context.

use cas_ast::{ExprId, RelOp, SolutionSet};

/// Guard recursion depth and then run a direct isolation-dispatch entrypoint
/// through the current runtime solve context and solver options.
#[allow(clippy::too_many_arguments)]
pub fn isolate_with_default_depth_guard_and_dispatch_with_state<T, FDispatchEntry>(
    state: &mut T,
    lhs: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    opts: crate::solver_options::SolverOptions,
    ctx: &crate::solve_runtime_types::RuntimeSolveCtx,
    mut dispatch_entry: FDispatchEntry,
) -> Result<
    (
        SolutionSet,
        Vec<crate::solve_runtime_mapping::DefaultSolveStep>,
    ),
    crate::error_model::CasError,
>
where
    FDispatchEntry: FnMut(
        ExprId,
        ExprId,
        RelOp,
        &str,
        &mut T,
        crate::solver_options::SolverOptions,
        &crate::solve_runtime_types::RuntimeSolveCtx,
    ) -> Result<
        (
            SolutionSet,
            Vec<crate::solve_runtime_mapping::DefaultSolveStep>,
        ),
        crate::error_model::CasError,
    >,
{
    crate::solve_runtime_isolation_entry_runtime::isolate_with_default_depth_guard_and_error_with_state(
        state,
        ctx.depth(),
        |state| dispatch_entry(lhs, rhs, op.clone(), var, state, opts, ctx),
    )
}
