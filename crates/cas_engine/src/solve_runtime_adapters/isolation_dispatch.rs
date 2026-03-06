use crate::solve_runtime_adapters::{SolveCtx, SolveStep, SolverOptions};
use crate::{CasError, Simplifier};
use cas_ast::{ExprId, RelOp, SolutionSet};

#[allow(clippy::too_many_arguments)]
pub(crate) fn isolate_with_default_depth(
    lhs: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    ctx: &SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    cas_solver_core::solve_runtime_isolation_entry_reentrant_context_runtime::isolate_with_default_depth_guard_and_dispatch_with_state(
        simplifier,
        lhs,
        rhs,
        op,
        var,
        opts.into(),
        ctx,
        dispatch_isolation_with_default_routes,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_isolation_with_default_routes(
    lhs: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    ctx: &SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    cas_solver_core::solve_runtime_isolation_dispatch_bound_runtime::dispatch_isolation_with_runtime_state_and_reentrant_entrypoints_and_state(
        simplifier,
        lhs,
        rhs,
        op,
        var,
        opts.into(),
        ctx,
        crate::solve_core_runtime::solve_inner,
        isolate_with_default_depth,
        crate::proof_runtime::prove_positive,
        crate::register_blocked_hint,
    )
}
