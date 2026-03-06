use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solve_runtime_adapters::{
    dispatch_isolation_with_default_routes, SolveCtx, SolveStep, SolverOptions,
};
use cas_ast::{ExprId, RelOp, SolutionSet};

#[allow(clippy::too_many_arguments)]
pub(crate) fn isolate(
    lhs: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    ctx: &SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    cas_solver_core::solve_runtime_flow::execute_isolation_with_default_depth_guard_and_dispatch_with_state(
        simplifier,
        ctx.depth(),
        || {
            CasError::SolverError(
                "Maximum solver recursion depth exceeded in isolation.".to_string(),
            )
        },
        |state| dispatch_isolation_with_default_routes(lhs, rhs, op, var, state, opts, ctx),
    )
}
