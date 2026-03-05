use crate::engine::Simplifier;
use crate::error::CasError;
use cas_ast::{ExprId, RelOp, SolutionSet};
use cas_solver_core::solve_analysis::ensure_recursion_depth_within_limit_or_error;
use cas_solver_core::solve_budget::MAX_SOLVE_RECURSION_DEPTH;

use super::isolation_dispatch::dispatch_isolation;
use super::{SolveCtx, SolveStep, SolverOptions};

pub(crate) fn isolate(
    lhs: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    ctx: &SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    // Check solve recursion depth tracked in SolveCtx.
    ensure_recursion_depth_within_limit_or_error(ctx.depth(), MAX_SOLVE_RECURSION_DEPTH, || {
        CasError::SolverError("Maximum solver recursion depth exceeded in isolation.".to_string())
    })?;

    dispatch_isolation(lhs, rhs, op, var, simplifier, opts, ctx)
}
