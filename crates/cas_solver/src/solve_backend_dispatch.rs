//! Solve backend dispatch layer.

use crate::{CasError, Simplifier, SolveCtx, SolveStep};
use cas_ast::{Equation, SolutionSet};

use crate::solve_backend::{CoreSolverOptions, SolveBackend};
use crate::solve_backend_active::ActiveSolveBackend;

/// Execute solve using the active engine-backed migration boundary.
pub fn solve_with_engine_backend(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: CoreSolverOptions,
    ctx: &SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    ActiveSolveBackend::solve_with_ctx_and_options(eq, var, simplifier, opts, ctx)
}
