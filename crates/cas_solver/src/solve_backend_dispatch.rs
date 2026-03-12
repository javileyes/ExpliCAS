//! Solve backend dispatch layer.

use crate::{CasError, Simplifier, SolveCtx, SolveStep};
use cas_ast::{Equation, SolutionSet};

use crate::solve_backend_contract::{CoreSolverOptions, SolveBackend};
use crate::solve_backend_local::LocalSolveBackend;

/// Execute solve using the currently selected active backend.
pub fn solve_with_active_backend(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: CoreSolverOptions,
    ctx: &SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    LocalSolveBackend::solve_with_ctx_and_options(eq, var, simplifier, opts, ctx)
}
