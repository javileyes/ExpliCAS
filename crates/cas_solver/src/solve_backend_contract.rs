//! Solve backend contract types.

use crate::{CasError, Simplifier, SolveCtx, SolveStep};
use cas_ast::{Equation, SolutionSet};

/// Core solve options shared by backend implementations.
pub type CoreSolverOptions = cas_solver_core::solver_options::SolverOptions;

/// Solver backend contract used by `cas_solver` facade entrypoints.
pub trait SolveBackend {
    fn solve_with_ctx_and_options(
        eq: &Equation,
        var: &str,
        simplifier: &mut Simplifier,
        opts: CoreSolverOptions,
        ctx: &SolveCtx,
    ) -> Result<(SolutionSet, Vec<SolveStep>), CasError>;
}
