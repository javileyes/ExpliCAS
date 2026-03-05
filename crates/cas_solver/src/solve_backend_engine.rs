//! Engine-backed solve backend implementation.

use crate::{CasError, Simplifier, SolveCtx, SolveStep};
use cas_ast::{Equation, SolutionSet};

use crate::solve_backend_contract::{CoreSolverOptions, SolveBackend};

/// Current backend implementation that delegates to `cas_engine`.
#[derive(Debug, Clone, Copy, Default)]
pub struct EngineSolveBackend;

impl SolveBackend for EngineSolveBackend {
    fn solve_with_ctx_and_options(
        eq: &Equation,
        var: &str,
        simplifier: &mut Simplifier,
        opts: CoreSolverOptions,
        ctx: &SolveCtx,
    ) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
        cas_engine::solve_with_ctx_and_options(eq, var, simplifier, opts, ctx)
    }
}
