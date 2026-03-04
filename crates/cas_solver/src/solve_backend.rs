//! Backend boundary for solve execution.
//!
//! Keeps `cas_engine` coupling in one place so `cas_solver` API can switch to
//! a local/native backend incrementally during migration.

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

/// Execute solve using the active engine-backed migration boundary.
pub fn solve_with_engine_backend(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: CoreSolverOptions,
    ctx: &SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    EngineSolveBackend::solve_with_ctx_and_options(eq, var, simplifier, opts, ctx)
}
