//! Solve-oriented runtime entrypoints for the engine facade.

use crate::solve_backend_contract::{
    DisplaySolveSteps, SolveCtx, SolveDiagnostics, SolveStep, SolverOptions,
};
use crate::{CasError, Simplifier};
use cas_ast::{Equation, SolutionSet};
use cas_solver_core::solve_types::cleanup_display_solve_steps;

pub(crate) fn solve(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let ctx = SolveCtx::default();
    crate::solve_core_runtime::solve_inner(eq, var, simplifier, SolverOptions::default(), &ctx)
}

pub(crate) fn solve_with_display_steps(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
) -> Result<(SolutionSet, DisplaySolveSteps, SolveDiagnostics), CasError> {
    let ctx = SolveCtx::default();
    let result = crate::solve_core_runtime::solve_inner(eq, var, simplifier, opts, &ctx);
    cas_solver_core::solve_types::finalize_display_solve_with_ctx(
        &ctx,
        result,
        crate::collect_assumption_records,
        |raw_steps| {
            cleanup_display_solve_steps(
                &mut simplifier.context,
                raw_steps,
                opts.detailed_steps,
                var,
            )
        },
    )
}
