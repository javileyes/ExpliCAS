//! Solve-oriented facade entrypoints.

use crate::{
    solve_backend_dispatch, CasError, DisplaySolveSteps, Simplifier, SolveCtx, SolveDiagnostics,
    SolveStep, SolverOptions,
};

/// Solve an equation for a variable.
pub fn solve(
    eq: &cas_ast::Equation,
    var: &str,
    simplifier: &mut Simplifier,
) -> Result<(cas_ast::SolutionSet, Vec<SolveStep>), CasError> {
    let ctx = SolveCtx::default();
    solve_backend_dispatch::solve_with_engine_backend(
        eq,
        var,
        simplifier,
        SolverOptions::default().to_core(),
        &ctx,
    )
}

/// Solve with display-ready steps and diagnostics.
pub fn solve_with_display_steps(
    eq: &cas_ast::Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
) -> Result<(cas_ast::SolutionSet, DisplaySolveSteps, SolveDiagnostics), CasError> {
    let ctx = SolveCtx::default();
    let result = solve_backend_dispatch::solve_with_engine_backend(
        eq,
        var,
        simplifier,
        opts.to_core(),
        &ctx,
    );
    cas_solver_core::solve_types::finalize_display_solve_with_ctx(
        &ctx,
        result,
        crate::collect_assumption_records,
        |raw_steps| {
            cas_solver_core::solve_types::cleanup_display_solve_steps(
                &mut simplifier.context,
                raw_steps,
                opts.detailed_steps,
                var,
            )
        },
    )
}
