//! Solve-oriented facade entrypoints.

use crate::{
    solve_backend_dispatch, CasError, DisplaySolveSteps, Simplifier, SolveDiagnostics, SolveStep,
    SolverOptions,
};

/// Solve an equation for a variable.
pub fn solve(
    eq: &cas_ast::Equation,
    var: &str,
    simplifier: &mut Simplifier,
) -> Result<(cas_ast::SolutionSet, Vec<SolveStep>), CasError> {
    cas_solver_core::solver_entrypoints_bound_runtime::solve_with_default_runtime_ctx_and_backend_with_state(
        eq,
        var,
        simplifier,
        SolverOptions::default().to_core(),
        solve_backend_dispatch::solve_with_active_backend,
    )
}

/// Solve with display-ready steps and diagnostics.
pub fn solve_with_display_steps(
    eq: &cas_ast::Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
) -> Result<(cas_ast::SolutionSet, DisplaySolveSteps, SolveDiagnostics), CasError> {
    cas_solver_core::solver_entrypoints_bound_runtime::solve_with_display_steps_with_default_runtime_ctx_and_backend_with_state(
        eq,
        var,
        simplifier,
        opts.to_core(),
        solve_backend_dispatch::solve_with_active_backend,
        crate::collect_assumption_records,
        |simplifier, raw_steps| {
            cas_solver_core::solver_entrypoints_bound_runtime::cleanup_display_solve_steps_with_runtime_state(
                simplifier,
                raw_steps,
                opts.detailed_steps,
                var,
            )
        },
    )
}
