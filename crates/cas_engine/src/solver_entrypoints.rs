//! Solve-oriented runtime entrypoints for the engine facade.

use crate::solve_backend_contract::{
    DisplaySolveSteps, SolveDiagnostics, SolveStep, SolverOptions,
};
use crate::{CasError, Simplifier};
use cas_ast::{Equation, SolutionSet};

pub(crate) fn solve(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    cas_solver_core::solver_entrypoints_bound_runtime::solve_with_default_runtime_ctx_and_backend_with_state(
        eq,
        var,
        simplifier,
        SolverOptions::default().into(),
        crate::solve_core_runtime::solve_inner,
    )
}

pub(crate) fn solve_with_display_steps(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
) -> Result<(SolutionSet, DisplaySolveSteps, SolveDiagnostics), CasError> {
    cas_solver_core::solver_entrypoints_bound_runtime::solve_with_display_steps_with_default_runtime_ctx_and_backend_with_state(
        eq,
        var,
        simplifier,
        opts.into(),
        crate::solve_core_runtime::solve_inner,
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
