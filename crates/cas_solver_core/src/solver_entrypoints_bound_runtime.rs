//! Shared solve-oriented entrypoint helpers bound to the current runtime solve
//! context and runtime solve type aliases.

use cas_ast::{Equation, SolutionSet};

/// Cleanup display-ready solve steps through the runtime state context.
pub fn cleanup_display_solve_steps_with_runtime_state<SState>(
    state: &mut SState,
    raw_steps: Vec<crate::solve_runtime_types::RuntimeSolveStep>,
    detailed: bool,
    var: &str,
) -> crate::solve_runtime_types::RuntimeDisplaySolveSteps
where
    SState: crate::solve_runtime_adapter_state_runtime::RuntimeSolveAdapterState,
{
    crate::solve_types::cleanup_display_solve_steps(
        state.runtime_context_mut(),
        raw_steps,
        detailed,
        var,
    )
}

/// Solve with a fresh default runtime solve context.
pub fn solve_with_default_runtime_ctx_and_backend_with_state<SState, FSolveBackend>(
    equation: &Equation,
    var: &str,
    state: &mut SState,
    opts: crate::solver_options::SolverOptions,
    mut solve_backend: FSolveBackend,
) -> Result<
    (
        SolutionSet,
        Vec<crate::solve_runtime_types::RuntimeSolveStep>,
    ),
    crate::error_model::CasError,
>
where
    FSolveBackend: FnMut(
        &Equation,
        &str,
        &mut SState,
        crate::solver_options::SolverOptions,
        &crate::solve_runtime_types::RuntimeSolveCtx,
    ) -> Result<
        (
            SolutionSet,
            Vec<crate::solve_runtime_types::RuntimeSolveStep>,
        ),
        crate::error_model::CasError,
    >,
{
    let ctx = crate::solve_runtime_types::RuntimeSolveCtx::default();
    solve_backend(equation, var, state, opts, &ctx)
}

/// Solve with a fresh default runtime solve context and finalize display steps
/// plus diagnostics through the shared runtime aliases.
pub fn solve_with_display_steps_with_default_runtime_ctx_and_backend_with_state<
    SState,
    FSolveBackend,
    FBuildAssumedRecords,
    FCleanupSteps,
>(
    equation: &Equation,
    var: &str,
    state: &mut SState,
    opts: crate::solver_options::SolverOptions,
    mut solve_backend: FSolveBackend,
    build_assumed_records: FBuildAssumedRecords,
    cleanup_steps: FCleanupSteps,
) -> Result<
    (
        SolutionSet,
        crate::solve_runtime_types::RuntimeDisplaySolveSteps,
        crate::solve_runtime_types::RuntimeSolveDiagnostics,
    ),
    crate::error_model::CasError,
>
where
    FSolveBackend: FnMut(
        &Equation,
        &str,
        &mut SState,
        crate::solver_options::SolverOptions,
        &crate::solve_runtime_types::RuntimeSolveCtx,
    ) -> Result<
        (
            SolutionSet,
            Vec<crate::solve_runtime_types::RuntimeSolveStep>,
        ),
        crate::error_model::CasError,
    >,
    FBuildAssumedRecords: FnMut(
        &[crate::assumption_model::AssumptionEvent],
    ) -> Vec<crate::assumption_model::AssumptionRecord>,
    FCleanupSteps: FnOnce(
        &mut SState,
        Vec<crate::solve_runtime_types::RuntimeSolveStep>,
    ) -> crate::solve_runtime_types::RuntimeDisplaySolveSteps,
{
    let ctx = crate::solve_runtime_types::RuntimeSolveCtx::default();
    let result = solve_backend(equation, var, state, opts, &ctx);
    crate::solve_types::finalize_display_solve_with_ctx(
        &ctx,
        result,
        build_assumed_records,
        |raw_steps| cleanup_steps(state, raw_steps),
    )
}
