//! Core solve dispatch pipeline.
//!
//! Contains the internal solve pipeline (`solve_inner`).

use crate::solve_backend_contract::CoreSolverOptions;
use crate::{CasError, Simplifier};
use cas_ast::{Equation, SolutionSet};

use crate::solve_runtime_adapters::{
    apply_strategy, build_solve_preflight_state, execute_strategy_pipeline,
    prepare_equation_for_strategy, SolveCtx, SolveStep,
};

/// Core solver implementation.
///
/// All public entry points delegate here. `parent_ctx` carries the shared
/// accumulator so that conditions from recursive calls are visible to the
/// top-level caller.
pub(crate) fn solve_inner(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: CoreSolverOptions,
    parent_ctx: &SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    cas_solver_core::solve_runtime_orchestration_context_runtime::solve_inner_with_runtime_ctx_and_default_rational_preflight_prepare_pipeline_with_state(
        simplifier,
        eq,
        var,
        opts,
        parent_ctx,
        |state| &state.context,
        || {
            CasError::SolverError(
                "Maximum solver recursion depth exceeded. The equation may be too complex."
                    .to_string(),
            )
        },
        || CasError::VariableNotFound(var.to_string()),
        |state, equation, solve_var, value_domain, solve_ctx| {
            build_solve_preflight_state(state, equation, solve_var, value_domain, solve_ctx)
        },
        |strategy_kind, equation, solve_var, state, solve_opts, solve_ctx| {
            apply_strategy(
                strategy_kind,
                equation,
                solve_var,
                state,
                solve_opts,
                solve_ctx,
            )
        },
        prepare_equation_for_strategy,
        |state, original_eq, prepared_eq, residual, solve_var, solve_opts, solve_ctx, exclusions| {
            execute_strategy_pipeline(
                state,
                original_eq,
                prepared_eq,
                residual,
                solve_var,
                solve_opts,
                solve_ctx,
                exclusions,
            )
        },
    )
}
