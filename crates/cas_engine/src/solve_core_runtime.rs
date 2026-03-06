//! Core solve dispatch pipeline.
//!
//! Contains the internal solve pipeline (`solve_inner`).

use crate::engine::Simplifier;
use crate::error::CasError;
use cas_ast::{Equation, SolutionSet};
use cas_solver_core::solve_analysis::guard_solved_result_with_exclusions;
use cas_solver_core::strategy_order::SolveStrategyKind;

use crate::solve_runtime_adapters::{
    apply_strategy, build_solve_preflight_state, execute_strategy_pipeline,
    prepare_equation_for_strategy, SolveCtx, SolveStep, SolverOptions,
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
    opts: SolverOptions,
    parent_ctx: &SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let current_depth = parent_ctx.depth().saturating_add(1);

    cas_solver_core::solve_runtime_flow::solve_inner_with_default_entry_preflight_prepare_and_pipeline_with_state(
        simplifier,
        eq,
        var,
        current_depth,
        |state| &state.context,
        || {
            CasError::SolverError(
                "Maximum solver recursion depth exceeded. The equation may be too complex."
                    .to_string(),
            )
        },
        || CasError::VariableNotFound(var.to_string()),
        |state| build_solve_preflight_state(state, eq, var, opts.value_domain, parent_ctx),
        |state, equation, solve_var, solve_ctx| {
            apply_strategy(
                SolveStrategyKind::RationalExponent,
                equation,
                solve_var,
                state,
                &opts,
                solve_ctx,
            )
        },
        guard_solved_result_with_exclusions,
        prepare_equation_for_strategy,
        |state, prepared_eq| {
            // NOTE: Pre-solve exponent normalization (Div(p,q) → Number(p/q)),
            // nested-pow folding, and additive cancellation are applied above.
            cas_solver_core::solve_analysis::debug_assert_equation_no_top_level_sub(
                &state.context,
                prepared_eq,
            );
        },
        |state, original_eq, prepared_eq, residual, solve_var, solve_ctx, exclusions| {
            // Resolve var-eliminated residuals early, otherwise guard cycle + run strategies.
            execute_strategy_pipeline(
                state,
                original_eq,
                prepared_eq,
                residual,
                solve_var,
                opts,
                solve_ctx,
                exclusions,
            )
        },
    )
}
