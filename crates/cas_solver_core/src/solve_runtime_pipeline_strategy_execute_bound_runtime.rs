//! Shared strategy-pipeline wrapper bound to [`RuntimeSolveAdapterState`].

use cas_ast::{Equation, ExprId, SolutionSet};

/// Execute the default strategy pipeline using default runtime-state helpers
/// while only requiring a direct `apply_strategy` entrypoint.
#[allow(clippy::too_many_arguments)]
pub fn execute_strategy_pipeline_with_runtime_state_and_apply_entrypoint_with_state<
    T,
    FApplyStrategyEntry,
>(
    state: &mut T,
    original_eq: &Equation,
    simplified_eq: &Equation,
    diff_simplified: ExprId,
    var: &str,
    opts: crate::solver_options::SolverOptions,
    ctx: &crate::solve_runtime_types::RuntimeSolveCtx,
    domain_exclusions: &[ExprId],
    apply_strategy_entry: FApplyStrategyEntry,
) -> Result<
    (
        SolutionSet,
        Vec<crate::solve_runtime_mapping::DefaultSolveStep>,
    ),
    crate::error_model::CasError,
>
where
    T: crate::solve_runtime_adapter_state_runtime::RuntimeSolveAdapterState,
    FApplyStrategyEntry: FnMut(
        crate::strategy_order::SolveStrategyKind,
        &Equation,
        &str,
        &mut T,
        &crate::solver_options::SolverOptions,
        &crate::solve_runtime_types::RuntimeSolveCtx,
    ) -> Option<
        Result<
            (
                SolutionSet,
                Vec<crate::solve_runtime_mapping::DefaultSolveStep>,
            ),
            crate::error_model::CasError,
        >,
    >,
{
    crate::solve_runtime_pipeline_strategy_execute_apply_entrypoint_context_runtime::execute_strategy_pipeline_with_apply_entrypoint_runtime_ctx_and_default_verification_with_state(
        state,
        original_eq,
        simplified_eq,
        diff_simplified,
        var,
        opts,
        ctx,
        domain_exclusions,
        crate::solve_runtime_adapter_state_runtime::simplifier_contains_var,
        crate::solve_runtime_adapter_state_runtime::simplifier_collect_steps,
        crate::solve_runtime_adapter_state_runtime::simplifier_context,
        crate::solve_runtime_adapter_state_runtime::simplifier_context_mut,
        crate::solve_runtime_adapter_state_runtime::context_render_expr,
        apply_strategy_entry,
        crate::solve_runtime_adapter_state_runtime::simplifier_simplify_expr,
        crate::solve_runtime_adapter_state_runtime::simplifier_are_equivalent,
    )
}
