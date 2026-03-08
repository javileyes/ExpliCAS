//! Shared strategy-pipeline wrapper that binds a direct `apply_strategy`
//! entrypoint to the current solver options/context.

use cas_ast::{Equation, ExprId, SolutionSet};

/// Execute the default strategy pipeline while wiring a direct `apply_strategy`
/// entrypoint through the current runtime solve context and solver options.
#[allow(clippy::too_many_arguments)]
pub fn execute_strategy_pipeline_with_apply_entrypoint_runtime_ctx_and_default_verification_with_state<
    SState,
    FContainsVar,
    FCollectSteps,
    FContextRef,
    FContextMut,
    FRenderExpr,
    FApplyStrategyEntry,
    FSimplifyExpr,
    FAreEquivalent,
>(
    state: &mut SState,
    original_eq: &Equation,
    simplified_eq: &Equation,
    diff_simplified: ExprId,
    var: &str,
    opts: crate::solver_options::SolverOptions,
    ctx: &crate::solve_runtime_types::RuntimeSolveCtx,
    domain_exclusions: &[ExprId],
    contains_var: FContainsVar,
    collect_steps: FCollectSteps,
    context_ref: FContextRef,
    context_mut: FContextMut,
    render_expr: FRenderExpr,
    mut apply_strategy_entry: FApplyStrategyEntry,
    simplify_expr: FSimplifyExpr,
    are_equivalent: FAreEquivalent,
) -> Result<
    (
        SolutionSet,
        Vec<crate::solve_runtime_mapping::DefaultSolveStep>,
    ),
    crate::error_model::CasError,
>
where
    FContainsVar: FnMut(&mut SState, ExprId, &str) -> bool,
    FCollectSteps: FnMut(&mut SState) -> bool,
    FContextRef: FnMut(&mut SState) -> &cas_ast::Context + Clone,
    FContextMut: FnMut(&mut SState) -> &mut cas_ast::Context,
    FRenderExpr: Fn(&cas_ast::Context, ExprId) -> String,
    FApplyStrategyEntry: FnMut(
        crate::strategy_order::SolveStrategyKind,
        &Equation,
        &str,
        &mut SState,
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
    FSimplifyExpr: FnMut(&mut SState, ExprId) -> ExprId,
    FAreEquivalent: FnMut(&mut SState, ExprId, ExprId) -> bool,
{
    crate::solve_runtime_pipeline_strategy_execute_context_runtime::execute_strategy_pipeline_with_apply_strategy_runtime_ctx_and_default_verification_with_state(
        state,
        original_eq,
        simplified_eq,
        diff_simplified,
        var,
        opts,
        ctx,
        domain_exclusions,
        contains_var,
        collect_steps,
        context_ref,
        context_mut,
        render_expr,
        |state, strategy_kind, equation, solve_var, solve_opts, solve_ctx| {
            apply_strategy_entry(strategy_kind, equation, solve_var, state, &solve_opts, solve_ctx)
        },
        simplify_expr,
        are_equivalent,
    )
}
