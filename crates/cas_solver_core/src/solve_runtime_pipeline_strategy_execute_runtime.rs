//! Shared runtime adapter for strategy pipeline execution.

use cas_ast::{Equation, ExprId, SolutionSet};

/// Execute strategy pipeline using shared default step/error mappers.
#[allow(clippy::too_many_arguments)]
pub fn execute_strategy_pipeline_with_default_mappers_and_state<
    SState,
    FContainsVar,
    FCollectSteps,
    FContextRef,
    FContextMut,
    FRenderExpr,
    FApplyStrategy,
    FSoftError,
    FSubstituteSides,
    FSimplifyExpr,
    FAreEquivalent,
>(
    state: &mut SState,
    original_eq: &Equation,
    simplified_eq: &Equation,
    diff_simplified: ExprId,
    var: &str,
    domain_exclusions: &[ExprId],
    contains_var: FContainsVar,
    collect_steps: FCollectSteps,
    context_ref: FContextRef,
    context_mut: FContextMut,
    render_expr: FRenderExpr,
    apply_strategy: FApplyStrategy,
    is_soft_error: FSoftError,
    substitute_sides: FSubstituteSides,
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
    FApplyStrategy: FnMut(
        &mut SState,
        crate::strategy_order::SolveStrategyKind,
    ) -> Option<
        Result<
            (
                SolutionSet,
                Vec<crate::solve_runtime_mapping::DefaultSolveStep>,
            ),
            crate::error_model::CasError,
        >,
    >,
    FSoftError: FnMut(&crate::error_model::CasError) -> bool,
    FSubstituteSides: FnMut(&mut SState, &Equation, &str, ExprId) -> (ExprId, ExprId),
    FSimplifyExpr: FnMut(&mut SState, ExprId) -> ExprId,
    FAreEquivalent: FnMut(&mut SState, ExprId, ExprId) -> bool,
{
    crate::solve_runtime_flow::execute_default_strategy_order_pipeline_with_default_cycle_guard_and_default_var_elimination_and_discrete_resolution_with_state(
        state,
        original_eq,
        simplified_eq,
        diff_simplified,
        var,
        domain_exclusions,
        contains_var,
        collect_steps,
        context_ref.clone(),
        context_ref,
        context_mut,
        render_expr,
        crate::solve_runtime_mapping::medium_step,
        crate::solve_runtime_mapping::solver_cycle_detected_error,
        apply_strategy,
        is_soft_error,
        substitute_sides,
        simplify_expr,
        are_equivalent,
        crate::solve_runtime_mapping::map_no_strategy_solved_error(),
    )
}
