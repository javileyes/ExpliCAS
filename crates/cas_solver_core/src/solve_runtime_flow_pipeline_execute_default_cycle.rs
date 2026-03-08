use super::solve_runtime_flow_pipeline_execute_default_resolve::execute_default_strategy_order_pipeline_with_default_var_elimination_and_discrete_resolution_with_state;
use cas_ast::{Equation, ExprId, SolutionSet};

/// Execute strategy pipeline with:
/// - default equation-fingerprint cycle guard,
/// - default var-elimination and discrete-result resolvers,
///
/// Strategy execution and verification kernels remain runtime-defined.
#[allow(clippy::too_many_arguments)]
pub fn execute_default_strategy_order_pipeline_with_default_cycle_guard_and_default_var_elimination_and_discrete_resolution_with_state<
    SState,
    S,
    E,
    FContainsVar,
    FCollectSteps,
    FContextRef,
    FContextRefForCycle,
    FContextMut,
    FRenderExpr,
    FMapStep,
    FMapCycleError,
    FApplyStrategy,
    FSoftError,
    FSubstituteSides,
    FSimplifyExpr,
    FAreEquivalent,
>(
    state: &mut SState,
    original_equation: &Equation,
    normalized_equation: &Equation,
    residual: ExprId,
    var: &str,
    domain_exclusions: &[ExprId],
    contains_var: FContainsVar,
    collect_steps: FCollectSteps,
    context_ref: FContextRef,
    context_ref_for_cycle: FContextRefForCycle,
    context_mut: FContextMut,
    render_expr: FRenderExpr,
    map_step: FMapStep,
    mut map_cycle_error: FMapCycleError,
    apply_strategy: FApplyStrategy,
    is_soft_error: FSoftError,
    substitute_sides: FSubstituteSides,
    simplify_expr: FSimplifyExpr,
    are_equivalent: FAreEquivalent,
    no_solution_error: E,
) -> Result<(SolutionSet, Vec<S>), E>
where
    FContainsVar: FnMut(&mut SState, ExprId, &str) -> bool,
    FCollectSteps: FnMut(&mut SState) -> bool,
    FContextRef: FnMut(&mut SState) -> &cas_ast::Context,
    FContextRefForCycle: FnMut(&mut SState) -> &cas_ast::Context,
    FContextMut: FnMut(&mut SState) -> &mut cas_ast::Context,
    FRenderExpr: Fn(&cas_ast::Context, ExprId) -> String,
    FMapStep: FnMut(String, Equation) -> S,
    FMapCycleError: FnMut() -> E,
    FApplyStrategy: FnMut(
        &mut SState,
        crate::strategy_order::SolveStrategyKind,
    ) -> Option<Result<(SolutionSet, Vec<S>), E>>,
    FSoftError: FnMut(&E) -> bool,
    FSubstituteSides: FnMut(&mut SState, &Equation, &str, ExprId) -> (ExprId, ExprId),
    FSimplifyExpr: FnMut(&mut SState, ExprId) -> ExprId,
    FAreEquivalent: FnMut(&mut SState, ExprId, ExprId) -> bool,
{
    let context_ref_for_cycle = std::cell::RefCell::new(context_ref_for_cycle);
    execute_default_strategy_order_pipeline_with_default_var_elimination_and_discrete_resolution_with_state(
        state,
        original_equation,
        normalized_equation,
        residual,
        var,
        domain_exclusions,
        contains_var,
        collect_steps,
        context_ref,
        context_mut,
        render_expr,
        map_step,
        |state, equation, var_name| {
            crate::solve_analysis::try_enter_equation_cycle_guard_with_error(
                (context_ref_for_cycle.borrow_mut())(state),
                equation,
                var_name,
                &mut map_cycle_error,
            )
        },
        apply_strategy,
        is_soft_error,
        substitute_sides,
        simplify_expr,
        are_equivalent,
        no_solution_error,
    )
}
