use super::super::solve_runtime_flow_pipeline_resolution::{
    resolve_discrete_strategy_result_against_equation_with_default_symbolic_and_substitution_verification_with_state,
    resolve_var_eliminated_residual_with_default_exclusion_policy_and_unified_step_mapper_with_state,
};
use super::solve_runtime_flow_pipeline_execute_core::execute_default_strategy_order_pipeline_with_state;
use cas_ast::{Equation, ExprId, SolutionSet};

/// Execute strategy pipeline with default var-elimination and discrete-result
/// resolvers, while keeping strategy execution and cycle guards runtime-defined.
#[allow(clippy::too_many_arguments)]
pub fn execute_default_strategy_order_pipeline_with_default_var_elimination_and_discrete_resolution_with_state<
    SState,
    S,
    E,
    Guard,
    FContainsVar,
    FCollectSteps,
    FContextRef,
    FContextMut,
    FRenderExpr,
    FMapStep,
    FEnterCycle,
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
    mut contains_var: FContainsVar,
    mut collect_steps: FCollectSteps,
    mut context_ref: FContextRef,
    mut context_mut: FContextMut,
    render_expr: FRenderExpr,
    mut map_step: FMapStep,
    enter_cycle: FEnterCycle,
    apply_strategy: FApplyStrategy,
    is_soft_error: FSoftError,
    mut substitute_sides: FSubstituteSides,
    mut simplify_expr: FSimplifyExpr,
    mut are_equivalent: FAreEquivalent,
    no_solution_error: E,
) -> Result<(SolutionSet, Vec<S>), E>
where
    FContainsVar: FnMut(&mut SState, ExprId, &str) -> bool,
    FCollectSteps: FnMut(&mut SState) -> bool,
    FContextRef: FnMut(&mut SState) -> &cas_ast::Context,
    FContextMut: FnMut(&mut SState) -> &mut cas_ast::Context,
    FRenderExpr: Fn(&cas_ast::Context, ExprId) -> String,
    FMapStep: FnMut(String, Equation) -> S,
    FEnterCycle: FnMut(&mut SState, &Equation, &str) -> Result<Guard, E>,
    FApplyStrategy: FnMut(
        &mut SState,
        crate::strategy_order::SolveStrategyKind,
    ) -> Option<Result<(SolutionSet, Vec<S>), E>>,
    FSoftError: FnMut(&E) -> bool,
    FSubstituteSides: FnMut(&mut SState, &Equation, &str, ExprId) -> (ExprId, ExprId),
    FSimplifyExpr: FnMut(&mut SState, ExprId) -> ExprId,
    FAreEquivalent: FnMut(&mut SState, ExprId, ExprId) -> bool,
{
    let render_expr = &render_expr;

    execute_default_strategy_order_pipeline_with_state(
        state,
        original_equation,
        normalized_equation,
        residual,
        var,
        |state, expr, var_name| contains_var(state, expr, var_name),
        |state, residual, var_name| {
            Ok(resolve_var_eliminated_residual_with_default_exclusion_policy_and_unified_step_mapper_with_state(
                state,
                residual,
                var_name,
                domain_exclusions,
                |state| collect_steps(state),
                |state| context_mut(state),
                |ctx, expr| render_expr(ctx, expr),
                &mut map_step,
            ))
        },
        enter_cycle,
        apply_strategy,
        is_soft_error,
        |state, equation, var_name, solutions, steps| {
            resolve_discrete_strategy_result_against_equation_with_default_symbolic_and_substitution_verification_with_state(
                state,
                equation,
                var_name,
                solutions,
                steps,
                |state| context_ref(state),
                |state, equation, solve_var, solution| {
                    substitute_sides(state, equation, solve_var, solution)
                },
                |state, expr| simplify_expr(state, expr),
                |state, lhs, rhs| are_equivalent(state, lhs, rhs),
            )
        },
        no_solution_error,
    )
}
