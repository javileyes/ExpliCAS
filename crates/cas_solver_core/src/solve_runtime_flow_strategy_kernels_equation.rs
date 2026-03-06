use cas_ast::{Equation, ExprId, SolutionSet};

/// Apply quadratic strategy using default factorized and candidate pipelines
/// with runtime-provided callbacks.
#[allow(clippy::too_many_arguments)]
pub fn apply_quadratic_strategy_with_default_kernels_and_state<
    T,
    S,
    SS,
    E,
    FCollectSteps,
    FContextRef,
    FContextMut,
    FSetCollect,
    FSimplify,
    FExpand,
    FRender,
    FSolveFactor,
    FMapStep,
    FMapSubstep,
    FMapPlanError,
    FOnQuadraticCoefficientPathSolved,
>(
    state: &mut T,
    equation: &Equation,
    var: &str,
    mut collect_steps: FCollectSteps,
    is_real_only: bool,
    context_ref: FContextRef,
    context_mut: FContextMut,
    set_collecting: FSetCollect,
    simplify_expr: FSimplify,
    expand_expr: FExpand,
    render_expr: FRender,
    solve_factor: FSolveFactor,
    map_step: FMapStep,
    map_substep: FMapSubstep,
    map_plan_error: FMapPlanError,
    on_quadratic_coefficient_path_solved: FOnQuadraticCoefficientPathSolved,
) -> Option<Result<(SolutionSet, Vec<S>), E>>
where
    SS: Clone,
    FCollectSteps: FnMut(&mut T) -> bool,
    FContextRef: Fn(&mut T) -> &cas_ast::Context,
    FContextMut: Fn(&mut T) -> &mut cas_ast::Context,
    FSetCollect: FnMut(&mut T, bool),
    FSimplify: FnMut(&mut T, ExprId) -> ExprId,
    FExpand: FnMut(&mut T, ExprId) -> ExprId,
    FRender: Fn(&cas_ast::Context, ExprId) -> String,
    FSolveFactor: FnMut(&mut T, &Equation) -> Result<(SolutionSet, Vec<S>), E>,
    FMapStep: FnMut(String, Equation, Option<Vec<SS>>) -> S,
    FMapSubstep: FnMut(String, Equation) -> SS,
    FMapPlanError: FnOnce(crate::quadratic_formula::QuadraticCoefficientSolvePlanError) -> E,
    FOnQuadraticCoefficientPathSolved: FnMut(&mut T),
{
    let include_items = collect_steps(state);
    crate::quadratic_strategy::execute_quadratic_strategy_with_default_factorized_and_candidate_pipelines_and_unified_step_mappers_with_state(
        state,
        equation,
        var,
        include_items,
        is_real_only,
        context_ref,
        context_mut,
        set_collecting,
        simplify_expr,
        expand_expr,
        render_expr,
        solve_factor,
        map_step,
        map_substep,
        map_plan_error,
        on_quadratic_coefficient_path_solved,
    )
}

/// Apply isolation strategy using default routing and runtime-provided
/// callbacks.
#[allow(clippy::too_many_arguments)]
pub fn apply_isolation_strategy_with_default_kernels_and_state<
    T,
    S,
    E,
    FCollectSteps,
    FContextRef,
    FSolveEquation,
    FMapStep,
    FVariableNotFoundError,
>(
    state: &mut T,
    equation: &Equation,
    var: &str,
    mut collect_steps: FCollectSteps,
    context_ref: FContextRef,
    solve_equation: FSolveEquation,
    map_step: FMapStep,
    variable_not_found_error: FVariableNotFoundError,
) -> Option<Result<(SolutionSet, Vec<S>), E>>
where
    FCollectSteps: FnMut(&mut T) -> bool,
    FContextRef: Fn(&mut T) -> &cas_ast::Context,
    FSolveEquation: FnMut(&mut T, &Equation, &str) -> Result<(SolutionSet, Vec<S>), E>,
    FMapStep: FnMut(String, Equation) -> S,
    FVariableNotFoundError: FnMut(&str) -> E,
{
    let include_item = collect_steps(state);
    crate::isolation_strategy::execute_isolation_strategy_with_default_routing_and_unified_step_mapper_with_state(
        state,
        equation,
        var,
        include_item,
        context_ref,
        solve_equation,
        map_step,
        variable_not_found_error,
    )
}
