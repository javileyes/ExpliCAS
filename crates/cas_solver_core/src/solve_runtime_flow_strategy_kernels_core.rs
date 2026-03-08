use cas_ast::{Equation, ExprId, SolutionSet};

/// Apply rational-exponent strategy using default kernel wiring and
/// runtime-provided callbacks.
#[allow(clippy::too_many_arguments)]
pub fn apply_rational_exponent_strategy_with_default_kernels_and_state<
    T,
    S,
    E,
    FCollectSteps,
    FContextMut,
    FSimplifyExpr,
    FSolveEquation,
    FMapStep,
>(
    state: &mut T,
    equation: &Equation,
    var: &str,
    mut collect_steps: FCollectSteps,
    context_mut: FContextMut,
    simplify_expr: FSimplifyExpr,
    solve_equation: FSolveEquation,
    map_step: FMapStep,
) -> Option<Result<(SolutionSet, Vec<S>), E>>
where
    FCollectSteps: FnMut(&mut T) -> bool,
    FContextMut: FnMut(&mut T) -> &mut cas_ast::Context,
    FSimplifyExpr: FnMut(&mut T, ExprId) -> ExprId,
    FSolveEquation: FnMut(&mut T, &Equation, &str) -> Result<(SolutionSet, Vec<S>), E>,
    FMapStep: FnMut(String, Equation) -> S,
{
    let include_item = collect_steps(state);
    crate::isolation_strategy::execute_rational_exponent_strategy_with_default_kernel_and_accept_all_solutions_and_unified_step_mapper_with_state(
        state,
        equation,
        var,
        include_item,
        context_mut,
        simplify_expr,
        solve_equation,
        map_step,
    )
}

/// Apply collect-terms strategy using default kernel wiring and
/// runtime-provided callbacks.
#[allow(clippy::too_many_arguments)]
pub fn apply_collect_terms_strategy_with_default_kernels_and_state<
    T,
    S,
    E,
    FCollectSteps,
    FContextMut,
    FSimplifyExpr,
    FRenderExpr,
    FSolveEquation,
    FMapStep,
>(
    state: &mut T,
    equation: &Equation,
    var: &str,
    mut collect_steps: FCollectSteps,
    context_mut: FContextMut,
    simplify_expr: FSimplifyExpr,
    render_expr: FRenderExpr,
    solve_equation: FSolveEquation,
    map_step: FMapStep,
) -> Option<Result<(SolutionSet, Vec<S>), E>>
where
    FCollectSteps: FnMut(&mut T) -> bool,
    FContextMut: FnMut(&mut T) -> &mut cas_ast::Context,
    FSimplifyExpr: FnMut(&mut T, ExprId) -> ExprId,
    FRenderExpr: FnMut(&mut T, ExprId) -> String,
    FSolveEquation: FnMut(&mut T, &Equation, &str) -> Result<(SolutionSet, Vec<S>), E>,
    FMapStep: FnMut(String, Equation) -> S,
{
    let include_item = collect_steps(state);
    crate::isolation_strategy::execute_collect_terms_strategy_with_default_kernel_and_unified_step_mapper_with_state(
        state,
        equation,
        var,
        include_item,
        context_mut,
        simplify_expr,
        render_expr,
        solve_equation,
        map_step,
    )
}

/// Apply exponential-substitution strategy using default substitution variable
/// and rewrite-plan derivation.
#[allow(clippy::too_many_arguments)]
pub fn apply_substitution_strategy_with_default_kernels_and_state<
    T,
    S,
    E,
    FCollectSteps,
    FContextMut,
    FRenderExpr,
    FSolveEquation,
    FMapStep,
>(
    state: &mut T,
    equation: &Equation,
    var: &str,
    mut collect_steps: FCollectSteps,
    context_mut: FContextMut,
    render_expr: FRenderExpr,
    solve_equation: FSolveEquation,
    map_step: FMapStep,
) -> Option<Result<(SolutionSet, Vec<S>), E>>
where
    FCollectSteps: FnMut(&mut T) -> bool,
    FContextMut: Fn(&mut T) -> &mut cas_ast::Context,
    FRenderExpr: FnMut(&mut T, ExprId) -> String,
    FSolveEquation: FnMut(&mut T, &Equation, &str) -> Result<(SolutionSet, Vec<S>), E>,
    FMapStep: FnMut(String, Equation) -> S,
{
    let include_item = collect_steps(state);
    crate::substitution::execute_exponential_substitution_strategy_result_pipeline_with_default_substitution_var_and_plan_with_state(
        state,
        equation,
        var,
        include_item,
        context_mut,
        render_expr,
        solve_equation,
        map_step,
    )
}

/// Apply rational-roots strategy using default limits, default root sorting,
/// and runtime-provided callbacks.
#[allow(clippy::too_many_arguments)]
pub fn apply_rational_roots_strategy_with_default_kernels_and_state<
    T,
    S,
    FCollectSteps,
    FContextMut,
    FSimplifyExpr,
    FExpandExpr,
    FMapStep,
>(
    state: &mut T,
    equation: &Equation,
    var: &str,
    mut collect_steps: FCollectSteps,
    context_mut: FContextMut,
    simplify_expr: FSimplifyExpr,
    expand_expr: FExpandExpr,
    map_step: FMapStep,
) -> Option<(SolutionSet, Vec<S>)>
where
    FCollectSteps: FnMut(&mut T) -> bool,
    FContextMut: Fn(&mut T) -> &mut cas_ast::Context,
    FSimplifyExpr: FnMut(&mut T, ExprId) -> ExprId,
    FExpandExpr: FnMut(&mut T, ExprId) -> ExprId,
    FMapStep: FnMut(String, Equation) -> S,
{
    let include_item = collect_steps(state);
    crate::rational_roots::execute_rational_roots_strategy_with_default_limits_and_default_root_sorting_and_unified_step_mapper_with_state(
        state,
        equation,
        var,
        include_item,
        context_mut,
        simplify_expr,
        expand_expr,
        map_step,
    )
}

/// Apply unwrap strategy using default route/residual-hint wiring and
/// runtime-provided callbacks.
#[allow(clippy::too_many_arguments)]
pub fn apply_unwrap_strategy_with_default_kernels_and_state<
    T,
    S,
    E,
    FCollectSteps,
    FContextMut,
    FClassifyLogSolve,
    FRenderExpr,
    FNoteAssumption,
    FSolveEquation,
    FMapStep,
>(
    state: &mut T,
    equation: &Equation,
    var: &str,
    mut collect_steps: FCollectSteps,
    context_mut: FContextMut,
    mode: crate::log_domain::DomainModeKind,
    wildcard_scope: bool,
    classify_log_solve: FClassifyLogSolve,
    render_expr: FRenderExpr,
    note_assumption: FNoteAssumption,
    solve_equation: FSolveEquation,
    map_step: FMapStep,
) -> Option<Result<(SolutionSet, Vec<S>), E>>
where
    FCollectSteps: FnMut(&mut T) -> bool,
    FContextMut: Fn(&mut T) -> &mut cas_ast::Context,
    FClassifyLogSolve:
        FnMut(&cas_ast::Context, ExprId, ExprId) -> crate::log_domain::LogSolveDecision,
    FRenderExpr: FnMut(&cas_ast::Context, ExprId) -> String,
    FNoteAssumption: FnMut(&mut T, crate::unwrap_plan::LogLinearAssumptionRecord),
    FSolveEquation: FnMut(&mut T, &Equation, &str) -> Result<(SolutionSet, Vec<S>), E>,
    FMapStep: FnMut(String, Equation) -> S,
{
    let include_item = collect_steps(state);
    crate::isolation_strategy::execute_unwrap_strategy_with_default_route_and_residual_hint_and_unified_step_mapper_with_state(
        state,
        equation,
        var,
        include_item,
        context_mut,
        mode,
        wildcard_scope,
        classify_log_solve,
        render_expr,
        note_assumption,
        solve_equation,
        map_step,
    )
}
