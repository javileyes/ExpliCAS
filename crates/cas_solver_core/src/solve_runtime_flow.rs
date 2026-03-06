//! Shared solve runtime orchestration for integration crates.
//!
//! This keeps default strategy-order and discrete-verification wiring out of
//! runtime wrappers (`cas_engine`, `cas_solver`) so they only provide kernels.

use cas_ast::{Equation, ExprId, SolutionSet};

pub use crate::solve_runtime_flow_preflight::*;
pub use crate::solve_runtime_flow_orchestration::*;
pub use crate::solve_runtime_flow_pipeline::*;
#[allow(unused_imports)]
pub use crate::solve_runtime_flow_isolation::*;
pub use crate::solve_runtime_flow_isolation_kernels::*;
pub use crate::solve_runtime_flow_strategy::*;

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
