//! Function-isolation orchestration helpers shared by engine-side solver.
//!
//! These wrappers keep equation-rewrite orchestration in `cas_solver_core`
//! while callers provide stateful hooks for simplification and recursion.

use cas_ast::{ExprId, RelOp, SolutionSet};

use crate::function_inverse::{FunctionIsolationRoute, FunctionIsolationRouteError};
use crate::log_isolation::LogIsolationRewritePlan;
use crate::solve_outcome::{
    execute_abs_isolation_plan_with_rhs_sign_pipeline_with_optional_items_and_solver_with_state,
    execute_log_isolation_result_pipeline_with_plan_and_error_and_merge_with_existing_steps,
    execute_unary_inverse_result_pipeline_or_else_with_and_merge_with_existing_steps_with_state,
    finalize_abs_split_solution_set_for_rhs, plan_abs_isolation_with_rhs_sign, AbsIsolationPlan,
    AbsSplitExecutionItem,
};

/// Dispatch function-isolation routing to caller-provided branch handlers.
pub fn execute_function_isolation_route_with<T, E, FAbs, FLog, FUnary, FVarMissing, FUnsupported>(
    routing: Result<FunctionIsolationRoute, FunctionIsolationRouteError>,
    on_abs_unary: FAbs,
    on_log_binary: FLog,
    on_unary_invertible: FUnary,
    on_variable_missing: FVarMissing,
    on_unsupported_arity: FUnsupported,
) -> Result<T, E>
where
    FAbs: FnOnce(ExprId) -> Result<T, E>,
    FLog: FnOnce(ExprId, ExprId) -> Result<T, E>,
    FUnary: FnOnce(ExprId) -> Result<T, E>,
    FVarMissing: FnOnce() -> E,
    FUnsupported: FnOnce() -> E,
{
    match routing {
        Ok(FunctionIsolationRoute::AbsUnary { arg }) => on_abs_unary(arg),
        Ok(FunctionIsolationRoute::LogBinary { base, arg }) => on_log_binary(base, arg),
        Ok(FunctionIsolationRoute::UnaryInvertible { arg }) => on_unary_invertible(arg),
        Err(FunctionIsolationRouteError::VariableNotFoundInUnaryArg) => Err(on_variable_missing()),
        Err(FunctionIsolationRouteError::UnsupportedArity) => Err(on_unsupported_arity()),
    }
}

/// Execute absolute-value function isolation (`|arg| op rhs`).
#[allow(clippy::too_many_arguments)]
pub fn execute_abs_function_isolation_with_state<
    T,
    S,
    E,
    FPlan,
    FRenderExpr,
    FSolveEquation,
    FMapStep,
    FFinalize,
>(
    state: &mut T,
    arg: ExprId,
    rhs: ExprId,
    op: RelOp,
    include_items: bool,
    existing_steps: Vec<S>,
    mut plan_abs: FPlan,
    mut render_expr: FRenderExpr,
    mut solve_equation: FSolveEquation,
    map_item_to_step: FMapStep,
    mut finalize_solved_sets: FFinalize,
) -> Result<(SolutionSet, Vec<S>), E>
where
    S: Clone,
    FPlan: FnMut(&mut T, ExprId, ExprId, RelOp) -> AbsIsolationPlan,
    FRenderExpr: FnMut(&mut T, ExprId) -> String,
    FSolveEquation: FnMut(&mut T, &cas_ast::Equation) -> Result<(SolutionSet, Vec<S>), E>,
    FMapStep: FnMut(AbsSplitExecutionItem) -> S,
    FFinalize: FnMut(&mut T, SolutionSet, SolutionSet) -> SolutionSet,
{
    execute_abs_isolation_plan_with_rhs_sign_pipeline_with_optional_items_and_solver_with_state(
        state,
        |state| plan_abs(state, arg, rhs, op.clone()),
        arg,
        include_items,
        existing_steps,
        |state, expr| render_expr(state, expr),
        |state, equation| solve_equation(state, equation),
        map_item_to_step,
        |state, positive_set, negative_set| finalize_solved_sets(state, positive_set, negative_set),
    )
}

/// Convenience variant for default absolute-value plan/finalizer.
#[allow(clippy::too_many_arguments)]
pub fn execute_abs_function_isolation_with_default_plan_and_finalizer_with_state<
    T,
    S,
    E,
    FRenderExpr,
    FSolveEquation,
    FMapStep,
>(
    state: &mut T,
    arg: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    include_items: bool,
    existing_steps: Vec<S>,
    render_expr: FRenderExpr,
    solve_equation: FSolveEquation,
    map_item_to_step: FMapStep,
    context_for_plan: impl FnMut(&mut T) -> &mut cas_ast::Context,
    context_for_finalize: impl FnMut(&mut T) -> &cas_ast::Context,
) -> Result<(SolutionSet, Vec<S>), E>
where
    S: Clone,
    FRenderExpr: FnMut(&mut T, ExprId) -> String,
    FSolveEquation: FnMut(&mut T, &cas_ast::Equation) -> Result<(SolutionSet, Vec<S>), E>,
    FMapStep: FnMut(AbsSplitExecutionItem) -> S,
{
    let mut context_for_plan = context_for_plan;
    let mut context_for_finalize = context_for_finalize;
    execute_abs_function_isolation_with_state(
        state,
        arg,
        rhs,
        op.clone(),
        include_items,
        existing_steps,
        |state, inner_arg, inner_rhs, inner_op| {
            plan_abs_isolation_with_rhs_sign(
                context_for_plan(state),
                inner_arg,
                inner_rhs,
                inner_op,
            )
        },
        render_expr,
        solve_equation,
        map_item_to_step,
        |state, positive_set, negative_set| {
            finalize_abs_split_solution_set_for_rhs(
                context_for_finalize(state),
                op.clone(),
                rhs,
                var,
                positive_set,
                negative_set,
            )
        },
    )
}

/// Execute logarithmic function isolation (`log(base, arg) op rhs`).
#[allow(clippy::too_many_arguments)]
pub fn execute_log_function_isolation_with_state<
    T,
    S,
    E,
    FPlanLog,
    FSolveEquation,
    FMapStep,
    FError,
>(
    state: &mut T,
    include_item: bool,
    existing_steps: Vec<S>,
    mut plan_log_step: FPlanLog,
    mut solve_equation: FSolveEquation,
    map_item_to_step: FMapStep,
    mut unsupported_error: FError,
) -> Result<(SolutionSet, Vec<S>), E>
where
    FPlanLog: FnMut(&mut T) -> Option<LogIsolationRewritePlan>,
    FSolveEquation: FnMut(&mut T, &cas_ast::Equation) -> Result<(SolutionSet, Vec<S>), E>,
    FMapStep: FnMut(crate::log_isolation::LogIsolationExecutionItem) -> S,
    FError: FnMut(&mut T) -> E,
{
    let rewrite = plan_log_step(state);
    let error = unsupported_error(state);
    execute_log_isolation_result_pipeline_with_plan_and_error_and_merge_with_existing_steps(
        include_item,
        existing_steps,
        rewrite,
        |equation| solve_equation(state, equation),
        map_item_to_step,
        error,
    )
}

/// Execute unary invertible-function isolation (`f(arg) op rhs`).
#[allow(clippy::too_many_arguments)]
pub fn execute_unary_function_isolation_with_state<
    T,
    S,
    E,
    FPlanUnary,
    FSimplifyRhs,
    FSolveIsolate,
    FMapStep,
    FUnknownFunction,
>(
    state: &mut T,
    fn_name: &str,
    arg: ExprId,
    rhs: ExprId,
    op: RelOp,
    is_lhs: bool,
    include_items: bool,
    existing_steps: Vec<S>,
    mut plan_unary_inverse_step: FPlanUnary,
    mut simplify_rhs: FSimplifyRhs,
    mut solve_isolate: FSolveIsolate,
    map_item_to_step: FMapStep,
    mut unknown_function_error: FUnknownFunction,
) -> Result<(SolutionSet, Vec<S>), E>
where
    FPlanUnary: FnMut(
        &mut T,
        &str,
        ExprId,
        ExprId,
        RelOp,
        bool,
    ) -> Option<crate::function_inverse::UnaryInverseIsolationStepPlan>,
    FSimplifyRhs: FnMut(&mut T, ExprId) -> (ExprId, Vec<(String, ExprId)>),
    FSolveIsolate: FnMut(&mut T, ExprId, ExprId, RelOp) -> Result<(SolutionSet, Vec<S>), E>,
    FMapStep: FnMut(crate::function_inverse::UnaryInverseSolveExecutionItem) -> S,
    FUnknownFunction: FnMut(&mut T) -> E,
{
    execute_unary_inverse_result_pipeline_or_else_with_and_merge_with_existing_steps_with_state(
        state,
        fn_name,
        arg,
        rhs,
        op,
        is_lhs,
        include_items,
        existing_steps,
        |state, name, lhs_expr, rhs_expr, rel_op, lhs_side| {
            plan_unary_inverse_step(state, name, lhs_expr, rhs_expr, rel_op, lhs_side)
        },
        |state, rhs_expr| simplify_rhs(state, rhs_expr),
        |state, lhs_expr, rhs_expr, rel_op| solve_isolate(state, lhs_expr, rhs_expr, rel_op),
        map_item_to_step,
        |state| unknown_function_error(state),
    )
}
