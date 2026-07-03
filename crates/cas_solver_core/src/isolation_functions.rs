//! Function-isolation orchestration helpers shared by engine-side solver.
//!
//! These wrappers keep equation-rewrite orchestration in `cas_solver_core`
//! while callers provide stateful hooks for simplification and recursion.

use cas_ast::symbol::SymbolId;
use cas_ast::{Context, ExprId, RelOp, SolutionSet};

use crate::function_inverse::{
    plan_unary_inverse_isolation_step, FunctionIsolationRoute, FunctionIsolationRouteError,
};
use crate::log_isolation::{plan_log_isolation_step_with, LogIsolationRewritePlan};
use crate::solve_outcome::{
    execute_abs_isolation_plan_with_rhs_sign_pipeline_with_optional_items_and_solver_with_state,
    execute_log_isolation_result_pipeline_with_plan_and_error_and_merge_with_existing_steps,
    execute_unary_inverse_result_pipeline_or_else_with_and_merge_with_existing_steps_with_state,
    finalize_abs_split_solution_set_for_rhs, plan_abs_isolation_with_rhs_sign, AbsIsolationPlan,
    AbsSplitExecutionItem,
};

/// Execute full function-isolation pipeline for `f(args) op rhs` using:
/// - default route derivation (`derive_function_isolation_route`)
/// - default abs/log/unary rewrite planners from core helpers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_function_isolation_with_default_kernels_for_var_with_state<
    T,
    S,
    E,
    FContextRef,
    FContextMut,
    FRenderExpr,
    FSolveIsolate,
    FSimplifyRhs,
    FMapAbsStep,
    FMapLogStep,
    FMapUnaryStep,
    FVariableMissingError,
    FUnsupportedArityError,
    FUnknownFunctionError,
>(
    state: &mut T,
    fn_id: SymbolId,
    args: &[ExprId],
    rhs: ExprId,
    op: RelOp,
    var: &str,
    include_items: bool,
    existing_steps: Vec<S>,
    context_ref: FContextRef,
    context_mut: FContextMut,
    render_expr: FRenderExpr,
    solve_isolate: FSolveIsolate,
    simplify_rhs_with_steps: FSimplifyRhs,
    map_abs_item_to_step: FMapAbsStep,
    map_log_item_to_step: FMapLogStep,
    map_unary_item_to_step: FMapUnaryStep,
    variable_missing_error: FVariableMissingError,
    unsupported_arity_error: FUnsupportedArityError,
    unknown_function_error: FUnknownFunctionError,
) -> Result<(SolutionSet, Vec<S>), E>
where
    S: Clone,
    FContextRef: Fn(&mut T) -> &Context,
    FContextMut: FnMut(&mut T) -> &mut Context,
    FRenderExpr: Fn(&Context, ExprId) -> String,
    FSolveIsolate: FnMut(&mut T, ExprId, ExprId, RelOp) -> Result<(SolutionSet, Vec<S>), E>,
    FSimplifyRhs: FnMut(&mut T, ExprId) -> (ExprId, Vec<(String, ExprId)>),
    FMapAbsStep: FnMut(AbsSplitExecutionItem) -> S,
    FMapLogStep: FnMut(crate::log_isolation::LogIsolationExecutionItem) -> S,
    FMapUnaryStep: FnMut(crate::function_inverse::UnaryInverseSolveExecutionItem) -> S,
    FVariableMissingError: FnMut(&mut T, &str) -> E,
    FUnsupportedArityError: FnMut(&mut T, SymbolId, usize, &str) -> E,
    FUnknownFunctionError: FnMut(&mut T, &str) -> E,
{
    let mut context_mut = context_mut;
    let mut solve_isolate = solve_isolate;
    let mut simplify_rhs_with_steps = simplify_rhs_with_steps;
    let map_abs_item_to_step = map_abs_item_to_step;
    let map_log_item_to_step = map_log_item_to_step;
    let map_unary_item_to_step = map_unary_item_to_step;
    let mut variable_missing_error = variable_missing_error;
    let mut unsupported_arity_error = unsupported_arity_error;
    let mut unknown_function_error = unknown_function_error;
    let route = crate::function_inverse::derive_function_isolation_route(
        context_ref(state),
        fn_id,
        args,
        var,
    );

    match route {
        Ok(FunctionIsolationRoute::AbsUnary { arg }) => {
            execute_abs_function_isolation_with_default_plan_and_finalizer_with_state(
                state,
                arg,
                rhs,
                op,
                var,
                include_items,
                existing_steps,
                |state, expr| render_expr(context_ref(state), expr),
                |state, equation| {
                    solve_isolate(state, equation.lhs, equation.rhs, equation.op.clone())
                },
                map_abs_item_to_step,
                |state| context_mut(state),
                |state| context_ref(state),
            )
        }
        Ok(FunctionIsolationRoute::LogBinary { base, arg }) => {
            execute_log_function_isolation_with_default_plan_with_state(
                state,
                base,
                arg,
                rhs,
                var,
                op,
                include_items,
                existing_steps,
                |state| context_mut(state),
                |core_ctx, expr| render_expr(core_ctx, expr),
                |state, equation| {
                    solve_isolate(state, equation.lhs, equation.rhs, equation.op.clone())
                },
                map_log_item_to_step,
                |state| unsupported_arity_error(state, fn_id, args.len(), var),
            )
        }
        Ok(FunctionIsolationRoute::UnaryInvertible { arg }) => {
            let fn_name = context_ref(state).sym_name(fn_id).to_string();
            // U4 soundness gate: inverting a PERIODIC trig function through
            // its principal branch (`sin → arcsin`) is only the accepted
            // convention when the argument is rational-AFFINE in the target
            // variable (`sin(x) = c → { arcsin(c) }`; affine cases with a
            // concrete RHS are owned by the periodic handler upstream). For
            // a NON-affine argument the principal branch asserts a finite
            // set while the true preimage is an infinite family the
            // SolutionSet cannot represent (`cos(π·x²) = 1` returned
            // `{ 0 }`, losing `±√(2k)`; `sin(π·sin(x)) = 1` returned
            // `{ π/6 }`, losing `5π/6 + 2kπ` — final-audit adjacents).
            // Decline to the honest operator-preserving residual instead.
            if matches!(fn_name.as_str(), "sin" | "cos" | "tan") {
                let ctx = context_ref(state);
                let arg_is_rational_affine =
                    cas_math::polynomial::Polynomial::from_expr(ctx, arg, var)
                        .map(|p| p.degree() == 1)
                        .unwrap_or(false)
                        || matches!(
                            ctx.get(arg),
                            cas_ast::Expr::Variable(sym) if ctx.sym_name(*sym) == var
                        );
                if !arg_is_rational_affine {
                    let ctx = context_mut(state);
                    let lhs_call = ctx.call(&fn_name, vec![arg]);
                    let residual =
                        crate::solve_outcome::residual_solution_set(ctx, lhs_call, rhs, op, var);
                    return Ok((residual, existing_steps));
                }
            }
            execute_unary_function_isolation_with_default_plan_with_state(
                state,
                &fn_name,
                arg,
                rhs,
                op,
                true,
                include_items,
                existing_steps,
                |state| context_mut(state),
                |state, rhs_expr| simplify_rhs_with_steps(state, rhs_expr),
                |state, lhs_expr, rhs_expr, inner_op| {
                    solve_isolate(state, lhs_expr, rhs_expr, inner_op)
                },
                map_unary_item_to_step,
                |state| unknown_function_error(state, &fn_name),
            )
        }
        Err(FunctionIsolationRouteError::VariableNotFoundInUnaryArg) => {
            Err(variable_missing_error(state, var))
        }
        Err(FunctionIsolationRouteError::UnsupportedArity) => {
            Err(unsupported_arity_error(state, fn_id, args.len(), var))
        }
    }
}

/// Execute full function isolation with default kernels and a single
/// step-mapper callback shared across abs/log/unary branches.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_function_isolation_with_default_kernels_and_unified_step_mapper_for_var_with_state<
    T,
    S,
    E,
    FContextRef,
    FContextMut,
    FRenderExpr,
    FSolveIsolate,
    FSimplifyRhs,
    FMapStep,
    FVariableMissingError,
    FUnsupportedArityError,
    FUnknownFunctionError,
>(
    state: &mut T,
    fn_id: SymbolId,
    args: &[ExprId],
    rhs: ExprId,
    op: RelOp,
    var: &str,
    include_items: bool,
    existing_steps: Vec<S>,
    context_ref: FContextRef,
    context_mut: FContextMut,
    render_expr: FRenderExpr,
    solve_isolate: FSolveIsolate,
    simplify_rhs_with_steps: FSimplifyRhs,
    map_step: FMapStep,
    variable_missing_error: FVariableMissingError,
    unsupported_arity_error: FUnsupportedArityError,
    unknown_function_error: FUnknownFunctionError,
) -> Result<(SolutionSet, Vec<S>), E>
where
    S: Clone,
    FContextRef: Fn(&mut T) -> &Context,
    FContextMut: FnMut(&mut T) -> &mut Context,
    FRenderExpr: Fn(&Context, ExprId) -> String,
    FSolveIsolate: FnMut(&mut T, ExprId, ExprId, RelOp) -> Result<(SolutionSet, Vec<S>), E>,
    FSimplifyRhs: FnMut(&mut T, ExprId) -> (ExprId, Vec<(String, ExprId)>),
    FMapStep: FnMut(String, cas_ast::Equation) -> S,
    FVariableMissingError: FnMut(&mut T, &str) -> E,
    FUnsupportedArityError: FnMut(&mut T, SymbolId, usize, &str) -> E,
    FUnknownFunctionError: FnMut(&mut T, &str) -> E,
{
    let map_step = std::cell::RefCell::new(map_step);
    execute_function_isolation_with_default_kernels_for_var_with_state(
        state,
        fn_id,
        args,
        rhs,
        op,
        var,
        include_items,
        existing_steps,
        context_ref,
        context_mut,
        render_expr,
        solve_isolate,
        simplify_rhs_with_steps,
        |item| (map_step.borrow_mut())(item.description().to_string(), item.equation),
        |item| (map_step.borrow_mut())(item.description().to_string(), item.equation),
        |item| (map_step.borrow_mut())(item.description().to_string(), item.equation),
        variable_missing_error,
        unsupported_arity_error,
        unknown_function_error,
    )
}

/// Execute absolute-value function isolation (`|arg| op rhs`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_abs_function_isolation_with_state<
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
pub(crate) fn execute_abs_function_isolation_with_default_plan_and_finalizer_with_state<
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
pub(crate) fn execute_log_function_isolation_with_state<
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

/// Execute logarithmic function isolation (`log(base, arg) op rhs`) using the
/// default core planning routine `plan_log_isolation_step_with`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_log_function_isolation_with_default_plan_with_state<
    T,
    S,
    E,
    FContextMut,
    FRenderExpr,
    FSolveEquation,
    FMapStep,
    FError,
>(
    state: &mut T,
    base: ExprId,
    arg: ExprId,
    rhs: ExprId,
    var: &str,
    op: RelOp,
    include_item: bool,
    existing_steps: Vec<S>,
    context_mut: FContextMut,
    render_expr: FRenderExpr,
    solve_equation: FSolveEquation,
    map_item_to_step: FMapStep,
    unsupported_error: FError,
) -> Result<(SolutionSet, Vec<S>), E>
where
    FContextMut: FnMut(&mut T) -> &mut cas_ast::Context,
    FRenderExpr: FnMut(&cas_ast::Context, ExprId) -> String,
    FSolveEquation: FnMut(&mut T, &cas_ast::Equation) -> Result<(SolutionSet, Vec<S>), E>,
    FMapStep: FnMut(crate::log_isolation::LogIsolationExecutionItem) -> S,
    FError: FnMut(&mut T) -> E,
{
    let mut context_mut = context_mut;
    let mut render_expr = render_expr;
    execute_log_function_isolation_with_state(
        state,
        include_item,
        existing_steps,
        |state| {
            plan_log_isolation_step_with(
                context_mut(state),
                base,
                arg,
                rhs,
                var,
                op.clone(),
                |core_ctx, expr| render_expr(core_ctx, expr),
            )
        },
        solve_equation,
        map_item_to_step,
        unsupported_error,
    )
}

/// Execute unary invertible-function isolation (`f(arg) op rhs`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_unary_function_isolation_with_state<
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

/// Execute unary invertible-function isolation (`f(arg) op rhs`) using the
/// default core planning routine `plan_unary_inverse_isolation_step`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_unary_function_isolation_with_default_plan_with_state<
    T,
    S,
    E,
    FContextMut,
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
    context_mut: FContextMut,
    simplify_rhs: FSimplifyRhs,
    solve_isolate: FSolveIsolate,
    map_item_to_step: FMapStep,
    unknown_function_error: FUnknownFunction,
) -> Result<(SolutionSet, Vec<S>), E>
where
    FContextMut: FnMut(&mut T) -> &mut cas_ast::Context,
    FSimplifyRhs: FnMut(&mut T, ExprId) -> (ExprId, Vec<(String, ExprId)>),
    FSolveIsolate: FnMut(&mut T, ExprId, ExprId, RelOp) -> Result<(SolutionSet, Vec<S>), E>,
    FMapStep: FnMut(crate::function_inverse::UnaryInverseSolveExecutionItem) -> S,
    FUnknownFunction: FnMut(&mut T) -> E,
{
    let mut context_mut = context_mut;
    execute_unary_function_isolation_with_state(
        state,
        fn_name,
        arg,
        rhs,
        op,
        is_lhs,
        include_items,
        existing_steps,
        |state, name, lhs_expr, rhs_expr, rel_op, lhs_side| {
            plan_unary_inverse_isolation_step(
                context_mut(state),
                name,
                lhs_expr,
                rhs_expr,
                rel_op,
                lhs_side,
            )
        },
        simplify_rhs,
        solve_isolate,
        map_item_to_step,
        unknown_function_error,
    )
}

#[cfg(test)]
mod tests {
    use super::{
        execute_log_function_isolation_with_default_plan_with_state,
        execute_unary_function_isolation_with_default_plan_with_state,
    };

    use cas_ast::RelOp;

    #[test]
    fn execute_log_function_isolation_with_default_plan_with_state_maps_error_when_no_plan() {
        let mut ctx = cas_ast::Context::new();
        let one = ctx.num(1);
        let two = ctx.num(2);
        let mut state = ctx;

        let err = execute_log_function_isolation_with_default_plan_with_state(
            &mut state,
            one,
            two,
            one,
            "x",
            RelOp::Eq,
            false,
            vec![],
            |state| state,
            |_, _| "render".to_string(),
            |_state, _equation| {
                Ok::<(cas_ast::SolutionSet, Vec<String>), &'static str>((
                    cas_ast::SolutionSet::AllReals,
                    vec![],
                ))
            },
            |item| item.description,
            |_state| "no-plan",
        )
        .expect_err("missing log-plan should map to provided error");

        assert_eq!(err, "no-plan");
    }

    #[test]
    fn execute_unary_function_isolation_with_default_plan_with_state_maps_unknown_fn_error() {
        let mut ctx = cas_ast::Context::new();
        let x = ctx.var("x");
        let y = ctx.num(2);
        let mut state = ctx;

        let err = execute_unary_function_isolation_with_default_plan_with_state(
            &mut state,
            "unknown_fn",
            x,
            y,
            RelOp::Eq,
            true,
            false,
            vec![],
            |state| state,
            |_state, rhs_expr| (rhs_expr, vec![]),
            |_state, _lhs, _rhs, _op| {
                Ok::<(cas_ast::SolutionSet, Vec<String>), &'static str>((
                    cas_ast::SolutionSet::AllReals,
                    vec![],
                ))
            },
            |item| item.description,
            |_state| "unknown-fn",
        )
        .expect_err("unknown unary function should map to provided error");

        assert_eq!(err, "unknown-fn");
    }
}
