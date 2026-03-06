use super::dispatch_isolation_with_default_kernels_and_route_callbacks_with_state;
use crate::solve_runtime_flow_isolation_kernels::{
    apply_add_isolation_strategy_with_default_kernels_and_unified_step_mapper_with_state,
    apply_div_isolation_strategy_with_default_kernels_and_unified_step_mapper_with_state,
    apply_mul_isolation_strategy_with_default_kernels_and_unified_step_mapper_with_state,
    apply_sub_isolation_strategy_with_default_kernels_and_unified_step_mapper_with_state,
};
use cas_ast::symbol::SymbolId;
use cas_ast::{Equation, ExprId, RelOp, SolutionSet};

/// Dispatch isolation with default kernels and default arithmetic route
/// handlers (`add/sub/mul/div`), while callers provide custom `pow/function`
/// handlers and recursive solve hooks.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_isolation_with_default_kernels_and_default_arithmetic_routes_and_custom_pow_function_with_state<
    T,
    S,
    E,
    FContextRef,
    FContextMut,
    FSimplifyRhs,
    FCollectLinearItem,
    FSimplifyLinearExpr,
    FProveNonzeroStatus,
    FRenderExpr,
    FCollectRouteItem,
    FSolveSplitCaseWithVar,
    FIsKnownNegative,
    FIsolateRewrittenWithVar,
    FOnPowRoute,
    FOnFunctionRoute,
    FCollectNegatedItem,
    FSolveNegatedRewritten,
    FMapStep,
    FMapUnsupportedError,
>(
    state: &mut T,
    lhs: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    context_ref: FContextRef,
    context_mut: FContextMut,
    simplify_rhs: FSimplifyRhs,
    collect_linear_item: FCollectLinearItem,
    simplify_linear_expr: FSimplifyLinearExpr,
    prove_nonzero_status: FProveNonzeroStatus,
    render_expr: FRenderExpr,
    collect_route_item: FCollectRouteItem,
    solve_split_case_with_var: FSolveSplitCaseWithVar,
    is_known_negative: FIsKnownNegative,
    isolate_rewritten_with_var: FIsolateRewrittenWithVar,
    mut on_pow_route: FOnPowRoute,
    mut on_function_route: FOnFunctionRoute,
    collect_negated_item: FCollectNegatedItem,
    solve_negated_rewritten: FSolveNegatedRewritten,
    map_step: FMapStep,
    map_unsupported_error: FMapUnsupportedError,
) -> Result<(SolutionSet, Vec<S>), E>
where
    S: Clone,
    FContextRef: Fn(&mut T) -> &cas_ast::Context,
    FContextMut: Fn(&mut T) -> &mut cas_ast::Context,
    FSimplifyRhs: FnMut(&mut T, ExprId) -> ExprId,
    FCollectLinearItem: FnMut(&mut T) -> bool,
    FSimplifyLinearExpr: FnMut(&mut T, ExprId) -> ExprId,
    FProveNonzeroStatus: FnMut(&mut T, ExprId) -> crate::linear_solution::NonZeroStatus,
    FRenderExpr: Fn(&cas_ast::Context, ExprId) -> String,
    FCollectRouteItem: FnMut(&mut T) -> bool,
    FSolveSplitCaseWithVar: FnMut(&mut T, &Equation, &str) -> Result<(SolutionSet, Vec<S>), E>,
    FIsKnownNegative: FnMut(&mut T, ExprId) -> bool,
    FIsolateRewrittenWithVar: FnMut(&mut T, Equation, &str) -> Result<(SolutionSet, Vec<S>), E>,
    FOnPowRoute: FnMut(
        &mut T,
        ExprId,
        ExprId,
        ExprId,
        ExprId,
        RelOp,
        &str,
        Vec<S>,
    ) -> Result<(SolutionSet, Vec<S>), E>,
    FOnFunctionRoute: FnMut(
        &mut T,
        SymbolId,
        Vec<ExprId>,
        ExprId,
        RelOp,
        &str,
        Vec<S>,
    ) -> Result<(SolutionSet, Vec<S>), E>,
    FCollectNegatedItem: FnMut(&mut T) -> bool,
    FSolveNegatedRewritten: FnMut(&mut T, Equation, &str) -> Result<(SolutionSet, Vec<S>), E>,
    FMapStep: FnMut(String, Equation) -> S,
    FMapUnsupportedError: FnMut(&mut T, cas_ast::Expr) -> E,
{
    let context_ref = &context_ref;
    let context_mut = &context_mut;
    let render_expr = &render_expr;
    let simplify_linear_expr = std::cell::RefCell::new(simplify_linear_expr);
    let prove_nonzero_status = std::cell::RefCell::new(prove_nonzero_status);
    let collect_route_item = std::cell::RefCell::new(collect_route_item);
    let solve_split_case_with_var = std::cell::RefCell::new(solve_split_case_with_var);
    let is_known_negative = std::cell::RefCell::new(is_known_negative);
    let isolate_rewritten_with_var = std::cell::RefCell::new(isolate_rewritten_with_var);
    let map_step = std::cell::RefCell::new(map_step);

    dispatch_isolation_with_default_kernels_and_route_callbacks_with_state(
        state,
        lhs,
        rhs,
        op.clone(),
        var,
        context_ref,
        context_mut,
        simplify_rhs,
        collect_linear_item,
        |state, expr| (simplify_linear_expr.borrow_mut())(state, expr),
        |state, expr| (prove_nonzero_status.borrow_mut())(state, expr),
        render_expr,
        |state, eq_lhs, left, right, eq_rhs, eq_op, solve_var, steps| {
            apply_add_isolation_strategy_with_default_kernels_and_unified_step_mapper_with_state(
                state,
                eq_lhs,
                left,
                right,
                eq_rhs,
                eq_op,
                solve_var,
                steps,
                |state| (collect_route_item.borrow_mut())(state),
                context_ref,
                context_mut,
                render_expr,
                |state, expr| (simplify_linear_expr.borrow_mut())(state, expr),
                |state, expr| (prove_nonzero_status.borrow_mut())(state, expr),
                |state, equation| {
                    (isolate_rewritten_with_var.borrow_mut())(state, equation, solve_var)
                },
                |description, equation| (map_step.borrow_mut())(description, equation),
            )
        },
        |state, left, right, eq_rhs, eq_op, solve_var, steps| {
            apply_sub_isolation_strategy_with_default_kernels_and_unified_step_mapper_with_state(
                state,
                left,
                right,
                eq_rhs,
                eq_op,
                solve_var,
                steps,
                |state| (collect_route_item.borrow_mut())(state),
                context_ref,
                context_mut,
                render_expr,
                |state, expr| (simplify_linear_expr.borrow_mut())(state, expr),
                |state, equation| {
                    (isolate_rewritten_with_var.borrow_mut())(state, equation, solve_var)
                },
                |description, equation| (map_step.borrow_mut())(description, equation),
            )
        },
        |state, eq_lhs, left, right, eq_rhs, eq_op, solve_var, steps| {
            apply_mul_isolation_strategy_with_default_kernels_and_unified_step_mapper_with_state(
                state,
                eq_lhs,
                left,
                right,
                eq_rhs,
                eq_op,
                solve_var,
                steps,
                |state| (collect_route_item.borrow_mut())(state),
                context_ref,
                context_mut,
                |state, equation| {
                    (solve_split_case_with_var.borrow_mut())(state, equation, solve_var)
                },
                |state, expr| (simplify_linear_expr.borrow_mut())(state, expr),
                |state, expr| (prove_nonzero_status.borrow_mut())(state, expr),
                |state, expr| (is_known_negative.borrow_mut())(state, expr),
                render_expr,
                |state, equation| {
                    (isolate_rewritten_with_var.borrow_mut())(state, equation, solve_var)
                },
                |description, equation| (map_step.borrow_mut())(description, equation),
            )
        },
        |state, eq_lhs, left, right, eq_rhs, eq_op, solve_var, steps| {
            apply_div_isolation_strategy_with_default_kernels_and_unified_step_mapper_with_state(
                state,
                eq_lhs,
                left,
                right,
                eq_rhs,
                eq_op,
                solve_var,
                steps,
                |state| (collect_route_item.borrow_mut())(state),
                |state| (collect_route_item.borrow_mut())(state),
                context_ref,
                context_mut,
                |state, expr| (is_known_negative.borrow_mut())(state, expr),
                render_expr,
                |state, expr| (simplify_linear_expr.borrow_mut())(state, expr),
                |state, equation| {
                    (isolate_rewritten_with_var.borrow_mut())(state, equation.clone(), solve_var)
                },
                |state, expr| (prove_nonzero_status.borrow_mut())(state, expr),
                |description, equation| (map_step.borrow_mut())(description, equation),
            )
        },
        |state, eq_lhs, base, exponent, eq_rhs, eq_op, solve_var, steps| {
            on_pow_route(
                state, eq_lhs, base, exponent, eq_rhs, eq_op, solve_var, steps,
            )
        },
        |state, fn_id, args, eq_rhs, eq_op, solve_var, steps| {
            on_function_route(state, fn_id, args, eq_rhs, eq_op, solve_var, steps)
        },
        collect_negated_item,
        solve_negated_rewritten,
        |description, equation| (map_step.borrow_mut())(description, equation),
        map_unsupported_error,
    )
}
