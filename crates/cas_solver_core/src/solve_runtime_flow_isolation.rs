//! Isolation runtime wrappers extracted from `solve_runtime_flow`.

use cas_ast::symbol::SymbolId;
use cas_ast::{Equation, ExprId, RelOp, SolutionSet};
use crate::solve_runtime_flow_isolation_kernels::{
    apply_add_isolation_strategy_with_default_kernels_and_unified_step_mapper_with_state,
    apply_div_isolation_strategy_with_default_kernels_and_unified_step_mapper_with_state,
    apply_function_isolation_with_default_kernels_and_runtime_trace_and_default_unsupported_inverse_message_and_recursive_equation_solver_with_state,
    apply_mul_isolation_strategy_with_default_kernels_and_unified_step_mapper_with_state,
    apply_pow_isolation_with_default_runtime_config_and_recursive_equation_solver_with_state,
    apply_sub_isolation_strategy_with_default_kernels_and_unified_step_mapper_with_state,
};

/// Dispatch one isolation step (`lhs op rhs`) using default isolated/negated
/// entries and default linear-collect kernels.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_isolation_with_default_isolated_and_negated_entries_and_default_linear_collect_kernels_for_var_and_unified_step_mapper_with_state<
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
    FOnAdd,
    FOnSub,
    FOnMul,
    FOnDiv,
    FOnPow,
    FOnFunction,
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
    on_add: FOnAdd,
    on_sub: FOnSub,
    on_mul: FOnMul,
    on_div: FOnDiv,
    on_pow: FOnPow,
    on_function: FOnFunction,
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
    FOnAdd: FnMut(&mut T, ExprId, ExprId) -> Result<(SolutionSet, Vec<S>), E>,
    FOnSub: FnMut(&mut T, ExprId, ExprId) -> Result<(SolutionSet, Vec<S>), E>,
    FOnMul: FnMut(&mut T, ExprId, ExprId) -> Result<(SolutionSet, Vec<S>), E>,
    FOnDiv: FnMut(&mut T, ExprId, ExprId) -> Result<(SolutionSet, Vec<S>), E>,
    FOnPow: FnMut(&mut T, ExprId, ExprId) -> Result<(SolutionSet, Vec<S>), E>,
    FOnFunction: FnMut(&mut T, SymbolId, Vec<ExprId>) -> Result<(SolutionSet, Vec<S>), E>,
    FCollectNegatedItem: FnMut(&mut T) -> bool,
    FSolveNegatedRewritten: FnMut(&mut T, Equation, &str) -> Result<(SolutionSet, Vec<S>), E>,
    FMapStep: FnMut(String, Equation) -> S,
    FMapUnsupportedError: FnMut(&mut T, cas_ast::Expr) -> E,
{
    crate::isolation_dispatch::execute_isolation_dispatch_with_default_isolated_and_negated_entries_and_default_linear_collect_kernels_for_var_and_unified_step_mapper_with_state(
        state,
        lhs,
        rhs,
        op,
        var,
        context_ref,
        context_mut,
        simplify_rhs,
        collect_linear_item,
        simplify_linear_expr,
        prove_nonzero_status,
        render_expr,
        on_add,
        on_sub,
        on_mul,
        on_div,
        on_pow,
        on_function,
        collect_negated_item,
        solve_negated_rewritten,
        map_step,
        map_unsupported_error,
    )
}

/// Dispatch isolation with default kernels while routing arithmetic/function
/// branches through caller-provided callbacks that receive full equation
/// context and start with empty branch-local step vectors.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_isolation_with_default_kernels_and_route_callbacks_with_state<
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
    FOnAddRoute,
    FOnSubRoute,
    FOnMulRoute,
    FOnDivRoute,
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
    mut on_add_route: FOnAddRoute,
    mut on_sub_route: FOnSubRoute,
    mut on_mul_route: FOnMulRoute,
    mut on_div_route: FOnDivRoute,
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
    FOnAddRoute: FnMut(
        &mut T,
        ExprId,
        ExprId,
        ExprId,
        ExprId,
        RelOp,
        &str,
        Vec<S>,
    ) -> Result<(SolutionSet, Vec<S>), E>,
    FOnSubRoute: FnMut(
        &mut T,
        ExprId,
        ExprId,
        ExprId,
        RelOp,
        &str,
        Vec<S>,
    ) -> Result<(SolutionSet, Vec<S>), E>,
    FOnMulRoute: FnMut(
        &mut T,
        ExprId,
        ExprId,
        ExprId,
        ExprId,
        RelOp,
        &str,
        Vec<S>,
    ) -> Result<(SolutionSet, Vec<S>), E>,
    FOnDivRoute: FnMut(
        &mut T,
        ExprId,
        ExprId,
        ExprId,
        ExprId,
        RelOp,
        &str,
        Vec<S>,
    ) -> Result<(SolutionSet, Vec<S>), E>,
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
    crate::isolation_dispatch::execute_isolation_dispatch_with_default_isolated_and_negated_entries_and_default_linear_collect_kernels_for_var_and_unified_step_mapper_with_state(
        state,
        lhs,
        rhs,
        op.clone(),
        var,
        context_ref,
        context_mut,
        simplify_rhs,
        collect_linear_item,
        simplify_linear_expr,
        prove_nonzero_status,
        render_expr,
        |state, left, right| on_add_route(state, lhs, left, right, rhs, op.clone(), var, Vec::new()),
        |state, left, right| on_sub_route(state, left, right, rhs, op.clone(), var, Vec::new()),
        |state, left, right| on_mul_route(state, lhs, left, right, rhs, op.clone(), var, Vec::new()),
        |state, left, right| on_div_route(state, lhs, left, right, rhs, op.clone(), var, Vec::new()),
        |state, base, exponent| on_pow_route(state, lhs, base, exponent, rhs, op.clone(), var, Vec::new()),
        |state, fn_id, args| on_function_route(state, fn_id, args, rhs, op.clone(), var, Vec::new()),
        collect_negated_item,
        solve_negated_rewritten,
        map_step,
        map_unsupported_error,
    )
}

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

/// Dispatch isolation with default kernels and default route handlers for
/// arithmetic (`add/sub/mul/div`), power, and function branches.
///
/// Callers provide one recursive equation solver and the domain/error hooks
/// required by pow/function specializations.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_isolation_with_default_kernels_and_default_arithmetic_pow_function_routes_with_state<
    T,
    S,
    E,
    TTacticOptions,
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
    FCollectPowItem,
    FSimplifyShortcut,
    FClearBlockedHints,
    FSimplifyWithTactic,
    FClassifyDecision,
    FMapPowEnsureError,
    FVisitPowAssumption,
    FRegisterBlockedHint,
    FMapPowUnsupportedErr,
    FCollectFunctionItems,
    FSimplifyWithTrace,
    FSymName,
    FMapVariableNotFound,
    FMapIsolationError,
    FMapUnknownFunction,
    FSolveEquationWithVar,
    FCollectNegatedItem,
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
    mode: crate::log_domain::DomainModeKind,
    wildcard_scope: bool,
    value_domain_real_only: bool,
    budget: crate::solve_budget::SolveBudget,
    collect_pow_item: FCollectPowItem,
    tactic_options: TTacticOptions,
    simplify_shortcut: FSimplifyShortcut,
    clear_blocked_hints: FClearBlockedHints,
    simplify_with_tactic: FSimplifyWithTactic,
    classify_decision: FClassifyDecision,
    map_pow_ensure_error: FMapPowEnsureError,
    visit_pow_assumption: FVisitPowAssumption,
    register_blocked_hint: FRegisterBlockedHint,
    map_pow_unsupported_err: FMapPowUnsupportedErr,
    collect_function_items: FCollectFunctionItems,
    simplify_with_trace: FSimplifyWithTrace,
    sym_name: FSymName,
    map_variable_not_found: FMapVariableNotFound,
    map_isolation_error: FMapIsolationError,
    map_unknown_function: FMapUnknownFunction,
    solve_equation_with_var: FSolveEquationWithVar,
    collect_negated_item: FCollectNegatedItem,
    map_step: FMapStep,
    map_unsupported_error: FMapUnsupportedError,
) -> Result<(SolutionSet, Vec<S>), E>
where
    S: Clone,
    TTacticOptions: Clone,
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
    FCollectPowItem: FnMut(&mut T) -> bool,
    FSimplifyShortcut: FnMut(&mut T, ExprId) -> ExprId,
    FClearBlockedHints: FnMut(&mut T),
    FSimplifyWithTactic: FnMut(&mut T, ExprId, &TTacticOptions) -> ExprId,
    FClassifyDecision: FnMut(&mut T, ExprId, ExprId) -> crate::log_domain::LogSolveDecision,
    FMapPowEnsureError: FnMut(&str, &'static str) -> E,
    FVisitPowAssumption: FnMut(&cas_ast::Context, ExprId, ExprId, crate::log_domain::LogAssumption),
    FRegisterBlockedHint: FnMut(&cas_ast::Context, crate::solve_outcome::LogBlockedHintRecord),
    FMapPowUnsupportedErr: FnMut(&'static str) -> E,
    FCollectFunctionItems: FnMut(&mut T) -> bool,
    FSimplifyWithTrace: FnMut(&mut T, ExprId) -> (ExprId, Vec<(String, ExprId)>),
    FSymName: FnMut(&mut T, SymbolId) -> String,
    FMapVariableNotFound: FnMut(&mut T, &str) -> E,
    FMapIsolationError: FnMut(&mut T, &str, String) -> E,
    FMapUnknownFunction: FnMut(&mut T, &str) -> E,
    FSolveEquationWithVar: FnMut(&mut T, Equation, &str) -> Result<(SolutionSet, Vec<S>), E>,
    FCollectNegatedItem: FnMut(&mut T) -> bool,
    FMapStep: FnMut(String, Equation) -> S,
    FMapUnsupportedError: FnMut(&mut T, cas_ast::Expr) -> E,
{
    let solve_equation_with_var = std::cell::RefCell::new(solve_equation_with_var);
    let collect_pow_item = std::cell::RefCell::new(collect_pow_item);
    let simplify_shortcut = std::cell::RefCell::new(simplify_shortcut);
    let clear_blocked_hints = std::cell::RefCell::new(clear_blocked_hints);
    let simplify_with_tactic = std::cell::RefCell::new(simplify_with_tactic);
    let classify_decision = std::cell::RefCell::new(classify_decision);
    let map_pow_ensure_error = std::cell::RefCell::new(map_pow_ensure_error);
    let visit_pow_assumption = std::cell::RefCell::new(visit_pow_assumption);
    let register_blocked_hint = std::cell::RefCell::new(register_blocked_hint);
    let map_pow_unsupported_err = std::cell::RefCell::new(map_pow_unsupported_err);
    let collect_function_items = std::cell::RefCell::new(collect_function_items);
    let simplify_with_trace = std::cell::RefCell::new(simplify_with_trace);
    let sym_name = std::cell::RefCell::new(sym_name);
    let map_variable_not_found = std::cell::RefCell::new(map_variable_not_found);
    let map_isolation_error = std::cell::RefCell::new(map_isolation_error);
    let map_unknown_function = std::cell::RefCell::new(map_unknown_function);
    let isolate_rewritten_with_var = std::cell::RefCell::new(isolate_rewritten_with_var);
    let map_step = std::cell::RefCell::new(map_step);
    let context_ref = &context_ref;
    let context_mut = &context_mut;
    let render_expr = &render_expr;

    dispatch_isolation_with_default_kernels_and_default_arithmetic_routes_and_custom_pow_function_with_state(
        state,
        lhs,
        rhs,
        op.clone(),
        var,
        context_ref,
        context_mut,
        simplify_rhs,
        collect_linear_item,
        simplify_linear_expr,
        prove_nonzero_status,
        render_expr,
        collect_route_item,
        solve_split_case_with_var,
        is_known_negative,
        |state, equation, local_var| {
            (isolate_rewritten_with_var.borrow_mut())(state, equation, local_var)
        },
        |state, eq_lhs, base, exponent, eq_rhs, eq_op, solve_var, steps| {
            apply_pow_isolation_with_default_runtime_config_and_recursive_equation_solver_with_state(
                state,
                eq_lhs,
                base,
                exponent,
                eq_rhs,
                eq_op,
                solve_var,
                mode,
                wildcard_scope,
                value_domain_real_only,
                budget,
                |state| (collect_pow_item.borrow_mut())(state),
                || tactic_options.clone(),
                steps,
                |state| context_ref(state),
                |state| context_mut(state),
                |core_ctx, expr| render_expr(core_ctx, expr),
                |state, expr| (simplify_shortcut.borrow_mut())(state, expr),
                |state| (clear_blocked_hints.borrow_mut())(state),
                |state, expr, tactic_opts| (simplify_with_tactic.borrow_mut())(state, expr, tactic_opts),
                |state, lhs_expr, rhs_expr| (classify_decision.borrow_mut())(state, lhs_expr, rhs_expr),
                |state, equation, local_var| (solve_equation_with_var.borrow_mut())(state, equation, local_var),
                |description, equation| (map_step.borrow_mut())(description, equation),
                |var_name, message| (map_pow_ensure_error.borrow_mut())(var_name, message),
                |core_ctx, assumption| {
                    (visit_pow_assumption.borrow_mut())(core_ctx, base, eq_rhs, assumption)
                },
                |core_ctx, hint| (register_blocked_hint.borrow_mut())(core_ctx, hint),
                |message| (map_pow_unsupported_err.borrow_mut())(message),
            )
        },
        |state, fn_id, args, eq_rhs, eq_op, solve_var, steps| {
            apply_function_isolation_with_default_kernels_and_runtime_trace_and_default_unsupported_inverse_message_and_recursive_equation_solver_with_state(
                state,
                fn_id,
                &args,
                eq_rhs,
                eq_op,
                solve_var,
                steps,
                |state| (collect_function_items.borrow_mut())(state),
                |state| context_ref(state),
                |state| context_mut(state),
                |core_ctx, expr| render_expr(core_ctx, expr),
                |state, equation, local_var| (solve_equation_with_var.borrow_mut())(state, equation, local_var),
                |state, expr| (simplify_with_trace.borrow_mut())(state, expr),
                |description, equation| (map_step.borrow_mut())(description, equation),
                |state, fn_symbol| (sym_name.borrow_mut())(state, fn_symbol),
                |state, missing_var| (map_variable_not_found.borrow_mut())(state, missing_var),
                |state, unsupported_var, message| {
                    (map_isolation_error.borrow_mut())(state, unsupported_var, message)
                },
                |state, fn_name| (map_unknown_function.borrow_mut())(state, fn_name),
            )
        },
        collect_negated_item,
        |state, equation, local_var| (solve_equation_with_var.borrow_mut())(state, equation, local_var),
        |description, equation| (map_step.borrow_mut())(description, equation),
        map_unsupported_error,
    )
}
