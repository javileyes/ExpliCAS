use super::dispatch_isolation_with_default_kernels_and_default_arithmetic_routes_and_custom_pow_function_with_state;
use crate::solve_runtime_flow_isolation_kernels::{
    apply_function_isolation_with_default_kernels_and_runtime_trace_and_default_unsupported_inverse_message_and_recursive_equation_solver_with_state,
    apply_pow_isolation_with_default_runtime_config_and_recursive_equation_solver_with_state,
};
use cas_ast::symbol::SymbolId;
use cas_ast::{Equation, ExprId, RelOp, SolutionSet};

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
                |state, expr, tactic_opts| {
                    (simplify_with_tactic.borrow_mut())(state, expr, tactic_opts)
                },
                |state, lhs_expr, rhs_expr| {
                    (classify_decision.borrow_mut())(state, lhs_expr, rhs_expr)
                },
                |state, equation, local_var| {
                    (solve_equation_with_var.borrow_mut())(state, equation, local_var)
                },
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
                |state, equation, local_var| {
                    (solve_equation_with_var.borrow_mut())(state, equation, local_var)
                },
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
        |state, equation, local_var| {
            (solve_equation_with_var.borrow_mut())(state, equation, local_var)
        },
        |description, equation| (map_step.borrow_mut())(description, equation),
        map_unsupported_error,
    )
}
