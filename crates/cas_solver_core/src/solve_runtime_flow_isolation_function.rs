//! Function-isolation runtime kernels extracted from
//! `solve_runtime_flow_isolation_kernels`.

use cas_ast::symbol::SymbolId;
use cas_ast::{Equation, ExprId, RelOp, SolutionSet};

#[allow(clippy::too_many_arguments)]
pub fn apply_function_isolation_with_default_kernels_and_runtime_trace_and_unified_step_mapper_for_var_with_state<
    T,
    S,
    E,
    FCollectItems,
    FContextRef,
    FContextMut,
    FRenderExpr,
    FIsolateRecursive,
    FSimplifyWithTrace,
    FMapStep,
    FMapVariableNotFound,
    FMapUnsupportedInverse,
    FMapUnknownFunction,
>(
    state: &mut T,
    fn_id: SymbolId,
    args: &[ExprId],
    rhs: ExprId,
    op: RelOp,
    var: &str,
    steps: Vec<S>,
    mut collect_items: FCollectItems,
    context_ref: FContextRef,
    context_mut: FContextMut,
    render_expr: FRenderExpr,
    isolate_recursive: FIsolateRecursive,
    simplify_with_trace: FSimplifyWithTrace,
    map_step: FMapStep,
    map_variable_not_found: FMapVariableNotFound,
    map_unsupported_inverse: FMapUnsupportedInverse,
    map_unknown_function: FMapUnknownFunction,
) -> Result<(SolutionSet, Vec<S>), E>
where
    S: Clone,
    FCollectItems: FnMut(&mut T) -> bool,
    FContextRef: Fn(&mut T) -> &cas_ast::Context,
    FContextMut: Fn(&mut T) -> &mut cas_ast::Context,
    FRenderExpr: Fn(&cas_ast::Context, ExprId) -> String,
    FIsolateRecursive: FnMut(&mut T, ExprId, ExprId, RelOp) -> Result<(SolutionSet, Vec<S>), E>,
    FSimplifyWithTrace: FnMut(&mut T, ExprId) -> (ExprId, Vec<(String, ExprId)>),
    FMapStep: FnMut(String, Equation) -> S,
    FMapVariableNotFound: FnMut(&mut T, &str) -> E,
    FMapUnsupportedInverse: FnMut(&mut T, SymbolId, usize, &str) -> E,
    FMapUnknownFunction: FnMut(&mut T, &str) -> E,
{
    let include_items = collect_items(state);
    crate::isolation_functions::execute_function_isolation_with_default_kernels_and_unified_step_mapper_for_var_with_state(
        state,
        fn_id,
        args,
        rhs,
        op,
        var,
        include_items,
        steps,
        context_ref,
        context_mut,
        render_expr,
        isolate_recursive,
        simplify_with_trace,
        map_step,
        map_variable_not_found,
        map_unsupported_inverse,
        map_unknown_function,
    )
}

/// Execute function isolation with default kernels and a standard unsupported
/// inverse error message built from function symbol name.
#[allow(clippy::too_many_arguments)]
pub fn apply_function_isolation_with_default_kernels_and_runtime_trace_and_default_unsupported_inverse_message_with_state<
    T,
    S,
    E,
    FCollectItems,
    FContextRef,
    FContextMut,
    FRenderExpr,
    FIsolateRecursive,
    FSimplifyWithTrace,
    FMapStep,
    FSymName,
    FMapVariableNotFound,
    FMapIsolationError,
    FMapUnknownFunction,
>(
    state: &mut T,
    fn_id: SymbolId,
    args: &[ExprId],
    rhs: ExprId,
    op: RelOp,
    var: &str,
    steps: Vec<S>,
    collect_items: FCollectItems,
    context_ref: FContextRef,
    context_mut: FContextMut,
    render_expr: FRenderExpr,
    isolate_recursive: FIsolateRecursive,
    simplify_with_trace: FSimplifyWithTrace,
    map_step: FMapStep,
    mut sym_name: FSymName,
    map_variable_not_found: FMapVariableNotFound,
    mut map_isolation_error: FMapIsolationError,
    map_unknown_function: FMapUnknownFunction,
) -> Result<(SolutionSet, Vec<S>), E>
where
    S: Clone,
    FCollectItems: FnMut(&mut T) -> bool,
    FContextRef: Fn(&mut T) -> &cas_ast::Context,
    FContextMut: Fn(&mut T) -> &mut cas_ast::Context,
    FRenderExpr: Fn(&cas_ast::Context, ExprId) -> String,
    FIsolateRecursive: FnMut(&mut T, ExprId, ExprId, RelOp) -> Result<(SolutionSet, Vec<S>), E>,
    FSimplifyWithTrace: FnMut(&mut T, ExprId) -> (ExprId, Vec<(String, ExprId)>),
    FMapStep: FnMut(String, Equation) -> S,
    FSymName: FnMut(&mut T, SymbolId) -> String,
    FMapVariableNotFound: FnMut(&mut T, &str) -> E,
    FMapIsolationError: FnMut(&mut T, &str, String) -> E,
    FMapUnknownFunction: FnMut(&mut T, &str) -> E,
{
    apply_function_isolation_with_default_kernels_and_runtime_trace_and_unified_step_mapper_for_var_with_state(
        state,
        fn_id,
        args,
        rhs,
        op,
        var,
        steps,
        collect_items,
        context_ref,
        context_mut,
        render_expr,
        isolate_recursive,
        simplify_with_trace,
        map_step,
        map_variable_not_found,
        |state, unsupported_fn_id, arity, unsupported_var| {
            let fn_name = sym_name(state, unsupported_fn_id);
            map_isolation_error(
                state,
                unsupported_var,
                format!(
                    "Cannot invert function '{}' with {} arguments",
                    fn_name, arity
                ),
            )
        },
        map_unknown_function,
    )
}

/// Execute function isolation with default unsupported-inverse messaging and a
/// single recursive equation solver reused for all inner isolate calls.
#[allow(clippy::too_many_arguments)]
pub fn apply_function_isolation_with_default_kernels_and_runtime_trace_and_default_unsupported_inverse_message_and_recursive_equation_solver_with_state<
    T,
    S,
    E,
    FCollectItems,
    FContextRef,
    FContextMut,
    FRenderExpr,
    FSolveEquation,
    FSimplifyWithTrace,
    FMapStep,
    FSymName,
    FMapVariableNotFound,
    FMapIsolationError,
    FMapUnknownFunction,
>(
    state: &mut T,
    fn_id: SymbolId,
    args: &[ExprId],
    rhs: ExprId,
    op: RelOp,
    var: &str,
    steps: Vec<S>,
    collect_items: FCollectItems,
    context_ref: FContextRef,
    context_mut: FContextMut,
    render_expr: FRenderExpr,
    solve_equation: FSolveEquation,
    simplify_with_trace: FSimplifyWithTrace,
    map_step: FMapStep,
    sym_name: FSymName,
    map_variable_not_found: FMapVariableNotFound,
    map_isolation_error: FMapIsolationError,
    map_unknown_function: FMapUnknownFunction,
) -> Result<(SolutionSet, Vec<S>), E>
where
    S: Clone,
    FCollectItems: FnMut(&mut T) -> bool,
    FContextRef: Fn(&mut T) -> &cas_ast::Context,
    FContextMut: Fn(&mut T) -> &mut cas_ast::Context,
    FRenderExpr: Fn(&cas_ast::Context, ExprId) -> String,
    FSolveEquation: FnMut(&mut T, Equation, &str) -> Result<(SolutionSet, Vec<S>), E>,
    FSimplifyWithTrace: FnMut(&mut T, ExprId) -> (ExprId, Vec<(String, ExprId)>),
    FMapStep: FnMut(String, Equation) -> S,
    FSymName: FnMut(&mut T, SymbolId) -> String,
    FMapVariableNotFound: FnMut(&mut T, &str) -> E,
    FMapIsolationError: FnMut(&mut T, &str, String) -> E,
    FMapUnknownFunction: FnMut(&mut T, &str) -> E,
{
    let solve_equation = std::cell::RefCell::new(solve_equation);
    apply_function_isolation_with_default_kernels_and_runtime_trace_and_default_unsupported_inverse_message_with_state(
        state,
        fn_id,
        args,
        rhs,
        op,
        var,
        steps,
        collect_items,
        context_ref,
        context_mut,
        render_expr,
        |state, lhs_expr, rhs_expr, inner_op| {
            (solve_equation.borrow_mut())(
                state,
                Equation {
                    lhs: lhs_expr,
                    rhs: rhs_expr,
                    op: inner_op,
                },
                var,
            )
        },
        simplify_with_trace,
        map_step,
        sym_name,
        map_variable_not_found,
        map_isolation_error,
        map_unknown_function,
    )
}
