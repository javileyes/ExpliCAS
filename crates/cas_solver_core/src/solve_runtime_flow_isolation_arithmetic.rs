//! Arithmetic-isolation runtime kernels extracted from
//! `solve_runtime_flow_isolation_kernels`.

use cas_ast::{Equation, ExprId, RelOp, SolutionSet};

/// Execute additive isolation (`left + right = rhs`) with default arithmetic
/// kernels and runtime-provided callbacks.
#[allow(clippy::too_many_arguments)]
pub fn apply_add_isolation_strategy_with_default_kernels_and_unified_step_mapper_with_state<
    T,
    S,
    E,
    FCollectItem,
    FContextRef,
    FContextMut,
    FRenderExpr,
    FSimplifyExpr,
    FProveNonzeroStatus,
    FIsolateRewritten,
    FMapStep,
>(
    state: &mut T,
    lhs: ExprId,
    left: ExprId,
    right: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    steps: Vec<S>,
    mut collect_item: FCollectItem,
    context_ref: FContextRef,
    context_mut: FContextMut,
    render_expr: FRenderExpr,
    simplify_expr: FSimplifyExpr,
    prove_nonzero_status: FProveNonzeroStatus,
    isolate_rewritten: FIsolateRewritten,
    map_step: FMapStep,
) -> Result<(SolutionSet, Vec<S>), E>
where
    FCollectItem: FnMut(&mut T) -> bool,
    FContextRef: Fn(&mut T) -> &cas_ast::Context,
    FContextMut: Fn(&mut T) -> &mut cas_ast::Context,
    FRenderExpr: Fn(&cas_ast::Context, ExprId) -> String,
    FSimplifyExpr: FnMut(&mut T, ExprId) -> ExprId,
    FProveNonzeroStatus: FnMut(&mut T, ExprId) -> crate::linear_solution::NonZeroStatus,
    FIsolateRewritten: FnMut(&mut T, Equation) -> Result<(SolutionSet, Vec<S>), E>,
    FMapStep: FnMut(String, Equation) -> S,
{
    let include_item = collect_item(state);
    crate::isolation_arithmetic::execute_add_isolation_pipeline_with_default_factored_linear_collect_and_unified_step_mapper_with_state(
        state,
        lhs,
        left,
        right,
        rhs,
        op,
        var,
        include_item,
        steps,
        context_ref,
        context_mut,
        render_expr,
        simplify_expr,
        prove_nonzero_status,
        isolate_rewritten,
        map_step,
    )
}

/// Execute subtractive isolation (`left - right = rhs`) with default arithmetic
/// kernels and runtime-provided callbacks.
#[allow(clippy::too_many_arguments)]
pub fn apply_sub_isolation_strategy_with_default_kernels_and_unified_step_mapper_with_state<
    T,
    S,
    E,
    FCollectItem,
    FContextRef,
    FContextMut,
    FRenderExpr,
    FSimplifyExpr,
    FIsolateRewritten,
    FMapStep,
>(
    state: &mut T,
    left: ExprId,
    right: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    steps: Vec<S>,
    mut collect_item: FCollectItem,
    context_ref: FContextRef,
    context_mut: FContextMut,
    render_expr: FRenderExpr,
    simplify_expr: FSimplifyExpr,
    isolate_rewritten: FIsolateRewritten,
    map_step: FMapStep,
) -> Result<(SolutionSet, Vec<S>), E>
where
    FCollectItem: FnMut(&mut T) -> bool,
    FContextRef: Fn(&mut T) -> &cas_ast::Context,
    FContextMut: Fn(&mut T) -> &mut cas_ast::Context,
    FRenderExpr: Fn(&cas_ast::Context, ExprId) -> String,
    FSimplifyExpr: FnMut(&mut T, ExprId) -> ExprId,
    FIsolateRewritten: FnMut(&mut T, Equation) -> Result<(SolutionSet, Vec<S>), E>,
    FMapStep: FnMut(String, Equation) -> S,
{
    let include_item = collect_item(state);
    crate::isolation_arithmetic::execute_sub_isolation_pipeline_with_default_plan_and_unified_step_mapper_with_state(
        state,
        left,
        right,
        rhs,
        op,
        var,
        include_item,
        steps,
        context_ref,
        context_mut,
        render_expr,
        simplify_expr,
        isolate_rewritten,
        map_step,
    )
}

/// Execute multiplicative isolation (`left * right = rhs`) with default
/// arithmetic kernels and runtime-provided callbacks.
#[allow(clippy::too_many_arguments)]
pub fn apply_mul_isolation_strategy_with_default_kernels_and_unified_step_mapper_with_state<
    T,
    S,
    E,
    FCollectItem,
    FContextRef,
    FContextMut,
    FSolveSplitCase,
    FSimplifyExpr,
    FProveNonzeroStatus,
    FIsKnownNegative,
    FRenderExpr,
    FIsolateRewritten,
    FMapStep,
>(
    state: &mut T,
    lhs: ExprId,
    left: ExprId,
    right: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    steps: Vec<S>,
    mut collect_item: FCollectItem,
    context_ref: FContextRef,
    context_mut: FContextMut,
    solve_split_case: FSolveSplitCase,
    simplify_expr: FSimplifyExpr,
    prove_nonzero_status: FProveNonzeroStatus,
    is_known_negative: FIsKnownNegative,
    render_expr: FRenderExpr,
    isolate_rewritten: FIsolateRewritten,
    map_step: FMapStep,
) -> Result<(SolutionSet, Vec<S>), E>
where
    FCollectItem: FnMut(&mut T) -> bool,
    FContextRef: Fn(&mut T) -> &cas_ast::Context,
    FContextMut: Fn(&mut T) -> &mut cas_ast::Context,
    FSolveSplitCase: FnMut(&mut T, &Equation) -> Result<(SolutionSet, Vec<S>), E>,
    FSimplifyExpr: FnMut(&mut T, ExprId) -> ExprId,
    FProveNonzeroStatus: FnMut(&mut T, ExprId) -> crate::linear_solution::NonZeroStatus,
    FIsKnownNegative: FnMut(&mut T, ExprId) -> bool,
    FRenderExpr: Fn(&cas_ast::Context, ExprId) -> String,
    FIsolateRewritten: FnMut(&mut T, Equation) -> Result<(SolutionSet, Vec<S>), E>,
    FMapStep: FnMut(String, Equation) -> S,
{
    let include_item = collect_item(state);
    crate::isolation_arithmetic::execute_mul_isolation_pipeline_with_default_additive_linear_collect_and_unified_step_mapper_with_state(
        state,
        lhs,
        left,
        right,
        rhs,
        op,
        var,
        include_item,
        steps,
        context_ref,
        context_mut,
        solve_split_case,
        simplify_expr,
        prove_nonzero_status,
        is_known_negative,
        render_expr,
        isolate_rewritten,
        map_step,
    )
}

/// Execute divisive isolation (`numerator / denominator = rhs`) with default
/// arithmetic kernels and runtime-provided callbacks.
#[allow(clippy::too_many_arguments)]
pub fn apply_div_isolation_strategy_with_default_kernels_and_unified_step_mapper_with_state<
    T,
    S,
    E,
    FCollectNumeratorItems,
    FCollectDenominatorItems,
    FContextRef,
    FContextMut,
    FIsKnownNegative,
    FRenderExpr,
    FSimplifyExpr,
    FIsolateRewritten,
    FProveNonzeroStatus,
    FMapStep,
>(
    state: &mut T,
    lhs: ExprId,
    numerator: ExprId,
    denominator: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    steps: Vec<S>,
    mut collect_numerator_items: FCollectNumeratorItems,
    mut collect_denominator_items: FCollectDenominatorItems,
    context_ref: FContextRef,
    context_mut: FContextMut,
    is_known_negative: FIsKnownNegative,
    render_expr: FRenderExpr,
    simplify_expr: FSimplifyExpr,
    isolate_rewritten: FIsolateRewritten,
    prove_nonzero_status: FProveNonzeroStatus,
    map_step: FMapStep,
) -> Result<(SolutionSet, Vec<S>), E>
where
    S: Clone,
    FCollectNumeratorItems: FnMut(&mut T) -> bool,
    FCollectDenominatorItems: FnMut(&mut T) -> bool,
    FContextRef: Fn(&mut T) -> &cas_ast::Context,
    FContextMut: Fn(&mut T) -> &mut cas_ast::Context,
    FIsKnownNegative: FnMut(&mut T, ExprId) -> bool,
    FRenderExpr: Fn(&cas_ast::Context, ExprId) -> String,
    FSimplifyExpr: FnMut(&mut T, ExprId) -> ExprId,
    FIsolateRewritten: FnMut(&mut T, &Equation) -> Result<(SolutionSet, Vec<S>), E>,
    FProveNonzeroStatus: FnMut(&mut T, ExprId) -> crate::linear_solution::NonZeroStatus,
    FMapStep: FnMut(String, Equation) -> S,
{
    let include_numerator_items = collect_numerator_items(state);
    let include_denominator_items = collect_denominator_items(state);
    crate::isolation_arithmetic::execute_div_isolation_pipeline_with_default_reciprocal_fallback_and_unified_step_mapper_with_state(
        state,
        lhs,
        numerator,
        denominator,
        rhs,
        op,
        var,
        include_numerator_items,
        include_denominator_items,
        steps,
        context_ref,
        context_mut,
        is_known_negative,
        render_expr,
        simplify_expr,
        isolate_rewritten,
        prove_nonzero_status,
        map_step,
    )
}
