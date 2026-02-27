//! Arithmetic isolation orchestration helpers shared by engine-side solver.
//!
//! These wrappers keep equation-rewrite orchestration in `cas_solver_core`
//! while callers provide stateful hooks for simplification and recursion.

use cas_ast::{Context, Equation, ExprId, RelOp, SolutionSet};

use crate::solve_outcome::{
    derive_add_isolation_operands, derive_mul_isolation_operands, derive_sub_isolation_operands,
    derive_div_isolation_route,
    execute_division_denominator_sign_split_or_term_isolation_plan_with_optional_items_and_merge_with_existing_steps_with_single_solver_with_state,
    execute_isolated_denominator_sign_split_or_division_denominator_plan_with_optional_items_and_merge_with_existing_steps_with_state,
    execute_product_zero_inequality_split_pipeline_with_existing_steps_with_context_snapshot,
    execute_term_isolation_plan_and_merge_with_existing_steps_with,
    execute_term_isolation_plan_with_rewritten_rhs_and_merge_with_existing_steps_with,
    merge_optional_solved_with_existing_steps_append_mut, mul_rhs_contains_variable,
    plan_add_operand_isolation_step_with, plan_div_denominator_isolation_with_zero_rhs_guard,
    plan_division_denominator_sign_split_or_div_numerator_isolation_with,
    plan_isolated_denominator_sign_split_or_division_denominator,
    plan_mul_factor_isolation_step_with, plan_product_zero_inequality_split_if_applicable,
    plan_sub_isolation_step_with, resolve_div_denominator_isolation_rhs_with, AddIsolationOperands,
    AddIsolationRoute, DivIsolationRoute, DivisionDenominatorDidacticPlan, DivisionDenominatorSignSplitPlan,
    DivisionDenominatorSignSplitSolvedCases, DivisionDidacticExecutionItem,
    IsolatedDenominatorSignSplitPlan, IsolatedDenominatorSignSplitSolvedCases,
    MulIsolationOperands, ProductZeroInequalityPlan, ProductZeroInequalitySolvedSets,
    TermIsolationRewriteExecutionItem, TermIsolationRewritePlan,
};

/// Execute additive isolation `(l + r) = rhs` with an optional
/// linear-collect fast path when both addends contain the solve variable.
#[allow(clippy::too_many_arguments)]
pub fn execute_add_isolation_pipeline_with_linear_collect_fallback_with_state<
    S,
    T,
    E,
    FDeriveOperands,
    FLinearCollect,
    FPlan,
    FSimplifyExpr,
    FIsolate,
    FMapStep,
>(
    state: &mut S,
    lhs: ExprId,
    left: ExprId,
    right: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    include_item: bool,
    mut existing_steps: Vec<T>,
    mut derive_operands: FDeriveOperands,
    mut try_linear_collect: FLinearCollect,
    mut plan_add: FPlan,
    mut simplify_expr: FSimplifyExpr,
    mut solve_rewritten: FIsolate,
    map_item_to_step: FMapStep,
) -> Result<(SolutionSet, Vec<T>), E>
where
    FDeriveOperands: FnMut(&mut S, ExprId, ExprId, &str) -> AddIsolationOperands,
    FLinearCollect: FnMut(&mut S, ExprId, ExprId, &str) -> Option<(SolutionSet, Vec<T>)>,
    FPlan: FnMut(&mut S, ExprId, ExprId, ExprId, RelOp) -> TermIsolationRewritePlan,
    FSimplifyExpr: FnMut(&mut S, ExprId) -> ExprId,
    FIsolate: FnMut(&mut S, Equation) -> Result<(SolutionSet, Vec<T>), E>,
    FMapStep: FnMut(TermIsolationRewriteExecutionItem) -> T,
{
    let operands = derive_operands(state, left, right, var);
    if matches!(operands.route, AddIsolationRoute::BothOperands) {
        if let Some(merged) = merge_optional_solved_with_existing_steps_append_mut(
            try_linear_collect(state, lhs, rhs, var),
            &mut existing_steps,
        ) {
            return Ok(merged);
        }
    }

    let plan = plan_add(
        state,
        operands.isolated_addend,
        operands.moved_addend,
        rhs,
        op,
    );
    let rewritten_rhs = simplify_expr(state, plan.equation.rhs);
    execute_term_isolation_plan_with_rewritten_rhs_and_merge_with_existing_steps_with(
        plan,
        rewritten_rhs,
        include_item,
        false,
        existing_steps,
        |expr| expr,
        |equation| solve_rewritten(state, equation),
        map_item_to_step,
    )
}

/// Execute additive isolation `(l + r) = rhs` using default operand derivation
/// and rewrite planning from `solve_outcome`.
#[allow(clippy::too_many_arguments)]
pub fn execute_add_isolation_pipeline_with_default_operands_and_plan_with_state<
    S,
    T,
    E,
    FContextRef,
    FContextMut,
    FRenderExpr,
    FLinearCollect,
    FSimplifyExpr,
    FIsolate,
    FMapStep,
>(
    state: &mut S,
    lhs: ExprId,
    left: ExprId,
    right: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    include_item: bool,
    existing_steps: Vec<T>,
    context_ref: FContextRef,
    context_mut: FContextMut,
    render_expr: FRenderExpr,
    try_linear_collect: FLinearCollect,
    simplify_expr: FSimplifyExpr,
    solve_rewritten: FIsolate,
    map_item_to_step: FMapStep,
) -> Result<(SolutionSet, Vec<T>), E>
where
    FContextRef: Fn(&mut S) -> &Context,
    FContextMut: Fn(&mut S) -> &mut Context,
    FRenderExpr: Fn(&Context, ExprId) -> String,
    FLinearCollect: FnMut(&mut S, ExprId, ExprId, &str) -> Option<(SolutionSet, Vec<T>)>,
    FSimplifyExpr: FnMut(&mut S, ExprId) -> ExprId,
    FIsolate: FnMut(&mut S, Equation) -> Result<(SolutionSet, Vec<T>), E>,
    FMapStep: FnMut(TermIsolationRewriteExecutionItem) -> T,
{
    execute_add_isolation_pipeline_with_linear_collect_fallback_with_state(
        state,
        lhs,
        left,
        right,
        rhs,
        op,
        var,
        include_item,
        existing_steps,
        |state, local_left, local_right, var_name| {
            derive_add_isolation_operands(context_ref(state), local_left, local_right, var_name)
        },
        try_linear_collect,
        |state, kept, moved, local_rhs, local_op| {
            let moved_desc = render_expr(context_ref(state), moved);
            plan_add_operand_isolation_step_with(
                context_mut(state),
                kept,
                moved,
                local_rhs,
                local_op,
                |_| moved_desc.clone(),
            )
        },
        simplify_expr,
        solve_rewritten,
        map_item_to_step,
    )
}

/// Execute subtractive isolation `(l - r) = rhs`.
#[allow(clippy::too_many_arguments)]
pub fn execute_sub_isolation_pipeline_with_state<
    S,
    T,
    E,
    FPlan,
    FSimplifyExpr,
    FIsolate,
    FMapStep,
>(
    state: &mut S,
    left: ExprId,
    right: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    include_item: bool,
    existing_steps: Vec<T>,
    mut plan_sub: FPlan,
    mut simplify_expr: FSimplifyExpr,
    mut solve_rewritten: FIsolate,
    map_item_to_step: FMapStep,
) -> Result<(SolutionSet, Vec<T>), E>
where
    FPlan: FnMut(&mut S, ExprId, ExprId, ExprId, RelOp, &str) -> TermIsolationRewritePlan,
    FSimplifyExpr: FnMut(&mut S, ExprId) -> ExprId,
    FIsolate: FnMut(&mut S, Equation) -> Result<(SolutionSet, Vec<T>), E>,
    FMapStep: FnMut(TermIsolationRewriteExecutionItem) -> T,
{
    let plan = plan_sub(state, left, right, rhs, op, var);
    let rewritten_rhs = simplify_expr(state, plan.equation.rhs);
    execute_term_isolation_plan_with_rewritten_rhs_and_merge_with_existing_steps_with(
        plan,
        rewritten_rhs,
        include_item,
        false,
        existing_steps,
        |expr| expr,
        |equation| solve_rewritten(state, equation),
        map_item_to_step,
    )
}

/// Execute subtractive isolation `(l - r) = rhs` using default route derivation
/// and rewrite planning from `solve_outcome`.
#[allow(clippy::too_many_arguments)]
pub fn execute_sub_isolation_pipeline_with_default_plan_with_state<
    S,
    T,
    E,
    FContextRef,
    FContextMut,
    FRenderExpr,
    FSimplifyExpr,
    FIsolate,
    FMapStep,
>(
    state: &mut S,
    left: ExprId,
    right: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    include_item: bool,
    existing_steps: Vec<T>,
    context_ref: FContextRef,
    context_mut: FContextMut,
    render_expr: FRenderExpr,
    simplify_expr: FSimplifyExpr,
    solve_rewritten: FIsolate,
    map_item_to_step: FMapStep,
) -> Result<(SolutionSet, Vec<T>), E>
where
    FContextRef: Fn(&mut S) -> &Context,
    FContextMut: Fn(&mut S) -> &mut Context,
    FRenderExpr: Fn(&Context, ExprId) -> String,
    FSimplifyExpr: FnMut(&mut S, ExprId) -> ExprId,
    FIsolate: FnMut(&mut S, Equation) -> Result<(SolutionSet, Vec<T>), E>,
    FMapStep: FnMut(TermIsolationRewriteExecutionItem) -> T,
{
    execute_sub_isolation_pipeline_with_state(
        state,
        left,
        right,
        rhs,
        op,
        var,
        include_item,
        existing_steps,
        |state, local_left, local_right, local_rhs, local_op, var_name| {
            let sub_operands = derive_sub_isolation_operands(
                context_ref(state),
                local_left,
                local_right,
                var_name,
            );
            let moved_desc = render_expr(context_ref(state), sub_operands.moved_term);
            plan_sub_isolation_step_with(
                context_mut(state),
                local_left,
                local_right,
                local_rhs,
                local_op,
                var_name,
                |_| moved_desc.clone(),
            )
        },
        simplify_expr,
        solve_rewritten,
        map_item_to_step,
    )
}

/// Execute multiplicative isolation `(l * r) = rhs`, including:
/// 1) product-zero inequality split when applicable,
/// 2) linear-collect fallback for variable-containing RHS,
/// 3) default factor-isolation rewrite path.
#[allow(clippy::too_many_arguments)]
pub fn execute_mul_isolation_pipeline_with_product_split_and_linear_collect_with_state<
    S,
    T,
    E,
    FPlanProductSplit,
    FSolveSplitCase,
    FContextSnapshot,
    FFinalizeSplit,
    FRhsContainsVar,
    FTryLinearCollect,
    FDeriveOperands,
    FIsKnownNegative,
    FRenderExpr,
    FPlanMul,
    FIsolate,
    FMapStep,
>(
    state: &mut S,
    lhs: ExprId,
    left: ExprId,
    right: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    include_item: bool,
    mut existing_steps: Vec<T>,
    mut plan_product_split_if_applicable: FPlanProductSplit,
    mut solve_split_case: FSolveSplitCase,
    mut context_snapshot: FContextSnapshot,
    mut finalize_split_with_context: FFinalizeSplit,
    mut rhs_contains_variable: FRhsContainsVar,
    mut try_linear_collect: FTryLinearCollect,
    mut derive_operands: FDeriveOperands,
    mut is_known_negative: FIsKnownNegative,
    mut render_expr: FRenderExpr,
    mut plan_mul: FPlanMul,
    mut isolate_rewritten: FIsolate,
    map_item_to_step: FMapStep,
) -> Result<(SolutionSet, Vec<T>), E>
where
    FPlanProductSplit:
        FnMut(&mut S, ExprId, ExprId, ExprId, RelOp, &str) -> Option<ProductZeroInequalityPlan>,
    FSolveSplitCase: FnMut(&mut S, &Equation) -> Result<(SolutionSet, Vec<T>), E>,
    FContextSnapshot: FnMut(&mut S) -> Context,
    FFinalizeSplit: FnMut(&Context, ProductZeroInequalitySolvedSets) -> SolutionSet,
    FRhsContainsVar: FnMut(&mut S, ExprId, &str) -> bool,
    FTryLinearCollect: FnMut(&mut S, ExprId, ExprId, &str) -> Option<(SolutionSet, Vec<T>)>,
    FDeriveOperands: FnMut(&mut S, ExprId, ExprId, &str) -> MulIsolationOperands,
    FIsKnownNegative: FnMut(&mut S, ExprId) -> bool,
    FRenderExpr: FnMut(&mut S, ExprId) -> String,
    FPlanMul:
        FnMut(&mut S, ExprId, ExprId, ExprId, RelOp, bool, String) -> TermIsolationRewritePlan,
    FIsolate: FnMut(&mut S, Equation) -> Result<(SolutionSet, Vec<T>), E>,
    FMapStep: FnMut(TermIsolationRewriteExecutionItem) -> T,
{
    let split_plan = plan_product_split_if_applicable(state, left, right, rhs, op.clone(), var);
    if let Some(split_plan) = split_plan {
        let snapshot = context_snapshot(state);
        return execute_product_zero_inequality_split_pipeline_with_existing_steps_with_context_snapshot(
            &split_plan,
            existing_steps,
            |equation| solve_split_case(state, equation),
            || snapshot.clone(),
            |ctx, solved_sets| finalize_split_with_context(ctx, solved_sets),
        );
    }

    if rhs_contains_variable(state, rhs, var) {
        if let Some(merged) = merge_optional_solved_with_existing_steps_append_mut(
            try_linear_collect(state, lhs, rhs, var),
            &mut existing_steps,
        ) {
            return Ok(merged);
        }
    }

    let operands = derive_operands(state, left, right, var);
    let moved_is_negative = is_known_negative(state, operands.moved_factor);
    let moved_desc = render_expr(state, operands.moved_factor);
    let plan = plan_mul(
        state,
        operands.isolated_factor,
        operands.moved_factor,
        rhs,
        op,
        moved_is_negative,
        moved_desc,
    );
    execute_term_isolation_plan_and_merge_with_existing_steps_with(
        plan,
        include_item,
        false,
        existing_steps,
        |expr| expr,
        |equation| isolate_rewritten(state, equation),
        map_item_to_step,
    )
}

/// Execute multiplicative isolation `(l * r) = rhs` using default split/route
/// planning from `solve_outcome`.
#[allow(clippy::too_many_arguments)]
pub fn execute_mul_isolation_pipeline_with_default_operands_and_plan_with_state<
    S,
    T,
    E,
    FContextRef,
    FContextMut,
    FSolveSplitCase,
    FContextSnapshot,
    FFinalizeSplit,
    FTryLinearCollect,
    FIsKnownNegative,
    FRenderExpr,
    FIsolate,
    FMapStep,
>(
    state: &mut S,
    lhs: ExprId,
    left: ExprId,
    right: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    include_item: bool,
    existing_steps: Vec<T>,
    context_ref: FContextRef,
    context_mut: FContextMut,
    solve_split_case: FSolveSplitCase,
    context_snapshot: FContextSnapshot,
    finalize_split_with_context: FFinalizeSplit,
    try_linear_collect: FTryLinearCollect,
    is_known_negative: FIsKnownNegative,
    render_expr: FRenderExpr,
    isolate_rewritten: FIsolate,
    map_item_to_step: FMapStep,
) -> Result<(SolutionSet, Vec<T>), E>
where
    FContextRef: Fn(&mut S) -> &Context,
    FContextMut: Fn(&mut S) -> &mut Context,
    FSolveSplitCase: FnMut(&mut S, &Equation) -> Result<(SolutionSet, Vec<T>), E>,
    FContextSnapshot: FnMut(&mut S) -> Context,
    FFinalizeSplit: FnMut(&Context, ProductZeroInequalitySolvedSets) -> SolutionSet,
    FTryLinearCollect: FnMut(&mut S, ExprId, ExprId, &str) -> Option<(SolutionSet, Vec<T>)>,
    FIsKnownNegative: FnMut(&mut S, ExprId) -> bool,
    FRenderExpr: Fn(&Context, ExprId) -> String,
    FIsolate: FnMut(&mut S, Equation) -> Result<(SolutionSet, Vec<T>), E>,
    FMapStep: FnMut(TermIsolationRewriteExecutionItem) -> T,
{
    let mut solve_split_case = solve_split_case;
    let mut context_snapshot = context_snapshot;
    let mut finalize_split_with_context = finalize_split_with_context;
    let mut try_linear_collect = try_linear_collect;
    let mut is_known_negative = is_known_negative;
    let mut isolate_rewritten = isolate_rewritten;
    execute_mul_isolation_pipeline_with_product_split_and_linear_collect_with_state(
        state,
        lhs,
        left,
        right,
        rhs,
        op,
        var,
        include_item,
        existing_steps,
        |state, local_left, local_right, local_rhs, local_op, var_name| {
            plan_product_zero_inequality_split_if_applicable(
                context_mut(state),
                local_left,
                local_right,
                local_rhs,
                local_op,
                var_name,
            )
        },
        |state, equation| solve_split_case(state, equation),
        |state| context_snapshot(state),
        |ctx, solved_sets| finalize_split_with_context(ctx, solved_sets),
        |state, local_rhs, var_name| {
            mul_rhs_contains_variable(context_ref(state), local_rhs, var_name)
        },
        |state, local_lhs, local_rhs, var_name| {
            try_linear_collect(state, local_lhs, local_rhs, var_name)
        },
        |state, local_left, local_right, var_name| {
            derive_mul_isolation_operands(context_ref(state), local_left, local_right, var_name)
        },
        |state, expr| is_known_negative(state, expr),
        |state, expr| render_expr(context_ref(state), expr),
        |state, kept, moved, local_rhs, local_op, moved_is_negative, moved_desc| {
            plan_mul_factor_isolation_step_with(
                context_mut(state),
                kept,
                moved,
                local_rhs,
                local_op,
                moved_is_negative,
                |_| moved_desc.clone(),
            )
        },
        |state, equation| isolate_rewritten(state, equation),
        map_item_to_step,
    )
}

/// Execute numerator-side division isolation:
/// 1) optional denominator-sign split for inequalities,
/// 2) fallback term-isolation rewrite when no split applies.
#[allow(clippy::too_many_arguments)]
pub fn execute_div_numerator_isolation_pipeline_with_state<
    S,
    TStep,
    E,
    FIsKnownNegative,
    FRenderExpr,
    FPlanNumeratorRoute,
    FSimplifyExpr,
    FRenderFallbackExpr,
    FSolveEquation,
    FMapDivisionStep,
    FMapTermStep,
    FFinalize,
>(
    state: &mut S,
    numerator: ExprId,
    denominator: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    include_items: bool,
    existing_steps: Vec<TStep>,
    mut is_known_negative: FIsKnownNegative,
    mut render_expr: FRenderExpr,
    mut plan_numerator_route: FPlanNumeratorRoute,
    simplify_expr: FSimplifyExpr,
    mut render_fallback_expr: FRenderFallbackExpr,
    solve_equation: FSolveEquation,
    map_division_item_to_step: FMapDivisionStep,
    map_term_item_to_step: FMapTermStep,
    finalize_solved_sets: FFinalize,
) -> Result<(SolutionSet, Vec<TStep>), E>
where
    TStep: Clone,
    FIsKnownNegative: FnMut(&mut S, ExprId) -> bool,
    FRenderExpr: FnMut(&mut S, ExprId) -> String,
    FPlanNumeratorRoute: FnMut(
        &mut S,
        ExprId,
        ExprId,
        ExprId,
        RelOp,
        &str,
        bool,
        String,
    ) -> (
        Option<DivisionDenominatorSignSplitPlan>,
        TermIsolationRewritePlan,
    ),
    FSimplifyExpr: FnMut(&mut S, ExprId) -> ExprId,
    FRenderFallbackExpr: FnMut(&mut S, ExprId) -> String,
    FSolveEquation: FnMut(&mut S, &Equation) -> Result<(SolutionSet, Vec<TStep>), E>,
    FMapDivisionStep: FnMut(DivisionDidacticExecutionItem) -> TStep,
    FMapTermStep: FnMut(TermIsolationRewriteExecutionItem) -> TStep,
    FFinalize: FnMut(
        &mut S,
        DivisionDenominatorSignSplitSolvedCases<SolutionSet, SolutionSet>,
    ) -> SolutionSet,
{
    let denominator_is_negative = is_known_negative(state, denominator);
    let denominator_desc = render_expr(state, denominator);
    let (split_plan, term_plan) = plan_numerator_route(
        state,
        numerator,
        denominator,
        rhs,
        op.clone(),
        var,
        denominator_is_negative,
        denominator_desc.clone(),
    );

    execute_division_denominator_sign_split_or_term_isolation_plan_with_optional_items_and_merge_with_existing_steps_with_single_solver_with_state(
        state,
        split_plan,
        denominator,
        op,
        numerator,
        include_items,
        term_plan,
        false,
        existing_steps,
        simplify_expr,
        move |state, expr| {
            if expr == denominator {
                denominator_desc.clone()
            } else {
                render_fallback_expr(state, expr)
            }
        },
        solve_equation,
        map_division_item_to_step,
        map_term_item_to_step,
        finalize_solved_sets,
    )
}

/// Dispatch division-isolation route to the corresponding branch handler.
pub fn execute_div_isolation_route_with_state<T, R, E, FNumerator, FDenominator>(
    state: &mut T,
    route: DivIsolationRoute,
    on_variable_in_numerator: FNumerator,
    on_variable_in_denominator: FDenominator,
) -> Result<R, E>
where
    FNumerator: FnOnce(&mut T) -> Result<R, E>,
    FDenominator: FnOnce(&mut T) -> Result<R, E>,
{
    match route {
        DivIsolationRoute::VariableInNumerator => on_variable_in_numerator(state),
        DivIsolationRoute::VariableInDenominator => on_variable_in_denominator(state),
    }
}

/// Derive and dispatch division-isolation route from `(ctx, numerator, var)`.
pub fn execute_div_isolation_route_for_var_with_state<
    T,
    R,
    E,
    FContext,
    FNumerator,
    FDenominator,
>(
    state: &mut T,
    context: FContext,
    numerator: ExprId,
    var: &str,
    on_variable_in_numerator: FNumerator,
    on_variable_in_denominator: FDenominator,
) -> Result<R, E>
where
    FContext: FnMut(&mut T) -> &Context,
    FNumerator: FnOnce(&mut T) -> Result<R, E>,
    FDenominator: FnOnce(&mut T) -> Result<R, E>,
{
    let mut context = context;
    let route = derive_div_isolation_route(context(state), numerator, var);
    execute_div_isolation_route_with_state(
        state,
        route,
        on_variable_in_numerator,
        on_variable_in_denominator,
    )
}

/// Execute numerator-side division isolation with default route planning from
/// `solve_outcome`.
#[allow(clippy::too_many_arguments)]
pub fn execute_div_numerator_isolation_pipeline_with_default_plan_with_state<
    S,
    TStep,
    E,
    FContextMut,
    FIsKnownNegative,
    FRenderExpr,
    FSimplifyExpr,
    FSolveEquation,
    FMapDivisionStep,
    FMapTermStep,
    FFinalize,
>(
    state: &mut S,
    numerator: ExprId,
    denominator: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    include_items: bool,
    existing_steps: Vec<TStep>,
    context_mut: FContextMut,
    is_known_negative: FIsKnownNegative,
    render_expr: FRenderExpr,
    simplify_expr: FSimplifyExpr,
    solve_equation: FSolveEquation,
    map_division_item_to_step: FMapDivisionStep,
    map_term_item_to_step: FMapTermStep,
    finalize_solved_sets: FFinalize,
) -> Result<(SolutionSet, Vec<TStep>), E>
where
    TStep: Clone,
    FContextMut: FnMut(&mut S) -> &mut Context,
    FIsKnownNegative: FnMut(&mut S, ExprId) -> bool,
    FRenderExpr: FnMut(&mut S, ExprId) -> String,
    FSimplifyExpr: FnMut(&mut S, ExprId) -> ExprId,
    FSolveEquation: FnMut(&mut S, &Equation) -> Result<(SolutionSet, Vec<TStep>), E>,
    FMapDivisionStep: FnMut(DivisionDidacticExecutionItem) -> TStep,
    FMapTermStep: FnMut(TermIsolationRewriteExecutionItem) -> TStep,
    FFinalize: FnMut(
        &mut S,
        DivisionDenominatorSignSplitSolvedCases<SolutionSet, SolutionSet>,
    ) -> SolutionSet,
{
    let mut context_mut = context_mut;
    execute_div_numerator_isolation_pipeline_with_state(
        state,
        numerator,
        denominator,
        rhs,
        op,
        var,
        include_items,
        existing_steps,
        is_known_negative,
        render_expr,
        |state,
         local_numerator,
         local_denominator,
         local_rhs,
         local_op,
         var_name,
         denominator_is_negative,
         denominator_desc| {
            plan_division_denominator_sign_split_or_div_numerator_isolation_with(
                context_mut(state),
                local_numerator,
                local_denominator,
                local_rhs,
                local_op,
                var_name,
                denominator_is_negative,
                |_| denominator_desc.clone(),
            )
        },
        simplify_expr,
        |_, expr| format!("#{expr}"),
        solve_equation,
        map_division_item_to_step,
        map_term_item_to_step,
        finalize_solved_sets,
    )
}

/// Execute denominator-side division isolation:
/// 1) optional reciprocal-solve fast path,
/// 2) isolated-denominator sign split or denominator didactic fallback.
#[allow(clippy::too_many_arguments)]
pub fn execute_div_denominator_isolation_pipeline_with_reciprocal_fallback_and_state<
    S,
    TStep,
    E,
    FShouldTryReciprocal,
    FTryReciprocal,
    FPlanSplitOrDidactic,
    FRenderExpr,
    FSimplifyExpr,
    FSolveBranch,
    FMapStep,
    FFinalize,
>(
    state: &mut S,
    lhs: ExprId,
    numerator: ExprId,
    denominator: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    include_items: bool,
    mut existing_steps: Vec<TStep>,
    mut should_try_reciprocal: FShouldTryReciprocal,
    mut try_reciprocal_solve: FTryReciprocal,
    mut plan_split_or_didactic: FPlanSplitOrDidactic,
    mut render_expr: FRenderExpr,
    simplify_expr: FSimplifyExpr,
    solve_branch: FSolveBranch,
    map_item_to_step: FMapStep,
    finalize_solved_sets: FFinalize,
) -> Result<(SolutionSet, Vec<TStep>), E>
where
    TStep: Clone,
    FShouldTryReciprocal: FnMut(&mut S, ExprId, &RelOp, &str) -> bool,
    FTryReciprocal: FnMut(&mut S, ExprId, ExprId, &str) -> Option<(SolutionSet, Vec<TStep>)>,
    FPlanSplitOrDidactic: FnMut(
        &mut S,
        ExprId,
        ExprId,
        ExprId,
        RelOp,
        &str,
    ) -> (
        Option<IsolatedDenominatorSignSplitPlan>,
        DivisionDenominatorDidacticPlan,
    ),
    FRenderExpr: FnMut(&mut S, ExprId) -> String,
    FSimplifyExpr: FnMut(&mut S, ExprId) -> ExprId,
    FSolveBranch: FnMut(&mut S, &Equation) -> Result<(SolutionSet, Vec<TStep>), E>,
    FMapStep: FnMut(DivisionDidacticExecutionItem) -> TStep,
    FFinalize: FnMut(&mut S, IsolatedDenominatorSignSplitSolvedCases<SolutionSet>) -> SolutionSet,
{
    if should_try_reciprocal(state, lhs, &op, var) {
        if let Some(merged) = merge_optional_solved_with_existing_steps_append_mut(
            try_reciprocal_solve(state, lhs, rhs, var),
            &mut existing_steps,
        ) {
            return Ok(merged);
        }
    }

    let (split_plan, didactic_plan) =
        plan_split_or_didactic(state, numerator, denominator, rhs, op.clone(), var);
    let multiply_by = didactic_plan.multiply_by;
    let divide_by = didactic_plan.divide_by;
    let denominator_desc = render_expr(state, denominator);
    let multiply_by_desc = render_expr(state, multiply_by);
    let divide_by_desc = render_expr(state, divide_by);

    execute_isolated_denominator_sign_split_or_division_denominator_plan_with_optional_items_and_merge_with_existing_steps_with_state(
        state,
        split_plan,
        denominator,
        op,
        include_items,
        didactic_plan,
        existing_steps,
        simplify_expr,
        move |_, expr| {
            if expr == denominator {
                denominator_desc.clone()
            } else if expr == multiply_by {
                multiply_by_desc.clone()
            } else if expr == divide_by {
                divide_by_desc.clone()
            } else {
                format!("#{expr}")
            }
        },
        solve_branch,
        map_item_to_step,
        finalize_solved_sets,
    )
}

/// Execute denominator-side division isolation with default split/didactic plan
/// resolution from `solve_outcome`.
#[allow(clippy::too_many_arguments)]
pub fn execute_div_denominator_isolation_pipeline_with_default_plan_with_state<
    S,
    TStep,
    E,
    FContextMut,
    FShouldTryReciprocal,
    FTryReciprocal,
    FRenderExpr,
    FSimplifyForRhs,
    FSimplifyExpr,
    FSolveBranch,
    FMapStep,
    FFinalize,
>(
    state: &mut S,
    lhs: ExprId,
    numerator: ExprId,
    denominator: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    include_items: bool,
    existing_steps: Vec<TStep>,
    context_mut: FContextMut,
    should_try_reciprocal: FShouldTryReciprocal,
    try_reciprocal_solve: FTryReciprocal,
    render_expr: FRenderExpr,
    simplify_for_rhs: FSimplifyForRhs,
    simplify_expr: FSimplifyExpr,
    solve_branch: FSolveBranch,
    map_item_to_step: FMapStep,
    finalize_solved_sets: FFinalize,
) -> Result<(SolutionSet, Vec<TStep>), E>
where
    TStep: Clone,
    FContextMut: FnMut(&mut S) -> &mut Context,
    FShouldTryReciprocal: FnMut(&mut S, ExprId, &RelOp, &str) -> bool,
    FTryReciprocal: FnMut(&mut S, ExprId, ExprId, &str) -> Option<(SolutionSet, Vec<TStep>)>,
    FRenderExpr: FnMut(&mut S, ExprId) -> String,
    FSimplifyForRhs: FnMut(&mut S, ExprId) -> ExprId,
    FSimplifyExpr: FnMut(&mut S, ExprId) -> ExprId,
    FSolveBranch: FnMut(&mut S, &Equation) -> Result<(SolutionSet, Vec<TStep>), E>,
    FMapStep: FnMut(DivisionDidacticExecutionItem) -> TStep,
    FFinalize: FnMut(&mut S, IsolatedDenominatorSignSplitSolvedCases<SolutionSet>) -> SolutionSet,
{
    let mut context_mut = context_mut;
    let mut simplify_for_rhs = simplify_for_rhs;
    execute_div_denominator_isolation_pipeline_with_reciprocal_fallback_and_state(
        state,
        lhs,
        numerator,
        denominator,
        rhs,
        op,
        var,
        include_items,
        existing_steps,
        should_try_reciprocal,
        try_reciprocal_solve,
        |state, local_numerator, local_denominator, local_rhs, local_op, var_name| {
            let isolation_plan = plan_div_denominator_isolation_with_zero_rhs_guard(
                context_mut(state),
                local_denominator,
                local_numerator,
                local_rhs,
                local_op.clone(),
            );
            let (_, simplified_rhs) =
                resolve_div_denominator_isolation_rhs_with(isolation_plan, |expr| {
                    simplify_for_rhs(state, expr)
                });
            plan_isolated_denominator_sign_split_or_division_denominator(
                context_mut(state),
                local_numerator,
                local_denominator,
                local_rhs,
                simplified_rhs,
                local_op,
                var_name,
            )
        },
        render_expr,
        simplify_expr,
        solve_branch,
        map_item_to_step,
        finalize_solved_sets,
    )
}

/// Execute full division isolation pipeline (`numerator / denominator = rhs`)
/// using default route derivation and default numerator/denominator kernels.
///
/// Route behavior:
/// - `VariableInNumerator`  -> numerator-side pipeline
/// - `VariableInDenominator` -> denominator-side pipeline
#[allow(clippy::too_many_arguments)]
pub fn execute_div_isolation_pipeline_with_default_route_and_kernels_with_state<
    S,
    TStep,
    E,
    FContextRef,
    FContextMut,
    FIsKnownNegative,
    FRenderExpr,
    FSimplifyExpr,
    FSimplifyForRhs,
    FSolveIsolate,
    FMapNumeratorDivisionStep,
    FMapNumeratorTermStep,
    FFinalizeNumerator,
    FShouldTryReciprocal,
    FTryReciprocal,
    FMapDenominatorStep,
    FFinalizeDenominator,
>(
    state: &mut S,
    lhs: ExprId,
    numerator: ExprId,
    denominator: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    include_numerator_items: bool,
    include_denominator_items: bool,
    existing_steps: Vec<TStep>,
    context_ref: FContextRef,
    context_mut: FContextMut,
    is_known_negative: FIsKnownNegative,
    render_expr: FRenderExpr,
    simplify_expr: FSimplifyExpr,
    simplify_for_rhs: FSimplifyForRhs,
    solve_isolate: FSolveIsolate,
    map_numerator_division_item_to_step: FMapNumeratorDivisionStep,
    map_numerator_term_item_to_step: FMapNumeratorTermStep,
    finalize_numerator_solved_sets: FFinalizeNumerator,
    should_try_reciprocal: FShouldTryReciprocal,
    try_reciprocal_solve: FTryReciprocal,
    map_denominator_item_to_step: FMapDenominatorStep,
    finalize_denominator_solved_sets: FFinalizeDenominator,
) -> Result<(SolutionSet, Vec<TStep>), E>
where
    TStep: Clone,
    FContextRef: Fn(&mut S) -> &Context,
    FContextMut: FnMut(&mut S) -> &mut Context,
    FIsKnownNegative: FnMut(&mut S, ExprId) -> bool,
    FRenderExpr: FnMut(&mut S, ExprId) -> String,
    FSimplifyExpr: FnMut(&mut S, ExprId) -> ExprId,
    FSimplifyForRhs: FnMut(&mut S, ExprId) -> ExprId,
    FSolveIsolate: FnMut(&mut S, &Equation) -> Result<(SolutionSet, Vec<TStep>), E>,
    FMapNumeratorDivisionStep: FnMut(DivisionDidacticExecutionItem) -> TStep,
    FMapNumeratorTermStep: FnMut(TermIsolationRewriteExecutionItem) -> TStep,
    FFinalizeNumerator: FnMut(
        &mut S,
        DivisionDenominatorSignSplitSolvedCases<SolutionSet, SolutionSet>,
    ) -> SolutionSet,
    FShouldTryReciprocal: FnMut(&mut S, ExprId, &RelOp, &str) -> bool,
    FTryReciprocal: FnMut(&mut S, ExprId, ExprId, &str) -> Option<(SolutionSet, Vec<TStep>)>,
    FMapDenominatorStep: FnMut(DivisionDidacticExecutionItem) -> TStep,
    FFinalizeDenominator:
        FnMut(&mut S, IsolatedDenominatorSignSplitSolvedCases<SolutionSet>) -> SolutionSet,
{
    let mut context_mut = context_mut;
    let mut is_known_negative = is_known_negative;
    let mut render_expr = render_expr;
    let mut simplify_expr = simplify_expr;
    let mut simplify_for_rhs = simplify_for_rhs;
    let mut solve_isolate = solve_isolate;
    let map_numerator_division_item_to_step = map_numerator_division_item_to_step;
    let map_numerator_term_item_to_step = map_numerator_term_item_to_step;
    let mut finalize_numerator_solved_sets = finalize_numerator_solved_sets;
    let mut should_try_reciprocal = should_try_reciprocal;
    let mut try_reciprocal_solve = try_reciprocal_solve;
    let map_denominator_item_to_step = map_denominator_item_to_step;
    let mut finalize_denominator_solved_sets = finalize_denominator_solved_sets;
    let route = derive_div_isolation_route(context_ref(state), numerator, var);

    match route {
        DivIsolationRoute::VariableInNumerator => {
            execute_div_numerator_isolation_pipeline_with_default_plan_with_state(
                state,
                numerator,
                denominator,
                rhs,
                op.clone(),
                var,
                include_numerator_items,
                existing_steps,
                |state| context_mut(state),
                |state, expr| is_known_negative(state, expr),
                |state, expr| render_expr(state, expr),
                |state, expr| simplify_expr(state, expr),
                |state, equation| solve_isolate(state, equation),
                map_numerator_division_item_to_step,
                map_numerator_term_item_to_step,
                |state, solved_sets| finalize_numerator_solved_sets(state, solved_sets),
            )
        }
        DivIsolationRoute::VariableInDenominator => {
            execute_div_denominator_isolation_pipeline_with_default_plan_with_state(
                state,
                lhs,
                numerator,
                denominator,
                rhs,
                op,
                var,
                include_denominator_items,
                existing_steps,
                |state| context_mut(state),
                |state, lhs_expr, local_op, var_name| {
                    should_try_reciprocal(state, lhs_expr, local_op, var_name)
                },
                |state, lhs_expr, local_rhs, var_name| {
                    try_reciprocal_solve(state, lhs_expr, local_rhs, var_name)
                },
                |state, expr| render_expr(state, expr),
                |state, expr| simplify_for_rhs(state, expr),
                |state, expr| simplify_expr(state, expr),
                |state, equation| solve_isolate(state, equation),
                map_denominator_item_to_step,
                |state, solved_sets| finalize_denominator_solved_sets(state, solved_sets),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        execute_add_isolation_pipeline_with_linear_collect_fallback_with_state,
        execute_div_denominator_isolation_pipeline_with_reciprocal_fallback_and_state,
        execute_div_isolation_route_for_var_with_state,
        execute_div_isolation_route_with_state,
        execute_div_numerator_isolation_pipeline_with_state,
        execute_mul_isolation_pipeline_with_product_split_and_linear_collect_with_state,
        execute_sub_isolation_pipeline_with_state,
    };
    use cas_ast::{Expr, ExprId, RelOp, SolutionSet};

    #[derive(Default)]
    struct SubHarness {
        context: cas_ast::Context,
        seen_lhs: Option<ExprId>,
    }

    #[derive(Default)]
    struct MulHarness {
        context: cas_ast::Context,
        split_solve_count: usize,
        isolate_count: usize,
        linear_collect_count: usize,
    }

    #[derive(Default)]
    struct DivHarness {
        context: cas_ast::Context,
        seen_lhs: Option<ExprId>,
    }

    #[test]
    fn execute_div_isolation_route_with_state_dispatches_numerator_branch() {
        let mut hit = 0usize;
        let out = execute_div_isolation_route_with_state(
            &mut hit,
            crate::solve_outcome::DivIsolationRoute::VariableInNumerator,
            |state| {
                *state += 1;
                Ok::<_, &'static str>("num")
            },
            |_state| Ok("den"),
        )
        .expect("numerator route should dispatch");

        assert_eq!(out, "num");
        assert_eq!(hit, 1);
    }

    #[test]
    fn execute_div_isolation_route_for_var_with_state_derives_denominator_branch() {
        let mut context = cas_ast::Context::new();
        let two = context.num(2);
        let x = context.var("x");
        let lhs = context.add(Expr::Div(two, x));
        let numerator = match context.get(lhs) {
            Expr::Div(num, _den) => *num,
            other => panic!("expected division expression, got {other:?}"),
        };

        let mut state = context;
        let out = execute_div_isolation_route_for_var_with_state(
            &mut state,
            |state| state,
            numerator,
            "x",
            |_state| Ok::<_, &'static str>("num"),
            |_state| Ok("den"),
        )
        .expect("route should derive denominator branch");

        assert_eq!(out, "den");
    }

    #[test]
    fn execute_add_isolation_pipeline_with_linear_collect_fallback_with_state_merges_existing_steps(
    ) {
        let mut ctx = cas_ast::Context::new();
        let x = ctx.var("x");
        let lhs = ctx.add(Expr::Add(x, x));
        let two = ctx.num(2);
        let mut state = ();

        let solved = execute_add_isolation_pipeline_with_linear_collect_fallback_with_state(
            &mut state,
            lhs,
            x,
            x,
            two,
            RelOp::Eq,
            "x",
            false,
            vec!["existing".to_string()],
            |_state, l, r, _var| crate::solve_outcome::AddIsolationOperands {
                route: crate::solve_outcome::AddIsolationRoute::BothOperands,
                isolated_addend: l,
                moved_addend: r,
            },
            |_state, _lhs, _rhs, _var| Some((SolutionSet::AllReals, vec!["fallback".to_string()])),
            |_state, _kept, _moved, _rhs, _op| {
                panic!("plan_add must not run when linear-collect fallback solves")
            },
            |_state, expr| expr,
            |_state, _equation| -> Result<(SolutionSet, Vec<String>), &'static str> {
                Err("recursive solve should not run")
            },
            |_item| "unused-step".to_string(),
        )
        .expect("fallback should solve");

        assert!(matches!(solved.0, SolutionSet::AllReals));
        assert_eq!(
            solved.1,
            vec!["existing".to_string(), "fallback".to_string()]
        );
    }

    #[test]
    fn execute_sub_isolation_pipeline_with_state_runs_recursive_solver_and_preserves_step_order() {
        let mut state = SubHarness::default();
        let x = state.context.var("x");
        let one = state.context.num(1);
        let two = state.context.num(2);

        let solved = execute_sub_isolation_pipeline_with_state(
            &mut state,
            x,
            one,
            two,
            RelOp::Eq,
            "x",
            false,
            vec!["existing".to_string()],
            |state, left, right, rhs, op, var| {
                crate::solve_outcome::plan_sub_isolation_step_with(
                    &mut state.context,
                    left,
                    right,
                    rhs,
                    op,
                    var,
                    |_| "move term".to_string(),
                )
            },
            |_state, expr| expr,
            |state, equation| -> Result<(SolutionSet, Vec<String>), &'static str> {
                state.seen_lhs = Some(equation.lhs);
                Ok((SolutionSet::Empty, vec!["solved".to_string()]))
            },
            |_item| "unused-step".to_string(),
        )
        .expect("sub isolation should call recursive solver");

        assert_eq!(state.seen_lhs, Some(x));
        assert!(matches!(solved.0, SolutionSet::Empty));
        assert_eq!(solved.1, vec!["solved".to_string(), "existing".to_string()]);
    }

    #[test]
    fn execute_mul_isolation_pipeline_with_product_split_and_linear_collect_with_state_prefers_product_split(
    ) {
        let mut state = MulHarness::default();
        let x = state.context.var("x");
        let zero = state.context.num(0);

        let solved =
            execute_mul_isolation_pipeline_with_product_split_and_linear_collect_with_state(
                &mut state,
                x,
                x,
                x,
                zero,
                RelOp::Gt,
                "x",
                false,
                vec!["existing".to_string()],
                |state, left, right, rhs, op, var| {
                    crate::solve_outcome::plan_product_zero_inequality_split_if_applicable(
                        &mut state.context,
                        left,
                        right,
                        rhs,
                        op,
                        var,
                    )
                },
                |state, _equation| -> Result<(SolutionSet, Vec<String>), &'static str> {
                    state.split_solve_count += 1;
                    Ok((SolutionSet::AllReals, vec!["split".to_string()]))
                },
                |state| state.context.clone(),
                |ctx, solved_sets| {
                    crate::solve_outcome::finalize_product_zero_inequality_solved_sets(
                        ctx,
                        solved_sets,
                    )
                },
                |_state, _rhs, _var| false,
                |state, _lhs, _rhs, _var| {
                    state.linear_collect_count += 1;
                    None
                },
                |state, left, right, var| {
                    crate::solve_outcome::derive_mul_isolation_operands(
                        &state.context,
                        left,
                        right,
                        var,
                    )
                },
                |_state, _expr| false,
                |_state, _expr| "unused".to_string(),
                |state, kept, moved, rhs, op, moved_is_negative, moved_desc| {
                    crate::solve_outcome::plan_mul_factor_isolation_step_with(
                        &mut state.context,
                        kept,
                        moved,
                        rhs,
                        op,
                        moved_is_negative,
                        |_| moved_desc.clone(),
                    )
                },
                |state, _equation| -> Result<(SolutionSet, Vec<String>), &'static str> {
                    state.isolate_count += 1;
                    Ok((SolutionSet::Empty, vec!["recurse".to_string()]))
                },
                |_item| "mapped".to_string(),
            )
            .expect("product split path should solve");

        assert_eq!(state.linear_collect_count, 0);
        assert_eq!(state.isolate_count, 0);
        assert_eq!(state.split_solve_count, 4);
        assert!(!matches!(solved.0, SolutionSet::Empty));
        assert_eq!(
            solved.1,
            vec![
                "split".to_string(),
                "split".to_string(),
                "split".to_string(),
                "split".to_string(),
                "existing".to_string()
            ]
        );
    }

    #[test]
    fn execute_mul_isolation_pipeline_with_product_split_and_linear_collect_with_state_uses_linear_collect_fallback(
    ) {
        let mut state = MulHarness::default();
        let x = state.context.var("x");
        let two = state.context.num(2);
        let rhs = state.context.add(Expr::Add(x, two));

        let solved =
            execute_mul_isolation_pipeline_with_product_split_and_linear_collect_with_state(
                &mut state,
                x,
                x,
                two,
                rhs,
                RelOp::Eq,
                "x",
                false,
                vec!["existing".to_string()],
                |_state, _left, _right, _rhs, _op, _var| None,
                |_state, _equation| -> Result<(SolutionSet, Vec<String>), &'static str> {
                    Err("split solver should not run")
                },
                |state| state.context.clone(),
                |_ctx, _solved_sets| SolutionSet::Empty,
                |_state, _rhs, _var| true,
                |state, _lhs, _rhs, _var| {
                    state.linear_collect_count += 1;
                    Some((SolutionSet::AllReals, vec!["fallback".to_string()]))
                },
                |_state, _left, _right, _var| crate::solve_outcome::MulIsolationOperands {
                    route: crate::solve_outcome::MulIsolationRoute::LeftFactor,
                    isolated_factor: x,
                    moved_factor: two,
                },
                |_state, _expr| false,
                |_state, _expr| "unused".to_string(),
                |_state, _kept, _moved, _rhs, _op, _neg, _desc| {
                    panic!("mul plan should not be built on linear fallback")
                },
                |_state, _equation| -> Result<(SolutionSet, Vec<String>), &'static str> {
                    Err("recursive isolate should not run")
                },
                |_item| "mapped".to_string(),
            )
            .expect("linear collect fallback should solve");

        assert_eq!(state.linear_collect_count, 1);
        assert!(matches!(solved.0, SolutionSet::AllReals));
        assert_eq!(
            solved.1,
            vec!["existing".to_string(), "fallback".to_string()]
        );
    }

    #[test]
    fn execute_div_numerator_isolation_pipeline_with_state_uses_term_fallback_when_no_split() {
        let mut state = DivHarness::default();
        let x = state.context.var("x");
        let y = state.context.var("y");
        let two = state.context.num(2);

        let solved = execute_div_numerator_isolation_pipeline_with_state(
            &mut state,
            x,
            y,
            two,
            RelOp::Eq,
            "x",
            false,
            vec!["existing".to_string()],
            |_state, _expr| false,
            |_state, expr| format!("#{expr}"),
            |state,
             numerator,
             denominator,
             rhs,
             op,
             _var,
             denominator_is_negative,
             denominator_desc| {
                let term_plan = crate::solve_outcome::plan_div_numerator_isolation_step_with(
                    &mut state.context,
                    numerator,
                    denominator,
                    rhs,
                    op,
                    denominator_is_negative,
                    |_| denominator_desc.clone(),
                );
                (None, term_plan)
            },
            |_state, expr| expr,
            |_state, expr| format!("#{expr}"),
            |state, equation| -> Result<(SolutionSet, Vec<String>), &'static str> {
                state.seen_lhs = Some(equation.lhs);
                Ok((SolutionSet::AllReals, vec!["recursive".to_string()]))
            },
            |_item| "unused-division-step".to_string(),
            |_item| "unused-term-step".to_string(),
            |_state, _solved| -> SolutionSet {
                panic!("finalizer must not run without split plan")
            },
        )
        .expect("fallback term path should solve");

        assert_eq!(state.seen_lhs, Some(x));
        assert!(matches!(solved.0, SolutionSet::AllReals));
        assert_eq!(
            solved.1,
            vec!["recursive".to_string(), "existing".to_string()]
        );
    }

    #[test]
    fn execute_div_denominator_isolation_pipeline_with_reciprocal_fallback_and_state_returns_reciprocal_result_when_available(
    ) {
        let mut ctx = cas_ast::Context::new();
        let lhs = ctx.var("x");
        let numerator = ctx.num(1);
        let denominator = ctx.var("y");
        let rhs = ctx.num(2);
        let mut state = ();

        let solved = execute_div_denominator_isolation_pipeline_with_reciprocal_fallback_and_state(
            &mut state,
            lhs,
            numerator,
            denominator,
            rhs,
            RelOp::Eq,
            "x",
            false,
            vec!["existing".to_string()],
            |_state, _lhs, _op, _var| true,
            |_state, _lhs, _rhs, _var| {
                Some((SolutionSet::AllReals, vec!["reciprocal".to_string()]))
            },
            |_state, _num, _den, _rhs, _op, _var| {
                panic!("planning must not run when reciprocal fallback solves")
            },
            |_state, _expr| "unused".to_string(),
            |_state, expr| expr,
            |_state, _equation| -> Result<(SolutionSet, Vec<String>), &'static str> {
                Err("recursive solve should not run")
            },
            |_item| "unused-step".to_string(),
            |_state, _solved| -> SolutionSet {
                panic!("finalizer must not run when reciprocal fallback solves")
            },
        )
        .expect("reciprocal fallback should solve");

        assert!(matches!(solved.0, SolutionSet::AllReals));
        assert_eq!(
            solved.1,
            vec!["existing".to_string(), "reciprocal".to_string()]
        );
    }
}
