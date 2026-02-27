//! Arithmetic isolation orchestration helpers shared by engine-side solver.
//!
//! These wrappers keep equation-rewrite orchestration in `cas_solver_core`
//! while callers provide stateful hooks for simplification and recursion.

use cas_ast::{Context, Equation, ExprId, RelOp, SolutionSet};

use crate::solve_outcome::{
    execute_product_zero_inequality_split_pipeline_with_existing_steps_with_context_snapshot,
    execute_term_isolation_plan_and_merge_with_existing_steps_with,
    execute_term_isolation_plan_with_rewritten_rhs_and_merge_with_existing_steps_with,
    merge_optional_solved_with_existing_steps_append_mut, AddIsolationOperands, AddIsolationRoute,
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

#[cfg(test)]
mod tests {
    use super::{
        execute_add_isolation_pipeline_with_linear_collect_fallback_with_state,
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
}
