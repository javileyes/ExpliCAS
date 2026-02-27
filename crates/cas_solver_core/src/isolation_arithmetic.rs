//! Arithmetic isolation orchestration helpers shared by engine-side solver.
//!
//! These wrappers keep equation-rewrite orchestration in `cas_solver_core`
//! while callers provide stateful hooks for simplification and recursion.

use cas_ast::{Equation, ExprId, RelOp, SolutionSet};

use crate::solve_outcome::{
    execute_term_isolation_plan_with_rewritten_rhs_and_merge_with_existing_steps_with,
    merge_optional_solved_with_existing_steps_append_mut, AddIsolationOperands, AddIsolationRoute,
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

#[cfg(test)]
mod tests {
    use super::{
        execute_add_isolation_pipeline_with_linear_collect_fallback_with_state,
        execute_sub_isolation_pipeline_with_state,
    };
    use cas_ast::{Expr, ExprId, RelOp, SolutionSet};

    #[derive(Default)]
    struct SubHarness {
        context: cas_ast::Context,
        seen_lhs: Option<ExprId>,
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
}
