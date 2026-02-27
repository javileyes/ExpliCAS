//! Power-isolation orchestration helpers shared by engine-side solver.
//!
//! These wrappers keep equation-rewrite orchestration in `cas_solver_core`
//! while callers provide stateful hooks for simplification and recursion.

use cas_ast::{Expr, ExprId, RelOp, SolutionSet};

use crate::solve_outcome::{
    execute_pow_base_isolation_pipeline_with_item_and_merge_with_existing_steps,
    execute_pow_exponent_shortcut_action_pipeline_with_item_and_finalize_with_existing_steps_with,
    execute_pow_exponent_shortcut_with_state, PowBaseIsolationEngineAction,
    PowExponentShortcutAction, PowExponentShortcutEngineAction, PowExponentShortcutExecutionItem,
};

/// Execute base-side power isolation (`b^e = rhs`) with caller-provided stateful hooks.
#[allow(clippy::too_many_arguments)]
pub fn execute_pow_base_isolation_pipeline_with_state<
    T,
    S,
    E,
    FPlanAction,
    FSolveIsolate,
    FMapStep,
>(
    state: &mut T,
    base: ExprId,
    exponent: ExprId,
    rhs: ExprId,
    op: RelOp,
    include_item: bool,
    existing_steps: Vec<S>,
    mut plan_action: FPlanAction,
    mut solve_isolate: FSolveIsolate,
    map_item_to_step: FMapStep,
) -> Result<(SolutionSet, Vec<S>), E>
where
    FPlanAction: FnMut(&mut T, ExprId, ExprId, ExprId, RelOp) -> PowBaseIsolationEngineAction,
    FSolveIsolate: FnMut(&mut T, ExprId, ExprId, RelOp) -> Result<(SolutionSet, Vec<S>), E>,
    FMapStep: FnMut(crate::solve_outcome::PowBaseIsolationExecutionItem) -> S,
{
    let action = plan_action(state, base, exponent, rhs, op);
    execute_pow_base_isolation_pipeline_with_item_and_merge_with_existing_steps(
        include_item,
        existing_steps,
        action,
        |isolated_lhs, isolated_rhs, isolated_op| {
            solve_isolate(state, isolated_lhs, isolated_rhs, isolated_op)
        },
        map_item_to_step,
    )
}

/// Execute exponent-side prelude for power isolation (`b^e = rhs`):
/// 1) exponent shortcut (`b^x = b`, `b^x = b^n`, etc.),
/// 2) RHS-variable safety guard for logarithmic inversion,
/// 3) base-one shortcut (`1^x = rhs`).
///
/// Returns:
/// - `Ok(Some(...))` when prelude produced a final solved result.
/// - `Ok(None)` when solver should continue with logarithmic isolation.
#[allow(clippy::too_many_arguments)]
pub fn execute_pow_exponent_shortcuts_and_guards_with_state<
    T,
    S,
    E,
    FClassifyBaseFlags,
    FReadExpr,
    FPlanShortcutAction,
    FBasesEquivalent,
    FRenderExpr,
    FSolveShortcut,
    FMapShortcutStep,
    FEnsureRhsNoVar,
    FTryBaseOneShortcut,
>(
    state: &mut T,
    lhs: ExprId,
    base: ExprId,
    exponent: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    can_branch: bool,
    include_shortcut_item: bool,
    include_base_one_item: bool,
    existing_steps: &mut Vec<S>,
    mut classify_base_flags: FClassifyBaseFlags,
    read_expr: FReadExpr,
    plan_shortcut_action: FPlanShortcutAction,
    bases_equivalent: FBasesEquivalent,
    render_expr: FRenderExpr,
    mut solve_shortcut: FSolveShortcut,
    map_shortcut_step: FMapShortcutStep,
    mut ensure_rhs_without_variable: FEnsureRhsNoVar,
    mut try_base_one_shortcut: FTryBaseOneShortcut,
) -> Result<Option<(SolutionSet, Vec<S>)>, E>
where
    FClassifyBaseFlags: FnMut(&mut T, ExprId) -> (bool, bool),
    FReadExpr: FnMut(&mut T, ExprId) -> Expr,
    FPlanShortcutAction: FnMut(
        &mut T,
        ExprId,
        RelOp,
        bool,
        Option<ExprId>,
        bool,
        bool,
        bool,
    ) -> PowExponentShortcutAction,
    FBasesEquivalent: FnMut(&mut T, ExprId, ExprId) -> bool,
    FRenderExpr: FnMut(&mut T, ExprId) -> String,
    FSolveShortcut: FnMut(&mut T, ExprId, RelOp) -> Result<(SolutionSet, Vec<S>), E>,
    FMapShortcutStep: FnMut(PowExponentShortcutExecutionItem) -> S,
    FEnsureRhsNoVar: FnMut(&mut T, ExprId, &str) -> Result<(), E>,
    FTryBaseOneShortcut: FnMut(
        &mut T,
        ExprId,
        ExprId,
        ExprId,
        RelOp,
        bool,
        &mut Vec<S>,
    ) -> Option<(SolutionSet, Vec<S>)>,
{
    let (base_is_zero, base_is_numeric) = classify_base_flags(state, base);
    let shortcut_action: PowExponentShortcutEngineAction = execute_pow_exponent_shortcut_with_state(
        state,
        exponent,
        base,
        rhs,
        op.clone(),
        var,
        base_is_zero,
        base_is_numeric,
        can_branch,
        read_expr,
        plan_shortcut_action,
        bases_equivalent,
        render_expr,
    );

    if let Some(solved_shortcut) =
        execute_pow_exponent_shortcut_action_pipeline_with_item_and_finalize_with_existing_steps_with(
            shortcut_action,
            include_shortcut_item,
            existing_steps,
            |shortcut_rhs, shortcut_op| solve_shortcut(state, shortcut_rhs, shortcut_op),
            map_shortcut_step,
        )?
    {
        return Ok(Some(solved_shortcut));
    }

    ensure_rhs_without_variable(state, rhs, var)?;

    if let Some(solved_base_one) = try_base_one_shortcut(
        state,
        base,
        lhs,
        rhs,
        op,
        include_base_one_item,
        existing_steps,
    ) {
        return Ok(Some(solved_base_one));
    }

    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::{
        execute_pow_base_isolation_pipeline_with_state,
        execute_pow_exponent_shortcuts_and_guards_with_state,
    };
    use crate::solve_outcome::{PowExponentShortcut, PowExponentShortcutAction};
    use cas_ast::{Expr, RelOp, SolutionSet};

    #[test]
    fn execute_pow_base_isolation_pipeline_with_state_merges_isolated_steps() {
        let mut ctx = cas_ast::Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let mut state = ();

        let solved = execute_pow_base_isolation_pipeline_with_state(
            &mut state,
            x,
            two,
            two,
            RelOp::Eq,
            false,
            vec!["existing".to_string()],
            |_state, base, _exp, rhs, op| {
                crate::solve_outcome::PowBaseIsolationEngineAction::IsolateBase {
                    lhs: base,
                    rhs,
                    op,
                    items: vec![],
                }
            },
            |_state, lhs, _rhs, _op| {
                Ok::<(SolutionSet, Vec<String>), &'static str>((
                    SolutionSet::Empty,
                    vec![format!("solve:{lhs}")],
                ))
            },
            |_item| "unused".to_string(),
        )
        .expect("isolation path should solve");

        assert!(matches!(solved.0, SolutionSet::Empty));
        assert_eq!(solved.1, vec![format!("solve:{x}"), "existing".to_string()]);
    }

    #[derive(Default)]
    struct PreludeHarness {
        context: cas_ast::Context,
        ensured_rhs_count: usize,
        base_one_count: usize,
    }

    #[test]
    fn execute_pow_exponent_shortcuts_and_guards_with_state_returns_shortcut_solution() {
        let mut state = PreludeHarness::default();
        let base = state.context.var("b");
        let exp = state.context.var("x");
        let two = state.context.num(2);
        let rhs = state.context.add(Expr::Pow(base, two));
        let lhs = state.context.add(Expr::Pow(base, exp));
        let mut steps = vec!["existing".to_string()];

        let solved = execute_pow_exponent_shortcuts_and_guards_with_state(
            &mut state,
            lhs,
            base,
            exp,
            rhs,
            RelOp::Eq,
            "x",
            true,
            false,
            false,
            &mut steps,
            |_state, _base| (false, false),
            |state, id| state.context.get(id).clone(),
            |_state,
             _base,
             _op,
             _bases_equal,
             _rhs_pow_base_equal,
             _base_is_zero,
             _base_is_numeric,
             _can_branch| {
                PowExponentShortcutAction::IsolateExponent {
                    shortcut: PowExponentShortcut::EqualPowBases { rhs_exp: two },
                    rhs: two,
                    op: RelOp::Eq,
                }
            },
            |_state, _left, _right| true,
            |_state, expr| format!("#{expr}"),
            |_state, _shortcut_rhs, _shortcut_op| {
                Ok::<(SolutionSet, Vec<String>), &'static str>((
                    SolutionSet::AllReals,
                    vec!["shortcut-solved".to_string()],
                ))
            },
            |_item| "shortcut-step".to_string(),
            |_state, _rhs, _var| -> Result<(), &'static str> {
                Err("RHS guard should not run when shortcut solved")
            },
            |_state, _base, _lhs, _rhs, _op, _include_item, _steps| {
                panic!("base-one shortcut should not run when shortcut solved")
            },
        )
        .expect("shortcut pipeline should solve");

        assert_eq!(state.ensured_rhs_count, 0);
        assert_eq!(state.base_one_count, 0);
        assert!(matches!(solved, Some((SolutionSet::AllReals, _))));
        let (_, solved_steps) = solved.expect("must contain solved output");
        assert_eq!(
            solved_steps,
            vec!["shortcut-solved".to_string(), "existing".to_string()]
        );
    }

    #[test]
    fn execute_pow_exponent_shortcuts_and_guards_with_state_uses_base_one_after_guard() {
        let mut state = PreludeHarness::default();
        let base = state.context.var("b");
        let exp = state.context.var("x");
        let rhs = state.context.num(3);
        let lhs = state.context.add(Expr::Pow(base, exp));
        let mut steps = vec!["existing".to_string()];

        let solved = execute_pow_exponent_shortcuts_and_guards_with_state(
            &mut state,
            lhs,
            base,
            exp,
            rhs,
            RelOp::Eq,
            "x",
            true,
            false,
            false,
            &mut steps,
            |_state, _base| (false, false),
            |state, id| state.context.get(id).clone(),
            |_state,
             _base,
             _op,
             _bases_equal,
             _rhs_pow_base_equal,
             _base_is_zero,
             _base_is_numeric,
             _can_branch| { PowExponentShortcutAction::Continue },
            |_state, _left, _right| false,
            |_state, expr| format!("#{expr}"),
            |_state,
             _shortcut_rhs,
             _shortcut_op|
             -> Result<(SolutionSet, Vec<String>), &'static str> {
                Err("shortcut solve callback should not run on continue route")
            },
            |_item| "shortcut-step".to_string(),
            |state, _rhs, _var| {
                state.ensured_rhs_count += 1;
                Ok::<(), &'static str>(())
            },
            |state, _base, _lhs, _rhs, _op, _include_item, existing_steps| {
                state.base_one_count += 1;
                let mut merged = std::mem::take(existing_steps);
                merged.push("base-one".to_string());
                Some((SolutionSet::AllReals, merged))
            },
        )
        .expect("prelude should return base-one shortcut solution");

        assert_eq!(state.ensured_rhs_count, 1);
        assert_eq!(state.base_one_count, 1);
        let (set, solved_steps) = solved.expect("must contain solved output");
        assert!(matches!(set, SolutionSet::AllReals));
        assert_eq!(
            solved_steps,
            vec!["existing".to_string(), "base-one".to_string()]
        );
    }

    #[test]
    fn execute_pow_exponent_shortcuts_and_guards_with_state_continues_when_no_shortcuts_apply() {
        let mut state = PreludeHarness::default();
        let base = state.context.var("b");
        let exp = state.context.var("x");
        let rhs = state.context.num(3);
        let lhs = state.context.add(Expr::Pow(base, exp));
        let mut steps = vec!["existing".to_string()];

        let solved = execute_pow_exponent_shortcuts_and_guards_with_state(
            &mut state,
            lhs,
            base,
            exp,
            rhs,
            RelOp::Eq,
            "x",
            true,
            false,
            false,
            &mut steps,
            |_state, _base| (false, false),
            |state, id| state.context.get(id).clone(),
            |_state,
             _base,
             _op,
             _bases_equal,
             _rhs_pow_base_equal,
             _base_is_zero,
             _base_is_numeric,
             _can_branch| { PowExponentShortcutAction::Continue },
            |_state, _left, _right| false,
            |_state, expr| format!("#{expr}"),
            |_state,
             _shortcut_rhs,
             _shortcut_op|
             -> Result<(SolutionSet, Vec<String>), &'static str> {
                Err("shortcut solve callback should not run on continue route")
            },
            |_item| "shortcut-step".to_string(),
            |state, _rhs, _var| {
                state.ensured_rhs_count += 1;
                Ok::<(), &'static str>(())
            },
            |state, _base, _lhs, _rhs, _op, _include_item, _existing_steps| {
                state.base_one_count += 1;
                None
            },
        )
        .expect("prelude should continue");

        assert_eq!(state.ensured_rhs_count, 1);
        assert_eq!(state.base_one_count, 1);
        assert!(solved.is_none());
        assert_eq!(steps, vec!["existing".to_string()]);
    }
}
