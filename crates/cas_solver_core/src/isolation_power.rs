//! Power-isolation orchestration helpers shared by engine-side solver.
//!
//! These wrappers keep equation-rewrite orchestration in `cas_solver_core`
//! while callers provide stateful hooks for simplification and recursion.

use cas_ast::{Context, Equation, Expr, ExprId, RelOp, SolutionSet};

use crate::log_domain::{DomainModeKind, LogAssumption, LogSolveDecision};

use crate::solve_outcome::{
    execute_and_resolve_pow_exponent_log_decision_pipeline_with_existing_steps_mut_with_unsupported_execution,
    execute_pow_base_isolation_pipeline_with_item_and_merge_with_existing_steps,
    execute_pow_exponent_log_isolation_pipeline_with_plan_and_merge_with_existing_steps_with,
    execute_pow_exponent_shortcut_action_pipeline_with_item_and_finalize_with_existing_steps_with,
    execute_pow_exponent_shortcut_with_state, LogBlockedHintRecord, PowBaseIsolationEngineAction,
    PowExponentLogIsolationExecutionItem, PowExponentLogIsolationRewritePlan,
    PowExponentLogUnsupportedExecution, PowExponentShortcutAction, PowExponentShortcutEngineAction,
    PowExponentShortcutExecutionItem, TermIsolationExecutionItem,
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

/// Execute exponent-side logarithmic decision resolution and, when needed,
/// continue with logarithmic rewrite isolation.
#[allow(clippy::too_many_arguments)]
pub fn execute_pow_exponent_log_decision_then_rewrite_with_state<
    T,
    S,
    E,
    FPlanUnsupported,
    FCloneContext,
    FMapTermStep,
    FVisitAssumption,
    FTryGuardedSolve,
    FRegisterBlockedHint,
    FMapUnsupportedErr,
    FPlanRewrite,
    FSolveRewrite,
    FMapLogStep,
>(
    state: &mut T,
    lhs: ExprId,
    base: ExprId,
    exponent: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    decision: &LogSolveDecision,
    mode: DomainModeKind,
    wildcard_scope: bool,
    can_branch: bool,
    include_terminal_items: bool,
    include_unsupported_items: bool,
    include_log_item: bool,
    mut existing_steps: Vec<S>,
    mut plan_unsupported_execution: FPlanUnsupported,
    mut clone_context: FCloneContext,
    map_term_item_to_step: FMapTermStep,
    mut visit_assumption: FVisitAssumption,
    mut try_guarded_solve: FTryGuardedSolve,
    mut register_blocked_hint: FRegisterBlockedHint,
    mut map_unsupported_err: FMapUnsupportedErr,
    mut plan_log_rewrite: FPlanRewrite,
    mut solve_rewrite: FSolveRewrite,
    map_log_item_to_step: FMapLogStep,
) -> Result<(SolutionSet, Vec<S>), E>
where
    FPlanUnsupported: FnMut(
        &mut T,
        &LogSolveDecision,
        bool,
        ExprId,
        ExprId,
        &str,
        ExprId,
        ExprId,
        ExprId,
        RelOp,
        Equation,
    ) -> Option<PowExponentLogUnsupportedExecution>,
    FCloneContext: FnMut(&mut T) -> Context,
    FMapTermStep: FnMut(TermIsolationExecutionItem) -> S,
    FVisitAssumption: FnMut(&Context, LogAssumption),
    FTryGuardedSolve: FnMut(&mut T, &Equation) -> Option<SolutionSet>,
    FRegisterBlockedHint: FnMut(&Context, LogBlockedHintRecord),
    FMapUnsupportedErr: FnMut(&'static str) -> E,
    FPlanRewrite:
        FnMut(&mut T, ExprId, ExprId, ExprId, RelOp) -> PowExponentLogIsolationRewritePlan,
    FSolveRewrite: FnMut(&mut T, &Equation) -> Result<(SolutionSet, Vec<S>), E>,
    FMapLogStep: FnMut(PowExponentLogIsolationExecutionItem) -> S,
{
    let source_equation = Equation {
        lhs,
        rhs,
        op: op.clone(),
    };
    let unsupported_execution = plan_unsupported_execution(
        state,
        decision,
        can_branch,
        lhs,
        rhs,
        var,
        exponent,
        base,
        rhs,
        op.clone(),
        source_equation.clone(),
    );

    let mut decision_ctx = clone_context(state);
    let resolved_decision =
        execute_and_resolve_pow_exponent_log_decision_pipeline_with_existing_steps_mut_with_unsupported_execution(
            &mut decision_ctx,
            decision,
            mode,
            wildcard_scope,
            lhs,
            rhs,
            var,
            source_equation,
            " (residual)",
            include_terminal_items,
            &mut existing_steps,
            map_term_item_to_step,
            |core_ctx, assumption| visit_assumption(core_ctx, assumption),
            include_unsupported_items,
            unsupported_execution,
            |equation| try_guarded_solve(state, equation),
            |core_ctx, hint| register_blocked_hint(core_ctx, hint),
        );
    match resolved_decision {
        Ok(Some((solution_set, merged_steps))) => return Ok((solution_set, merged_steps)),
        Ok(None) => {}
        Err(message) => return Err(map_unsupported_err(message)),
    }

    let rewrite = plan_log_rewrite(state, exponent, base, rhs, op);
    execute_pow_exponent_log_isolation_pipeline_with_plan_and_merge_with_existing_steps_with(
        include_log_item,
        existing_steps,
        rewrite,
        |equation| solve_rewrite(state, equation),
        map_log_item_to_step,
    )
}

#[cfg(test)]
mod tests {
    use super::{
        execute_pow_base_isolation_pipeline_with_state,
        execute_pow_exponent_log_decision_then_rewrite_with_state,
        execute_pow_exponent_shortcuts_and_guards_with_state,
    };
    use crate::log_domain::{DomainModeKind, LogSolveDecision};
    use crate::solve_outcome::{
        PowExponentLogIsolationRewritePlan, PowExponentShortcut, PowExponentShortcutAction,
    };
    use cas_ast::{Equation, Expr, RelOp, SolutionSet};

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

    #[derive(Default)]
    struct DecisionHarness {
        context: cas_ast::Context,
        rewrite_solve_count: usize,
    }

    #[test]
    fn execute_pow_exponent_log_decision_then_rewrite_with_state_runs_rewrite_on_continue() {
        let mut state = DecisionHarness::default();
        let base = state.context.var("b");
        let exponent = state.context.var("x");
        let rhs = state.context.num(8);
        let lhs = state.context.add(Expr::Pow(base, exponent));
        let mut visit_assumption_count = 0usize;
        let mut blocked_hint_count = 0usize;

        let solved = execute_pow_exponent_log_decision_then_rewrite_with_state(
            &mut state,
            lhs,
            base,
            exponent,
            rhs,
            RelOp::Eq,
            "x",
            &LogSolveDecision::Ok,
            DomainModeKind::Assume,
            false,
            true,
            false,
            false,
            false,
            vec!["existing".to_string()],
            |_state,
             _decision,
             _can_branch,
             _lhs,
             _rhs,
             _var,
             _exponent,
             _base,
             _raw_rhs,
             _op,
             _equation| None,
            |state| state.context.clone(),
            |_term_item| "unused-term-step".to_string(),
            |_ctx, _assumption| {
                visit_assumption_count += 1;
            },
            |_state, _equation| None,
            |_ctx, _hint| {
                blocked_hint_count += 1;
            },
            |_message| "unexpected-needs-complex",
            |_state, exp, base, rhs, op| PowExponentLogIsolationRewritePlan {
                equation: Equation {
                    lhs: exp,
                    rhs: base, // payload shape doesn't matter in this harness.
                    op,
                },
                items: {
                    let _ = rhs;
                    vec![]
                },
            },
            |state, _equation| {
                state.rewrite_solve_count += 1;
                Ok::<(SolutionSet, Vec<String>), &'static str>((
                    SolutionSet::AllReals,
                    vec!["rewrite-solved".to_string()],
                ))
            },
            |_log_item| "unused-log-step".to_string(),
        )
        .expect("continue route should execute rewrite");

        assert_eq!(visit_assumption_count, 0);
        assert_eq!(blocked_hint_count, 0);
        assert_eq!(state.rewrite_solve_count, 1);
        assert!(matches!(solved.0, SolutionSet::AllReals));
        assert_eq!(
            solved.1,
            vec!["rewrite-solved".to_string(), "existing".to_string()]
        );
    }

    #[test]
    fn execute_pow_exponent_log_decision_then_rewrite_with_state_maps_needs_complex_error() {
        let mut state = DecisionHarness::default();
        let base = state.context.var("b");
        let exponent = state.context.var("x");
        let rhs = state.context.num(8);
        let lhs = state.context.add(Expr::Pow(base, exponent));

        let err = execute_pow_exponent_log_decision_then_rewrite_with_state(
            &mut state,
            lhs,
            base,
            exponent,
            rhs,
            RelOp::Eq,
            "x",
            &LogSolveDecision::NeedsComplex("need complex domain"),
            DomainModeKind::Strict,
            false,
            true,
            false,
            false,
            false,
            vec!["existing".to_string()],
            |_state,
             _decision,
             _can_branch,
             _lhs,
             _rhs,
             _var,
             _exponent,
             _base,
             _raw_rhs,
             _op,
             _equation| None,
            |state| state.context.clone(),
            |_term_item| "unused-term-step".to_string(),
            |_ctx, _assumption| {},
            |_state, _equation| None,
            |_ctx, _hint| {},
            |message| format!("mapped:{message}"),
            |_state, exp, base, _rhs, op| PowExponentLogIsolationRewritePlan {
                equation: Equation {
                    lhs: exp,
                    rhs: base,
                    op,
                },
                items: vec![],
            },
            |_state, _equation| -> Result<(SolutionSet, Vec<String>), String> {
                Err("rewrite should not run on needs-complex".to_string())
            },
            |_log_item| "unused-log-step".to_string(),
        )
        .expect_err("needs-complex decision must map to caller error");

        assert_eq!(err, "mapped:need complex domain".to_string());
        assert_eq!(state.rewrite_solve_count, 0);
    }
}
