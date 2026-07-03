use cas_ast::{Case, ConditionPredicate, Equation, ExprId, SolutionSet};

/// Result of verifying a single solution.
#[derive(Debug, Clone)]
pub enum VerifyStatus {
    /// Solution verified: equation simplifies to 0 after substitution.
    Verified,
    /// Solution could not be verified (residual remains).
    Unverifiable {
        /// The residual expression that didn't simplify to 0.
        residual: ExprId,
        /// Human-readable reason.
        reason: String,
        /// Optional concrete probe showing a residual counterexample.
        counterexample_hint: Option<String>,
    },
    /// Solution type not checkable (intervals, AllReals, residual).
    NotCheckable {
        /// Reason why verification is not possible.
        reason: &'static str,
    },
}

/// Summary of verification for a solution set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifySummary {
    /// All solutions verified.
    AllVerified,
    /// Some solutions verified, some not.
    PartiallyVerified,
    /// A non-discrete solution is justified symbolically by explicit guards.
    VerifiedUnderGuard,
    /// Verification would require numeric sampling for non-discrete solutions.
    NeedsSampling,
    /// No solutions verified.
    NoneVerified,
    /// Solution type not checkable.
    NotCheckable,
    /// Empty solution set (trivially verified).
    Empty,
}

/// Result of verifying an entire solution set.
#[derive(Debug, Clone)]
pub struct VerifyResult {
    /// Status for each discrete solution (if applicable).
    pub solutions: Vec<(ExprId, VerifyStatus)>,
    /// Overall summary.
    pub summary: VerifySummary,
    /// Guard under which verification was performed (for Conditional).
    pub guard_description: Option<String>,
}

/// Stateful variant of [`verify_solution_set_with`].
///
/// Useful for call-sites where per-candidate verification shares mutable state
/// (e.g., one simplifier instance).
pub(crate) fn verify_solution_set_with_state<T, F>(
    state: &mut T,
    solutions: &SolutionSet,
    verify_discrete: &mut F,
) -> VerifyResult
where
    F: FnMut(&mut T, ExprId) -> VerifyStatus,
{
    match solutions {
        SolutionSet::Empty => VerifyResult {
            solutions: vec![],
            summary: VerifySummary::Empty,
            guard_description: None,
        },

        SolutionSet::Discrete(sols) => {
            let mut results = Vec::with_capacity(sols.len());
            let mut verified_count = 0;

            for &sol in sols {
                let status = verify_discrete(state, sol);
                if matches!(status, VerifyStatus::Verified) {
                    verified_count += 1;
                }
                results.push((sol, status));
            }

            VerifyResult {
                summary: discrete_summary(results.len(), verified_count),
                solutions: results,
                guard_description: None,
            }
        }

        SolutionSet::AllReals => VerifyResult {
            solutions: vec![],
            summary: VerifySummary::NotCheckable,
            guard_description: Some("not checkable (infinite set: all reals)".to_string()),
        },

        SolutionSet::Continuous(_interval) => VerifyResult {
            solutions: vec![],
            summary: VerifySummary::NeedsSampling,
            guard_description: Some(
                "verification requires numeric sampling (continuous interval)".to_string(),
            ),
        },

        SolutionSet::Union(_intervals) => VerifyResult {
            solutions: vec![],
            summary: VerifySummary::NeedsSampling,
            guard_description: Some(
                "verification requires numeric sampling (union of intervals)".to_string(),
            ),
        },

        SolutionSet::Residual(_expr) => VerifyResult {
            solutions: vec![],
            summary: VerifySummary::NotCheckable,
            guard_description: Some("unverifiable (residual expression)".to_string()),
        },

        SolutionSet::Periodic { .. } => VerifyResult {
            solutions: vec![],
            summary: VerifySummary::NotCheckable,
            guard_description: Some("not checkable (infinite periodic family)".to_string()),
        },

        SolutionSet::PeriodicIntervalUnion { .. } => VerifyResult {
            solutions: vec![],
            summary: VerifySummary::NotCheckable,
            guard_description: Some(
                "not checkable (infinite periodic interval family)".to_string(),
            ),
        },

        SolutionSet::Conditional(cases) => {
            if let Some(description) = classify_guard_verified_conditional(cases) {
                return VerifyResult {
                    solutions: vec![],
                    summary: VerifySummary::VerifiedUnderGuard,
                    guard_description: Some(description),
                };
            }

            let mut all_results = Vec::new();
            let mut has_verified = false;
            let mut has_needs_sampling = false;
            let mut has_not_checkable = false;

            for case in cases {
                let case_result =
                    verify_solution_set_with_state(state, &case.then.solutions, verify_discrete);

                match case_result.summary {
                    VerifySummary::AllVerified | VerifySummary::PartiallyVerified => {
                        has_verified = true;
                    }
                    VerifySummary::VerifiedUnderGuard => {
                        has_verified = true;
                    }
                    VerifySummary::NeedsSampling => {
                        has_needs_sampling = true;
                    }
                    VerifySummary::NotCheckable => {
                        has_not_checkable = true;
                    }
                    _ => {}
                }

                all_results.extend(case_result.solutions);
            }

            VerifyResult {
                solutions: all_results,
                summary: conditional_summary(has_verified, has_needs_sampling, has_not_checkable),
                guard_description: conditional_guard_description(
                    has_verified,
                    has_needs_sampling,
                    has_not_checkable,
                ),
            }
        }
    }
}

/// Stateful variant of
/// [`verify_substituted_residual_with_strict_fold_and_generic_fallback`].
///
/// This form lets callers thread one mutable state object across all hooks
/// without interior mutability wrappers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_substituted_residual_with_strict_fold_and_generic_fallback_with_state<
    T,
    FSimplifyStrict,
    FSimplifyGeneric,
    FContainsVariable,
    FFoldNumericIslands,
    FIsZero,
    FRecordAttempted,
    FRecordChanged,
    FRecordVerified,
>(
    state: &mut T,
    diff: ExprId,
    mut simplify_strict: FSimplifyStrict,
    mut simplify_generic: FSimplifyGeneric,
    mut contains_variable: FContainsVariable,
    mut fold_numeric_islands: FFoldNumericIslands,
    mut is_zero: FIsZero,
    mut record_attempted: FRecordAttempted,
    mut record_changed: FRecordChanged,
    mut record_verified: FRecordVerified,
) -> (bool, ExprId)
where
    FSimplifyStrict: FnMut(&mut T, ExprId) -> ExprId,
    FSimplifyGeneric: FnMut(&mut T, ExprId) -> ExprId,
    FContainsVariable: FnMut(&mut T, ExprId) -> bool,
    FFoldNumericIslands: FnMut(&mut T, ExprId) -> ExprId,
    FIsZero: FnMut(&mut T, ExprId) -> bool,
    FRecordAttempted: FnMut(&mut T),
    FRecordChanged: FnMut(&mut T),
    FRecordVerified: FnMut(&mut T),
{
    let strict_result = simplify_strict(state, diff);
    if is_zero(state, strict_result) {
        return (true, strict_result);
    }

    if contains_variable(state, strict_result) {
        record_attempted(state);
        let folded = fold_numeric_islands(state, strict_result);
        if folded != strict_result {
            record_changed(state);
            let folded_result = simplify_strict(state, folded);
            if is_zero(state, folded_result) {
                record_verified(state);
                return (true, strict_result);
            }
        }
    }

    if !contains_variable(state, strict_result) {
        let generic_result = simplify_generic(state, diff);
        if is_zero(state, generic_result) {
            return (true, strict_result);
        }
    }

    (false, strict_result)
}

/// Verify one candidate solution from `(equation, var, solution)` by:
/// 1) substituting into `lhs - rhs`,
/// 2) running strict/fold/generic residual verification,
/// 3) materializing `VerifyStatus` with rendered residual on failure.
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_solution_with_strict_fold_and_generic_fallback_with_state<
    T,
    FSubstituteDiff,
    FSimplifyStrict,
    FSimplifyGeneric,
    FContainsVariable,
    FFoldNumericIslands,
    FIsZero,
    FRecordAttempted,
    FRecordChanged,
    FRecordVerified,
    FRenderExpr,
>(
    state: &mut T,
    equation: &Equation,
    var: &str,
    solution: ExprId,
    mut substitute_diff: FSubstituteDiff,
    simplify_strict: FSimplifyStrict,
    simplify_generic: FSimplifyGeneric,
    contains_variable: FContainsVariable,
    fold_numeric_islands: FFoldNumericIslands,
    is_zero: FIsZero,
    record_attempted: FRecordAttempted,
    record_changed: FRecordChanged,
    record_verified: FRecordVerified,
    mut render_expr: FRenderExpr,
) -> VerifyStatus
where
    FSubstituteDiff: FnMut(&mut T, &Equation, &str, ExprId) -> ExprId,
    FSimplifyStrict: FnMut(&mut T, ExprId) -> ExprId,
    FSimplifyGeneric: FnMut(&mut T, ExprId) -> ExprId,
    FContainsVariable: FnMut(&mut T, ExprId) -> bool,
    FFoldNumericIslands: FnMut(&mut T, ExprId) -> ExprId,
    FIsZero: FnMut(&mut T, ExprId) -> bool,
    FRecordAttempted: FnMut(&mut T),
    FRecordChanged: FnMut(&mut T),
    FRecordVerified: FnMut(&mut T),
    FRenderExpr: FnMut(&mut T, ExprId) -> String,
{
    let diff = substitute_diff(state, equation, var, solution);
    let (verified, strict_result) =
        verify_substituted_residual_with_strict_fold_and_generic_fallback_with_state(
            state,
            diff,
            simplify_strict,
            simplify_generic,
            contains_variable,
            fold_numeric_islands,
            is_zero,
            record_attempted,
            record_changed,
            record_verified,
        );

    if verified {
        VerifyStatus::Verified
    } else {
        VerifyStatus::Unverifiable {
            residual: strict_result,
            reason: format!("residual: {}", render_expr(state, strict_result)),
            counterexample_hint: None,
        }
    }
}

/// Same as [`verify_solution_with_strict_fold_and_generic_fallback_with_state`],
/// but wires verification telemetry to `verify_stats` by default.
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_solution_with_strict_fold_and_generic_fallback_with_default_stats_and_state<
    T,
    FSubstituteDiff,
    FSimplifyStrict,
    FSimplifyGeneric,
    FContainsVariable,
    FFoldNumericIslands,
    FIsZero,
    FRenderExpr,
>(
    state: &mut T,
    equation: &Equation,
    var: &str,
    solution: ExprId,
    substitute_diff: FSubstituteDiff,
    simplify_strict: FSimplifyStrict,
    simplify_generic: FSimplifyGeneric,
    contains_variable: FContainsVariable,
    fold_numeric_islands: FFoldNumericIslands,
    is_zero: FIsZero,
    render_expr: FRenderExpr,
) -> VerifyStatus
where
    FSubstituteDiff: FnMut(&mut T, &Equation, &str, ExprId) -> ExprId,
    FSimplifyStrict: FnMut(&mut T, ExprId) -> ExprId,
    FSimplifyGeneric: FnMut(&mut T, ExprId) -> ExprId,
    FContainsVariable: FnMut(&mut T, ExprId) -> bool,
    FFoldNumericIslands: FnMut(&mut T, ExprId) -> ExprId,
    FIsZero: FnMut(&mut T, ExprId) -> bool,
    FRenderExpr: FnMut(&mut T, ExprId) -> String,
{
    verify_solution_with_strict_fold_and_generic_fallback_with_state(
        state,
        equation,
        var,
        solution,
        substitute_diff,
        simplify_strict,
        simplify_generic,
        contains_variable,
        fold_numeric_islands,
        is_zero,
        |_state| crate::verify_stats::record_attempted(),
        |_state| crate::verify_stats::record_changed(),
        |_state| crate::verify_stats::record_verified(),
        render_expr,
    )
}

fn classify_guard_verified_conditional(cases: &[Case]) -> Option<String> {
    if cases.is_empty() {
        return None;
    }

    let mut guarded_branch_count = 0usize;

    for case in cases {
        if case.then.residual.is_some() {
            return None;
        }

        match &case.then.solutions {
            SolutionSet::AllReals | SolutionSet::Continuous(_) | SolutionSet::Union(_) => {
                if case.when.is_empty() || !simple_guard_set(case.when.predicates()) {
                    return None;
                }
                guarded_branch_count += 1;
            }
            SolutionSet::Empty => {
                if !case.when.is_empty() && !simple_guard_set(case.when.predicates()) {
                    return None;
                }
            }
            SolutionSet::Discrete(_)
            | SolutionSet::Residual(_)
            | SolutionSet::Conditional(_)
            | SolutionSet::Periodic { .. }
            | SolutionSet::PeriodicIntervalUnion { .. } => {
                return None;
            }
        }
    }

    if guarded_branch_count == 0 {
        return None;
    }

    Some(format!(
        "verified symbolically under guard ({} guarded non-discrete branch{})",
        guarded_branch_count,
        if guarded_branch_count == 1 { "" } else { "es" }
    ))
}

fn simple_guard_set(predicates: &[ConditionPredicate]) -> bool {
    !predicates.is_empty()
        && predicates.iter().all(|pred| {
            matches!(
                pred,
                ConditionPredicate::NonZero(_)
                    | ConditionPredicate::Positive(_)
                    | ConditionPredicate::NonNegative(_)
            )
        })
}

/// Compute summary for discrete verification outcomes.
pub(crate) fn discrete_summary(total: usize, verified_count: usize) -> VerifySummary {
    if total == 0 {
        VerifySummary::Empty
    } else if verified_count == total {
        VerifySummary::AllVerified
    } else if verified_count > 0 {
        VerifySummary::PartiallyVerified
    } else {
        VerifySummary::NoneVerified
    }
}

/// Compute summary for aggregated conditional verification outcomes.
pub(crate) fn conditional_summary(
    has_verified: bool,
    has_needs_sampling: bool,
    has_not_checkable: bool,
) -> VerifySummary {
    if has_verified && !has_needs_sampling && !has_not_checkable {
        VerifySummary::AllVerified
    } else if has_verified {
        VerifySummary::PartiallyVerified
    } else if has_needs_sampling {
        VerifySummary::NeedsSampling
    } else if has_not_checkable {
        VerifySummary::NotCheckable
    } else {
        VerifySummary::NoneVerified
    }
}

fn conditional_guard_description(
    has_verified: bool,
    has_needs_sampling: bool,
    has_not_checkable: bool,
) -> Option<String> {
    if has_verified && has_needs_sampling && has_not_checkable {
        Some(
            "some non-discrete branches require numeric sampling and others remain not checkable"
                .to_string(),
        )
    } else if has_verified && has_needs_sampling {
        Some("some non-discrete branches require numeric sampling".to_string())
    } else if has_verified && has_not_checkable {
        Some("some non-discrete branches remain not checkable".to_string())
    } else if has_needs_sampling && has_not_checkable {
        Some("verification requires numeric sampling for some non-discrete branches".to_string())
    } else if has_needs_sampling {
        Some("verification requires numeric sampling for non-discrete branches".to_string())
    } else if has_not_checkable {
        Some("solution type not checkable in non-discrete branches".to_string())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Context;
    use cas_ast::RelOp;

    #[test]
    fn test_discrete_summary() {
        assert_eq!(discrete_summary(0, 0), VerifySummary::Empty);
        assert_eq!(discrete_summary(3, 3), VerifySummary::AllVerified);
        assert_eq!(discrete_summary(3, 1), VerifySummary::PartiallyVerified);
        assert_eq!(discrete_summary(3, 0), VerifySummary::NoneVerified);
    }

    #[test]
    fn test_conditional_summary() {
        assert_eq!(
            conditional_summary(true, false, false),
            VerifySummary::AllVerified
        );
        assert_eq!(
            conditional_summary(true, false, true),
            VerifySummary::PartiallyVerified
        );
        assert_eq!(
            conditional_summary(true, true, false),
            VerifySummary::PartiallyVerified
        );
        assert_eq!(
            conditional_summary(false, true, false),
            VerifySummary::NeedsSampling
        );
        assert_eq!(
            conditional_summary(false, false, true),
            VerifySummary::NotCheckable
        );
        assert_eq!(
            conditional_summary(false, false, false),
            VerifySummary::NoneVerified
        );
    }

    #[test]
    fn conditional_guard_description_keeps_non_discrete_note_when_discrete_branches_verify() {
        assert_eq!(
            conditional_guard_description(true, true, true).as_deref(),
            Some(
                "some non-discrete branches require numeric sampling and others remain not checkable"
            )
        );
        assert_eq!(
            conditional_guard_description(true, true, false).as_deref(),
            Some("some non-discrete branches require numeric sampling")
        );
        assert_eq!(
            conditional_guard_description(true, false, true).as_deref(),
            Some("some non-discrete branches remain not checkable")
        );
    }

    #[test]
    fn test_verify_solution_set_with_state_discrete() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let two = ctx.num(2);
        let set = SolutionSet::Discrete(vec![one, two]);

        let mut calls = 0usize;
        let mut verify = |counter: &mut usize, _id: ExprId| {
            *counter += 1;
            VerifyStatus::Verified
        };

        let result = verify_solution_set_with_state(&mut calls, &set, &mut verify);
        assert_eq!(calls, 2);
        assert_eq!(result.summary, VerifySummary::AllVerified);
        assert_eq!(result.solutions.len(), 2);
    }

    #[test]
    fn verify_solution_with_default_stats_with_state_reports_verified() {
        let mut ctx = Context::new();
        let lhs = ctx.var("x");
        let rhs = ctx.num(0);
        let eq = Equation {
            lhs,
            rhs,
            op: RelOp::Eq,
        };
        let zero = ctx.num(0);
        let mut substitute_calls = 0usize;

        let status =
            verify_solution_with_strict_fold_and_generic_fallback_with_default_stats_and_state(
                &mut substitute_calls,
                &eq,
                "x",
                lhs,
                |calls, _eq, _var, _solution| {
                    *calls += 1;
                    zero
                },
                |_calls, expr| expr,
                |_calls, expr| expr,
                |_calls, _expr| false,
                |_calls, expr| expr,
                |_calls, expr| expr == zero,
                |_calls, _expr| "0".to_string(),
            );

        assert!(matches!(status, VerifyStatus::Verified));
        assert_eq!(substitute_calls, 1);
    }

    #[test]
    fn verify_solution_with_default_stats_with_state_reports_unverifiable() {
        let mut ctx = Context::new();
        let lhs = ctx.var("x");
        let rhs = ctx.num(0);
        let eq = Equation {
            lhs,
            rhs,
            op: RelOp::Eq,
        };
        let one = ctx.num(1);
        let zero = ctx.num(0);
        let mut substitute_calls = 0usize;

        let status =
            verify_solution_with_strict_fold_and_generic_fallback_with_default_stats_and_state(
                &mut substitute_calls,
                &eq,
                "x",
                lhs,
                |calls, _eq, _var, _solution| {
                    *calls += 1;
                    one
                },
                |_calls, expr| expr,
                |_calls, expr| expr,
                |_calls, _expr| false,
                |_calls, expr| expr,
                |_calls, expr| expr == zero,
                |_calls, expr| format!("expr:{expr:?}"),
            );

        assert!(matches!(
            status,
            VerifyStatus::Unverifiable {
                residual,
                reason,
                counterexample_hint
            } if residual == one
                && reason.starts_with("residual: expr:")
                && counterexample_hint.is_none()
        ));
        assert_eq!(substitute_calls, 1);
    }

    #[test]
    fn verify_substituted_residual_with_strict_fold_and_generic_fallback_with_state_verifies_after_fold(
    ) {
        let mut ctx = Context::new();
        let diff = ctx.num(103);
        let strict = ctx.var("x");
        let folded = ctx.num(8);
        let zero = ctx.num(0);

        #[derive(Default)]
        struct VerifyState {
            strict_calls: usize,
            generic_calls: usize,
            attempted_calls: usize,
            changed_calls: usize,
            verified_calls: usize,
        }
        let mut state = VerifyState::default();

        let (verified, strict_residual) =
            verify_substituted_residual_with_strict_fold_and_generic_fallback_with_state(
                &mut state,
                diff,
                |hooks, expr| {
                    hooks.strict_calls += 1;
                    if expr == diff {
                        strict
                    } else if expr == folded {
                        zero
                    } else {
                        expr
                    }
                },
                |hooks, expr| {
                    hooks.generic_calls += 1;
                    expr
                },
                |_hooks, expr| expr == strict,
                |_hooks, expr| if expr == strict { folded } else { expr },
                |_hooks, expr| expr == zero,
                |hooks| hooks.attempted_calls += 1,
                |hooks| hooks.changed_calls += 1,
                |hooks| hooks.verified_calls += 1,
            );

        assert!(verified);
        assert_eq!(strict_residual, strict);
        assert_eq!(state.strict_calls, 2);
        assert_eq!(state.generic_calls, 0);
        assert_eq!(state.attempted_calls, 1);
        assert_eq!(state.changed_calls, 1);
        assert_eq!(state.verified_calls, 1);
    }

    #[test]
    fn verify_solution_with_strict_fold_and_generic_fallback_with_state_returns_verified() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let equation = Equation {
            lhs: x,
            rhs: one,
            op: RelOp::Eq,
        };

        let status = verify_solution_with_strict_fold_and_generic_fallback_with_state(
            &mut ctx,
            &equation,
            "x",
            one,
            |_ctx, _equation, _var, _solution| one,
            |_ctx, expr| expr,
            |_ctx, expr| expr,
            |_ctx, _expr| false,
            |_ctx, expr| expr,
            |_ctx, expr| expr == one,
            |_ctx| {},
            |_ctx| {},
            |_ctx| {},
            |_ctx, expr| format!("expr#{expr:?}"),
        );

        assert!(matches!(status, VerifyStatus::Verified));
    }

    #[test]
    fn verify_solution_with_strict_fold_and_generic_fallback_with_state_returns_unverifiable() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let two = ctx.num(2);
        let equation = Equation {
            lhs: x,
            rhs: one,
            op: RelOp::Eq,
        };

        let status = verify_solution_with_strict_fold_and_generic_fallback_with_state(
            &mut ctx,
            &equation,
            "x",
            one,
            |_ctx, _equation, _var, _solution| two,
            |_ctx, expr| expr,
            |_ctx, expr| expr,
            |_ctx, _expr| false,
            |_ctx, expr| expr,
            |_ctx, _expr| false,
            |_ctx| {},
            |_ctx| {},
            |_ctx| {},
            |_ctx, expr| format!("expr#{expr:?}"),
        );

        match status {
            VerifyStatus::Unverifiable {
                residual,
                reason,
                counterexample_hint,
            } => {
                assert_eq!(residual, two);
                assert!(reason.contains("residual: expr#"));
                assert!(counterexample_hint.is_none());
            }
            other => panic!("expected unverifiable, got {other:?}"),
        }
    }
}
