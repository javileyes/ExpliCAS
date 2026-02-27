use cas_ast::{ExprId, SolutionSet};

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

/// Verify a solution set using a callback for each discrete candidate.
///
/// The callback is invoked only for `SolutionSet::Discrete` entries.
/// Non-discrete sets are mapped to `NotCheckable` summaries.
pub fn verify_solution_set_with<F>(solutions: &SolutionSet, verify_discrete: &mut F) -> VerifyResult
where
    F: FnMut(ExprId) -> VerifyStatus,
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
                let status = verify_discrete(sol);
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
            summary: VerifySummary::NotCheckable,
            guard_description: Some("not checkable (continuous interval)".to_string()),
        },

        SolutionSet::Union(_intervals) => VerifyResult {
            solutions: vec![],
            summary: VerifySummary::NotCheckable,
            guard_description: Some("not checkable (union of intervals)".to_string()),
        },

        SolutionSet::Residual(_expr) => VerifyResult {
            solutions: vec![],
            summary: VerifySummary::NotCheckable,
            guard_description: Some("unverifiable (residual expression)".to_string()),
        },

        SolutionSet::Conditional(cases) => {
            let mut all_results = Vec::new();
            let mut has_verified = false;
            let mut has_not_checkable = false;

            for case in cases {
                let case_result = verify_solution_set_with(&case.then.solutions, verify_discrete);

                match case_result.summary {
                    VerifySummary::AllVerified | VerifySummary::PartiallyVerified => {
                        has_verified = true;
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
                summary: conditional_summary(has_verified, has_not_checkable),
                guard_description: None,
            }
        }
    }
}

/// Verify one substituted residual with the engine's 2-phase flow:
/// 1) strict simplify, 2) variable-only island fold + strict retry,
/// 3) generic simplify fallback only for variable-free strict residuals.
///
/// Returns `(verified, strict_residual)` where `strict_residual` is the phase-1
/// strict simplification output and is intended for error reporting when
/// verification fails.
#[allow(clippy::too_many_arguments)]
pub fn verify_substituted_residual_with_strict_fold_and_generic_fallback<
    FSimplifyStrict,
    FSimplifyGeneric,
    FContainsVariable,
    FFoldNumericIslands,
    FIsZero,
    FRecordAttempted,
    FRecordChanged,
    FRecordVerified,
>(
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
    FSimplifyStrict: FnMut(ExprId) -> ExprId,
    FSimplifyGeneric: FnMut(ExprId) -> ExprId,
    FContainsVariable: FnMut(ExprId) -> bool,
    FFoldNumericIslands: FnMut(ExprId) -> ExprId,
    FIsZero: FnMut(ExprId) -> bool,
    FRecordAttempted: FnMut(),
    FRecordChanged: FnMut(),
    FRecordVerified: FnMut(),
{
    let strict_result = simplify_strict(diff);
    if is_zero(strict_result) {
        return (true, strict_result);
    }

    if contains_variable(strict_result) {
        record_attempted();
        let folded = fold_numeric_islands(strict_result);
        if folded != strict_result {
            record_changed();
            let folded_result = simplify_strict(folded);
            if is_zero(folded_result) {
                record_verified();
                return (true, strict_result);
            }
        }
    }

    if !contains_variable(strict_result) {
        let generic_result = simplify_generic(diff);
        if is_zero(generic_result) {
            return (true, strict_result);
        }
    }

    (false, strict_result)
}

/// Stateful variant of
/// [`verify_substituted_residual_with_strict_fold_and_generic_fallback`].
///
/// This form lets callers thread one mutable state object across all hooks
/// without interior mutability wrappers.
#[allow(clippy::too_many_arguments)]
pub fn verify_substituted_residual_with_strict_fold_and_generic_fallback_with_state<
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

/// Compute summary for discrete verification outcomes.
pub fn discrete_summary(total: usize, verified_count: usize) -> VerifySummary {
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
pub fn conditional_summary(has_verified: bool, has_not_checkable: bool) -> VerifySummary {
    if has_verified && !has_not_checkable {
        VerifySummary::AllVerified
    } else if has_verified {
        VerifySummary::PartiallyVerified
    } else if has_not_checkable {
        VerifySummary::NotCheckable
    } else {
        VerifySummary::NoneVerified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Context;
    use std::cell::Cell;

    #[test]
    fn test_discrete_summary() {
        assert_eq!(discrete_summary(0, 0), VerifySummary::Empty);
        assert_eq!(discrete_summary(3, 3), VerifySummary::AllVerified);
        assert_eq!(discrete_summary(3, 1), VerifySummary::PartiallyVerified);
        assert_eq!(discrete_summary(3, 0), VerifySummary::NoneVerified);
    }

    #[test]
    fn test_conditional_summary() {
        assert_eq!(conditional_summary(true, false), VerifySummary::AllVerified);
        assert_eq!(
            conditional_summary(true, true),
            VerifySummary::PartiallyVerified
        );
        assert_eq!(
            conditional_summary(false, true),
            VerifySummary::NotCheckable
        );
        assert_eq!(
            conditional_summary(false, false),
            VerifySummary::NoneVerified
        );
    }

    #[test]
    fn test_verify_solution_set_with_discrete() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let two = ctx.num(2);
        let set = SolutionSet::Discrete(vec![one, two]);
        let mut calls = 0usize;
        let mut verify = |_id: ExprId| {
            calls += 1;
            VerifyStatus::Verified
        };

        let result = verify_solution_set_with(&set, &mut verify);
        assert_eq!(calls, 2);
        assert_eq!(result.summary, VerifySummary::AllVerified);
        assert_eq!(result.solutions.len(), 2);
    }

    #[test]
    fn verify_substituted_residual_with_strict_fold_and_generic_fallback_verifies_on_strict() {
        let mut ctx = Context::new();
        let diff = ctx.num(99);
        let zero = ctx.num(0);
        let nonzero = ctx.num(2);

        let (verified, strict_residual) =
            verify_substituted_residual_with_strict_fold_and_generic_fallback(
                diff,
                |_expr| zero,
                |_expr| nonzero,
                |_expr| false,
                |expr| expr,
                |expr| expr == zero,
                || {},
                || {},
                || {},
            );

        assert!(verified);
        assert_eq!(strict_residual, zero);
    }

    #[test]
    fn verify_substituted_residual_with_strict_fold_and_generic_fallback_verifies_after_fold() {
        let mut ctx = Context::new();
        let diff = ctx.num(100);
        let strict = ctx.var("x");
        let folded = ctx.num(7);
        let zero = ctx.num(0);

        let attempted = Cell::new(0usize);
        let changed = Cell::new(0usize);
        let verified_counter = Cell::new(0usize);

        let (verified, strict_residual) =
            verify_substituted_residual_with_strict_fold_and_generic_fallback(
                diff,
                |expr| {
                    if expr == diff {
                        strict
                    } else if expr == folded {
                        zero
                    } else {
                        expr
                    }
                },
                |_expr| ctx.num(5),
                |expr| expr == strict,
                |expr| if expr == strict { folded } else { expr },
                |expr| expr == zero,
                || attempted.set(attempted.get() + 1),
                || changed.set(changed.get() + 1),
                || verified_counter.set(verified_counter.get() + 1),
            );

        assert!(verified);
        assert_eq!(strict_residual, strict);
        assert_eq!(attempted.get(), 1);
        assert_eq!(changed.get(), 1);
        assert_eq!(verified_counter.get(), 1);
    }

    #[test]
    fn verify_substituted_residual_with_strict_fold_and_generic_fallback_verifies_on_generic_ground(
    ) {
        let mut ctx = Context::new();
        let diff = ctx.num(101);
        let strict = ctx.num(9);
        let zero = ctx.num(0);

        let (verified, strict_residual) =
            verify_substituted_residual_with_strict_fold_and_generic_fallback(
                diff,
                |_expr| strict,
                |_expr| zero,
                |_expr| false,
                |expr| expr,
                |expr| expr == zero,
                || {},
                || {},
                || {},
            );

        assert!(verified);
        assert_eq!(strict_residual, strict);
    }

    #[test]
    fn verify_substituted_residual_with_strict_fold_and_generic_fallback_returns_unverified() {
        let mut ctx = Context::new();
        let diff = ctx.num(102);
        let strict = ctx.var("x");
        let zero = ctx.num(0);

        let attempted = Cell::new(0usize);
        let changed = Cell::new(0usize);
        let verified_counter = Cell::new(0usize);

        let (verified, strict_residual) =
            verify_substituted_residual_with_strict_fold_and_generic_fallback(
                diff,
                |_expr| strict,
                |_expr| zero,
                |expr| expr == strict,
                |expr| expr,
                |expr| expr == zero,
                || attempted.set(attempted.get() + 1),
                || changed.set(changed.get() + 1),
                || verified_counter.set(verified_counter.get() + 1),
            );

        assert!(!verified);
        assert_eq!(strict_residual, strict);
        assert_eq!(attempted.get(), 1);
        assert_eq!(changed.get(), 0);
        assert_eq!(verified_counter.get(), 0);
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
}
