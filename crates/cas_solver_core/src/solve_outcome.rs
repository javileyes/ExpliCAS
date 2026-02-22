use crate::isolation_utils::{mk_residual_solve, NumericSign};
use crate::log_domain::{
    classify_terminal_action, DomainModeKind, LogSolveDecision, LogTerminalAction,
};
use crate::solution_set::open_positive_domain;
use cas_ast::{
    Case, ConditionPredicate, ConditionSet, Context, Expr, ExprId, RelOp, SolutionSet, SolveResult,
};

/// Classification of a variable-free equation residual `diff = lhs - rhs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarFreeDiffKind {
    /// `diff == 0`: identity (`0 = 0`)
    IdentityZero,
    /// `diff != 0` as a concrete number: contradiction (`c = 0`)
    ContradictionNonZero,
    /// Non-numeric residual over other symbols: constraint on parameters
    Constraint,
}

/// Generic terminal solve outcome (message + solution set).
#[derive(Debug, Clone, PartialEq)]
pub struct TerminalSolveOutcome {
    pub message: &'static str,
    pub solutions: SolutionSet,
}

/// Route for handling `base^x = base` shortcuts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerEqualsBaseRoute {
    /// `0^x = 0` -> `x > 0`.
    ExponentGreaterThanZero,
    /// Numeric base (non-zero): `x = 1`.
    ExponentEqualsOneNumericBase,
    /// Symbolic base with no branch budget: fallback `x = 1`.
    ExponentEqualsOneNoBranchBudget,
    /// Symbolic base with branching budget: produce conditional case split.
    SymbolicCaseSplit,
}

/// Classify the simplified variable-free residual.
pub fn classify_var_free_difference(ctx: &Context, diff: ExprId) -> VarFreeDiffKind {
    match ctx.get(diff) {
        Expr::Number(n) if *n == num_rational::BigRational::from_integer(0.into()) => {
            VarFreeDiffKind::IdentityZero
        }
        Expr::Number(_) => VarFreeDiffKind::ContradictionNonZero,
        _ => VarFreeDiffKind::Constraint,
    }
}

/// Solve outcome for `B^E op RHS` when `E` is even and `RHS` is proven negative.
pub fn even_power_negative_rhs_outcome(op: RelOp) -> SolutionSet {
    match op {
        RelOp::Eq => SolutionSet::Empty,
        RelOp::Gt | RelOp::Geq | RelOp::Neq => SolutionSet::AllReals,
        RelOp::Lt | RelOp::Leq => SolutionSet::Empty,
    }
}

/// Outcome for `1^x op rhs` in real arithmetic.
pub fn power_base_one_outcome(rhs_is_one: bool) -> SolutionSet {
    if rhs_is_one {
        SolutionSet::AllReals
    } else {
        SolutionSet::Empty
    }
}

/// Decide how to handle `base^x = base`.
pub fn classify_power_equals_base_route(
    base_is_zero: bool,
    base_is_numeric: bool,
    can_branch: bool,
) -> PowerEqualsBaseRoute {
    if base_is_zero {
        PowerEqualsBaseRoute::ExponentGreaterThanZero
    } else if base_is_numeric {
        PowerEqualsBaseRoute::ExponentEqualsOneNumericBase
    } else if can_branch {
        PowerEqualsBaseRoute::SymbolicCaseSplit
    } else {
        PowerEqualsBaseRoute::ExponentEqualsOneNoBranchBudget
    }
}

/// Build a residual solution set `solve(__eq__(lhs, rhs), var)`.
pub fn residual_solution_set(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
) -> SolutionSet {
    SolutionSet::Residual(residual_expression(ctx, lhs, rhs, var))
}

/// Build residual expression `solve(__eq__(lhs, rhs), var)`.
pub fn residual_expression(ctx: &mut Context, lhs: ExprId, rhs: ExprId, var: &str) -> ExprId {
    mk_residual_solve(ctx, lhs, rhs, var)
}

/// Resolve terminal log-solve actions into concrete solution sets.
pub fn resolve_log_terminal_outcome(
    ctx: &mut Context,
    decision: &LogSolveDecision,
    mode: DomainModeKind,
    wildcard_scope: bool,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
) -> Option<TerminalSolveOutcome> {
    match classify_terminal_action(decision, mode, wildcard_scope) {
        LogTerminalAction::ReturnEmptySet => {
            let LogSolveDecision::EmptySet(message) = decision else {
                return None;
            };
            Some(TerminalSolveOutcome {
                message,
                solutions: SolutionSet::Empty,
            })
        }
        LogTerminalAction::ReturnResidualInWildcard => {
            let LogSolveDecision::NeedsComplex(message) = decision else {
                return None;
            };
            Some(TerminalSolveOutcome {
                message,
                solutions: residual_solution_set(ctx, lhs, rhs, var),
            })
        }
        LogTerminalAction::Continue => None,
    }
}

/// Pre-check decision for absolute-value equalities `|A| = RHS`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbsEqualityPrecheck {
    /// `RHS < 0` -> impossible.
    ReturnEmptySet,
    /// `RHS = 0` -> reduce to `A = 0`.
    CollapseToZero,
    /// `RHS > 0` -> keep normal branch split.
    Continue,
}

/// Fast-path routing for absolute-value isolation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbsIsolationFastPath {
    ReturnEmptySet,
    CollapseToZero,
    Continue,
}

/// Classify numeric RHS sign for `|A| = RHS`.
pub fn abs_equality_precheck(sign: NumericSign) -> AbsEqualityPrecheck {
    match sign {
        NumericSign::Negative => AbsEqualityPrecheck::ReturnEmptySet,
        NumericSign::Zero => AbsEqualityPrecheck::CollapseToZero,
        NumericSign::Positive => AbsEqualityPrecheck::Continue,
    }
}

/// Decide fast-path behavior for absolute-value isolation from operator and RHS sign.
pub fn classify_abs_isolation_fast_path(
    op: RelOp,
    rhs_sign: Option<NumericSign>,
) -> AbsIsolationFastPath {
    if op != RelOp::Eq {
        return AbsIsolationFastPath::Continue;
    }
    match rhs_sign.map(abs_equality_precheck) {
        Some(AbsEqualityPrecheck::ReturnEmptySet) => AbsIsolationFastPath::ReturnEmptySet,
        Some(AbsEqualityPrecheck::CollapseToZero) => AbsIsolationFastPath::CollapseToZero,
        _ => AbsIsolationFastPath::Continue,
    }
}

/// For `|A| = rhs`, attach the soundness guard `rhs >= 0` when
/// `rhs` depends on the solve variable.
pub fn guard_abs_solution_with_nonnegative_rhs(
    rhs_contains_var: bool,
    rhs: ExprId,
    combined: SolutionSet,
) -> SolutionSet {
    if rhs_contains_var {
        let guard = ConditionSet::single(ConditionPredicate::NonNegative(rhs));
        SolutionSet::Conditional(vec![Case::new(guard, combined)])
    } else {
        combined
    }
}

/// Build `Conditional([guard -> guarded_solutions, else -> Residual(original_eq)])`.
pub fn guarded_solutions_with_residual_fallback(
    guard: ConditionSet,
    guarded_solutions: SolutionSet,
    residual_expr: ExprId,
) -> SolutionSet {
    SolutionSet::Conditional(vec![
        Case::new(guard, guarded_solutions),
        Case::new(ConditionSet::empty(), SolutionSet::Residual(residual_expr)),
    ])
}

/// Build guarded conditional solutions when both pieces are available;
/// otherwise fall back to residual.
pub fn guarded_or_residual(
    guard: Option<ConditionSet>,
    guarded_solutions: Option<SolutionSet>,
    residual_expr: ExprId,
) -> SolutionSet {
    match (guard, guarded_solutions) {
        (Some(guard), Some(guarded_solutions)) => {
            guarded_solutions_with_residual_fallback(guard, guarded_solutions, residual_expr)
        }
        _ => SolutionSet::Residual(residual_expr),
    }
}

/// Outcome for symbolic `a^x = a` (with `a` symbolic).
///
/// Returns:
/// - `a = 1`  -> `AllReals`
/// - `a = 0`  -> `x in (0, +inf)`
/// - otherwise -> `x = 1`
pub fn power_equals_base_symbolic_outcome(ctx: &mut Context, base: ExprId) -> SolutionSet {
    let one = ctx.num(1);

    let case_one_guard = ConditionSet::single(ConditionPredicate::EqOne(base));
    let case_one = Case::with_result(case_one_guard, SolveResult::solved(SolutionSet::AllReals));

    let case_zero_guard = ConditionSet::single(ConditionPredicate::EqZero(base));
    let case_zero = Case::with_result(
        case_zero_guard,
        SolveResult::solved(open_positive_domain(ctx)),
    );

    let case_default_guard = ConditionSet::empty();
    let case_default = Case::with_result(
        case_default_guard,
        SolveResult::solved(SolutionSet::Discrete(vec![one])),
    );

    SolutionSet::Conditional(vec![case_one, case_zero, case_default])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_zero_number() {
        let mut ctx = Context::new();
        let zero = ctx.num(0);
        assert_eq!(
            classify_var_free_difference(&ctx, zero),
            VarFreeDiffKind::IdentityZero
        );
    }

    #[test]
    fn classify_nonzero_number() {
        let mut ctx = Context::new();
        let two = ctx.num(2);
        assert_eq!(
            classify_var_free_difference(&ctx, two),
            VarFreeDiffKind::ContradictionNonZero
        );
    }

    #[test]
    fn classify_symbolic_constraint() {
        let mut ctx = Context::new();
        let y = ctx.var("y");
        assert_eq!(
            classify_var_free_difference(&ctx, y),
            VarFreeDiffKind::Constraint
        );
    }

    #[test]
    fn even_power_negative_rhs_outcome_for_eq_is_empty() {
        assert!(matches!(
            even_power_negative_rhs_outcome(RelOp::Eq),
            SolutionSet::Empty
        ));
    }

    #[test]
    fn even_power_negative_rhs_outcome_for_neq_is_all_reals() {
        assert!(matches!(
            even_power_negative_rhs_outcome(RelOp::Neq),
            SolutionSet::AllReals
        ));
    }

    #[test]
    fn power_equals_base_symbolic_outcome_has_three_cases() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let out = power_equals_base_symbolic_outcome(&mut ctx, a);
        match out {
            SolutionSet::Conditional(cases) => assert_eq!(cases.len(), 3),
            other => panic!("Expected conditional set, got {:?}", other),
        }
    }

    #[test]
    fn power_base_one_outcome_all_reals_when_rhs_is_one() {
        assert!(matches!(
            power_base_one_outcome(true),
            SolutionSet::AllReals
        ));
    }

    #[test]
    fn power_base_one_outcome_empty_when_rhs_not_one() {
        assert!(matches!(power_base_one_outcome(false), SolutionSet::Empty));
    }

    #[test]
    fn abs_equality_precheck_negative_is_empty() {
        assert_eq!(
            abs_equality_precheck(NumericSign::Negative),
            AbsEqualityPrecheck::ReturnEmptySet
        );
    }

    #[test]
    fn abs_equality_precheck_zero_collapses_to_zero() {
        assert_eq!(
            abs_equality_precheck(NumericSign::Zero),
            AbsEqualityPrecheck::CollapseToZero
        );
    }

    #[test]
    fn abs_equality_precheck_positive_continues() {
        assert_eq!(
            abs_equality_precheck(NumericSign::Positive),
            AbsEqualityPrecheck::Continue
        );
    }

    #[test]
    fn abs_fast_path_non_equality_continues() {
        assert_eq!(
            classify_abs_isolation_fast_path(RelOp::Geq, Some(NumericSign::Negative)),
            AbsIsolationFastPath::Continue
        );
    }

    #[test]
    fn abs_fast_path_negative_rhs_returns_empty() {
        assert_eq!(
            classify_abs_isolation_fast_path(RelOp::Eq, Some(NumericSign::Negative)),
            AbsIsolationFastPath::ReturnEmptySet
        );
    }

    #[test]
    fn abs_fast_path_zero_rhs_collapses_to_zero() {
        assert_eq!(
            classify_abs_isolation_fast_path(RelOp::Eq, Some(NumericSign::Zero)),
            AbsIsolationFastPath::CollapseToZero
        );
    }

    #[test]
    fn abs_fast_path_missing_sign_continues() {
        assert_eq!(
            classify_abs_isolation_fast_path(RelOp::Eq, None),
            AbsIsolationFastPath::Continue
        );
    }

    #[test]
    fn abs_guard_wraps_solution_when_rhs_contains_var() {
        let mut ctx = Context::new();
        let rhs = ctx.var("x");
        let out = guard_abs_solution_with_nonnegative_rhs(true, rhs, SolutionSet::AllReals);
        match out {
            SolutionSet::Conditional(cases) => {
                assert_eq!(cases.len(), 1);
                assert_eq!(
                    cases[0].when,
                    ConditionSet::single(ConditionPredicate::NonNegative(rhs))
                );
                assert!(matches!(cases[0].then.solutions, SolutionSet::AllReals));
            }
            other => panic!("expected conditional guard, got {:?}", other),
        }
    }

    #[test]
    fn abs_guard_leaves_solution_unchanged_for_var_free_rhs() {
        let mut ctx = Context::new();
        let rhs = ctx.num(2);
        let out = guard_abs_solution_with_nonnegative_rhs(false, rhs, SolutionSet::Empty);
        assert!(matches!(out, SolutionSet::Empty));
    }

    #[test]
    fn resolve_log_terminal_empty_set() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let decision = LogSolveDecision::EmptySet("no real solutions");
        let outcome = resolve_log_terminal_outcome(
            &mut ctx,
            &decision,
            DomainModeKind::Generic,
            false,
            x,
            y,
            "x",
        )
        .expect("empty-set terminal outcome");
        assert_eq!(outcome.message, "no real solutions");
        assert!(matches!(outcome.solutions, SolutionSet::Empty));
    }

    #[test]
    fn resolve_log_terminal_residual_in_wildcard() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let decision = LogSolveDecision::NeedsComplex("needs complex log");
        let outcome = resolve_log_terminal_outcome(
            &mut ctx,
            &decision,
            DomainModeKind::Assume,
            true,
            x,
            y,
            "x",
        )
        .expect("residual terminal outcome");
        assert_eq!(outcome.message, "needs complex log");
        assert!(matches!(outcome.solutions, SolutionSet::Residual(_)));
    }

    #[test]
    fn resolve_log_terminal_continue_returns_none() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let decision = LogSolveDecision::Ok;
        let out = resolve_log_terminal_outcome(
            &mut ctx,
            &decision,
            DomainModeKind::Generic,
            false,
            x,
            y,
            "x",
        );
        assert!(out.is_none());
    }

    #[test]
    fn guarded_solutions_with_residual_fallback_builds_two_cases() {
        let mut ctx = Context::new();
        let b = ctx.var("b");
        let residual = ctx.var("residual");
        let guard = ConditionSet::single(ConditionPredicate::Positive(b));
        let out = guarded_solutions_with_residual_fallback(guard, SolutionSet::AllReals, residual);
        match out {
            SolutionSet::Conditional(cases) => {
                assert_eq!(cases.len(), 2);
                assert_eq!(
                    cases[0].when,
                    ConditionSet::single(ConditionPredicate::Positive(b))
                );
                assert!(matches!(cases[0].then.solutions, SolutionSet::AllReals));
                assert_eq!(cases[1].when, ConditionSet::empty());
                assert!(
                    matches!(cases[1].then.solutions, SolutionSet::Residual(id) if id == residual)
                );
            }
            other => panic!("expected conditional, got {:?}", other),
        }
    }

    #[test]
    fn guarded_or_residual_returns_conditional_when_guard_and_solutions_exist() {
        let mut ctx = Context::new();
        let b = ctx.var("b");
        let residual = ctx.var("residual");
        let guard = ConditionSet::single(ConditionPredicate::Positive(b));
        let out = guarded_or_residual(Some(guard), Some(SolutionSet::AllReals), residual);
        assert!(matches!(out, SolutionSet::Conditional(_)));
    }

    #[test]
    fn guarded_or_residual_returns_residual_without_guard() {
        let mut ctx = Context::new();
        let residual = ctx.var("residual");
        let out = guarded_or_residual(None, Some(SolutionSet::AllReals), residual);
        assert!(matches!(out, SolutionSet::Residual(id) if id == residual));
    }

    #[test]
    fn guarded_or_residual_returns_residual_without_guarded_solutions() {
        let mut ctx = Context::new();
        let b = ctx.var("b");
        let residual = ctx.var("residual");
        let guard = ConditionSet::single(ConditionPredicate::Positive(b));
        let out = guarded_or_residual(Some(guard), None, residual);
        assert!(matches!(out, SolutionSet::Residual(id) if id == residual));
    }

    #[test]
    fn residual_solution_set_wraps_residual_expression() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let out = residual_solution_set(&mut ctx, x, y, "x");
        assert!(matches!(out, SolutionSet::Residual(_)));
    }

    #[test]
    fn residual_expression_builds_solve_call() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let expr = residual_expression(&mut ctx, x, y, "x");
        assert!(matches!(ctx.get(expr), Expr::Function(_, _)));
    }

    #[test]
    fn classify_power_equals_base_route_zero_base() {
        assert_eq!(
            classify_power_equals_base_route(true, false, true),
            PowerEqualsBaseRoute::ExponentGreaterThanZero
        );
    }

    #[test]
    fn classify_power_equals_base_route_numeric_base() {
        assert_eq!(
            classify_power_equals_base_route(false, true, false),
            PowerEqualsBaseRoute::ExponentEqualsOneNumericBase
        );
    }

    #[test]
    fn classify_power_equals_base_route_symbolic_no_budget() {
        assert_eq!(
            classify_power_equals_base_route(false, false, false),
            PowerEqualsBaseRoute::ExponentEqualsOneNoBranchBudget
        );
    }

    #[test]
    fn classify_power_equals_base_route_symbolic_with_budget() {
        assert_eq!(
            classify_power_equals_base_route(false, false, true),
            PowerEqualsBaseRoute::SymbolicCaseSplit
        );
    }
}
