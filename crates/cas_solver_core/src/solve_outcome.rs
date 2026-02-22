use crate::isolation_utils::{mk_residual_solve, NumericSign};
use crate::log_domain::{
    assumptions_to_condition_set, classify_log_unsupported_route, classify_terminal_action,
    DomainModeKind, LogAssumption, LogSolveDecision, LogTerminalAction, LogUnsupportedRoute,
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

/// Shortcut routing for equations with variable in exponent (`base^x = rhs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowExponentShortcut {
    None,
    /// `base^x = base`
    PowerEqualsBase(PowerEqualsBaseRoute),
    /// `base^x = base^n`
    EqualPowBases {
        rhs_exp: ExprId,
    },
}

/// Operational resolution for exponent shortcut routes.
#[derive(Debug, Clone, PartialEq)]
pub enum PowExponentShortcutResolution {
    Continue,
    IsolateExponent { rhs: ExprId, op: RelOp },
    ReturnSolutionSet(SolutionSet),
}

/// Structured outcome for unsupported logarithmic rewrites.
#[derive(Debug, Clone, PartialEq)]
pub enum LogUnsupportedOutcome<'a> {
    ResidualBudgetExhausted {
        message: &'a str,
        solutions: SolutionSet,
    },
    Guarded {
        message: &'a str,
        missing_conditions: &'a [LogAssumption],
        guard: ConditionSet,
        residual: ExprId,
    },
}

/// Routing for isolation when the variable is in the power base (`B^E = RHS`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowBaseIsolationRoute {
    /// `E` is even and `RHS` is known negative.
    EvenExponentNegativeRhsImpossible,
    /// `E` is even and root isolation must use absolute value.
    EvenExponentUseAbsRoot,
    /// General root isolation, optionally flipping inequality for negative exponent.
    GeneralRoot {
        flip_inequality_for_negative_exponent: bool,
    },
}

/// Shortcut outcomes for `1^x = rhs`-style equations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerBaseOneShortcut {
    NotApplicable,
    AllReals,
    Empty,
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

/// Classify whether the base-one shortcut applies and which outcome it yields.
pub fn classify_power_base_one_shortcut(
    base_is_one: bool,
    rhs_is_one: bool,
) -> PowerBaseOneShortcut {
    if !base_is_one {
        return PowerBaseOneShortcut::NotApplicable;
    }
    if rhs_is_one {
        PowerBaseOneShortcut::AllReals
    } else {
        PowerBaseOneShortcut::Empty
    }
}

/// Convert shortcut classification to solution set, if applicable.
pub fn power_base_one_shortcut_solutions(shortcut: PowerBaseOneShortcut) -> Option<SolutionSet> {
    match shortcut {
        PowerBaseOneShortcut::NotApplicable => None,
        PowerBaseOneShortcut::AllReals => Some(SolutionSet::AllReals),
        PowerBaseOneShortcut::Empty => Some(SolutionSet::Empty),
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

/// Compute shortcut inputs for exponent isolation (`base^x = rhs`) from `rhs` shape.
///
/// The caller provides `same_base`, which decides whether the original base and
/// a candidate expression are equivalent under its local notion of equivalence.
pub fn detect_pow_exponent_shortcut_inputs<F>(
    rhs: ExprId,
    rhs_expr: &Expr,
    mut same_base: F,
) -> (bool, Option<ExprId>)
where
    F: FnMut(ExprId) -> bool,
{
    let bases_equal = same_base(rhs);
    let rhs_pow_base_equal = match rhs_expr {
        Expr::Pow(rhs_base, rhs_exp) if same_base(*rhs_base) => Some(*rhs_exp),
        _ => None,
    };
    (bases_equal, rhs_pow_base_equal)
}

/// Classify exponent-isolation shortcuts before generic logarithmic isolation.
pub fn classify_pow_exponent_shortcut(
    op: RelOp,
    bases_equal: bool,
    rhs_pow_base_equal: Option<ExprId>,
    base_is_zero: bool,
    base_is_numeric: bool,
    can_branch: bool,
) -> PowExponentShortcut {
    if bases_equal && op == RelOp::Eq {
        return PowExponentShortcut::PowerEqualsBase(classify_power_equals_base_route(
            base_is_zero,
            base_is_numeric,
            can_branch,
        ));
    }
    if let Some(rhs_exp) = rhs_pow_base_equal {
        return PowExponentShortcut::EqualPowBases { rhs_exp };
    }
    PowExponentShortcut::None
}

/// Resolve a classified exponent shortcut into either:
/// - an equation target for exponent isolation, or
/// - a terminal solution set.
pub fn resolve_pow_exponent_shortcut(
    ctx: &mut Context,
    shortcut: PowExponentShortcut,
    base: ExprId,
    op: RelOp,
) -> PowExponentShortcutResolution {
    match shortcut {
        PowExponentShortcut::PowerEqualsBase(route) => match route {
            PowerEqualsBaseRoute::ExponentGreaterThanZero => {
                PowExponentShortcutResolution::IsolateExponent {
                    rhs: ctx.num(0),
                    op: RelOp::Gt,
                }
            }
            PowerEqualsBaseRoute::ExponentEqualsOneNumericBase
            | PowerEqualsBaseRoute::ExponentEqualsOneNoBranchBudget => {
                PowExponentShortcutResolution::IsolateExponent {
                    rhs: ctx.num(1),
                    op,
                }
            }
            PowerEqualsBaseRoute::SymbolicCaseSplit => {
                PowExponentShortcutResolution::ReturnSolutionSet(
                    power_equals_base_symbolic_outcome(ctx, base),
                )
            }
        },
        PowExponentShortcut::EqualPowBases { rhs_exp } => {
            PowExponentShortcutResolution::IsolateExponent { rhs: rhs_exp, op }
        }
        PowExponentShortcut::None => PowExponentShortcutResolution::Continue,
    }
}

/// Classify route for base-isolation in a power equation.
pub fn classify_pow_base_isolation_route(
    exponent_is_even: bool,
    rhs_is_known_negative: bool,
    exponent_is_known_negative: bool,
) -> PowBaseIsolationRoute {
    if exponent_is_even && rhs_is_known_negative {
        PowBaseIsolationRoute::EvenExponentNegativeRhsImpossible
    } else if exponent_is_even {
        PowBaseIsolationRoute::EvenExponentUseAbsRoot
    } else {
        PowBaseIsolationRoute::GeneralRoot {
            flip_inequality_for_negative_exponent: exponent_is_known_negative,
        }
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

/// Classify log-domain terminal decision for equations that have exactly one
/// exponential side with the solve variable in exponent position.
pub fn classify_single_side_exponential_log_decision<F>(
    ctx: &Context,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    lhs_has_var: bool,
    rhs_has_var: bool,
    mut classify_log_solve: F,
) -> Option<LogSolveDecision>
where
    F: FnMut(ExprId, ExprId) -> LogSolveDecision,
{
    let candidate = crate::isolation_utils::find_single_side_exponential_var_in_exponent(
        ctx,
        lhs,
        rhs,
        var,
        lhs_has_var,
        rhs_has_var,
    )?;
    Some(classify_log_solve(candidate.base, candidate.other_side))
}

/// Resolve terminal outcome for equations that have exactly one exponential side
/// with the solve variable in exponent position.
#[allow(clippy::too_many_arguments)]
pub fn resolve_single_side_exponential_terminal_outcome<F>(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    lhs_has_var: bool,
    rhs_has_var: bool,
    mode: DomainModeKind,
    wildcard_scope: bool,
    mut classify_log_solve: F,
) -> Option<TerminalSolveOutcome>
where
    F: FnMut(ExprId, ExprId) -> LogSolveDecision,
{
    let decision = classify_single_side_exponential_log_decision(
        ctx,
        lhs,
        rhs,
        var,
        lhs_has_var,
        rhs_has_var,
        &mut classify_log_solve,
    )?;
    resolve_log_terminal_outcome(ctx, &decision, mode, wildcard_scope, lhs, rhs, var)
}

/// Build a user-facing message for terminal outcomes, appending
/// `residual_suffix` only when the outcome is residual.
pub fn terminal_outcome_message(outcome: &TerminalSolveOutcome, residual_suffix: &str) -> String {
    if matches!(outcome.solutions, SolutionSet::Residual(_)) {
        format!("{}{}", outcome.message, residual_suffix)
    } else {
        outcome.message.to_string()
    }
}

/// Resolve unsupported-log rewrite decisions into concrete residual/guarded outcomes.
pub fn resolve_log_unsupported_outcome<'a>(
    ctx: &mut Context,
    decision: &'a LogSolveDecision,
    can_branch: bool,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    base: ExprId,
    rhs_expr: ExprId,
) -> Option<LogUnsupportedOutcome<'a>> {
    match classify_log_unsupported_route(decision, can_branch) {
        LogUnsupportedRoute::NotUnsupported => None,
        LogUnsupportedRoute::ResidualBudgetExhausted { message } => {
            Some(LogUnsupportedOutcome::ResidualBudgetExhausted {
                message,
                solutions: residual_solution_set(ctx, lhs, rhs, var),
            })
        }
        LogUnsupportedRoute::Guarded {
            message,
            missing_conditions,
        } => Some(LogUnsupportedOutcome::Guarded {
            message,
            missing_conditions,
            guard: assumptions_to_condition_set(missing_conditions, base, rhs_expr),
            residual: residual_expression(ctx, lhs, rhs, var),
        }),
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
    fn classify_power_base_one_shortcut_not_applicable_when_base_not_one() {
        assert_eq!(
            classify_power_base_one_shortcut(false, true),
            PowerBaseOneShortcut::NotApplicable
        );
    }

    #[test]
    fn classify_power_base_one_shortcut_all_reals_when_rhs_one() {
        assert_eq!(
            classify_power_base_one_shortcut(true, true),
            PowerBaseOneShortcut::AllReals
        );
    }

    #[test]
    fn classify_power_base_one_shortcut_empty_when_rhs_not_one() {
        assert_eq!(
            classify_power_base_one_shortcut(true, false),
            PowerBaseOneShortcut::Empty
        );
    }

    #[test]
    fn power_base_one_shortcut_solutions_maps_correctly() {
        assert!(matches!(
            power_base_one_shortcut_solutions(PowerBaseOneShortcut::AllReals),
            Some(SolutionSet::AllReals)
        ));
        assert!(matches!(
            power_base_one_shortcut_solutions(PowerBaseOneShortcut::Empty),
            Some(SolutionSet::Empty)
        ));
        assert!(power_base_one_shortcut_solutions(PowerBaseOneShortcut::NotApplicable).is_none());
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
    fn resolve_single_side_exponential_terminal_outcome_returns_empty_when_classifier_says_empty() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let pow = ctx.add(Expr::Pow(two, x));
        let neg_five = ctx.num(-5);

        let out = resolve_single_side_exponential_terminal_outcome(
            &mut ctx,
            pow,
            neg_five,
            "x",
            true,
            false,
            DomainModeKind::Generic,
            false,
            |_base, _rhs| LogSolveDecision::EmptySet("no real solutions"),
        )
        .expect("must produce terminal outcome");

        assert_eq!(out.message, "no real solutions");
        assert!(matches!(out.solutions, SolutionSet::Empty));
    }

    #[test]
    fn classify_single_side_exponential_log_decision_returns_none_without_single_side_candidate() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let out = classify_single_side_exponential_log_decision(
            &ctx,
            x,
            y,
            "x",
            true,
            true,
            |_base, _rhs| LogSolveDecision::Ok,
        );
        assert!(out.is_none());
    }

    #[test]
    fn terminal_outcome_message_appends_suffix_for_residual() {
        let mut ctx = Context::new();
        let residual = ctx.var("residual");
        let outcome = TerminalSolveOutcome {
            message: "needs complex log",
            solutions: SolutionSet::Residual(residual),
        };
        assert_eq!(
            terminal_outcome_message(&outcome, " (residual)"),
            "needs complex log (residual)"
        );
    }

    #[test]
    fn terminal_outcome_message_keeps_message_for_non_residual() {
        let outcome = TerminalSolveOutcome {
            message: "no real solutions",
            solutions: SolutionSet::Empty,
        };
        assert_eq!(
            terminal_outcome_message(&outcome, " (residual)"),
            "no real solutions"
        );
    }

    #[test]
    fn resolve_log_unsupported_outcome_returns_none_for_supported_decision() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let out =
            resolve_log_unsupported_outcome(&mut ctx, &LogSolveDecision::Ok, true, x, x, "x", x, x);
        assert!(out.is_none());
    }

    #[test]
    fn resolve_log_unsupported_outcome_returns_residual_when_budget_exhausted() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let decision = LogSolveDecision::Unsupported(
            "Cannot prove RHS > 0 for logarithm",
            vec![crate::log_domain::LogAssumption::PositiveRhs],
        );
        let out = resolve_log_unsupported_outcome(&mut ctx, &decision, false, x, y, "x", x, y)
            .expect("must produce residual outcome");
        match out {
            LogUnsupportedOutcome::ResidualBudgetExhausted { message, solutions } => {
                assert_eq!(message, "Cannot prove RHS > 0 for logarithm");
                assert!(matches!(solutions, SolutionSet::Residual(_)));
            }
            other => panic!("expected residual outcome, got {:?}", other),
        }
    }

    #[test]
    fn resolve_log_unsupported_outcome_returns_guarded_when_branching_allowed() {
        let mut ctx = Context::new();
        let base = ctx.var("a");
        let rhs = ctx.var("b");
        let decision = LogSolveDecision::Unsupported(
            "Cannot prove base > 0 and RHS > 0 for logarithm",
            vec![
                crate::log_domain::LogAssumption::PositiveBase,
                crate::log_domain::LogAssumption::PositiveRhs,
            ],
        );
        let out =
            resolve_log_unsupported_outcome(&mut ctx, &decision, true, base, rhs, "x", base, rhs)
                .expect("must produce guarded outcome");
        match out {
            LogUnsupportedOutcome::Guarded {
                message,
                missing_conditions,
                guard,
                residual,
            } => {
                assert_eq!(message, "Cannot prove base > 0 and RHS > 0 for logarithm");
                assert_eq!(missing_conditions.len(), 2);
                assert_eq!(guard.predicates().len(), 2);
                assert!(matches!(ctx.get(residual), Expr::Function(_, _)));
            }
            other => panic!("expected guarded outcome, got {:?}", other),
        }
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

    #[test]
    fn classify_pow_exponent_shortcut_prefers_power_equals_base_in_eq() {
        let mut ctx = Context::new();
        let rhs_exp = ctx.var("n");
        let out =
            classify_pow_exponent_shortcut(RelOp::Eq, true, Some(rhs_exp), false, true, false);
        assert!(matches!(
            out,
            PowExponentShortcut::PowerEqualsBase(
                PowerEqualsBaseRoute::ExponentEqualsOneNumericBase
            )
        ));
    }

    #[test]
    fn detect_pow_exponent_shortcut_inputs_detects_equal_pow_bases() {
        let mut ctx = Context::new();
        let b = ctx.var("b");
        let n = ctx.var("n");
        let rhs = ctx.add(Expr::Pow(b, n));
        let rhs_expr = ctx.get(rhs).clone();
        let (bases_equal, rhs_pow_base_equal) =
            detect_pow_exponent_shortcut_inputs(rhs, &rhs_expr, |candidate| candidate == b);
        assert!(!bases_equal);
        assert_eq!(rhs_pow_base_equal, Some(n));
    }

    #[test]
    fn detect_pow_exponent_shortcut_inputs_handles_non_pow_rhs() {
        let mut ctx = Context::new();
        let b = ctx.var("b");
        let rhs = ctx.var("y");
        let rhs_expr = ctx.get(rhs).clone();
        let (bases_equal, rhs_pow_base_equal) =
            detect_pow_exponent_shortcut_inputs(rhs, &rhs_expr, |candidate| candidate == b);
        assert!(!bases_equal);
        assert_eq!(rhs_pow_base_equal, None);
    }

    #[test]
    fn classify_pow_exponent_shortcut_allows_equal_pow_bases_for_inequality() {
        let mut ctx = Context::new();
        let rhs_exp = ctx.var("n");
        let out =
            classify_pow_exponent_shortcut(RelOp::Geq, false, Some(rhs_exp), false, false, false);
        assert!(matches!(
            out,
            PowExponentShortcut::EqualPowBases { rhs_exp: exp } if exp == rhs_exp
        ));
    }

    #[test]
    fn classify_pow_exponent_shortcut_returns_none_when_no_shortcut_applies() {
        let out = classify_pow_exponent_shortcut(RelOp::Eq, false, None, false, false, false);
        assert_eq!(out, PowExponentShortcut::None);
    }

    #[test]
    fn resolve_pow_exponent_shortcut_maps_zero_base_route_to_gt_zero() {
        let mut ctx = Context::new();
        let base = ctx.num(0);
        let out = resolve_pow_exponent_shortcut(
            &mut ctx,
            PowExponentShortcut::PowerEqualsBase(PowerEqualsBaseRoute::ExponentGreaterThanZero),
            base,
            RelOp::Eq,
        );
        match out {
            PowExponentShortcutResolution::IsolateExponent { rhs, op } => {
                assert_eq!(op, RelOp::Gt);
                assert!(matches!(
                    ctx.get(rhs),
                    Expr::Number(n) if *n == num_rational::BigRational::from_integer(0.into())
                ));
            }
            other => panic!("expected isolate route, got {:?}", other),
        }
    }

    #[test]
    fn resolve_pow_exponent_shortcut_maps_equal_pow_bases_to_rhs_exponent() {
        let mut ctx = Context::new();
        let base = ctx.var("b");
        let rhs_exp = ctx.var("n");
        let out = resolve_pow_exponent_shortcut(
            &mut ctx,
            PowExponentShortcut::EqualPowBases { rhs_exp },
            base,
            RelOp::Leq,
        );
        match out {
            PowExponentShortcutResolution::IsolateExponent { rhs, op } => {
                assert_eq!(rhs, rhs_exp);
                assert_eq!(op, RelOp::Leq);
            }
            other => panic!("expected isolate route, got {:?}", other),
        }
    }

    #[test]
    fn resolve_pow_exponent_shortcut_maps_symbolic_case_split_to_terminal_set() {
        let mut ctx = Context::new();
        let base = ctx.var("a");
        let out = resolve_pow_exponent_shortcut(
            &mut ctx,
            PowExponentShortcut::PowerEqualsBase(PowerEqualsBaseRoute::SymbolicCaseSplit),
            base,
            RelOp::Eq,
        );
        assert!(matches!(
            out,
            PowExponentShortcutResolution::ReturnSolutionSet(SolutionSet::Conditional(_))
        ));
    }

    #[test]
    fn classify_pow_base_isolation_route_impossible_even_negative_rhs() {
        assert_eq!(
            classify_pow_base_isolation_route(true, true, false),
            PowBaseIsolationRoute::EvenExponentNegativeRhsImpossible
        );
    }

    #[test]
    fn classify_pow_base_isolation_route_even_uses_abs_root() {
        assert_eq!(
            classify_pow_base_isolation_route(true, false, true),
            PowBaseIsolationRoute::EvenExponentUseAbsRoot
        );
    }

    #[test]
    fn classify_pow_base_isolation_route_general_tracks_negative_exponent() {
        assert_eq!(
            classify_pow_base_isolation_route(false, false, true),
            PowBaseIsolationRoute::GeneralRoot {
                flip_inequality_for_negative_exponent: true
            }
        );
    }
}
