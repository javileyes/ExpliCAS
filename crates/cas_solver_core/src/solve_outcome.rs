use crate::isolation_utils::NumericSign;
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

/// Classify numeric RHS sign for `|A| = RHS`.
pub fn abs_equality_precheck(sign: NumericSign) -> AbsEqualityPrecheck {
    match sign {
        NumericSign::Negative => AbsEqualityPrecheck::ReturnEmptySet,
        NumericSign::Zero => AbsEqualityPrecheck::CollapseToZero,
        NumericSign::Positive => AbsEqualityPrecheck::Continue,
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
}
