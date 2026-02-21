use cas_ast::{Case, ConditionPredicate, ConditionSet, ExprId, SolutionSet};

/// Ternary status for proofs about whether an expression is non-zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonZeroStatus {
    NonZero,
    Zero,
    Unknown,
}

/// Build the canonical solution set for a linear solve kernel:
/// `coef * var = constant`, with candidate `var = constant / coef`.
///
/// `coef_status` and `constant_status` are proof statuses for non-zero checks.
/// Unknown statuses produce a guarded conditional fallback.
pub fn build_linear_solution_set(
    coef: ExprId,
    constant: ExprId,
    solution: ExprId,
    coef_status: NonZeroStatus,
    constant_status: NonZeroStatus,
) -> SolutionSet {
    match coef_status {
        NonZeroStatus::NonZero => {
            return SolutionSet::Discrete(vec![solution]);
        }
        NonZeroStatus::Zero => match constant_status {
            NonZeroStatus::Zero => return SolutionSet::AllReals,
            NonZeroStatus::NonZero => return SolutionSet::Empty,
            NonZeroStatus::Unknown => {}
        },
        NonZeroStatus::Unknown => {}
    }

    // Fallback: conditional split on coefficient being zero or non-zero.
    let primary_guard = ConditionSet::single(ConditionPredicate::NonZero(coef));
    let primary_case = Case::new(primary_guard, SolutionSet::Discrete(vec![solution]));

    let mut both_zero_guard = ConditionSet::single(ConditionPredicate::EqZero(coef));
    both_zero_guard.push(ConditionPredicate::EqZero(constant));
    let all_reals_case = Case::new(both_zero_guard, SolutionSet::AllReals);

    let otherwise_case = Case::new(ConditionSet::empty(), SolutionSet::Empty);

    SolutionSet::Conditional(vec![primary_case, all_reals_case, otherwise_case])
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Context;

    #[test]
    fn nonzero_coef_returns_discrete_solution() {
        let mut ctx = Context::new();
        let coef = ctx.var("a");
        let constant = ctx.var("b");
        let solution = ctx.var("x0");
        let set = build_linear_solution_set(
            coef,
            constant,
            solution,
            NonZeroStatus::NonZero,
            NonZeroStatus::Unknown,
        );
        assert_eq!(set, SolutionSet::Discrete(vec![solution]));
    }

    #[test]
    fn zero_coef_and_zero_constant_returns_all_reals() {
        let mut ctx = Context::new();
        let coef = ctx.var("a");
        let constant = ctx.var("b");
        let solution = ctx.var("x0");
        let set = build_linear_solution_set(
            coef,
            constant,
            solution,
            NonZeroStatus::Zero,
            NonZeroStatus::Zero,
        );
        assert_eq!(set, SolutionSet::AllReals);
    }

    #[test]
    fn zero_coef_and_nonzero_constant_returns_empty() {
        let mut ctx = Context::new();
        let coef = ctx.var("a");
        let constant = ctx.var("b");
        let solution = ctx.var("x0");
        let set = build_linear_solution_set(
            coef,
            constant,
            solution,
            NonZeroStatus::Zero,
            NonZeroStatus::NonZero,
        );
        assert_eq!(set, SolutionSet::Empty);
    }

    #[test]
    fn unknown_statuses_return_conditional() {
        let mut ctx = Context::new();
        let coef = ctx.var("a");
        let constant = ctx.var("b");
        let solution = ctx.var("x0");
        let set = build_linear_solution_set(
            coef,
            constant,
            solution,
            NonZeroStatus::Unknown,
            NonZeroStatus::Unknown,
        );
        assert!(matches!(set, SolutionSet::Conditional(_)));
    }
}
