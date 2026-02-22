use cas_ast::{Case, ConditionPredicate, ConditionSet, ExprId, SolutionSet};

/// Ternary status for proofs about whether an expression is non-zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonZeroStatus {
    NonZero,
    Zero,
    Unknown,
}

/// Derive `(coef_status, constant_status)` for linear solve degeneracy checks.
///
/// Policy:
/// - If the coefficient still contains solve variable, both statuses remain unknown.
/// - Otherwise prove coefficient non-zero first.
/// - Only when coefficient is proven zero, prove the constant term.
pub fn derive_linear_nonzero_statuses<F>(
    coef_contains_var: bool,
    coef: ExprId,
    constant: ExprId,
    mut prove_nonzero_status: F,
) -> (NonZeroStatus, NonZeroStatus)
where
    F: FnMut(ExprId) -> NonZeroStatus,
{
    let mut coef_status = NonZeroStatus::Unknown;
    let mut constant_status = NonZeroStatus::Unknown;

    if !coef_contains_var {
        coef_status = prove_nonzero_status(coef);
        if coef_status == NonZeroStatus::Zero {
            constant_status = prove_nonzero_status(constant);
        }
    }

    (coef_status, constant_status)
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

    #[test]
    fn derive_statuses_skips_proofs_when_coefficient_contains_var() {
        let mut ctx = Context::new();
        let coef = ctx.var("a");
        let constant = ctx.var("b");
        let mut calls = 0usize;

        let (coef_status, constant_status) =
            derive_linear_nonzero_statuses(true, coef, constant, |_expr| {
                calls += 1;
                NonZeroStatus::Unknown
            });

        assert_eq!(coef_status, NonZeroStatus::Unknown);
        assert_eq!(constant_status, NonZeroStatus::Unknown);
        assert_eq!(calls, 0);
    }

    #[test]
    fn derive_statuses_only_proves_constant_when_coefficient_is_zero() {
        let mut ctx = Context::new();
        let coef = ctx.var("a");
        let constant = ctx.var("b");
        let mut seen = Vec::new();

        let (coef_status, constant_status) =
            derive_linear_nonzero_statuses(false, coef, constant, |expr| {
                seen.push(expr);
                if expr == coef {
                    NonZeroStatus::Zero
                } else {
                    NonZeroStatus::NonZero
                }
            });

        assert_eq!(coef_status, NonZeroStatus::Zero);
        assert_eq!(constant_status, NonZeroStatus::NonZero);
        assert_eq!(seen, vec![coef, constant]);
    }
}
