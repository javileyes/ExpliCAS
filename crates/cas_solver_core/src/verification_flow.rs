use crate::domain_mode::DomainMode;
use crate::verification::{
    verify_solution_set_with_state,
    verify_solution_with_strict_fold_and_generic_fallback_with_default_stats_and_state,
    VerifyResult, VerifyStatus,
};
use cas_ast::{Equation, ExprId, SolutionSet};
use std::cell::RefCell;

/// Shared runtime flow for solver candidate verification.
///
/// This orchestrates the strict/generic two-pass verification pipeline while
/// delegating runtime details (substitution, simplification, rendering) via
/// closures so higher layers can plug their own engine state.
#[allow(clippy::too_many_arguments)]
pub fn verify_solution_with_domain_modes_with_state<
    T,
    FSubstituteDiff,
    FSimplifyByDomain,
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
    simplify_by_domain: FSimplifyByDomain,
    contains_variable: FContainsVariable,
    fold_numeric_islands: FFoldNumericIslands,
    is_zero: FIsZero,
    render_expr: FRenderExpr,
) -> VerifyStatus
where
    FSubstituteDiff: FnMut(&mut T, &Equation, &str, ExprId) -> ExprId,
    FSimplifyByDomain: FnMut(&mut T, ExprId, DomainMode) -> ExprId,
    FContainsVariable: FnMut(&mut T, ExprId) -> bool,
    FFoldNumericIslands: FnMut(&mut T, ExprId) -> ExprId,
    FIsZero: FnMut(&mut T, ExprId) -> bool,
    FRenderExpr: FnMut(&mut T, ExprId) -> String,
{
    let simplify_by_domain = RefCell::new(simplify_by_domain);
    verify_solution_with_strict_fold_and_generic_fallback_with_default_stats_and_state(
        state,
        equation,
        var,
        solution,
        substitute_diff,
        |state, expr| (simplify_by_domain.borrow_mut())(state, expr, DomainMode::Strict),
        |state, expr| (simplify_by_domain.borrow_mut())(state, expr, DomainMode::Generic),
        contains_variable,
        fold_numeric_islands,
        is_zero,
        render_expr,
    )
}

/// Verify a full solution set using a stateful candidate verifier.
pub fn verify_solution_set_for_equation_with_state<T, FVerifySolution>(
    state: &mut T,
    equation: &Equation,
    var: &str,
    solutions: &SolutionSet,
    mut verify_solution: FVerifySolution,
) -> VerifyResult
where
    FVerifySolution: FnMut(&mut T, &Equation, &str, ExprId) -> VerifyStatus,
{
    let mut verify_discrete =
        |state: &mut T, solution: ExprId| verify_solution(state, equation, var, solution);
    verify_solution_set_with_state(state, solutions, &mut verify_discrete)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::{Context, RelOp};

    #[test]
    fn verify_solution_set_for_equation_with_state_runs_per_solution() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let eq = Equation {
            lhs: x,
            rhs: x,
            op: RelOp::Eq,
        };
        let solutions = SolutionSet::Discrete(vec![ctx.num(1), ctx.num(2)]);

        let mut calls = 0usize;
        let result = verify_solution_set_for_equation_with_state(
            &mut calls,
            &eq,
            "x",
            &solutions,
            |counter, _eq, _var, _solution| {
                *counter += 1;
                VerifyStatus::Verified
            },
        );

        assert_eq!(calls, 2);
        assert_eq!(result.solutions.len(), 2);
    }
}
