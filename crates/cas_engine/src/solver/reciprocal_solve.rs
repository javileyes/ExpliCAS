//! Reciprocal Solve Strategy
//!
//! Handles equations of the form `1/var = sum_of_fractions` with pedagogical steps:
//! 1. "Combine fractions on RHS (common denominator)"
//! 2. "Take reciprocal"
//!
//! Example: `1/R = 1/R1 + 1/R2` → `R = R1·R2/(R1+R2)`

use cas_ast::{Context, Equation, Expr, ExprId, RelOp};

use crate::engine::Simplifier;
use crate::solver::{contains_var, SolveStep};

/// Check if expr is `1/var` pattern (simple reciprocal of target variable)
pub(crate) fn is_simple_reciprocal(ctx: &Context, expr: ExprId, var: &str) -> bool {
    cas_solver_core::isolation_utils::is_simple_reciprocal(ctx, expr, var)
}

/// Combine multiple fractions into a single fraction (numerator, denominator).
///
/// Uses common denominator D = ∏ den_i
/// Numerator N = Σ (num_i × (D/den_i))
///
/// Returns (N, D) after light normalization (canonicalized order).
pub(crate) fn combine_fractions_deterministic(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    cas_solver_core::reciprocal::combine_fractions_deterministic(ctx, expr)
}

/// Try to solve `1/var = expr` using pedagogical steps.
///
/// Returns `Some((SolutionSet, steps))` if pattern matches,
/// `None` to fall through to standard isolation.
pub(crate) fn try_reciprocal_solve(
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    simplifier: &mut Simplifier,
) -> Option<(crate::solver::SolutionSet, Vec<SolveStep>)> {
    // Check pattern: LHS must be 1/var
    if !is_simple_reciprocal(&simplifier.context, lhs, var) {
        return None;
    }

    // Check: RHS must NOT contain var (otherwise this is more complex)
    if contains_var(&simplifier.context, rhs, var) {
        return None;
    }

    // Combine fractions on RHS
    let (numerator, denominator) = combine_fractions_deterministic(&mut simplifier.context, rhs)?;

    let mut steps = Vec::new();

    // Step 1: Combine fractions on RHS
    // 1/R = N/D
    let combined_rhs = simplifier.context.add(Expr::Div(numerator, denominator));

    // Light simplification for display (order terms)
    let (display_rhs, _) = simplifier.simplify(combined_rhs);

    if simplifier.collect_steps() {
        let var_id = simplifier.context.var(var);
        let one = simplifier.context.num(1);
        let reciprocal_lhs = simplifier.context.add(Expr::Div(one, var_id));

        steps.push(SolveStep {
            description: "Combine fractions on RHS (common denominator)".to_string(),
            equation_after: Equation {
                lhs: reciprocal_lhs,
                rhs: display_rhs,
                op: RelOp::Eq,
            },
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        });
    }

    // Step 2: Take reciprocal
    // R = D/N
    let solution = simplifier.context.add(Expr::Div(denominator, numerator));
    let (simplified_solution, _) = simplifier.simplify(solution);

    if simplifier.collect_steps() {
        let var_id = simplifier.context.var(var);
        steps.push(SolveStep {
            description: "Take reciprocal".to_string(),
            equation_after: Equation {
                lhs: var_id,
                rhs: simplified_solution,
                op: RelOp::Eq,
            },
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        });
    }

    // Build solution set - may need guard for numerator ≠ 0
    use crate::domain::Proof;
    use crate::helpers::prove_nonzero;
    use cas_ast::{Case, ConditionPredicate, ConditionSet, SolutionSet};

    // Simplify the numerator for cleaner display and proof checking
    let (simplified_numerator, _) = simplifier.simplify(numerator);

    // Check if numerator is provably non-zero (e.g., literal 2 in "1/x = 2")
    // If so, we can return a simple Discrete solution without conditional guard
    let solution_set = if prove_nonzero(&simplifier.context, simplified_numerator) == Proof::Proven
    {
        // Trivially satisfied - no guard needed
        SolutionSet::Discrete(vec![simplified_solution])
    } else {
        // Need conditional guard: numerator ≠ 0
        let guard = ConditionSet::single(ConditionPredicate::NonZero(simplified_numerator));
        let case = Case::new(guard, SolutionSet::Discrete(vec![simplified_solution]));
        SolutionSet::Conditional(vec![case])
    };

    Some((solution_set, steps))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Context;

    #[test]
    fn test_is_simple_reciprocal() {
        let mut ctx = Context::new();
        let r = ctx.var("R");
        let one = ctx.num(1);
        let reciprocal = ctx.add(Expr::Div(one, r));

        assert!(is_simple_reciprocal(&ctx, reciprocal, "R"));
        assert!(!is_simple_reciprocal(&ctx, reciprocal, "X"));
        assert!(!is_simple_reciprocal(&ctx, r, "R"));
    }

    #[test]
    fn test_combine_fractions_simple() {
        let mut ctx = Context::new();
        let r1 = ctx.var("R1");
        let r2 = ctx.var("R2");
        let one = ctx.num(1);

        // 1/R1 + 1/R2
        let frac1 = ctx.add(Expr::Div(one, r1));
        let one2 = ctx.num(1);
        let frac2 = ctx.add(Expr::Div(one2, r2));
        let sum = ctx.add(Expr::Add(frac1, frac2));

        let result = combine_fractions_deterministic(&mut ctx, sum);
        assert!(result.is_some());

        let (num, denom) = result.unwrap();
        // Numerator should contain R1 and R2
        // Denominator should be R1*R2
        assert!(contains_var(&ctx, num, "R1") || contains_var(&ctx, num, "R2"));
        assert!(contains_var(&ctx, denom, "R1"));
        assert!(contains_var(&ctx, denom, "R2"));
    }
}
