//! Solver utility functions: verification, substitution, domain guards.

use cas_ast::{Context, ExprId, SolutionSet};

use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::SolveStep;

/// Verify a candidate solution by substitution into the original equation.
pub(crate) fn verify_solution(
    eq: &cas_ast::Equation,
    var: &str,
    sol: ExprId,
    simplifier: &mut Simplifier,
) -> bool {
    // 1. Substitute
    let lhs_sub = cas_solver_core::substitution::substitute_named_var(
        &mut simplifier.context,
        eq.lhs,
        var,
        sol,
    );
    let rhs_sub = cas_solver_core::substitution::substitute_named_var(
        &mut simplifier.context,
        eq.rhs,
        var,
        sol,
    );

    // 2. Simplify
    let (lhs_sim, _) = simplifier.simplify(lhs_sub);
    let (rhs_sim, _) = simplifier.simplify(rhs_sub);

    // 3. Check equality
    simplifier.are_equivalent(lhs_sim, rhs_sim)
}

/// Check if an expression is "symbolic" (contains functions or variables).
/// Symbolic expressions cannot be verified by substitution because they don't
/// simplify to pure numbers. Examples: ln(c/d)/ln(a/b), x + a, sqrt(y)
pub(crate) fn is_symbolic_expr(ctx: &Context, expr: ExprId) -> bool {
    cas_solver_core::solve_analysis::is_symbolic_expr(ctx, expr)
}

/// V2.1 Issue #10: Extract all denominators from an expression that contain the given variable.
///
/// Used to detect domain restrictions when solving equations with fractions.
/// Returns a list of ExprIds that appear as denominators and contain the variable.
///
/// Example: `(x*y)/x` returns `[x]` (the denominator x contains var "x")
pub(crate) fn extract_denominators_with_var(ctx: &Context, expr: ExprId, var: &str) -> Vec<ExprId> {
    cas_solver_core::solve_analysis::extract_denominators_with_var(ctx, expr, var)
}

/// V2.1 Issue #10: Wrap a solve result with domain guards for denominators.
///
/// If there are domain exclusions (denominators that must be non-zero),
/// this wraps the result in a Conditional with NonZero guards.
pub(crate) fn wrap_with_domain_guards(
    result: Result<(SolutionSet, Vec<SolveStep>), CasError>,
    exclusions: &[ExprId],
    _simplifier: &mut Simplifier,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    // If no exclusions, return as-is
    if exclusions.is_empty() {
        return result;
    }

    let (solution_set, steps) = result?;
    let guarded =
        cas_solver_core::solve_analysis::apply_nonzero_exclusion_guards(solution_set, exclusions);
    Ok((guarded, steps))
}
