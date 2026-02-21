//! Solver utility functions: verification, substitution, domain guards.

use cas_ast::ExprId;

use crate::engine::Simplifier;

/// Verify a candidate solution by substitution into the original equation.
pub(crate) fn verify_solution(
    eq: &cas_ast::Equation,
    var: &str,
    sol: ExprId,
    simplifier: &mut Simplifier,
) -> bool {
    // 1. Substitute candidate into equation sides
    let (lhs_sub, rhs_sub) = cas_solver_core::verify_substitution::substitute_equation_sides(
        &mut simplifier.context,
        eq,
        var,
        sol,
    );

    // 2. Simplify
    let (lhs_sim, _) = simplifier.simplify(lhs_sub);
    let (rhs_sim, _) = simplifier.simplify(rhs_sub);

    // 3. Check equality
    simplifier.are_equivalent(lhs_sim, rhs_sim)
}
