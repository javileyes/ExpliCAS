//! Solution verification module.
//!
//! Verifies solver solutions by substituting back into the original equation
//! and checking if the result simplifies to zero.
//!
//! Uses a 2-phase approach:
//! - Phase 1 (Strict): domain-honest simplification.
//! - Phase 2 (Generic fallback): only when strict residual is variable-free.

use cas_ast::{Equation, ExprId, SolutionSet};
use cas_math::expr_predicates::contains_variable;
use cas_solver_core::isolation_utils::is_numeric_zero;
use cas_solver_core::verification_flow::{
    verify_solution_set_for_equation_with_state, verify_solution_with_domain_modes_with_state,
};
use cas_solver_core::verify_substitution::substitute_equation_diff;

use crate::engine::Simplifier;
use crate::solver::check_helpers::{fold_numeric_islands, simplify_options_for_domain};
use crate::solver::simplifier_render_expr;

pub use cas_solver_core::verification::{VerifyResult, VerifyStatus, VerifySummary};

/// Verify a single solution by substituting into the equation.
pub fn verify_solution(
    simplifier: &mut Simplifier,
    equation: &Equation,
    var: &str,
    solution: ExprId,
) -> VerifyStatus {
    verify_solution_with_domain_modes_with_state(
        simplifier,
        equation,
        var,
        solution,
        |state, eq, solve_var, candidate| {
            substitute_equation_diff(&mut state.context, eq, solve_var, candidate)
        },
        |state, expr, domain_mode| {
            let opts = simplify_options_for_domain(domain_mode);
            state.simplify_with_stats(expr, opts).0
        },
        |state, expr| contains_variable(&state.context, expr),
        |state, expr| fold_numeric_islands(&mut state.context, expr),
        |state, expr| is_numeric_zero(&state.context, expr),
        simplifier_render_expr,
    )
}

/// Verify a solution set, handling all [`SolutionSet`] variants.
pub fn verify_solution_set(
    simplifier: &mut Simplifier,
    equation: &Equation,
    var: &str,
    solutions: &SolutionSet,
) -> VerifyResult {
    verify_solution_set_for_equation_with_state(
        simplifier,
        equation,
        var,
        solutions,
        verify_solution,
    )
}
