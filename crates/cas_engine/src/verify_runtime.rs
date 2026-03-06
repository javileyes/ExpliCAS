//! Solution verification runtime helpers.

use crate::engine::Simplifier;
use cas_ast::{Equation, ExprId, SolutionSet};
use cas_solver_core::verification::{VerifyResult, VerifyStatus};

/// Verify a single solution by substituting into the equation.
pub(crate) fn verify_solution(
    simplifier: &mut Simplifier,
    equation: &Equation,
    var: &str,
    solution: ExprId,
) -> VerifyStatus {
    cas_solver_core::verification_runtime_flow::verify_solution_with_runtime_kernels_with_state(
        simplifier,
        equation,
        var,
        solution,
        |state| &state.context,
        |state| &mut state.context,
        |state, expr, opts| state.simplify_with_stats(expr, opts).0,
        crate::runtime_ground_eval::ground_eval_candidate,
    )
}

/// Verify a solution set, handling all [`SolutionSet`] variants.
pub(crate) fn verify_solution_set(
    simplifier: &mut Simplifier,
    equation: &Equation,
    var: &str,
    solutions: &SolutionSet,
) -> VerifyResult {
    cas_solver_core::verification_runtime_flow::verify_solution_set_with_runtime_kernels_with_state(
        simplifier,
        equation,
        var,
        solutions,
        |state| &state.context,
        |state| &mut state.context,
        |state, expr, opts| state.simplify_with_stats(expr, opts).0,
        crate::runtime_ground_eval::ground_eval_candidate,
    )
}
