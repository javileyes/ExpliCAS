//! Proof and verification runtime entrypoints for the engine facade.

use crate::engine::Simplifier;
use cas_solver_core::verification::{VerifyResult, VerifyStatus};

pub(crate) fn verify_solution(
    simplifier: &mut Simplifier,
    equation: &cas_ast::Equation,
    var: &str,
    solution: cas_ast::ExprId,
) -> VerifyStatus {
    cas_solver_core::verification_runtime_bound_runtime::verify_solution_with_runtime_state_and_ground_eval_with_state(
        simplifier,
        equation,
        var,
        solution,
        crate::proof_runtime::ground_eval_candidate,
    )
}

pub(crate) fn verify_solution_set(
    simplifier: &mut Simplifier,
    equation: &cas_ast::Equation,
    var: &str,
    solutions: &cas_ast::SolutionSet,
) -> VerifyResult {
    cas_solver_core::verification_runtime_bound_runtime::verify_solution_set_with_runtime_state_and_ground_eval_with_state(
        simplifier,
        equation,
        var,
        solutions,
        crate::proof_runtime::ground_eval_candidate,
    )
}
