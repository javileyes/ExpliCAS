//! Shared verification wrappers bound to [`RuntimeSolveAdapterState`].

use crate::verification::{VerifyResult, VerifyStatus};
use cas_ast::{Equation, ExprId};

/// Verify one candidate solution using the same state type as both runtime
/// adapter and proof-simplifier factory.
pub fn verify_solution_with_runtime_state_and_runtime_proof_simplifier_with_state<S>(
    state: &mut S,
    equation: &Equation,
    var: &str,
    solution: ExprId,
) -> VerifyStatus
where
    S: crate::proof_runtime_bound_runtime::RuntimeProofSimplifierFactory
        + crate::solve_runtime_adapter_state_runtime::RuntimeSolveAdapterState,
{
    verify_solution_with_runtime_state_and_ground_eval_with_state(
        state,
        equation,
        var,
        solution,
        crate::proof_runtime_bound_runtime::ground_eval_candidate_with_runtime_proof_simplifier::<S>,
    )
}

/// Verify one candidate solution using the runtime-state simplifier contract.
pub fn verify_solution_with_runtime_state_and_ground_eval_with_state<S, FGroundEvalCandidate>(
    state: &mut S,
    equation: &Equation,
    var: &str,
    solution: ExprId,
    ground_eval_candidate: FGroundEvalCandidate,
) -> VerifyStatus
where
    S: crate::solve_runtime_adapter_state_runtime::RuntimeSolveAdapterState,
    FGroundEvalCandidate: Fn(
            &cas_ast::Context,
            ExprId,
            &crate::simplify_options::SimplifyOptions,
        ) -> Option<(cas_ast::Context, ExprId)>
        + Copy,
{
    crate::verification_runtime_flow::verify_solution_with_runtime_kernels_with_state(
        state,
        equation,
        var,
        solution,
        |s| s.runtime_context(),
        |s| s.runtime_context_mut(),
        |s, expr, opts| s.runtime_simplify_with_options_expr(expr, opts),
        ground_eval_candidate,
    )
}

/// Verify a full solution set using the same state type as both runtime
/// adapter and proof-simplifier factory.
pub fn verify_solution_set_with_runtime_state_and_runtime_proof_simplifier_with_state<S>(
    state: &mut S,
    equation: &Equation,
    var: &str,
    solutions: &cas_ast::SolutionSet,
) -> VerifyResult
where
    S: crate::proof_runtime_bound_runtime::RuntimeProofSimplifierFactory
        + crate::solve_runtime_adapter_state_runtime::RuntimeSolveAdapterState,
{
    verify_solution_set_with_runtime_state_and_ground_eval_with_state(
        state,
        equation,
        var,
        solutions,
        crate::proof_runtime_bound_runtime::ground_eval_candidate_with_runtime_proof_simplifier::<S>,
    )
}

/// Verify a full solution set using the runtime-state simplifier contract.
pub fn verify_solution_set_with_runtime_state_and_ground_eval_with_state<S, FGroundEvalCandidate>(
    state: &mut S,
    equation: &Equation,
    var: &str,
    solutions: &cas_ast::SolutionSet,
    ground_eval_candidate: FGroundEvalCandidate,
) -> VerifyResult
where
    S: crate::solve_runtime_adapter_state_runtime::RuntimeSolveAdapterState,
    FGroundEvalCandidate: Fn(
            &cas_ast::Context,
            ExprId,
            &crate::simplify_options::SimplifyOptions,
        ) -> Option<(cas_ast::Context, ExprId)>
        + Copy,
{
    crate::verification_runtime_flow::verify_solution_set_with_runtime_kernels_with_state(
        state,
        equation,
        var,
        solutions,
        |s| s.runtime_context(),
        |s| s.runtime_context_mut(),
        |s, expr, opts| s.runtime_simplify_with_options_expr(expr, opts),
        ground_eval_candidate,
    )
}
