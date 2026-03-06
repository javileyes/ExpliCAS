//! Proof and verification entrypoints.

use crate::{Proof, Simplifier, ValueDomain, VerifyResult, VerifyStatus};

/// Result shape for equation-level additive cancellation.
pub type CancelResult = cas_solver_core::cancel_common_terms::CancelResult;

/// Equation-level additive cancellation primitives owned by `cas_solver`.
pub use crate::cancel_runtime::cancel_additive_terms_semantic;
pub use cas_solver_core::cancel_common_terms::cancel_common_additive_terms;

/// Attempt to prove that an expression is non-zero.
pub fn prove_nonzero(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> Proof {
    crate::proof_runtime::prove_nonzero(ctx, expr)
}

/// Attempt to prove that an expression is strictly positive.
pub fn prove_positive(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
    value_domain: ValueDomain,
) -> Proof {
    crate::proof_runtime::prove_positive(ctx, expr, value_domain)
}

/// Verify a single solution by substituting into the equation.
pub fn verify_solution(
    simplifier: &mut Simplifier,
    equation: &cas_ast::Equation,
    var: &str,
    solution: cas_ast::ExprId,
) -> VerifyStatus {
    cas_solver_core::verification_runtime_flow::verify_solution_with_runtime_kernels_with_state(
        simplifier,
        equation,
        var,
        solution,
        |state| &state.context,
        |state| &mut state.context,
        |state, expr, opts| state.simplify_with_stats(expr, opts).0,
        crate::proof_runtime::ground_eval_candidate,
    )
}

/// Verify an entire solution set against the source equation.
pub fn verify_solution_set(
    simplifier: &mut Simplifier,
    equation: &cas_ast::Equation,
    var: &str,
    solutions: &cas_ast::SolutionSet,
) -> VerifyResult {
    cas_solver_core::verification_runtime_flow::verify_solution_set_with_runtime_kernels_with_state(
        simplifier,
        equation,
        var,
        solutions,
        |state| &state.context,
        |state| &mut state.context,
        |state, expr, opts| state.simplify_with_stats(expr, opts).0,
        crate::proof_runtime::ground_eval_candidate,
    )
}
