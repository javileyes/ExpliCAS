//! Solution verification runtime helpers.

use crate::engine::Simplifier;
use cas_ast::{Equation, ExprId, SolutionSet};
use cas_solver_core::verification::{VerifyResult, VerifyStatus};

fn context_ref(simplifier: &Simplifier) -> &cas_ast::Context {
    &simplifier.context
}

fn context_mut(simplifier: &mut Simplifier) -> &mut cas_ast::Context {
    &mut simplifier.context
}

fn simplify_with_options(
    simplifier: &mut Simplifier,
    expr: ExprId,
    opts: cas_solver_core::simplify_options::SimplifyOptions,
) -> ExprId {
    simplifier.simplify_with_stats(expr, opts).0
}

/// Verify a single solution by substituting into the equation.
pub fn verify_solution(
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
        context_ref,
        context_mut,
        simplify_with_options,
        crate::runtime_ground_eval::ground_eval_candidate,
    )
}

/// Verify a solution set, handling all [`SolutionSet`] variants.
pub fn verify_solution_set(
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
        context_ref,
        context_mut,
        simplify_with_options,
        crate::runtime_ground_eval::ground_eval_candidate,
    )
}
