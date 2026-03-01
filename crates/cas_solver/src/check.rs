//! Solution verification helpers exposed by `cas_solver`.
//!
//! Delegates to `cas_engine::api` to keep behavior in one canonical
//! implementation while consuming only stable engine surface.

use cas_ast::{Equation, ExprId, SolutionSet};
use cas_engine::Simplifier;

pub use cas_engine::api::{VerifyResult, VerifyStatus, VerifySummary};

/// Verify a single solution by substituting into the equation.
pub fn verify_solution(
    simplifier: &mut Simplifier,
    equation: &Equation,
    var: &str,
    solution: ExprId,
) -> VerifyStatus {
    cas_engine::api::verify_solution(simplifier, equation, var, solution)
}

/// Verify a solution set, handling all [`SolutionSet`] variants.
pub fn verify_solution_set(
    simplifier: &mut Simplifier,
    equation: &Equation,
    var: &str,
    solutions: &SolutionSet,
) -> VerifyResult {
    cas_engine::api::verify_solution_set(simplifier, equation, var, solutions)
}
