//! High-level solver facade entrypoint exports.

pub use crate::solver_entrypoints_eval::{expand_with_stats, fold_constants, limit, LimitResult};
pub use crate::solver_entrypoints_proof_verify::{
    cancel_additive_terms_semantic, cancel_common_additive_terms, prove_nonzero, prove_positive,
    verify_solution, verify_solution_set, CancelResult,
};
/// Number-theory helpers exposed by the solver facade without pulling engine rule modules.
pub mod number_theory {
    pub use crate::solver_number_theory::{compute_gcd, explain_gcd, GcdResult};
}
