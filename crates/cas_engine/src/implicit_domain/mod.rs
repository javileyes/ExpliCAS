//! Implicit Domain Inference.
//!
//! This module infers domain constraints that are implicitly required by
//! expression structure. For example, `sqrt(x)` in RealOnly mode implies `x >= 0`.
//!
//! # Key Concepts
//!
//! - **Implicit Domain**: Constraints derived from expression structure, not assumptions.
//! - **Witness**: The subexpression that enforces a constraint (e.g., `sqrt(x)` for `x >= 0`).
//! - **Witness Survival**: A constraint is only valid if its witness survives in the output.
//!
//! # Usage
//!
//! ```ignore
//! let implicit = infer_implicit_domain(ctx, root, ValueDomain::RealOnly);
//! // Later, when checking if x >= 0 is valid:
//! if implicit.contains_nonnegative(x) && witness_survives(ctx, x, output, WitnessKind::Sqrt) {
//!     // Can use ProvenImplicit
//! }
//! ```

mod inference;

pub use cas_solver_core::domain_condition::{
    filter_requires_for_display, ImplicitCondition, ImplicitDomain, RequiresDisplayLevel,
};

// Re-export all public items from submodules.
pub use cas_solver_core::domain_assumption_classification::{
    classify_assumption, classify_assumptions_in_place,
};
pub use cas_solver_core::domain_context::DomainContext;
pub use cas_solver_core::domain_inference_counter::{
    get as infer_domain_calls_get, reset as infer_domain_calls_reset,
};
pub use cas_solver_core::domain_normalization::{
    normalize_and_dedupe_conditions, normalize_condition, normalize_condition_expr,
    render_conditions_normalized,
};
pub use cas_solver_core::domain_witness::{
    witness_survives, witness_survives_in_context, WitnessKind,
};
pub use inference::{
    check_analytic_expansion, derive_requires_from_equation, domain_delta_check,
    expands_analytic_domain, expands_analytic_in_context, infer_implicit_domain,
    AnalyticExpansionResult, DomainDelta,
};

#[cfg(test)]
#[path = "witness/tests.rs"]
mod witness_tests;
