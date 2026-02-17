//! # Helpers Module
//!
//! Shared utility functions for expression manipulation and pattern matching.
//! Used across multiple rule files to avoid code duplication.
//!
//! ## Organization
//!
//! - **Core utilities** (generic): `destructure`, `extraction`, `numeric`, `numeric_eval`,
//!   `complexity`, `pi`, `predicates`
//! - **Trig helpers** (re-exported from `rules::trigonometry`): `trig_matchers`
//! - **Flatten/roots**: `trig_roots_flatten`
//! - **Scoring**: `nf_scoring`

mod complexity;
mod destructure;
mod extraction;
pub(crate) mod ground_eval;
mod nf_scoring;
mod numeric;
mod numeric_eval;
mod pi;
mod predicates;
mod trig_roots_flatten;

// Re-export all items for internal use
pub(crate) use complexity::*;
pub(crate) use destructure::*;
pub(crate) use extraction::*;
pub(crate) use nf_scoring::*;
pub(crate) use numeric::*;
pub(crate) use numeric_eval::*;
// eval_f64_with_substitution is used by integration tests (metamorphic divisor safety guard)
pub use numeric_eval::eval_f64_with_substitution;
pub(crate) use pi::*;
// predicates has `is_zero` and `prove_nonzero` used by integration tests — keep pub
pub use predicates::*;
// trig_matchers moved to rules/trigonometry/trig_matchers.rs — re-export for backwards compat
pub(crate) use crate::rules::trigonometry::trig_matchers::*;
pub(crate) use trig_roots_flatten::*;

#[cfg(test)]
mod tests;
