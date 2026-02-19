//! # Helpers Module
//!
//! Shared utility functions for expression manipulation and pattern matching.
//! Used across multiple rule files to avoid code duplication.
//!
//! ## Organization
//!
//! - **Core utilities** (generic): `destructure`, `extraction`,
//!   `complexity`, `pi`, `predicates`
//! - **Trig helpers** (re-exported from `rules::trigonometry`): `trig_matchers`
//! - **Flatten/roots**: `trig_roots_flatten`
//! - **Scoring**: `nf_scoring`

mod complexity;
mod destructure;
pub(crate) mod ground_eval;
mod nf_scoring;
mod predicates;
mod trig_roots_flatten;

// Re-export all items for internal use
pub(crate) use cas_math::expr_extract::{
    extract_i64_integer as get_integer, extract_integer_exact as get_integer_exact,
};
pub(crate) use cas_math::pi_helpers::*;
pub(crate) use complexity::*;
pub(crate) use destructure::*;
pub(crate) use nf_scoring::*;
// predicates has `is_zero` and `prove_nonzero` used by integration tests — keep pub
pub use predicates::*;
// trig_matchers moved to rules/trigonometry/trig_matchers.rs — re-export for backwards compat
pub(crate) use crate::rules::trigonometry::trig_matchers::*;
pub(crate) use trig_roots_flatten::*;

#[cfg(test)]
mod tests;
