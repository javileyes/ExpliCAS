//! # Helpers Module
//!
//! Shared utility functions for expression manipulation and pattern matching.
//! Used across multiple rule files to avoid code duplication.
//!
//! ## Organization
//!
//! - **Core utilities** (generic): `destructure`, `complexity`, `pi`, `predicates`
//! - **Flatten/roots**: `cas_math::trig_roots_flatten`
//! - **Scoring**: `nf_scoring`

pub(crate) mod ground_eval;
mod predicates;

// predicates has `is_zero` and `prove_nonzero` used by integration tests — keep pub
pub use predicates::*;

#[cfg(test)]
mod tests;
