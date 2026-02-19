//! # Helpers Module
//!
//! Shared utility functions for expression manipulation and pattern matching.
//! Used across multiple rule files to avoid code duplication.
//!
//! ## Organization
//!
//! - **Core utilities** (generic): `destructure`, `complexity`, `pi`, `predicates`
//! - **Trig helpers** (re-exported from `rules::trigonometry`): `trig_matchers`
//! - **Flatten/roots**: `cas_math::expr_trig_roots_flatten`
//! - **Scoring**: `nf_scoring`

pub(crate) mod ground_eval;
mod predicates;

// Re-export all items for internal use
pub(crate) use cas_math::expr_complexity::*;
pub(crate) use cas_math::expr_destructure::*;
pub(crate) use cas_math::expr_extract::{
    extract_i64_integer as get_integer, extract_integer_exact as get_integer_exact,
};
pub(crate) use cas_math::expr_nf_scoring::*;
pub(crate) use cas_math::expr_predicates::is_one_expr as is_one;
pub use cas_math::expr_predicates::is_zero_expr as is_zero;
pub(crate) use cas_math::expr_trig_roots_flatten::*;
pub(crate) use cas_math::pi_helpers::*;
// predicates has `is_zero` and `prove_nonzero` used by integration tests — keep pub
pub use predicates::*;
// trig_matchers moved to rules/trigonometry/trig_matchers.rs — re-export for backwards compat
pub(crate) use crate::rules::trigonometry::trig_matchers::*;

#[cfg(test)]
mod tests;
