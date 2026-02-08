//! # Helpers Module
//!
//! This module provides shared utility functions for expression manipulation
//! and pattern matching. These functions are used across multiple rule files
//! to avoid code duplication.
//!
//! ## Categories
//!
//! - **Expression Predicates**: `is_one`, `is_zero`, `is_negative`, `is_half`
//! - **Value Extraction**: `get_integer`, `get_parts`, `get_variant_name`
//! - **Flattening**: `flatten_add`, `flatten_add_sub_chain`, `flatten_mul`, `flatten_mul_chain`
//! - **Trigonometry**: `is_trig_pow`, `get_trig_arg`, `extract_double_angle_arg`
//! - **Pi Helpers**: `is_pi`, `is_pi_over_n`, `build_pi_over_n`
//! - **Roots**: `get_square_root`

mod complexity;
mod destructure;
mod nf_scoring;
mod numeric;
mod numeric_eval;
mod pi;
mod predicates;
mod solver_domain;
mod trig_matchers;
mod trig_roots_flatten;

// Re-export all items for internal use
pub(crate) use complexity::*;
pub(crate) use destructure::*;
pub(crate) use nf_scoring::*;
pub(crate) use numeric::*;
pub(crate) use numeric_eval::*;
pub(crate) use pi::*;
// predicates has `is_zero` and `prove_nonzero` used by integration tests — keep pub
pub use predicates::*;
pub(crate) use solver_domain::*;
pub(crate) use trig_matchers::*;
pub(crate) use trig_roots_flatten::*;

#[cfg(test)]
mod tests;
