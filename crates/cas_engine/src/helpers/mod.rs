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

// Re-export all public items for backward compatibility
pub use complexity::*;
pub use destructure::*;
pub use nf_scoring::*;
pub use numeric::*;
pub use numeric_eval::*;
pub use pi::*;
pub use predicates::*;
pub use solver_domain::*;
pub use trig_matchers::*;
pub use trig_roots_flatten::*;

#[cfg(test)]
mod tests;
