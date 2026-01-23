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

use cas_ast::{Context, Expr, ExprId};
use num_traits::{One, Signed, ToPrimitive, Zero};

include!("helpers/01_destructure.rs");
include!("helpers/02_numeric.rs");
include!("helpers/03_trig_roots_flatten.rs");
include!("helpers/04_pi.rs");
include!("helpers/05_predicates.rs");
include!("helpers/06_solver_domain.rs");
include!("helpers/07_nf_scoring.rs");
include!("helpers/08_numeric_eval.rs");
include!("helpers/09_trig_matchers.rs");
include!("helpers/10_complexity.rs");

#[cfg(test)]
mod tests {
    include!("helpers/tests.rs");
}
