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

include!("helpers/destructure.rs");
include!("helpers/numeric.rs");
include!("helpers/trig_roots_flatten.rs");
include!("helpers/pi.rs");
include!("helpers/predicates.rs");
include!("helpers/solver_domain.rs");
include!("helpers/nf_scoring.rs");
include!("helpers/numeric_eval.rs");
include!("helpers/trig_matchers.rs");
include!("helpers/complexity.rs");

#[cfg(test)]
mod tests {
    include!("helpers/tests.rs");
}
