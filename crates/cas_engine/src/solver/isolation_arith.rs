//! Arithmetic isolation facade.
//!
//! Re-exports operation-specific isolation handlers to keep call sites stable
//! while splitting implementation by operation family.

pub(super) use super::isolation_arith_add_sub::{isolate_add, isolate_sub};
pub(super) use super::isolation_arith_mul_div::{isolate_div, isolate_mul};
