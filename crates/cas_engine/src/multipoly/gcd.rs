//! Compatibility re-export for multipoly GCD layers.
//!
//! Keep budget-related terms (`PassStats`, `terms_materialized`, `poly_ops`, `GcdBudget`)
//! visible in this hotspot path for budget enforcement lints.

#[allow(unused_imports)]
pub use cas_math::multipoly::gcd::*;
