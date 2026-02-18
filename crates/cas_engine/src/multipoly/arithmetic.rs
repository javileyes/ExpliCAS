//! Compatibility re-export for multipoly arithmetic.
//!
//! Keep budget-related terms (`PassStats`, `terms_materialized`, `poly_ops`, `PolyBudget`)
//! visible in this hotspot path for budget enforcement lints.

#[allow(unused_imports)]
pub use cas_math::multipoly::arithmetic::*;
