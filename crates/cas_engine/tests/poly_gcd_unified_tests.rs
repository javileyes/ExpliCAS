// Compatibility wrapper: migrated to `cas_solver`.
pub use cas_engine::*;
extern crate self as cas_solver;

#[path = "../../cas_solver/tests/poly_gcd_unified_tests.rs"]
mod poly_gcd_unified_tests;
