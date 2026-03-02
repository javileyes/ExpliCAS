// Compatibility wrapper: migrated to `cas_solver`.
pub use cas_engine::*;
extern crate self as cas_solver;

#[path = "../../cas_solver/tests/multivar_gcd_tests.rs"]
mod multivar_gcd_tests;
