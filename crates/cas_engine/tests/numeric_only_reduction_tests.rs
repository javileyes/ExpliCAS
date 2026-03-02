// Compatibility wrapper: migrated to `cas_solver`.
pub use cas_engine::*;
extern crate self as cas_solver;

#[path = "../../cas_solver/tests/numeric_only_reduction_tests.rs"]
mod numeric_only_reduction_tests;
