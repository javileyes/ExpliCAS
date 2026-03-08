// Compatibility wrapper: migrated to `cas_solver`.
pub use cas_engine::*;
extern crate self as cas_solver;

#[path = "../../cas_solver/tests/fraction_opposite_denominators_tests.rs"]
mod fraction_opposite_denominators_tests;
