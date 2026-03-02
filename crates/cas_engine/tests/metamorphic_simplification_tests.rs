// Compatibility wrapper: migrated to `cas_solver`.
pub use cas_engine::*;
extern crate self as cas_solver;

#[path = "../../cas_solver/tests/metamorphic_simplification_tests.rs"]
mod metamorphic_simplification_tests;
