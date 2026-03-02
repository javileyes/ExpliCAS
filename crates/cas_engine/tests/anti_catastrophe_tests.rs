// Compatibility wrapper: migrated to `cas_solver`.
pub use cas_engine::*;
extern crate self as cas_solver;

#[path = "../../cas_solver/tests/anti_catastrophe_tests.rs"]
mod anti_catastrophe_tests;
