// Compatibility wrapper: migrated to `cas_solver`.
pub use cas_engine::*;
extern crate self as cas_solver;

#[path = "../../cas_solver/tests/conditional_flatten_tests.rs"]
mod conditional_flatten_tests;
