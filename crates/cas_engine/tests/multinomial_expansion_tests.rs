// Compatibility wrapper: migrated to `cas_solver`.
pub use cas_engine::*;
extern crate self as cas_solver;

#[path = "../../cas_solver/tests/multinomial_expansion_tests.rs"]
mod multinomial_expansion_tests;
