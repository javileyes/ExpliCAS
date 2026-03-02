// Compatibility wrapper: migrated to `cas_solver`.
pub use cas_engine::*;
extern crate self as cas_solver;

#[path = "../../cas_solver/tests/assumption_kind_display_tests.rs"]
mod assumption_kind_display_tests;
