//! Compatibility wrapper.
//!
//! Canonical numeric-only diagnostic tests live in `cas_solver`.

pub use cas_engine::*;
extern crate self as cas_solver;

#[path = "../../cas_solver/tests/numeric_only_diagnostic.rs"]
mod solver_numeric_only_diagnostic_tests;
