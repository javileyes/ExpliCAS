//! Compatibility wrapper.
//!
//! Canonical prefer-division tests live in `cas_solver`.

pub use cas_engine::Simplifier;
extern crate self as cas_solver;

#[path = "../../cas_solver/tests/prefer_division_tests.rs"]
mod solver_prefer_division_tests;
