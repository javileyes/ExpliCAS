//! Compatibility wrapper.
//!
//! Canonical rationalization semantic tests live in `cas_solver`.

pub use cas_engine::Simplifier;
extern crate self as cas_solver;

#[path = "../../cas_solver/tests/rationalize_semantic_tests.rs"]
mod solver_rationalize_semantic_tests;
