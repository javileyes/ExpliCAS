//! Compatibility wrapper.
//!
//! Canonical golden corpus tests live in `cas_solver`.
//! Keep this shim so legacy commands targeting `cas_engine` still work.

pub use cas_engine::api::solve;
pub use cas_engine::{Engine, Simplifier};
extern crate self as cas_solver;

#[path = "../../cas_solver/tests/golden_corpus_tests.rs"]
mod solver_golden_corpus_tests;
