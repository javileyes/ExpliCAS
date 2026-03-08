//! Compatibility wrapper.
//!
//! Canonical profiler_tests lives in `cas_solver`.

pub use cas_engine::*;
extern crate self as cas_solver;

#[path = "../../cas_solver/tests/profiler_tests.rs"]
mod solver_profiler_tests;
