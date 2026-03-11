//! Compatibility wrapper.
//!
//! Canonical phase-based tests live in `cas_solver`.

pub use cas_engine::*;
extern crate cas_engine as cas_solver;

#[path = "../../cas_solver/tests/phase_based_tests.rs"]
mod solver_phase_based_tests;
