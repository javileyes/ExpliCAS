//! Compatibility wrapper.
//!
//! Canonical pipeline tests live in `cas_solver`.

pub use cas_engine::{Orchestrator, Simplifier};
extern crate self as cas_solver;

#[path = "../../cas_solver/tests/pipeline_tests.rs"]
mod solver_pipeline_tests;
