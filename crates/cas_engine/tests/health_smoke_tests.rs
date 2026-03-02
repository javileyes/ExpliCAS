//! Compatibility wrapper.
//!
//! Canonical health smoke tests live in `cas_solver`.

pub use cas_engine::{PipelineStats, Simplifier, SimplifyOptions};
extern crate self as cas_solver;

#[path = "../../cas_solver/tests/health_smoke_tests.rs"]
mod solver_health_smoke_tests;
