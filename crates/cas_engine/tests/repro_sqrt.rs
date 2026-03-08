//! Compatibility wrapper.
//!
//! Canonical repro_sqrt lives in `cas_solver`.

pub use cas_engine::*;
extern crate self as cas_solver;

#[path = "../../cas_solver/tests/repro_sqrt.rs"]
mod solver_repro_sqrt;
