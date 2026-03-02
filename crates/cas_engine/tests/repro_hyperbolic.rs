//! Compatibility wrapper.
//!
//! Canonical repro_hyperbolic lives in `cas_solver`.

pub use cas_engine::*;
extern crate self as cas_solver;

#[path = "../../cas_solver/tests/repro_hyperbolic.rs"]
mod solver_repro_hyperbolic;
