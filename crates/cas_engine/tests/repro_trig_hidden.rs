//! Compatibility wrapper.
//!
//! Canonical repro_trig_hidden lives in `cas_solver`.

pub use cas_engine::*;
extern crate self as cas_solver;

#[path = "../../cas_solver/tests/repro_trig_hidden.rs"]
mod solver_repro_trig_hidden;
