//! Compatibility wrapper.
//!
//! Canonical repro_trig_hidden lives in `cas_solver`.

extern crate cas_engine as cas_solver;

#[path = "../../cas_solver/tests/repro_trig_hidden.rs"]
mod solver_repro_trig_hidden;
