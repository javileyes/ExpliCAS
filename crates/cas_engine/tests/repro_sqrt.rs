//! Compatibility wrapper.
//!
//! Canonical repro_sqrt lives in `cas_solver`.

extern crate cas_engine as cas_solver;

#[path = "../../cas_solver/tests/repro_sqrt.rs"]
mod solver_repro_sqrt;
