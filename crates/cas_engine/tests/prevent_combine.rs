//! Compatibility wrapper.
//!
//! Canonical prevent_combine lives in `cas_solver`.

extern crate cas_engine as cas_solver;

#[path = "../../cas_solver/tests/prevent_combine.rs"]
mod solver_prevent_combine;
