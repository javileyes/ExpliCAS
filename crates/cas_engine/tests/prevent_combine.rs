//! Compatibility wrapper.
//!
//! Canonical prevent_combine lives in `cas_solver`.

pub use cas_engine::*;
extern crate self as cas_solver;

#[path = "../../cas_solver/tests/prevent_combine.rs"]
mod solver_prevent_combine;
