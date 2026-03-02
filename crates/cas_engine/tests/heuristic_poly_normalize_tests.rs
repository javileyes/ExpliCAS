//! Compatibility wrapper.
//!
//! Canonical heuristic_poly_normalize_tests lives in `cas_solver`.

pub use cas_engine::*;
extern crate self as cas_solver;

#[path = "../../cas_solver/tests/heuristic_poly_normalize_tests.rs"]
mod solver_heuristic_poly_normalize_tests;
