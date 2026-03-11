//! Compatibility wrapper.
//!
//! Canonical heuristic_poly_normalize_tests lives in `cas_solver`.

extern crate cas_engine as cas_solver;

#[path = "../../cas_solver/tests/heuristic_poly_normalize_tests.rs"]
mod solver_heuristic_poly_normalize_tests;
