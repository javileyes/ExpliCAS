//! Compatibility wrapper.
//!
//! Canonical expand_integration_test lives in `cas_solver`.

extern crate cas_engine as cas_solver;

#[path = "../../cas_solver/tests/expand_integration_test.rs"]
mod solver_expand_integration_test;
