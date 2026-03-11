//! Compatibility wrapper.
//!
//! Canonical factor_integration_test lives in `cas_solver`.

extern crate cas_engine as cas_solver;

#[path = "../../cas_solver/tests/factor_integration_test.rs"]
mod solver_factor_integration_test;
