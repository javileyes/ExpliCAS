//! Compatibility wrapper.
//!
//! Canonical factor_integration_test lives in `cas_solver`.

pub use cas_engine::*;
extern crate self as cas_solver;

#[path = "../../cas_solver/tests/factor_integration_test.rs"]
mod solver_factor_integration_test;
