//! Compatibility wrapper.
//!
//! Canonical assumptions_contract_tests lives in `cas_solver`.

pub use cas_engine::*;
extern crate self as cas_solver;

#[path = "../../cas_solver/tests/assumptions_contract_tests.rs"]
mod solver_assumptions_contract_tests;
