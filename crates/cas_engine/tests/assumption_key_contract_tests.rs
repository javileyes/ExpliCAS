//! Compatibility wrapper.
//!
//! Canonical assumption_key_contract_tests lives in `cas_solver`.

pub use cas_engine::*;
extern crate self as cas_solver;

#[path = "../../cas_solver/tests/assumption_key_contract_tests.rs"]
mod solver_assumption_key_contract_tests;
