//! Compatibility wrapper.
//!
//! Canonical assumption_key_contract_tests lives in `cas_solver`.

extern crate cas_engine as cas_solver;

#[path = "../../cas_solver/tests/assumption_key_contract_tests.rs"]
mod solver_assumption_key_contract_tests;
