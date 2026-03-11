//! Compatibility wrapper.
//!
//! Canonical semantics_contract_tests lives in `cas_solver`.

extern crate cas_engine as cas_solver;

#[path = "../../cas_solver/tests/semantics_contract_tests.rs"]
mod solver_semantics_contract_tests;
