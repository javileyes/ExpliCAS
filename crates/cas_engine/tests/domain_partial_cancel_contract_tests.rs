//! Compatibility wrapper.
//!
//! Canonical domain_partial_cancel_contract_tests lives in `cas_solver`.

extern crate cas_engine as cas_solver;

#[path = "../../cas_solver/tests/domain_partial_cancel_contract_tests.rs"]
mod solver_domain_partial_cancel_contract_tests;
