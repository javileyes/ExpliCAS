//! Compatibility wrapper.
//!
//! Canonical domain_assume_warnings_contract_tests lives in `cas_solver`.

pub use cas_engine::*;
extern crate self as cas_solver;

#[path = "../../cas_solver/tests/domain_assume_warnings_contract_tests.rs"]
mod solver_domain_assume_warnings_contract_tests;
