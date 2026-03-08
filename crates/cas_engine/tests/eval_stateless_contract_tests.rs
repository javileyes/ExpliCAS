// Compatibility wrapper: migrated to `cas_solver`.
pub use cas_engine::*;
extern crate self as cas_solver;

#[path = "../../cas_solver/tests/eval_stateless_contract_tests.rs"]
mod eval_stateless_contract_tests;
