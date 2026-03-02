// Compatibility wrapper: migrated to `cas_solver`.
pub use cas_engine::*;
extern crate self as cas_solver;

#[path = "../../cas_solver/tests/const_fold_contract_tests.rs"]
mod const_fold_contract_tests;
