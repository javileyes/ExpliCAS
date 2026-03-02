// Compatibility wrapper: migrated to `cas_solver`.
pub use cas_engine::*;
extern crate self as cas_solver;

#[path = "../../cas_solver/tests/mul_zero_contract_tests.rs"]
mod mul_zero_contract_tests;
