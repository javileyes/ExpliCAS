// Compatibility wrapper: migrated to `cas_solver`.
pub use cas_engine::*;
extern crate self as cas_solver;

#[path = "../../cas_solver/tests/inv_trig_contract_tests.rs"]
mod inv_trig_contract_tests;
