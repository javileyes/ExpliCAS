// Compatibility wrapper: migrated to `cas_solver`.
pub use cas_engine::*;
extern crate self as cas_solver;

#[path = "../../cas_solver/tests/zippel_preset_contract_tests.rs"]
mod zippel_preset_contract_tests;
