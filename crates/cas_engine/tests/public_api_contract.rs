// Compatibility wrapper: migrated to `cas_solver`.
extern crate cas_engine as cas_solver;

#[path = "../../cas_solver/tests/public_api_contract.rs"]
mod public_api_contract;
