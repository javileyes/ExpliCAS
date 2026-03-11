// Compatibility wrapper: migrated to `cas_solver`.
extern crate cas_engine as cas_solver;

#[path = "../../cas_solver/tests/stress_test.rs"]
mod stress_test;
