// Compatibility wrapper: migrated to `cas_solver`.
extern crate cas_engine as cas_solver;

#[path = "../../cas_solver/tests/regression_distribute_loop.rs"]
mod regression_distribute_loop;
