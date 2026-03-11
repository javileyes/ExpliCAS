// Compatibility wrapper: migrated to `cas_solver`.
extern crate cas_engine as cas_solver;

#[path = "../../cas_solver/tests/stack_overflow_investigation.rs"]
mod stack_overflow_investigation;
