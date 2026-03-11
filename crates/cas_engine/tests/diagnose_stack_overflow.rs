// Compatibility wrapper: migrated to `cas_solver`.
extern crate cas_engine as cas_solver;

#[path = "../../cas_solver/tests/diagnose_stack_overflow.rs"]
mod diagnose_stack_overflow;
