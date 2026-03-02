// Compatibility wrapper: migrated to `cas_solver`.
pub use cas_engine::*;
extern crate self as cas_solver;

#[path = "../../cas_solver/tests/stack_overflow_investigation.rs"]
mod stack_overflow_investigation;
