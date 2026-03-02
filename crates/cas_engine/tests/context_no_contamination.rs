// Compatibility wrapper: migrated to `cas_solver`.
pub use cas_engine::*;
extern crate self as cas_solver;

#[path = "../../cas_solver/tests/context_no_contamination.rs"]
mod context_no_contamination;
