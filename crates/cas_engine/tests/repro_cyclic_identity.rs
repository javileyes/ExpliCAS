// Compatibility wrapper: migrated to `cas_solver`.
pub use cas_engine::*;
extern crate self as cas_solver;

#[path = "../../cas_solver/tests/repro_cyclic_identity.rs"]
mod repro_cyclic_identity;
