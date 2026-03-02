// Compatibility shim: migrated strategy generators now live in `cas_solver/tests/strategies`.
#[path = "../../../cas_solver/tests/strategies/mod.rs"]
mod migrated;
pub use migrated::*;
