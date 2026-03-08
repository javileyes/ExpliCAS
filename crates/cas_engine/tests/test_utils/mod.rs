// Compatibility shim: migrated helpers now live in `cas_solver/tests/test_utils`.
use crate as cas_solver;

#[path = "../../../cas_solver/tests/test_utils/mod.rs"]
mod migrated;
pub use migrated::*;
