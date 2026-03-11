//! Compatibility wrapper.
//!
//! Canonical rationalize+autoexpand tests live in `cas_solver`.

pub use cas_engine::{Engine, EvalOptions, ExpandPolicy, SharedSemanticConfig};
pub mod phase {
    pub use cas_engine::SharedSemanticConfig;
}
extern crate cas_engine as cas_solver;

#[path = "../../cas_solver/tests/rationalize_autoexpand_tests.rs"]
mod solver_rationalize_autoexpand_tests;
