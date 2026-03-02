//! Compatibility wrapper.
//!
//! Canonical rationalize+autoexpand tests live in `cas_solver`.

pub use cas_engine::{Engine, EvalOptions, ExpandPolicy};
pub mod phase {
    pub use cas_engine::SharedSemanticConfig;
}
extern crate self as cas_solver;

#[path = "../../cas_solver/tests/rationalize_autoexpand_tests.rs"]
mod solver_rationalize_autoexpand_tests;
