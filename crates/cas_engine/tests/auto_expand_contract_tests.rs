//! Compatibility wrapper.
//!
//! Canonical auto-expand contract tests live in `cas_solver`.

pub use cas_engine::{ContextMode, ExpandPolicy, HeuristicPoly, Simplifier, SimplifyOptions};
extern crate self as cas_solver;

#[path = "../../cas_solver/tests/auto_expand_contract_tests.rs"]
mod solver_auto_expand_contract_tests;
