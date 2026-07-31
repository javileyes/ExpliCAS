//! CLI contract tests for the unified budget system and eval command.
//!
//! These tests validate the CLI behavior including:
//! - Help output shows correct commands
//! - wire output parsing and schema version
//! - Budget presets and strict mode
//!
//! The cases live in `cli_contract_tests/`, one submodule per mathematical
//! domain. This file stays the single test binary: it holds the shared
//! imports and the `cli()` helper, which submodules pick up via
//! `use super::*`. Adding a case means editing its domain submodule, not
//! this file. Note that test paths are now module-qualified, so an exact
//! filter needs the prefix (`--exact solving::test_eval_...`).

use assert_cmd::cargo;
use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::Value;

/// Get the CLI command
fn cli() -> Command {
    Command::new(cargo::cargo_bin!("cas_cli"))
}

mod absolute_value;
mod cli_wire;
mod complex;
mod core_simplification;
mod differentiation;
mod integration;
mod limits;
mod linear_algebra;
mod number_theory;
mod odes;
mod radicals_powers;
mod series_sums;
mod solving;
mod trigonometry;
mod vector_calculus;
