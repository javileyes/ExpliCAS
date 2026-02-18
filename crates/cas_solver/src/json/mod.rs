//! Canonical JSON API entry points for solver responses.
//!
//! This module provides stable, serializable JSON entry points for CLI and FFI consumers.
//! All callsites should use these types to ensure consistent JSON schema.
//!
//! # Schema Version
//!
//! Current schema version: **1**
//!
//! # Stability Contract
//!
//! - `schema_version`, `ok`, `kind`, `code` are **stable** - do not change
//! - `message` is human-readable and may change between versions
//! - `details` is extensible (new keys may be added)

mod envelope;
mod eval;
mod mappers;
mod substitute;

pub use envelope::{eval_str_to_output_envelope, EnvelopeEvalOptions};
pub use eval::eval_str_to_json;
pub use substitute::substitute_str_to_json;
