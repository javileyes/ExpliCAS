//! Canonical JSON API types for engine responses.
//!
//! This module provides stable, serializable types for CLI and FFI consumers.
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

mod eval;
mod substitute;

pub use cas_api_models::SCHEMA_VERSION;
pub use eval::eval_str_to_json;
pub use substitute::substitute_str_to_json;
