//! Wire model adapter for REPL output.
//!
//! Canonical wire DTOs live in `cas_api_models::wire`. This module keeps only
//! REPL-specific conversions.

mod convert;

pub use cas_api_models::wire::{WireKind, WireMsg, WireReply, WireSpan, SCHEMA_VERSION};
pub use convert::wire_reply_from_repl;

#[cfg(test)]
mod tests;
