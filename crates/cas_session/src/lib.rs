//! Session-related components extracted from `cas_engine`.

pub mod env;

pub use env::{is_reserved, substitute, substitute_with_shadow, Environment};
