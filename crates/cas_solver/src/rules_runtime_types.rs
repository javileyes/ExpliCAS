//! Local passthrough for engine rules module exports.
//!
//! This keeps public compatibility while making the dependency surface explicit
//! and easy to replace during later migration iterations.

pub use crate::engine_bridge::rules;
