//! Substitute API with optional step rendering for upper layers (CLI/JSON).
//!
//! Core substitution logic remains in `cas_math::substitute`.

mod eval;
mod parse;
mod steps;
mod strategy;
mod types;

pub use self::eval::*;
pub use self::parse::*;
pub use self::steps::*;
pub use self::strategy::*;
pub use self::types::*;
