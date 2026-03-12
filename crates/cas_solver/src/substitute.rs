//! Substitute API with optional step rendering for upper layers (CLI/JSON).
//!
//! Core substitution logic remains in `cas_math::substitute`.

mod eval;
mod parse;
mod steps;
mod strategy;
mod types;

#[cfg_attr(not(test), allow(unused_imports))]
pub(crate) use self::eval::evaluate_substitute_and_simplify;
#[cfg_attr(not(test), allow(unused_imports))]
pub(crate) use self::parse::parse_substitute_args;
pub use self::steps::substitute_with_steps;
pub use self::strategy::{
    detect_substitute_strategy, substitute_auto, substitute_auto_with_strategy,
    substitute_power_aware,
};
pub use self::types::{
    SubstituteOptions, SubstituteParseError, SubstituteResult, SubstituteSimplifyEvalOutput,
    SubstituteStep, SubstituteStrategy,
};
