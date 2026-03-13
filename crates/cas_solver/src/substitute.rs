//! Substitute API with optional step rendering for upper layers (CLI/JSON).
//!
//! Core substitution logic remains in `cas_math::substitute`.

mod eval;
mod parse;
mod steps;
mod strategy;

#[cfg_attr(not(test), allow(unused_imports))]
pub(crate) use self::eval::evaluate_substitute_and_simplify;
#[cfg_attr(not(test), allow(unused_imports))]
pub(crate) use self::parse::parse_substitute_args;
pub use self::steps::substitute_with_steps;
pub use self::strategy::{
    detect_substitute_strategy, substitute_auto, substitute_auto_with_strategy,
    substitute_power_aware,
};
use cas_ast::ExprId;
pub use cas_math::substitute::SubstituteOptions;
pub use cas_solver_core::substitute_command_types::SubstituteParseError;

/// Evaluated payload for REPL-style `subst` followed by simplify.
#[derive(Debug, Clone)]
#[cfg_attr(not(test), allow(dead_code))]
pub struct SubstituteSimplifyEvalOutput {
    pub simplified_expr: ExprId,
    pub strategy: SubstituteStrategy,
    pub steps: Vec<crate::Step>,
}

/// A single substitution step for traceability.
#[derive(Clone, Debug)]
pub struct SubstituteStep {
    /// Rule name: "SubstituteExact", "SubstitutePowerMultiple", "SubstitutePowOfTarget"
    pub rule: String,
    /// Expression before substitution (formatted)
    pub before: String,
    /// Expression after substitution (formatted)
    pub after: String,
    /// Optional note (e.g., "n=4, k=2, m=2")
    pub note: Option<String>,
}

/// Result of substitution including optional steps.
#[derive(Clone, Debug)]
pub struct SubstituteResult {
    pub expr: ExprId,
    pub steps: Vec<SubstituteStep>,
}

/// Strategy chosen for substitution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubstituteStrategy {
    /// Direct variable replacement by node id.
    Variable,
    /// Power-aware expression substitution.
    PowerAware,
}
