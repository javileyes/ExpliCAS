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
