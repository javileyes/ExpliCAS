//! Shared types for limit computation configuration.

use crate::infinity_support::InfSign;
use cas_ast::ExprId;

/// Direction of limit approach.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Approach {
    /// x -> +inf
    PosInfinity,
    /// x -> -inf
    NegInfinity,
    // V3 future: Point(ExprId) for x -> a
}

impl Approach {
    /// Convert an approach direction to the corresponding infinity sign.
    pub fn inf_sign(self) -> InfSign {
        match self {
            Approach::PosInfinity => InfSign::Pos,
            Approach::NegInfinity => InfSign::Neg,
        }
    }
}

/// Pre-simplification mode for limits.
///
/// Controls whether expressions are pre-processed before limit rules.
/// Default is Off for maximum conservatism and reproducibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PreSimplifyMode {
    /// No pre-simplification (most conservative).
    #[default]
    Off,
    /// Safe pre-simplification using allowlist-only transforms.
    /// Does NOT: rationalize, introduce domain assumptions, expand aggressively.
    Safe,
}

/// Options for limit computation.
#[derive(Debug, Clone, Default)]
pub struct LimitOptions {
    /// Whether to collect steps.
    pub steps: bool,
    /// Whether to use more aggressive (but still safe) rules.
    pub aggressive: bool,
    /// Pre-simplification mode (default: Off).
    pub presimplify: PreSimplifyMode,
}

/// Result produced by pure limit evaluation.
#[derive(Debug, Clone)]
pub struct LimitEvalOutcome {
    /// The computed expression (or residual `limit(...)` call if unresolved).
    pub expr: ExprId,
    /// Warning when no safe limit was found.
    pub warning: Option<String>,
}
