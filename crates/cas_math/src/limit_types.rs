//! Shared types for limit computation configuration.

use crate::infinity_support::InfSign;
use cas_ast::ExprId;

/// Side for a one-sided finite limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FiniteLimitSide {
    /// x -> a^-.
    Left,
    /// x -> a^+.
    Right,
}

impl FiniteLimitSide {
    pub const fn tail_sign(self) -> InfSign {
        match self {
            FiniteLimitSide::Left => InfSign::Neg,
            FiniteLimitSide::Right => InfSign::Pos,
        }
    }

    pub const fn marker(self) -> &'static str {
        match self {
            FiniteLimitSide::Left => "left",
            FiniteLimitSide::Right => "right",
        }
    }
}

/// Direction of limit approach.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Approach {
    /// x -> +inf
    PosInfinity,
    /// x -> -inf
    NegInfinity,
    /// x -> finite point from both sides.
    Finite(ExprId),
    /// x -> finite point from an explicit side.
    FiniteOneSided(ExprId, FiniteLimitSide),
}

impl Approach {
    /// Convert an approach direction to the corresponding infinity sign.
    pub fn inf_sign(self) -> Option<InfSign> {
        match self {
            Approach::PosInfinity => Some(InfSign::Pos),
            Approach::NegInfinity => Some(InfSign::Neg),
            Approach::Finite(_) | Approach::FiniteOneSided(_, _) => None,
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
