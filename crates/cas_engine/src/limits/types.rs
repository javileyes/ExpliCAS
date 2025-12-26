//! Types for the limit framework.

use crate::Step;
use cas_ast::ExprId;

/// Direction of limit approach.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Approach {
    /// x → +∞
    PosInfinity,
    /// x → -∞
    NegInfinity,
    // V3 future: Point(ExprId) for x → a
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

/// Result of limit computation.
#[derive(Debug)]
pub struct LimitResult {
    /// The computed limit expression (or residual `limit(...)` if unresolved).
    pub expr: ExprId,
    /// Steps taken during computation (if requested).
    pub steps: Vec<Step>,
    /// Warning message if limit could not be determined safely.
    pub warning: Option<String>,
}

/// Internal classification of limit value.
/// Used for composing limits in V2+.
#[allow(dead_code)] // Reserved for V2 composition
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LimitValueKind {
    /// Known finite value (number, π, e, i)
    FiniteLiteral,
    /// Positive infinity
    InfinityPos,
    /// Negative infinity
    InfinityNeg,
    /// Indeterminate form (∞-∞, 0/0, etc.)
    Undefined,
    /// Cannot determine
    Unknown,
}
