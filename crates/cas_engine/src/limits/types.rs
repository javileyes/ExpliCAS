//! Types for the limit framework.

use crate::Step;
use cas_ast::ExprId;
pub use cas_math::limit_types::{Approach, LimitOptions, PreSimplifyMode};

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
