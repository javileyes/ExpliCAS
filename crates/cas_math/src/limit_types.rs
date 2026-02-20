//! Shared types for limit computation configuration.

/// Direction of limit approach.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Approach {
    /// x -> +inf
    PosInfinity,
    /// x -> -inf
    NegInfinity,
    // V3 future: Point(ExprId) for x -> a
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
