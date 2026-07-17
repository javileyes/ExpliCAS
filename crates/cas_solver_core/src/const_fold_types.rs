//! Shared constant-fold configuration/result types.

use cas_ast::ExprId;

/// Constant folding mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConstFoldMode {
    /// No constant folding (default, preserves all expressions).
    #[default]
    Off,
    /// Safe constant folding on allowlisted constant subtrees.
    Safe,
}

/// Numeric display mode for results — PRESENTATION ONLY. Lives outside
/// `SharedSemanticConfig` on purpose: no rule can consult it, so it can
/// never change a computed value (the engine stays exact and symbolic;
/// `Decimal` approximates at the output boundary).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NumericDisplayMode {
    /// Exact fractions and radicals (default).
    #[default]
    Exact,
    /// Approximate numeric parts of results at the output boundary.
    Decimal,
}

/// Constant folding result with statistics.
#[derive(Debug, Clone)]
pub struct ConstFoldResult {
    /// The resulting expression (may be same as input if no folding occurred).
    pub expr: ExprId,
    /// Number of nodes created during folding.
    pub nodes_created: u64,
    /// Number of fold operations performed.
    pub folds_performed: u64,
}
