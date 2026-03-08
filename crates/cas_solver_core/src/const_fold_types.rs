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
