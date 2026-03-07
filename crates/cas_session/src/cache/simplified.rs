use cas_ast::ExprId;
use cas_solver_core::diagnostics_model::RequiredItem;
use std::sync::Arc;

use super::SimplifyCacheKey;

/// Cached simplification result for a session entry.
#[derive(Debug, Clone)]
pub struct SimplifiedCache {
    /// Key for invalidation (must match current context).
    pub key: SimplifyCacheKey,
    /// Simplified expression.
    pub expr: ExprId,
    /// Domain requirements from this entry (for propagation).
    pub requires: Vec<RequiredItem>,
    /// Derivation steps (None = light cache, steps omitted for large entries).
    pub steps: Option<Arc<Vec<cas_solver::Step>>>,
}
