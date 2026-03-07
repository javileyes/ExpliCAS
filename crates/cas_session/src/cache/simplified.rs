use cas_ast::ExprId;
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
    pub requires: Vec<cas_solver::RequiredItem>,
    /// Derivation steps (None = light cache, steps omitted for large entries).
    pub steps: Option<Arc<Vec<cas_solver::Step>>>,
}
