//! Resolution cache traces shared by session resolution helpers.

use crate::types::EntryId;
use cas_ast::ExprId;

/// Record of a single cache hit during resolution.
#[derive(Debug, Clone)]
pub struct CacheHitTrace<RequiredItem> {
    /// The entry ID that was resolved from cache.
    pub entry_id: EntryId,
    /// The ExprId of the `#N` node in the AST before resolution.
    pub before_ref_expr: ExprId,
    /// The cached simplified ExprId that replaced the reference.
    pub after_expr: ExprId,
    /// Domain requirements from the cached entry.
    pub requires: Vec<RequiredItem>,
}

/// Result of resolving session references with accumulated requirements.
#[derive(Debug, Clone)]
pub struct ResolvedExpr<RequiredItem> {
    /// The resolved expression.
    pub expr: ExprId,
    /// Accumulated domain requirements from all referenced entries.
    pub requires: Vec<RequiredItem>,
    /// Whether cache was used (for timeline step generation).
    pub used_cache: bool,
    /// Chain of referenced entry IDs (for debugging).
    pub ref_chain: smallvec::SmallVec<[EntryId; 4]>,
    /// Cache hits recorded during resolution (for synthetic step generation).
    pub cache_hits: Vec<CacheHitTrace<RequiredItem>>,
}
