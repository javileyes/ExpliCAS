//! Session store engine-specific cache payload and type aliases.

use cas_ast::ExprId;
pub use cas_session_core::store::{
    Entry as CoreEntry, SessionCacheValue, SessionStore as CoreStore,
};
pub use cas_session_core::types::{CacheConfig, EntryId, EntryKind, RefMode};

// =============================================================================
// Session Reference Caching (V2.15.36)
// =============================================================================

/// Key for cache invalidation - must match for cache hit.
///
/// If any of these settings change between when the cache was created
/// and when it's being used, the cache is invalid.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct SimplifyCacheKey {
    /// Domain mode at time of simplification.
    pub domain: crate::domain::DomainMode,
    /// Build/version hash for ruleset (currently static).
    pub ruleset_rev: u64,
}

impl SimplifyCacheKey {
    /// Create a cache key from current context settings.
    pub fn from_context(domain: crate::domain::DomainMode) -> Self {
        Self {
            domain,
            // For now, use a static value. In the future, could hash ruleset config.
            ruleset_rev: 1,
        }
    }

    /// Check if this key is compatible with another (for cache hit).
    pub fn is_compatible(&self, other: &Self) -> bool {
        self == other
    }
}

/// Cached simplification result for a session entry.
///
/// Stored after evaluation to enable fast resolution of `#N` references
/// without re-running the simplification pipeline.
#[derive(Debug, Clone)]
pub struct SimplifiedCache {
    /// Key for invalidation (must match current context).
    pub key: SimplifyCacheKey,
    /// Simplified expression.
    pub expr: ExprId,
    /// Domain requirements from this entry (for propagation).
    pub requires: Vec<crate::diagnostics::RequiredItem>,
    /// Derivation steps (None = light cache, steps omitted for large entries).
    pub steps: Option<std::sync::Arc<Vec<crate::step::Step>>>,
}

impl SessionCacheValue for SimplifiedCache {
    fn steps_len(&self) -> usize {
        self.steps.as_ref().map(|s| s.len()).unwrap_or(0)
    }

    fn apply_light_cache(mut self, light_cache_threshold: Option<usize>) -> Self {
        if let Some(threshold) = light_cache_threshold {
            if self.steps_len() > threshold {
                self.steps = None;
            }
        }
        self
    }
}

/// Record of a single cache hit during resolution.
///
/// Used to generate synthetic timeline steps showing which
/// cached results were used and what they resolved to.
#[derive(Debug, Clone)]
pub struct CacheHitTrace {
    /// The entry ID that was resolved from cache.
    pub entry_id: EntryId,
    /// The ExprId of the `#N` node in the AST before resolution.
    pub before_ref_expr: ExprId,
    /// The cached simplified ExprId that replaced the reference.
    pub after_expr: ExprId,
    /// Domain requirements from the cached entry.
    pub requires: Vec<crate::diagnostics::RequiredItem>,
}

/// Result of resolving session references with accumulated requires.
#[derive(Debug, Clone)]
pub struct ResolvedExpr {
    /// The resolved expression.
    pub expr: ExprId,
    /// Accumulated domain requirements from all referenced entries.
    pub requires: Vec<crate::diagnostics::RequiredItem>,
    /// Whether cache was used (for timeline step generation).
    pub used_cache: bool,
    /// Chain of referenced entry IDs (for debugging).
    pub ref_chain: smallvec::SmallVec<[EntryId; 4]>,
    /// Cache hits recorded during resolution (for synthetic step generation).
    pub cache_hits: Vec<CacheHitTrace>,
}

/// Engine specialization of a session entry.
pub type Entry = CoreEntry<crate::diagnostics::Diagnostics, SimplifiedCache>;

/// Engine specialization of the generic session store.
pub type SessionStore = CoreStore<crate::diagnostics::Diagnostics, SimplifiedCache>;
