//! Generic session cache payload and resolution traces.
//!
//! These types are shared across crates to avoid duplicating session-cache
//! structures in both `cas_engine` and `cas_session`.

use crate::store::SessionCacheValue;
use crate::types::EntryId;
use cas_ast::ExprId;
use std::sync::Arc;

/// Key for cache invalidation.
///
/// If any setting changes between cache creation and cache usage, the cache is
/// considered invalid.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct SimplifyCacheKey<Domain> {
    /// Domain mode at time of simplification.
    pub domain: Domain,
    /// Build/version hash for ruleset (currently static).
    pub ruleset_rev: u64,
}

impl<Domain> SimplifyCacheKey<Domain> {
    /// Create a cache key from current context settings.
    pub fn from_context(domain: Domain) -> Self {
        Self {
            domain,
            // For now, use a static value. In the future, this could hash ruleset config.
            ruleset_rev: 1,
        }
    }
}

impl<Domain: PartialEq> SimplifyCacheKey<Domain> {
    /// Check if this key is compatible with another (for cache hit).
    pub fn is_compatible(&self, other: &Self) -> bool {
        self == other
    }
}

/// Cached simplification result for a session entry.
#[derive(Debug, Clone)]
pub struct SimplifiedCache<Domain, RequiredItem, Step> {
    /// Key for invalidation (must match current context).
    pub key: SimplifyCacheKey<Domain>,
    /// Simplified expression.
    pub expr: ExprId,
    /// Domain requirements from this entry (for propagation).
    pub requires: Vec<RequiredItem>,
    /// Derivation steps (None = light cache, steps omitted for large entries).
    pub steps: Option<Arc<Vec<Step>>>,
}

impl<Domain: Clone, RequiredItem: Clone, Step: Clone> SessionCacheValue
    for SimplifiedCache<Domain, RequiredItem, Step>
{
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
