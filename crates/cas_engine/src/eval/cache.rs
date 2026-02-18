//! Engine-owned cache payload for session-level simplified results.
//!
//! `cas_session_core` stays generic (store/resolve infrastructure), while this
//! module defines the concrete cache shape used by `cas_engine`.

use cas_ast::ExprId;
use std::sync::Arc;

/// Key for cache invalidation.
///
/// If any setting changes between cache creation and cache usage, the cache is
/// considered invalid.
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
            // For now, use a static value. In the future this can hash ruleset config.
            ruleset_rev: 1,
        }
    }

    /// Check if this key is compatible with another (for cache hit).
    pub fn is_compatible(&self, other: &Self) -> bool {
        self == other
    }
}

/// Cached simplification result for a session entry.
#[derive(Debug, Clone)]
pub struct SimplifiedCache {
    /// Key for invalidation (must match current context).
    pub key: SimplifyCacheKey,
    /// Simplified expression.
    pub expr: ExprId,
    /// Domain requirements from this entry (for propagation).
    pub requires: Vec<crate::diagnostics::RequiredItem>,
    /// Derivation steps (None = light cache, steps omitted for large entries).
    pub steps: Option<Arc<Vec<crate::step::Step>>>,
}
