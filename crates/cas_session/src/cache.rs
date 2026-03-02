//! Session-owned simplified cache models.
//!
//! These payloads live in `cas_session` (application state layer), while
//! `cas_engine` only reports simplification artifacts through `EvalStore`.

use cas_ast::ExprId;
use std::sync::Arc;

/// Session-local domain axis used by cache keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CacheDomainMode {
    Strict,
    Assume,
    Generic,
}

impl From<cas_solver::DomainMode> for CacheDomainMode {
    fn from(mode: cas_solver::DomainMode) -> Self {
        match mode {
            cas_solver::DomainMode::Strict => Self::Strict,
            cas_solver::DomainMode::Assume => Self::Assume,
            cas_solver::DomainMode::Generic => Self::Generic,
        }
    }
}

/// Key for cache invalidation.
///
/// If any setting changes between cache creation and cache usage, the cache is
/// considered invalid.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct SimplifyCacheKey {
    /// Domain mode at time of simplification.
    pub domain: CacheDomainMode,
    /// Build/version hash for ruleset (currently static).
    pub ruleset_rev: u64,
}

impl SimplifyCacheKey {
    /// Create a cache key from current context settings.
    pub fn from_context(domain: cas_solver::DomainMode) -> Self {
        Self {
            domain: domain.into(),
            // For now, use a static value. In the future this can hash ruleset config.
            ruleset_rev: 1,
        }
    }

    /// Create a cache key from a CLI-style domain flag (`strict|assume|generic`).
    pub fn from_domain_flag(domain: &str) -> Self {
        let mode = match domain {
            "strict" => cas_solver::DomainMode::Strict,
            "assume" => cas_solver::DomainMode::Assume,
            _ => cas_solver::DomainMode::Generic,
        };
        Self::from_context(mode)
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
    pub requires: Vec<cas_solver::RequiredItem>,
    /// Derivation steps (None = light cache, steps omitted for large entries).
    pub steps: Option<Arc<Vec<cas_solver::Step>>>,
}

#[cfg(test)]
mod tests {
    use super::{CacheDomainMode, SimplifyCacheKey};

    #[test]
    fn cache_key_from_domain_flag_maps_known_values() {
        assert_eq!(
            SimplifyCacheKey::from_domain_flag("strict").domain,
            CacheDomainMode::Strict
        );
        assert_eq!(
            SimplifyCacheKey::from_domain_flag("assume").domain,
            CacheDomainMode::Assume
        );
        assert_eq!(
            SimplifyCacheKey::from_domain_flag("generic").domain,
            CacheDomainMode::Generic
        );
    }

    #[test]
    fn cache_key_from_domain_flag_defaults_to_generic() {
        assert_eq!(
            SimplifyCacheKey::from_domain_flag("unknown").domain,
            CacheDomainMode::Generic
        );
    }
}
