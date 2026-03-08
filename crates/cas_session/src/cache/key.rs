use cas_solver_core::domain_mode::DomainMode;

use super::CacheDomainMode;

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
    pub fn from_context(domain: DomainMode) -> Self {
        Self {
            domain: domain.into(),
            // For now, use a static value. In the future this can hash ruleset config.
            ruleset_rev: 1,
        }
    }

    /// Create a cache key from a CLI-style domain flag (`strict|assume|generic`).
    pub fn from_domain_flag(domain: &str) -> Self {
        let mode = match domain {
            "strict" => DomainMode::Strict,
            "assume" => DomainMode::Assume,
            _ => DomainMode::Generic,
        };
        Self::from_context(mode)
    }

    /// Check if this key is compatible with another (for cache hit).
    pub fn is_compatible(&self, other: &Self) -> bool {
        self == other
    }
}
