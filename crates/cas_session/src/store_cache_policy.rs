use crate::{CacheConfig, SessionStore, SimplifiedCache};

pub(crate) fn simplify_cache_steps_len(cache: &SimplifiedCache) -> usize {
    cache.steps.as_ref().map(|s| s.len()).unwrap_or(0)
}

pub(crate) fn apply_simplified_light_cache(
    mut cache: SimplifiedCache,
    light_cache_threshold: Option<usize>,
) -> SimplifiedCache {
    if let Some(threshold) = light_cache_threshold {
        if simplify_cache_steps_len(&cache) > threshold {
            cache.steps = None;
        }
    }
    cache
}

pub(crate) fn session_store_with_cache_config(cache_config: CacheConfig) -> SessionStore {
    SessionStore::with_cache_config_and_policy(
        cache_config,
        simplify_cache_steps_len,
        apply_simplified_light_cache,
    )
}
