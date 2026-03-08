use crate::ReplCore;

impl ReplCore {
    /// Clear engine profile cache.
    pub(crate) fn clear_profile_cache(&mut self) {
        self.engine.clear_profile_cache();
    }

    /// Number of cached engine profiles.
    pub(crate) fn profile_cache_len(&self) -> usize {
        self.engine.profile_cache_len()
    }
}
