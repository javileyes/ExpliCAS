/// Minimal profile-cache surface needed by `cache` REPL commands.
pub trait ProfileCacheStore {
    fn profile_cache_len(&self) -> usize;
    fn clear_profile_cache(&mut self);
}

impl ProfileCacheStore for crate::Engine {
    fn profile_cache_len(&self) -> usize {
        crate::Engine::profile_cache_len(self)
    }

    fn clear_profile_cache(&mut self) {
        crate::Engine::clear_profile_cache(self);
    }
}
