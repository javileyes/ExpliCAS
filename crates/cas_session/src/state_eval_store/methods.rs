/// Local adapter that bridges `cas_session::SessionStore` to `cas_session_core::EvalStore`.
#[derive(Debug, Default)]
pub struct SessionEvalStore(pub(crate) crate::SessionStore);

impl SessionEvalStore {
    pub(crate) fn new() -> Self {
        Self(crate::store_cache_policy::session_store_with_cache_config(
            cas_session_core::types::CacheConfig::default(),
        ))
    }

    pub(crate) fn from_store(store: crate::SessionStore) -> Self {
        Self(store)
    }
}

impl std::ops::Deref for SessionEvalStore {
    type Target = crate::SessionStore;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for SessionEvalStore {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
