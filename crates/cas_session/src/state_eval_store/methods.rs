/// Local adapter that bridges `cas_session::SessionStore` to `cas_session_core::EvalStore`.
#[derive(Debug, Default)]
pub struct SessionEvalStore {
    pub(crate) store: crate::SessionStore,
    dirty: bool,
}

impl SessionEvalStore {
    pub(crate) fn new() -> Self {
        Self {
            store: crate::store_cache_policy::session_store_with_cache_config(
                cas_session_core::types::CacheConfig::default(),
            ),
            dirty: false,
        }
    }

    pub(crate) fn from_store(store: crate::SessionStore) -> Self {
        Self {
            store,
            dirty: false,
        }
    }

    pub(crate) fn is_dirty(&self) -> bool {
        self.dirty
    }

    pub(crate) fn mark_clean(&mut self) {
        self.dirty = false;
    }

    pub(crate) fn push(
        &mut self,
        kind: cas_session_core::types::EntryKind,
        raw_text: String,
    ) -> u64 {
        self.dirty = true;
        self.store.push(kind, raw_text)
    }

    pub(crate) fn clear(&mut self) {
        if !self.store.is_empty() {
            self.dirty = true;
        }
        self.store.clear();
    }

    pub(crate) fn remove(&mut self, ids: &[u64]) {
        let before = self.store.len();
        self.store.remove(ids);
        if self.store.len() != before {
            self.dirty = true;
        }
    }

    pub(crate) fn update_diagnostics(&mut self, id: u64, diagnostics: crate::Diagnostics) {
        self.dirty = true;
        self.store.update_diagnostics(id, diagnostics);
    }

    pub(crate) fn update_simplified(&mut self, id: u64, simplified: crate::cache::SimplifiedCache) {
        self.dirty = true;
        self.store.update_simplified(id, simplified);
    }

    pub(crate) fn touch_cached(&mut self, entry_id: u64) {
        self.store.touch_cached(entry_id);
    }
}

impl std::ops::Deref for SessionEvalStore {
    type Target = crate::SessionStore;

    fn deref(&self) -> &Self::Target {
        &self.store
    }
}
