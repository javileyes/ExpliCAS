use crate::state_core::SessionState;

impl SessionState {
    pub fn history_entries(&self) -> &[crate::Entry] {
        self.store.list()
    }

    pub fn history_push<S: Into<String>>(
        &mut self,
        kind: cas_session_core::types::EntryKind,
        raw_text: S,
    ) -> cas_session_core::types::EntryId {
        self.store.push(kind, raw_text.into())
    }

    pub fn history_get(&self, id: cas_session_core::types::EntryId) -> Option<&crate::Entry> {
        self.store.get(id)
    }

    pub fn history_len(&self) -> usize {
        self.store.len()
    }

    pub fn history_remove(&mut self, ids: &[cas_session_core::types::EntryId]) {
        self.store.remove(ids);
    }
}
