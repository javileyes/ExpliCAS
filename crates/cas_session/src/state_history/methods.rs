use crate::state_core::SessionState;

impl SessionState {
    pub fn history_entries(&self) -> &[crate::Entry] {
        self.store.list()
    }

    pub fn history_push<S: Into<String>>(
        &mut self,
        kind: crate::EntryKind,
        raw_text: S,
    ) -> crate::EntryId {
        self.store.push(kind, raw_text.into())
    }

    pub fn history_get(&self, id: crate::EntryId) -> Option<&crate::Entry> {
        self.store.get(id)
    }

    pub fn history_len(&self) -> usize {
        self.store.len()
    }

    pub fn history_remove(&mut self, ids: &[crate::EntryId]) {
        self.store.remove(ids);
    }
}
