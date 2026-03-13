use super::{SessionEvalStore, SessionState};
use crate::env::Environment;
use crate::SessionStore;
use cas_solver_core::eval_options::EvalOptions;

impl SessionState {
    pub fn assumptions(&self) -> &EvalOptions {
        &self.options
    }

    pub fn options(&self) -> &EvalOptions {
        &self.options
    }

    pub fn options_mut(&mut self) -> &mut EvalOptions {
        self.dirty = true;
        &mut self.options
    }

    pub(crate) fn is_dirty(&self) -> bool {
        self.dirty || self.store.is_dirty()
    }

    pub(crate) fn mark_clean(&mut self) {
        self.dirty = false;
        self.store.mark_clean();
    }

    pub fn new() -> Self {
        Self {
            store: SessionEvalStore::new(),
            env: Environment::new(),
            options: EvalOptions::default(),
            dirty: false,
        }
    }

    /// Create a state from an existing session store (for snapshot restoration).
    pub(crate) fn from_store(store: SessionStore) -> Self {
        Self {
            store: SessionEvalStore::from_store(store),
            env: Environment::new(),
            options: EvalOptions::default(),
            dirty: false,
        }
    }
}
