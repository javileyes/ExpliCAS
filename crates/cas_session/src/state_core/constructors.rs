use super::SessionState;
use crate::{state_eval_store::SessionEvalStore, SessionStore};
use cas_solver_core::eval_options::EvalOptions;

impl SessionState {
    pub fn assumptions(&self) -> &EvalOptions {
        &self.options
    }

    pub fn options(&self) -> &EvalOptions {
        &self.options
    }

    pub fn options_mut(&mut self) -> &mut EvalOptions {
        &mut self.options
    }

    pub fn new() -> Self {
        Self {
            store: SessionEvalStore::new(),
            env: crate::Environment::new(),
            options: EvalOptions::default(),
        }
    }

    /// Create a state from an existing session store (for snapshot restoration).
    pub(crate) fn from_store(store: SessionStore) -> Self {
        Self {
            store: SessionEvalStore::from_store(store),
            env: crate::Environment::new(),
            options: EvalOptions::default(),
        }
    }
}
