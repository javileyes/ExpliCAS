use super::SessionState;
use crate::{state_eval_store::SessionEvalStore, SessionStore};

impl SessionState {
    pub fn assumptions(&self) -> &cas_solver::EvalOptions {
        &self.options
    }

    pub fn options(&self) -> &cas_solver::EvalOptions {
        &self.options
    }

    pub fn options_mut(&mut self) -> &mut cas_solver::EvalOptions {
        &mut self.options
    }

    pub fn new() -> Self {
        Self {
            store: SessionEvalStore::new(),
            env: crate::Environment::new(),
            options: cas_solver::EvalOptions::default(),
        }
    }

    /// Create a state from an existing session store (for snapshot restoration).
    pub(crate) fn from_store(store: SessionStore) -> Self {
        Self {
            store: SessionEvalStore::from_store(store),
            env: crate::Environment::new(),
            options: cas_solver::EvalOptions::default(),
        }
    }
}
