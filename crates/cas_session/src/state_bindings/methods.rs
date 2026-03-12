use cas_ast::ExprId;

use crate::state_core::SessionState;

impl SessionState {
    pub fn get_binding(&self, name: &str) -> Option<ExprId> {
        self.env.get(name)
    }

    pub fn set_binding<S: Into<String>>(&mut self, name: S, expr: ExprId) {
        let name = name.into();
        if self.env.get(&name) != Some(expr) {
            self.dirty = true;
            self.env.set(name, expr);
        }
    }

    pub fn unset_binding(&mut self, name: &str) -> bool {
        let changed = self.env.unset(name);
        if changed {
            self.dirty = true;
        }
        changed
    }

    pub fn bindings(&self) -> Vec<(String, ExprId)> {
        self.env
            .list()
            .into_iter()
            .map(|(name, expr)| (name.to_string(), expr))
            .collect()
    }

    pub fn binding_count(&self) -> usize {
        self.env.len()
    }

    pub fn clear_bindings(&mut self) {
        if !self.env.is_empty() {
            self.dirty = true;
            self.env.clear_all();
        }
    }

    /// Clear all session data (history + env bindings).
    /// Note: options are intentionally preserved.
    pub fn clear(&mut self) {
        self.store.clear();
        if !self.env.is_empty() {
            self.dirty = true;
            self.env.clear_all();
        }
    }
}
