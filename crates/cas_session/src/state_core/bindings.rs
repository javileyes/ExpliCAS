use cas_ast::ExprId;
use cas_solver_core::session_runtime::{AssignmentApplyContext, BindingsContext};

use crate::state_core::SessionState;

impl SessionState {
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

impl BindingsContext for SessionState {
    fn binding_count(&self) -> usize {
        SessionState::binding_count(self)
    }

    fn clear_bindings(&mut self) {
        SessionState::clear_bindings(self);
    }

    fn unset_binding(&mut self, name: &str) -> bool {
        SessionState::unset_binding(self, name)
    }

    fn bindings(&self) -> Vec<(String, ExprId)> {
        SessionState::bindings(self)
    }
}

impl AssignmentApplyContext for SessionState {
    fn assignment_unset_binding(&mut self, name: &str) -> bool {
        SessionState::unset_binding(self, name)
    }

    fn assignment_set_binding(&mut self, name: String, expr: ExprId) {
        SessionState::set_binding(self, name, expr);
    }

    fn assignment_unset_function(&mut self, name: &str) -> bool {
        let changed = self.env.unset_function(name);
        if changed {
            self.dirty = true;
        }
        changed
    }

    fn assignment_set_function(&mut self, name: String, params: Vec<String>, expr: ExprId) {
        let same = self
            .env
            .get_function(&name)
            .is_some_and(|binding| binding.params == params && binding.expr == expr);
        if !same {
            self.dirty = true;
            self.env.set_function(name, params, expr);
        }
    }

    fn assignment_resolve_session_refs(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
    ) -> Result<ExprId, String> {
        crate::resolve_refs::resolve_session_refs(ctx, expr, &self.store)
            .map_err(|error| error.to_string())
    }

    fn assignment_substitute_bindings_with_shadow(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        shadow: &[&str],
    ) -> ExprId {
        crate::env::substitute_with_shadow(ctx, &self.env, expr, shadow)
    }

    fn assignment_is_reserved_name(&self, name: &str) -> bool {
        crate::env::is_reserved(name)
    }
}
