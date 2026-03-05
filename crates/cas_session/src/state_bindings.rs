use cas_ast::ExprId;

use crate::state_core::SessionState;

impl SessionState {
    pub fn get_binding(&self, name: &str) -> Option<ExprId> {
        self.env.get(name)
    }

    pub fn set_binding<S: Into<String>>(&mut self, name: S, expr: ExprId) {
        self.env.set(name.into(), expr);
    }

    pub fn unset_binding(&mut self, name: &str) -> bool {
        self.env.unset(name)
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
        self.env.clear_all();
    }

    /// Clear all session data (history + env bindings).
    /// Note: options are intentionally preserved.
    pub fn clear(&mut self) {
        self.store.clear();
        self.env.clear_all();
    }
}

impl cas_solver::BindingsContext for SessionState {
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

impl cas_solver::AssignmentApplyContext for SessionState {
    fn assignment_unset_binding(&mut self, name: &str) -> bool {
        SessionState::unset_binding(self, name)
    }

    fn assignment_set_binding(&mut self, name: String, expr: ExprId) {
        SessionState::set_binding(self, name, expr);
    }

    fn assignment_resolve_state_refs(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
    ) -> Result<ExprId, String> {
        self.resolve_state_refs(ctx, expr)
            .map_err(|error| error.to_string())
    }

    fn assignment_is_reserved_name(&self, name: &str) -> bool {
        crate::env::is_reserved(name)
    }
}
