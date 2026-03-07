use cas_ast::ExprId;

use crate::state_core::SessionState;

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
