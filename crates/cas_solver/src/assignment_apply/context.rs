use cas_ast::ExprId;

/// Context required to apply assignment updates from stateful layers.
pub trait AssignmentApplyContext {
    fn assignment_unset_binding(&mut self, name: &str) -> bool;
    fn assignment_set_binding(&mut self, name: String, expr: ExprId);
    fn assignment_resolve_state_refs(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
    ) -> Result<ExprId, String>;
    fn assignment_is_reserved_name(&self, name: &str) -> bool;
}
