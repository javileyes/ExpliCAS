/// Mutable context required by bindings command helpers.
pub trait BindingsContext {
    fn binding_count(&self) -> usize;
    fn clear_bindings(&mut self);
    fn unset_binding(&mut self, name: &str) -> bool;
    fn bindings(&self) -> Vec<(String, cas_ast::ExprId)>;
}
