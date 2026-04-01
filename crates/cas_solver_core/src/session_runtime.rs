use cas_ast::{Context, ExprId};

/// Context required to apply assignment updates from stateful layers.
pub trait AssignmentApplyContext {
    fn assignment_unset_binding(&mut self, name: &str) -> bool;
    fn assignment_set_binding(&mut self, name: String, expr: ExprId);
    fn assignment_unset_function(&mut self, _name: &str) -> bool {
        false
    }
    fn assignment_set_function(&mut self, _name: String, _params: Vec<String>, _expr: ExprId) {}
    fn assignment_resolve_session_refs(
        &self,
        ctx: &mut Context,
        expr: ExprId,
    ) -> Result<ExprId, String>;
    fn assignment_substitute_bindings_with_shadow(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        shadow: &[&str],
    ) -> ExprId;
    fn assignment_resolve_state_refs(
        &self,
        ctx: &mut Context,
        expr: ExprId,
    ) -> Result<ExprId, String> {
        let resolved = self.assignment_resolve_session_refs(ctx, expr)?;
        Ok(self.assignment_substitute_bindings_with_shadow(ctx, resolved, &[]))
    }
    fn assignment_is_reserved_name(&self, name: &str) -> bool;
}

/// Mutable context required by bindings command helpers.
pub trait BindingsContext {
    fn binding_count(&self) -> usize;
    fn clear_bindings(&mut self);
    fn unset_binding(&mut self, name: &str) -> bool;
    fn bindings(&self) -> Vec<(String, ExprId)>;
}

/// Lightweight binding view for presentation layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindingOverviewEntry {
    pub name: String,
    pub expr: ExprId,
}

/// Result of applying a `clear`-style binding command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClearBindingsResult {
    All {
        cleared_count: usize,
    },
    Selected {
        cleared_count: usize,
        missing_names: Vec<String>,
    },
}

/// Mutable context required to apply `budget` command updates.
pub trait SolveBudgetContext {
    fn solve_budget_max_branches(&self) -> usize;
    fn set_solve_budget_max_branches(&mut self, max_branches: usize);
}

/// Result of applying a `budget` command against session options.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolveBudgetCommandResult {
    Current { max_branches: usize },
    Updated { max_branches: usize },
    Invalid { raw_value: String },
}
