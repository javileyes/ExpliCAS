use cas_solver_core::session_runtime::BindingOverviewEntry;

use super::BindingsContext;

/// Return a stable, presentation-friendly view of bindings.
pub fn binding_overview_entries<C: BindingsContext>(context: &C) -> Vec<BindingOverviewEntry> {
    context
        .bindings()
        .into_iter()
        .map(|(name, expr)| BindingOverviewEntry { name, expr })
        .collect()
}
