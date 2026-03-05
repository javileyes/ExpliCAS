use crate::{BindingOverviewEntry, ClearBindingsResult};

/// Mutable context required by bindings command helpers.
pub trait BindingsContext {
    fn binding_count(&self) -> usize;
    fn clear_bindings(&mut self);
    fn unset_binding(&mut self, name: &str) -> bool;
    fn bindings(&self) -> Vec<(String, cas_ast::ExprId)>;
}

/// Apply a `clear` command over session bindings.
///
/// Accepts either:
/// - `clear` (clear all bindings)
/// - `clear <name> [<name> ...]` (clear selected bindings)
pub fn clear_bindings_command<C: BindingsContext>(
    context: &mut C,
    input: &str,
) -> ClearBindingsResult {
    let trimmed = input.trim();
    let all_mode = trimmed == "clear" || trimmed.is_empty();

    if all_mode {
        let cleared_count = context.binding_count();
        context.clear_bindings();
        return ClearBindingsResult::All { cleared_count };
    }

    let names_part = trimmed.strip_prefix("clear").unwrap_or(trimmed).trim();
    let names: Vec<&str> = names_part.split_whitespace().collect();

    let mut cleared_count = 0;
    let mut missing_names = Vec::new();
    for name in names {
        if context.unset_binding(name) {
            cleared_count += 1;
        } else {
            missing_names.push(name.to_string());
        }
    }

    ClearBindingsResult::Selected {
        cleared_count,
        missing_names,
    }
}

/// Return a stable, presentation-friendly view of bindings.
pub fn binding_overview_entries<C: BindingsContext>(context: &C) -> Vec<BindingOverviewEntry> {
    context
        .bindings()
        .into_iter()
        .map(|(name, expr)| BindingOverviewEntry { name, expr })
        .collect()
}
