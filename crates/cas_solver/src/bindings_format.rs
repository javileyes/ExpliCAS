use cas_solver_core::session_runtime::{BindingOverviewEntry, ClearBindingsResult};

/// Message used when no variables exist.
pub fn vars_empty_message() -> &'static str {
    "No variables defined."
}

/// Format binding overview lines using an expression renderer callback.
pub fn format_binding_overview_lines<F>(
    entries: &[BindingOverviewEntry],
    mut render_expr: F,
) -> Vec<String>
where
    F: FnMut(cas_ast::ExprId) -> String,
{
    let mut lines = vec!["Variables:".to_string()];
    for entry in entries {
        lines.push(format!("  {} = {}", entry.name, render_expr(entry.expr)));
    }
    lines
}

/// Format clear-bindings command result into output lines.
pub fn format_clear_bindings_result_lines(result: &ClearBindingsResult) -> Vec<String> {
    match result {
        ClearBindingsResult::All { cleared_count } => {
            if *cleared_count == 0 {
                vec!["No variables to clear.".to_string()]
            } else {
                vec![format!("Cleared {} variable(s).", cleared_count)]
            }
        }
        ClearBindingsResult::Selected {
            cleared_count,
            missing_names,
        } => {
            let mut lines: Vec<String> = missing_names
                .iter()
                .map(|name| format!("Warning: '{}' was not defined", name))
                .collect();
            if *cleared_count > 0 {
                lines.push(format!("Cleared {} variable(s).", cleared_count));
            }
            lines
        }
    }
}
