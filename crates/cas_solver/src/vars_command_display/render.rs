use crate::{format_binding_overview_lines, vars_empty_message, BindingOverviewEntry};

/// Evaluate `vars` command lines using an expression renderer callback.
pub fn evaluate_vars_command_lines<F>(
    bindings: &[BindingOverviewEntry],
    render_expr: F,
) -> Vec<String>
where
    F: FnMut(cas_ast::ExprId) -> String,
{
    if bindings.is_empty() {
        vec![vars_empty_message().to_string()]
    } else {
        format_binding_overview_lines(bindings, render_expr)
    }
}
