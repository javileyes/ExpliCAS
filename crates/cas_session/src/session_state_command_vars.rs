use crate::{format_binding_overview_lines, vars_empty_message, SessionState};

/// Evaluate `vars` command lines using an expression renderer callback.
pub fn evaluate_vars_command_lines<F>(state: &SessionState, render_expr: F) -> Vec<String>
where
    F: FnMut(cas_ast::ExprId) -> String,
{
    let bindings = crate::binding_overview_entries(state);
    if bindings.is_empty() {
        vec![vars_empty_message().to_string()]
    } else {
        format_binding_overview_lines(&bindings, render_expr)
    }
}

/// Evaluate `vars` command lines using an explicit AST context.
pub fn evaluate_vars_command_lines_with_context(
    state: &SessionState,
    context: &cas_ast::Context,
) -> Vec<String> {
    evaluate_vars_command_lines(state, |id| {
        format!("{}", cas_formatter::DisplayExpr { context, id })
    })
}
