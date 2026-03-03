use crate::{
    clear_bindings_command, delete_history_entries, format_binding_overview_lines,
    format_clear_bindings_result_lines, format_delete_history_error_message,
    format_delete_history_result_message, format_history_entry_inspection_lines,
    format_history_overview_lines, history_empty_message, history_overview_entries,
    vars_empty_message, HistoryEntryDetails, HistoryEntryInspection, HistoryExprInspection,
    SessionState,
};

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

/// Evaluate `history` command lines using an expression renderer callback.
pub fn evaluate_history_command_lines<F>(state: &SessionState, render_expr: F) -> Vec<String>
where
    F: FnMut(cas_ast::ExprId) -> String,
{
    let entries = history_overview_entries(state);
    if entries.is_empty() {
        vec![history_empty_message().to_string()]
    } else {
        format_history_overview_lines(&entries, render_expr)
    }
}

/// Evaluate `history` command lines using an explicit AST context.
pub fn evaluate_history_command_lines_with_context(
    state: &SessionState,
    context: &cas_ast::Context,
) -> Vec<String> {
    evaluate_history_command_lines(state, |id| {
        format!("{}", cas_formatter::DisplayExpr { context, id })
    })
}

/// Evaluate `clear` command and return output lines.
pub fn evaluate_clear_command_lines(state: &mut SessionState, input: &str) -> Vec<String> {
    let result = clear_bindings_command(state, input);
    format_clear_bindings_result_lines(&result)
}

/// Evaluate `del` command and return a user-facing message.
pub fn evaluate_delete_history_command_message(state: &mut SessionState, input: &str) -> String {
    match delete_history_entries(state, input) {
        Ok(result) => format_delete_history_result_message(&result),
        Err(error) => format_delete_history_error_message(&error),
    }
}

/// Format `show` command lines from a pre-computed inspection.
pub fn format_show_history_command_lines<F, M>(
    inspection: &HistoryEntryInspection,
    render_expr: F,
    mut metadata_lines: M,
) -> Vec<String>
where
    F: FnMut(cas_ast::ExprId) -> String,
    M: FnMut(&HistoryExprInspection) -> Vec<String>,
{
    let mut lines = format_history_entry_inspection_lines(inspection, render_expr);
    if let HistoryEntryDetails::Expr(expr_info) = &inspection.details {
        lines.extend(metadata_lines(expr_info));
    }
    lines
}

/// Format `show` command lines from a pre-computed inspection using explicit context.
pub fn format_show_history_command_lines_with_context<M>(
    inspection: &HistoryEntryInspection,
    context: &cas_ast::Context,
    mut metadata_lines: M,
) -> Vec<String>
where
    M: FnMut(&cas_ast::Context, &HistoryExprInspection) -> Vec<String>,
{
    let mut lines = format_history_entry_inspection_lines(inspection, |id| {
        format!("{}", cas_formatter::DisplayExpr { context, id })
    });
    if let HistoryEntryDetails::Expr(expr_info) = &inspection.details {
        lines.extend(metadata_lines(context, expr_info));
    }
    lines
}
