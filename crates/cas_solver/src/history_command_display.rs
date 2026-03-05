use crate::{format_history_overview_lines, history_empty_message, HistoryOverviewEntry};

/// Evaluate `history` command lines using an expression renderer callback.
pub fn evaluate_history_command_lines<F>(
    entries: &[HistoryOverviewEntry],
    render_expr: F,
) -> Vec<String>
where
    F: FnMut(cas_ast::ExprId) -> String,
{
    if entries.is_empty() {
        vec![history_empty_message().to_string()]
    } else {
        format_history_overview_lines(entries, render_expr)
    }
}

/// Evaluate `history` command lines using an explicit AST context.
pub fn evaluate_history_command_lines_with_context(
    entries: &[HistoryOverviewEntry],
    context: &cas_ast::Context,
) -> Vec<String> {
    evaluate_history_command_lines(entries, |id| {
        format!("{}", cas_formatter::DisplayExpr { context, id })
    })
}
