use crate::{
    evaluate_history_command_lines, evaluate_history_command_lines_with_context,
    history_overview_entries, HistoryOverviewContext,
};

/// Evaluate `history` command lines using an expression renderer callback.
pub fn evaluate_history_command_lines_from_history<C, F>(context: &C, render_expr: F) -> Vec<String>
where
    C: HistoryOverviewContext,
    F: FnMut(cas_ast::ExprId) -> String,
{
    let entries = history_overview_entries(context);
    evaluate_history_command_lines(&entries, render_expr)
}

/// Evaluate `history` command lines using an explicit AST context.
pub fn evaluate_history_command_lines_from_history_with_context<C>(
    context: &C,
    ast_context: &cas_ast::Context,
) -> Vec<String>
where
    C: HistoryOverviewContext,
{
    let entries = history_overview_entries(context);
    evaluate_history_command_lines_with_context(&entries, ast_context)
}
