//! History/inspect/session-facing API re-exported for session clients.

use crate::history_overview::HistoryOverviewContext;

pub use crate::history_delete::delete_history_entries;
pub use crate::history_delete::evaluate_delete_history_command_message;
pub use crate::history_format::{
    format_delete_history_error_message, format_delete_history_result_message,
    format_history_overview_lines, history_empty_message,
};
pub use crate::history_metadata_format::format_history_eval_metadata_sections;
pub use crate::history_models::{
    DeleteHistoryError, DeleteHistoryResult, HistoryEntryDetails, HistoryEntryInspection,
    HistoryExprInspection, HistoryOverviewEntry, HistoryOverviewKind,
    InspectHistoryEntryInputError, ParseHistoryEntryIdError,
};
pub use crate::history_overview::history_overview_entries;
pub use crate::history_parse::parse_history_ids;
pub use crate::history_show_format::{
    format_show_history_command_lines, format_show_history_command_lines_with_context,
};
pub use crate::inspect_format::{
    format_history_entry_inspection_lines, format_inspect_history_entry_error_message,
};
pub use crate::inspect_parse::parse_history_entry_id;
pub use crate::inspect_runtime::{
    inspect_history_entry, inspect_history_entry_input, InspectHistoryContext,
};
pub use crate::repl_session_runtime::{
    evaluate_delete_history_command_message_on_runtime as evaluate_delete_history_command_message_on_repl_core,
    evaluate_history_command_message_on_runtime as evaluate_history_command_message_on_repl_core,
    evaluate_show_command_lines_on_runtime as evaluate_show_command_lines_on_repl_core,
};
pub use crate::show_command::evaluate_show_command_lines;

/// Evaluate `history` command lines using an expression renderer callback.
pub fn evaluate_history_command_lines<C, F>(context: &C, render_expr: F) -> Vec<String>
where
    C: HistoryOverviewContext,
    F: FnMut(cas_ast::ExprId) -> String,
{
    let entries = history_overview_entries(context);
    crate::evaluate_history_command_lines(&entries, render_expr)
}

/// Evaluate `history` command lines using an explicit AST context.
pub fn evaluate_history_command_lines_with_context<C>(
    context: &C,
    ast_context: &cas_ast::Context,
) -> Vec<String>
where
    C: HistoryOverviewContext,
{
    let entries = history_overview_entries(context);
    crate::evaluate_history_command_lines_with_context(&entries, ast_context)
}
