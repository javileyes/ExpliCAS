use crate::{DeleteHistoryError, DeleteHistoryResult, HistoryOverviewEntry, HistoryOverviewKind};

/// Message used when there are no history entries to list.
pub fn history_empty_message() -> &'static str {
    "No entries in session history."
}

/// Format history overview lines using an expression renderer callback.
pub fn format_history_overview_lines<F>(
    entries: &[HistoryOverviewEntry],
    mut render_expr: F,
) -> Vec<String>
where
    F: FnMut(cas_ast::ExprId) -> String,
{
    let mut lines = vec![format!("Session history ({} entries):", entries.len())];
    for entry in entries {
        let (type_indicator, display) = match entry.kind {
            HistoryOverviewKind::Expr { expr } => ("Expr", render_expr(expr)),
            HistoryOverviewKind::Eq { lhs, rhs } => (
                "Eq  ",
                format!("{} = {}", render_expr(lhs), render_expr(rhs)),
            ),
        };
        lines.push(format!(
            "  #{:<3} [{}] {}",
            entry.id, type_indicator, display
        ));
    }
    lines
}

/// Format a delete-history result as a user-facing message.
pub fn format_delete_history_result_message(result: &DeleteHistoryResult) -> String {
    if result.removed_count > 0 {
        let id_str: Vec<String> = result
            .requested_ids
            .iter()
            .map(|id| format!("#{}", id))
            .collect();
        format!(
            "Deleted {} entry/entries: {}",
            result.removed_count,
            id_str.join(", ")
        )
    } else {
        "No entries found with the specified IDs.".to_string()
    }
}

/// Format delete-history command errors.
pub fn format_delete_history_error_message(error: &DeleteHistoryError) -> String {
    match error {
        DeleteHistoryError::NoValidIds => {
            "Error: No valid IDs specified. Use 'del #1 #2' or 'del 1 2'.".to_string()
        }
    }
}
