use crate::{EntryId, SessionState};

/// Lightweight history entry view for presentation layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HistoryOverviewKind {
    Expr {
        expr: cas_ast::ExprId,
    },
    Eq {
        lhs: cas_ast::ExprId,
        rhs: cas_ast::ExprId,
    },
}

/// Lightweight history entry view without exposing store internals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoryOverviewEntry {
    pub id: EntryId,
    pub kind: HistoryOverviewKind,
}

/// Error while deleting history entries from command-style input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeleteHistoryError {
    NoValidIds,
}

/// Summary of deleting history entries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeleteHistoryResult {
    pub requested_ids: Vec<EntryId>,
    pub removed_count: usize,
}

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

/// Parse history IDs from whitespace-separated tokens (supports optional `#` prefix).
pub fn parse_history_ids(input: &str) -> Vec<EntryId> {
    input
        .split_whitespace()
        .filter_map(|token| token.trim_start_matches('#').parse::<EntryId>().ok())
        .collect()
}

/// Delete history entries based on command-style ID input.
pub fn delete_history_entries(
    state: &mut SessionState,
    input: &str,
) -> Result<DeleteHistoryResult, DeleteHistoryError> {
    let requested_ids = parse_history_ids(input);
    if requested_ids.is_empty() {
        return Err(DeleteHistoryError::NoValidIds);
    }

    let before = state.history_len();
    state.history_remove(&requested_ids);
    let removed_count = before.saturating_sub(state.history_len());

    Ok(DeleteHistoryResult {
        requested_ids,
        removed_count,
    })
}

/// Return a stable, presentation-friendly view of history entries.
pub fn history_overview_entries(state: &SessionState) -> Vec<HistoryOverviewEntry> {
    state
        .history_entries()
        .iter()
        .map(|entry| {
            let kind = match entry.kind {
                crate::EntryKind::Expr(expr) => HistoryOverviewKind::Expr { expr },
                crate::EntryKind::Eq { lhs, rhs } => HistoryOverviewKind::Eq { lhs, rhs },
            };
            HistoryOverviewEntry { id: entry.id, kind }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        delete_history_entries, format_delete_history_error_message,
        format_delete_history_result_message, format_history_overview_lines, history_empty_message,
        history_overview_entries, parse_history_ids, DeleteHistoryError, HistoryOverviewKind,
    };
    use crate::{EntryKind, SessionState};

    #[test]
    fn parse_history_ids_accepts_hash_prefix() {
        let ids = parse_history_ids("#1 #2 3 nope");
        assert_eq!(ids, vec![1, 2, 3]);
    }

    #[test]
    fn delete_history_entries_errors_without_ids() {
        let mut state = SessionState::new();
        let err = delete_history_entries(&mut state, "abc").expect_err("expected no ids error");
        assert_eq!(err, DeleteHistoryError::NoValidIds);
    }

    #[test]
    fn delete_history_entries_removes_existing_entries() {
        let mut state = SessionState::new();
        let mut ctx = cas_ast::Context::new();
        let x = cas_parser::parse("x", &mut ctx).expect("parse x");
        let y = cas_parser::parse("y", &mut ctx).expect("parse y");
        state.history_push(EntryKind::Expr(x), "x");
        state.history_push(EntryKind::Expr(y), "y");

        let result = delete_history_entries(&mut state, "#1 #99").expect("delete");
        assert_eq!(result.requested_ids, vec![1, 99]);
        assert_eq!(result.removed_count, 1);
        assert_eq!(state.history_len(), 1);
    }

    #[test]
    fn history_overview_entries_returns_typed_view() {
        let mut state = SessionState::new();
        let mut ctx = cas_ast::Context::new();
        let x = cas_parser::parse("x", &mut ctx).expect("parse x");
        let y = cas_parser::parse("y", &mut ctx).expect("parse y");
        state.history_push(EntryKind::Expr(x), "x");
        state.history_push(EntryKind::Eq { lhs: x, rhs: y }, "x = y");

        let overview = history_overview_entries(&state);
        assert_eq!(overview.len(), 2);
        assert!(matches!(overview[0].kind, HistoryOverviewKind::Expr { expr } if expr == x));
        assert!(matches!(
            overview[1].kind,
            HistoryOverviewKind::Eq { lhs, rhs } if lhs == x && rhs == y
        ));
    }

    #[test]
    fn format_delete_history_result_message_lists_ids() {
        let message = format_delete_history_result_message(&super::DeleteHistoryResult {
            requested_ids: vec![1, 3],
            removed_count: 1,
        });
        assert_eq!(message, "Deleted 1 entry/entries: #1, #3");
    }

    #[test]
    fn format_delete_history_error_message_no_valid_ids() {
        let message = format_delete_history_error_message(&DeleteHistoryError::NoValidIds);
        assert!(message.contains("Use 'del #1 #2'"));
    }

    #[test]
    fn history_empty_message_is_stable() {
        assert_eq!(history_empty_message(), "No entries in session history.");
    }

    #[test]
    fn format_history_overview_lines_renders_entries() {
        let entries = vec![
            super::HistoryOverviewEntry {
                id: 1,
                kind: HistoryOverviewKind::Expr {
                    expr: cas_ast::ExprId::from_raw(10),
                },
            },
            super::HistoryOverviewEntry {
                id: 2,
                kind: HistoryOverviewKind::Eq {
                    lhs: cas_ast::ExprId::from_raw(11),
                    rhs: cas_ast::ExprId::from_raw(12),
                },
            },
        ];
        let lines = format_history_overview_lines(&entries, |id| format!("E{}", id.index()));
        assert_eq!(lines[0], "Session history (2 entries):");
        assert_eq!(lines[1], "  #1   [Expr] E10");
        assert_eq!(lines[2], "  #2   [Eq  ] E11 = E12");
    }
}
