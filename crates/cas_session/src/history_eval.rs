use crate::{
    history_types::{
        DeleteHistoryError, DeleteHistoryResult, HistoryOverviewEntry, HistoryOverviewKind,
    },
    EntryId, SessionState,
};

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
