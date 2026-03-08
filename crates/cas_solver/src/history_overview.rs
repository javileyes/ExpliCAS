use crate::{HistoryOverviewEntry, HistoryOverviewKind};
pub use cas_solver_core::history_runtime::{
    HistoryEntryKindRaw, HistoryEntryRaw, HistoryOverviewContext,
};

/// Return a stable, presentation-friendly view of history entries.
pub fn history_overview_entries<C: HistoryOverviewContext>(
    context: &C,
) -> Vec<HistoryOverviewEntry> {
    context
        .history_entries_raw()
        .into_iter()
        .map(|entry| {
            let kind = match entry.kind {
                HistoryEntryKindRaw::Expr(expr) => HistoryOverviewKind::Expr { expr },
                HistoryEntryKindRaw::Eq { lhs, rhs } => HistoryOverviewKind::Eq { lhs, rhs },
            };
            HistoryOverviewEntry { id: entry.id, kind }
        })
        .collect()
}
