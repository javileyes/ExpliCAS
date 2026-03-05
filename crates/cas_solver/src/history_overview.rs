use crate::{HistoryOverviewEntry, HistoryOverviewKind};

/// Raw history kind used to decouple overview mapping from storage models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoryEntryKindRaw {
    Expr(cas_ast::ExprId),
    Eq {
        lhs: cas_ast::ExprId,
        rhs: cas_ast::ExprId,
    },
}

/// Raw history entry used to decouple overview mapping from storage models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HistoryEntryRaw {
    pub id: u64,
    pub kind: HistoryEntryKindRaw,
}

/// Context that can expose history entries in a raw, storage-agnostic shape.
pub trait HistoryOverviewContext {
    fn history_entries_raw(&self) -> Vec<HistoryEntryRaw>;
}

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
