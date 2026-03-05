use cas_session_core::types::EntryId;

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
