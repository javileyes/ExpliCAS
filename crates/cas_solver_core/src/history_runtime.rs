use cas_ast::ExprId;

/// Raw history kind used to decouple overview/inspection mapping from storage models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoryEntryKindRaw {
    Expr(ExprId),
    Eq { lhs: ExprId, rhs: ExprId },
}

/// Raw history entry used to decouple overview mapping from storage models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HistoryEntryRaw {
    pub id: u64,
    pub kind: HistoryEntryKindRaw,
}

/// Raw history entry payload needed for `show`/inspection flows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoryInspectEntryRaw {
    pub id: u64,
    pub type_str: String,
    pub raw_text: String,
    pub kind: HistoryEntryKindRaw,
}

/// Context that can expose history entries in a raw, storage-agnostic shape.
pub trait HistoryOverviewContext {
    fn history_entries_raw(&self) -> Vec<HistoryEntryRaw>;
}

/// Mutable context required to delete entries from command-style history.
pub trait HistoryDeleteContext {
    fn history_len(&self) -> usize;
    fn history_remove(&mut self, ids: &[u64]);
}
