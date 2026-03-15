use cas_solver_core::blocked_hint::BlockedHint;
use cas_solver_core::domain_condition::ImplicitCondition;
use cas_solver_core::domain_warning::DomainWarning;

/// Unique identifier for a session history entry.
pub type EntryId = u64;

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

/// Errors when parsing a `show`-style history entry identifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseHistoryEntryIdError {
    Invalid,
}

/// Errors when inspecting a history entry from command-style input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InspectHistoryEntryInputError {
    InvalidId,
    NotFound { id: EntryId },
}

/// High-level inspection payload for a history entry.
#[derive(Debug, Clone)]
pub struct HistoryEntryInspection {
    pub id: EntryId,
    pub type_str: String,
    pub raw_text: String,
    pub details: HistoryEntryDetails,
}

/// Per-kind inspection details.
#[derive(Debug, Clone)]
pub enum HistoryEntryDetails {
    Expr(HistoryExprInspection),
    Eq {
        lhs: cas_ast::ExprId,
        rhs: cas_ast::ExprId,
    },
}

/// Expression entry diagnostics and derived forms.
#[derive(Debug, Clone)]
pub struct HistoryExprInspection {
    pub parsed: cas_ast::ExprId,
    pub resolved: Option<cas_ast::ExprId>,
    pub simplified: Option<cas_ast::ExprId>,
    pub required_conditions: Vec<ImplicitCondition>,
    pub domain_warnings: Vec<DomainWarning>,
    pub blocked_hints: Vec<BlockedHint>,
}
