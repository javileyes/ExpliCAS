use cas_ast::ExprId;
use cas_session_core::types::EntryId;

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
    Eq { lhs: ExprId, rhs: ExprId },
}

/// Expression entry diagnostics and derived forms.
#[derive(Debug, Clone)]
pub struct HistoryExprInspection {
    pub parsed: ExprId,
    pub resolved: Option<ExprId>,
    pub simplified: Option<ExprId>,
    pub required_conditions: Vec<crate::ImplicitCondition>,
    pub domain_warnings: Vec<crate::DomainWarning>,
    pub blocked_hints: Vec<crate::BlockedHint>,
}
