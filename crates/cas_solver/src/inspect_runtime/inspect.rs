mod expr;

use cas_session_core::types::EntryId;

use super::InspectHistoryContext;
use crate::{Engine, HistoryEntryDetails, HistoryEntryInspection, HistoryEntryKindRaw};

/// Inspect a history entry and compute resolved/simplified forms plus metadata.
///
/// Returns `None` when `id` does not exist.
pub fn inspect_history_entry<C: InspectHistoryContext>(
    context: &mut C,
    engine: &mut Engine,
    id: EntryId,
) -> Option<HistoryEntryInspection> {
    let entry = context.history_entry_raw(id)?;

    let details = match entry.kind {
        HistoryEntryKindRaw::Expr(expr_id) => HistoryEntryDetails::Expr(
            expr::inspect_history_expr(context, engine, &entry.raw_text, expr_id),
        ),
        HistoryEntryKindRaw::Eq { lhs, rhs } => HistoryEntryDetails::Eq { lhs, rhs },
    };

    Some(HistoryEntryInspection {
        id: entry.id,
        type_str: entry.type_str,
        raw_text: entry.raw_text,
        details,
    })
}
