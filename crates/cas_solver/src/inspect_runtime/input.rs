use super::{inspect_history_entry, InspectHistoryContext};
use crate::{Engine, HistoryEntryInspection, InspectHistoryEntryInputError};

/// Parse and inspect a history entry in one call for command handlers.
pub fn inspect_history_entry_input<C: InspectHistoryContext>(
    context: &mut C,
    engine: &mut Engine,
    input: &str,
) -> Result<HistoryEntryInspection, InspectHistoryEntryInputError> {
    let id = crate::parse_history_entry_id(input)
        .map_err(|_| InspectHistoryEntryInputError::InvalidId)?;
    inspect_history_entry(context, engine, id).ok_or(InspectHistoryEntryInputError::NotFound { id })
}
