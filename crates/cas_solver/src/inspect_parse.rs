use cas_session_core::types::EntryId;

use crate::ParseHistoryEntryIdError;

/// Parse a `show`-style ID token (supports optional `#` prefix).
pub fn parse_history_entry_id(input: &str) -> Result<EntryId, ParseHistoryEntryIdError> {
    let token = input.trim().trim_start_matches('#');
    if token.is_empty() {
        return Err(ParseHistoryEntryIdError::Invalid);
    }
    token
        .parse::<EntryId>()
        .map_err(|_| ParseHistoryEntryIdError::Invalid)
}
