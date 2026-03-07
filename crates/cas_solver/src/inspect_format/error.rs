use crate::InspectHistoryEntryInputError;

/// Format inspection command errors as user-facing messages.
pub fn format_inspect_history_entry_error_message(error: &InspectHistoryEntryInputError) -> String {
    match error {
        InspectHistoryEntryInputError::InvalidId => {
            "Error: Invalid entry ID. Use 'show #N' or 'show N'.".to_string()
        }
        InspectHistoryEntryInputError::NotFound { id } => format!(
            "Error: Entry #{} not found.\nHint: Use 'history' to see available entries.",
            id
        ),
    }
}
