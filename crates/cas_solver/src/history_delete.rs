use crate::{
    format_delete_history_error_message, format_delete_history_result_message, parse_history_ids,
    DeleteHistoryError, DeleteHistoryResult,
};

/// Mutable context required to delete entries from command-style history.
pub trait HistoryDeleteContext {
    fn history_len(&self) -> usize;
    fn history_remove(&mut self, ids: &[u64]);
}

/// Delete history entries based on command-style ID input.
pub fn delete_history_entries<C: HistoryDeleteContext>(
    context: &mut C,
    input: &str,
) -> Result<DeleteHistoryResult, DeleteHistoryError> {
    let requested_ids = parse_history_ids(input);
    if requested_ids.is_empty() {
        return Err(DeleteHistoryError::NoValidIds);
    }

    let before = context.history_len();
    context.history_remove(&requested_ids);
    let removed_count = before.saturating_sub(context.history_len());

    Ok(DeleteHistoryResult {
        requested_ids,
        removed_count,
    })
}

/// Evaluate `del` command and return a user-facing message.
pub fn evaluate_delete_history_command_message<C: HistoryDeleteContext>(
    context: &mut C,
    input: &str,
) -> String {
    match delete_history_entries(context, input) {
        Ok(result) => format_delete_history_result_message(&result),
        Err(error) => format_delete_history_error_message(&error),
    }
}
