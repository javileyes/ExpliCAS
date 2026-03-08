use cas_session_core::types::EntryId;

/// Parse history IDs from whitespace-separated tokens (supports optional `#` prefix).
pub fn parse_history_ids(input: &str) -> Vec<EntryId> {
    input
        .split_whitespace()
        .filter_map(|token| token.trim_start_matches('#').parse::<EntryId>().ok())
        .collect()
}
