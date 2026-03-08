pub(super) fn extract_limit_command_tail(line: &str) -> &str {
    line.strip_prefix("limit").unwrap_or(line).trim()
}
