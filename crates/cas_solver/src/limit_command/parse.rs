pub(super) fn extract_limit_command_tail(line: &str) -> &str {
    let line = line.trim();

    if line.starts_with("limit(") && line.ends_with(')') {
        return &line["limit(".len()..line.len() - 1];
    }

    if line.starts_with("lim(") && line.ends_with(')') {
        return &line["lim(".len()..line.len() - 1];
    }

    line.strip_prefix("limit").unwrap_or(line).trim()
}
