pub(crate) fn parse_rationalize_input(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("rationalize").unwrap_or(line).trim();
    if rest.is_empty() {
        None
    } else {
        Some(rest)
    }
}
