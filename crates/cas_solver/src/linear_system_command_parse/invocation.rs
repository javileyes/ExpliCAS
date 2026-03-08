pub(super) fn parse_linear_system_invocation_input(line: &str) -> String {
    let rest = line.strip_prefix("solve_system").unwrap_or(line).trim();
    let inner = if rest.starts_with('(') && rest.ends_with(')') && rest.len() >= 2 {
        &rest[1..rest.len() - 1]
    } else {
        rest
    };
    inner.trim().to_string()
}
