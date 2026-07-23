pub(super) fn parse_linear_system_invocation_input(line: &str) -> String {
    let rest = line.strip_prefix("solve_system").unwrap_or(line).trim();
    let inner = if rest.starts_with('(') && rest.ends_with(')') && rest.len() >= 2 {
        &rest[1..rest.len() - 1]
    } else {
        rest
    };
    let inner = inner.trim();
    // List-form parity (S4): the REPL accepts `solve_system([eqs], [vars])`
    // with the same desugar the wire uses.
    if inner.starts_with('[') {
        if let Some(spec) = cas_api_models::system_list_body_to_spec(inner) {
            return spec;
        }
    }
    inner.to_string()
}
