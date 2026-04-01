pub(super) fn parse_telescope_invocation_input(line: &str) -> Option<String> {
    parse_command_invocation_input(line, "telescope")
}

pub(super) fn parse_expand_invocation_input(line: &str) -> Option<String> {
    parse_command_invocation_input(line, "expand")
}

pub(super) fn parse_collect_invocation_input(line: &str) -> Option<(String, String)> {
    let rest = parse_command_invocation_input(line, "collect")?;
    let (expr, var) = crate::input_parse_common::rsplit_ignoring_parens(&rest, ',')?;
    let expr = expr.trim();
    let var = var.trim();
    if expr.is_empty() || var.is_empty() {
        return None;
    }
    Some((expr.to_string(), var.to_string()))
}

pub(super) fn parse_expand_log_invocation_input(line: &str) -> Option<String> {
    parse_command_invocation_input(line, "expand_log")
}

fn parse_command_invocation_input(line: &str, command: &str) -> Option<String> {
    let rest = line.strip_prefix(command).unwrap_or(line).trim();
    if rest.is_empty() {
        None
    } else {
        Some(rest.to_string())
    }
}
