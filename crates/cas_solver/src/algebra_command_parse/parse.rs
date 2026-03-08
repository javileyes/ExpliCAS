pub(super) fn parse_telescope_invocation_input(line: &str) -> Option<String> {
    parse_command_invocation_input(line, "telescope")
}

pub(super) fn parse_expand_invocation_input(line: &str) -> Option<String> {
    parse_command_invocation_input(line, "expand")
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
