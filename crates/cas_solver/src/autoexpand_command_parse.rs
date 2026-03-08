use crate::autoexpand_command_types::AutoexpandCommandInput;

/// Parse raw `autoexpand ...` command input.
pub fn parse_autoexpand_command_input(line: &str) -> AutoexpandCommandInput {
    let args: Vec<&str> = line.split_whitespace().collect();
    match args.get(1) {
        None => AutoexpandCommandInput::ShowCurrent,
        Some(&"on") => AutoexpandCommandInput::SetPolicy(crate::ExpandPolicy::Auto),
        Some(&"off") => AutoexpandCommandInput::SetPolicy(crate::ExpandPolicy::Off),
        Some(other) => AutoexpandCommandInput::UnknownMode((*other).to_string()),
    }
}
