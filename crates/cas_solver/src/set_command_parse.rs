use crate::SetCommandInput;

/// Parse raw `set ...` input.
pub fn parse_set_command_input(line: &str) -> SetCommandInput<'_> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() == 1 || (parts.len() == 2 && parts[1] == "show") {
        return SetCommandInput::ShowAll;
    }
    if parts.len() == 2 {
        return SetCommandInput::ShowOption(parts[1]);
    }
    SetCommandInput::SetOption {
        option: parts[1],
        value: parts[2],
    }
}
