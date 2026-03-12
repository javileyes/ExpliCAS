use cas_solver_core::profile_command_types::ProfileCommandInput;

/// Parse raw `profile ...` command input.
pub fn parse_profile_command_input(line: &str) -> ProfileCommandInput {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() == 1 {
        return ProfileCommandInput::ShowReport;
    }
    match parts[1] {
        "enable" => ProfileCommandInput::Enable,
        "disable" => ProfileCommandInput::Disable,
        "clear" => ProfileCommandInput::Clear,
        _ => ProfileCommandInput::Invalid,
    }
}
