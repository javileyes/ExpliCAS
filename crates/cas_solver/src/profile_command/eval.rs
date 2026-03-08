use super::parse::parse_profile_command_input;
use super::types::{ProfileCommandInput, ProfileCommandResult};

/// Evaluate a `profile` command into effect + message.
pub fn evaluate_profile_command_input(line: &str) -> ProfileCommandResult {
    match parse_profile_command_input(line) {
        ProfileCommandInput::ShowReport => ProfileCommandResult::ShowReport,
        ProfileCommandInput::Enable => ProfileCommandResult::SetEnabled {
            enabled: true,
            message: "Profiler enabled.".to_string(),
        },
        ProfileCommandInput::Disable => ProfileCommandResult::SetEnabled {
            enabled: false,
            message: "Profiler disabled.".to_string(),
        },
        ProfileCommandInput::Clear => ProfileCommandResult::Clear {
            message: "Profiler statistics cleared.".to_string(),
        },
        ProfileCommandInput::Invalid => ProfileCommandResult::Invalid {
            message: "Usage: profile [enable|disable|clear]".to_string(),
        },
    }
}
