/// Parsed input for the `profile` command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProfileCommandInput {
    ShowReport,
    Enable,
    Disable,
    Clear,
    Invalid,
}

/// Normalized result for `profile` command handling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProfileCommandResult {
    ShowReport,
    SetEnabled { enabled: bool, message: String },
    Clear { message: String },
    Invalid { message: String },
}

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

/// Apply a `profile` command directly to a simplifier and return user-facing text.
pub fn apply_profile_command(simplifier: &mut crate::Simplifier, line: &str) -> String {
    match evaluate_profile_command_input(line) {
        ProfileCommandResult::ShowReport => simplifier.profiler.report(),
        ProfileCommandResult::SetEnabled { enabled, message } => {
            if enabled {
                simplifier.profiler.enable();
            } else {
                simplifier.profiler.disable();
            }
            message
        }
        ProfileCommandResult::Clear { message } => {
            simplifier.profiler.clear();
            message
        }
        ProfileCommandResult::Invalid { message } => message,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        apply_profile_command, evaluate_profile_command_input, parse_profile_command_input,
        ProfileCommandInput,
    };

    #[test]
    fn parse_profile_command_input_enable() {
        assert_eq!(
            parse_profile_command_input("profile enable"),
            ProfileCommandInput::Enable
        );
    }

    #[test]
    fn parse_profile_command_input_invalid() {
        assert_eq!(
            parse_profile_command_input("profile nope"),
            ProfileCommandInput::Invalid
        );
    }

    #[test]
    fn evaluate_profile_command_input_returns_usage_for_invalid() {
        let out = evaluate_profile_command_input("profile nope");
        let text = format!("{out:?}");
        assert!(text.contains("Usage: profile [enable|disable|clear]"));
    }

    #[test]
    fn apply_profile_command_returns_usage_for_invalid() {
        let mut s = crate::Simplifier::with_default_rules();
        let out = apply_profile_command(&mut s, "profile nope");
        assert_eq!(out, "Usage: profile [enable|disable|clear]");
    }

    #[test]
    fn apply_profile_command_enable_and_disable_report_messages() {
        let mut s = crate::Simplifier::with_default_rules();
        assert_eq!(
            apply_profile_command(&mut s, "profile enable"),
            "Profiler enabled."
        );
        assert_eq!(
            apply_profile_command(&mut s, "profile disable"),
            "Profiler disabled."
        );
    }
}
