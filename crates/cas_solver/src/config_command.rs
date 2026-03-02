/// Parsed input for `config ...` command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigCommandInput {
    List,
    Save,
    Restore,
    SetRule { rule: String, enable: bool },
    MissingRuleArg { action: String },
    InvalidUsage,
    UnknownSubcommand { subcommand: String },
}

/// Evaluated result for `config ...` command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigCommandResult {
    ShowList {
        message: String,
    },
    SaveRequested,
    RestoreRequested,
    ApplyToggleConfig {
        toggles: crate::SimplifierToggleConfig,
        message: String,
    },
    Error {
        message: String,
    },
}

/// Parse raw `config ...` command input.
pub fn parse_config_command_input(line: &str) -> ConfigCommandInput {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 2 {
        return ConfigCommandInput::InvalidUsage;
    }
    match parts[1] {
        "list" => ConfigCommandInput::List,
        "save" => ConfigCommandInput::Save,
        "restore" => ConfigCommandInput::Restore,
        "enable" | "disable" => {
            if parts.len() < 3 {
                return ConfigCommandInput::MissingRuleArg {
                    action: parts[1].to_string(),
                };
            }
            ConfigCommandInput::SetRule {
                rule: parts[2].to_string(),
                enable: parts[1] == "enable",
            }
        }
        other => ConfigCommandInput::UnknownSubcommand {
            subcommand: other.to_string(),
        },
    }
}

pub fn config_usage_message() -> &'static str {
    "Usage: config <list|enable|disable|save|restore> [rule]"
}

pub fn config_rule_usage_message(action: &str) -> String {
    format!("Usage: config {} <rule>", action)
}

pub fn config_unknown_subcommand_message(subcommand: &str) -> String {
    format!("Unknown config command: {}", subcommand)
}

/// Evaluate `config ...` command into an actionable result.
///
/// This keeps parsing, validation, and user-facing messaging in one place,
/// leaving callers to apply only infrastructure effects (save/restore/sync).
pub fn evaluate_config_command(
    line: &str,
    toggles: crate::SimplifierToggleConfig,
) -> ConfigCommandResult {
    match parse_config_command_input(line) {
        ConfigCommandInput::List => ConfigCommandResult::ShowList {
            message: crate::format_simplifier_toggle_config(toggles),
        },
        ConfigCommandInput::Save => ConfigCommandResult::SaveRequested,
        ConfigCommandInput::Restore => ConfigCommandResult::RestoreRequested,
        ConfigCommandInput::SetRule { rule, enable } => {
            let mut next = toggles;
            match crate::set_simplifier_toggle_rule(&mut next, &rule, enable) {
                Ok(()) => ConfigCommandResult::ApplyToggleConfig {
                    toggles: next,
                    message: format!("Rule '{}' set to {}.", rule, enable),
                },
                Err(message) => ConfigCommandResult::Error { message },
            }
        }
        ConfigCommandInput::MissingRuleArg { action } => ConfigCommandResult::Error {
            message: config_rule_usage_message(&action),
        },
        ConfigCommandInput::InvalidUsage => ConfigCommandResult::Error {
            message: config_usage_message().to_string(),
        },
        ConfigCommandInput::UnknownSubcommand { subcommand } => ConfigCommandResult::Error {
            message: config_unknown_subcommand_message(&subcommand),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{
        evaluate_config_command, parse_config_command_input, ConfigCommandInput,
        ConfigCommandResult,
    };

    #[test]
    fn parse_config_command_input_reads_enable_rule() {
        assert_eq!(
            parse_config_command_input("config enable distribute"),
            ConfigCommandInput::SetRule {
                rule: "distribute".to_string(),
                enable: true,
            }
        );
    }

    #[test]
    fn parse_config_command_input_detects_missing_rule() {
        assert_eq!(
            parse_config_command_input("config disable"),
            ConfigCommandInput::MissingRuleArg {
                action: "disable".to_string(),
            }
        );
    }

    #[test]
    fn parse_config_command_input_detects_unknown_subcommand() {
        assert_eq!(
            parse_config_command_input("config nope"),
            ConfigCommandInput::UnknownSubcommand {
                subcommand: "nope".to_string(),
            }
        );
    }

    #[test]
    fn evaluate_config_command_set_rule_applies_toggle_config() {
        let toggles = crate::SimplifierToggleConfig::default();
        let result = evaluate_config_command("config enable distribute", toggles);
        match result {
            ConfigCommandResult::ApplyToggleConfig { toggles, message } => {
                assert!(toggles.distribute);
                assert!(message.contains("Rule 'distribute' set to true."));
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn evaluate_config_command_list_formats_message() {
        let toggles = crate::SimplifierToggleConfig::default();
        let result = evaluate_config_command("config list", toggles);
        match result {
            ConfigCommandResult::ShowList { message } => {
                assert!(message.contains("distribute:"));
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn evaluate_config_command_invalid_usage_returns_error() {
        let toggles = crate::SimplifierToggleConfig::default();
        let result = evaluate_config_command("config", toggles);
        assert!(matches!(result, ConfigCommandResult::Error { .. }));
    }
}
