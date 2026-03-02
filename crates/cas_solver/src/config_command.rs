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

#[cfg(test)]
mod tests {
    use super::{parse_config_command_input, ConfigCommandInput};

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
}
