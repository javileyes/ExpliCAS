use crate::config_command_types::ConfigCommandInput;

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
