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

pub fn config_usage_message() -> &'static str {
    "Usage: config <list|enable|disable|save|restore> [rule]"
}

pub fn config_rule_usage_message(action: &str) -> String {
    format!("Usage: config {} <rule>", action)
}

pub fn config_unknown_subcommand_message(subcommand: &str) -> String {
    format!("Unknown config command: {}", subcommand)
}

pub fn format_simplifier_toggle_config(config: crate::SimplifierToggleConfig) -> String {
    format!(
        "Current Configuration:\n\
           distribute: {}\n\
           expand_binomials: {}\n\
           distribute_constants: {}\n\
           factor_difference_squares: {}\n\
           root_denesting: {}\n\
           trig_double_angle: {}\n\
           trig_angle_sum: {}\n\
           log_split_exponents: {}\n\
           rationalize_denominator: {}\n\
           canonicalize_trig_square: {}\n\
           auto_factor: {}",
        config.distribute,
        config.expand_binomials,
        config.distribute_constants,
        config.factor_difference_squares,
        config.root_denesting,
        config.trig_double_angle,
        config.trig_angle_sum,
        config.log_split_exponents,
        config.rationalize_denominator,
        config.canonicalize_trig_square,
        config.auto_factor
    )
}
