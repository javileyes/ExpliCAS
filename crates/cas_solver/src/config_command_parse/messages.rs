pub(crate) fn config_usage_message() -> &'static str {
    "Usage: config <list|enable|disable|save|restore> [rule]"
}

pub(crate) fn config_rule_usage_message(action: &str) -> String {
    format!("Usage: config {} <rule>", action)
}

pub(crate) fn config_unknown_subcommand_message(subcommand: &str) -> String {
    format!("Unknown config command: {}", subcommand)
}
