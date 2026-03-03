use crate::config_command_parse::{
    config_rule_usage_message, config_unknown_subcommand_message, config_usage_message,
    format_simplifier_toggle_config, parse_config_command_input,
};
use crate::config_command_types::{
    ConfigCommandApplyOutput, ConfigCommandInput, ConfigCommandResult,
};

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
            message: format_simplifier_toggle_config(toggles),
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

/// Evaluate and apply `config ...` directly to persisted config.
///
/// This consolidates command parsing, config persistence, and toggle updates.
pub fn evaluate_and_apply_config_command(
    line: &str,
    config: &mut crate::CasConfig,
) -> ConfigCommandApplyOutput {
    match evaluate_config_command(line, crate::solver_toggle_config_from_cas_config(config)) {
        ConfigCommandResult::ShowList { message } => ConfigCommandApplyOutput {
            message,
            sync_simplifier: false,
        },
        ConfigCommandResult::SaveRequested => match config.save() {
            Ok(_) => ConfigCommandApplyOutput {
                message: "Configuration saved to cas_config.toml".to_string(),
                sync_simplifier: false,
            },
            Err(error) => ConfigCommandApplyOutput {
                message: format!("Error saving configuration: {error}"),
                sync_simplifier: false,
            },
        },
        ConfigCommandResult::RestoreRequested => {
            *config = crate::CasConfig::restore();
            ConfigCommandApplyOutput {
                message: "Configuration restored to defaults.".to_string(),
                sync_simplifier: true,
            }
        }
        ConfigCommandResult::ApplyToggleConfig { toggles, message } => {
            crate::apply_solver_toggle_to_cas_config(config, toggles);
            ConfigCommandApplyOutput {
                message,
                sync_simplifier: true,
            }
        }
        ConfigCommandResult::Error { message } => ConfigCommandApplyOutput {
            message,
            sync_simplifier: false,
        },
    }
}
