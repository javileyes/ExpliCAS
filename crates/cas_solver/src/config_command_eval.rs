use crate::config_command_parse::{
    config_rule_usage_message, config_unknown_subcommand_message, config_usage_message,
    format_simplifier_toggle_config, parse_config_command_input,
};
use cas_api_models::{ConfigCommandInput, ConfigCommandResult, SimplifierToggleState};
use cas_solver_core::simplifier_config::SimplifierToggleConfig;

/// Evaluate `config ...` command into an actionable result.
///
/// This keeps parsing, validation, and user-facing messaging in one place,
/// leaving callers to apply only infrastructure effects (save/restore/sync).
pub fn evaluate_config_command(line: &str, toggles: SimplifierToggleConfig) -> ConfigCommandResult {
    match parse_config_command_input(line) {
        ConfigCommandInput::List => ConfigCommandResult::ShowList {
            message: format_simplifier_toggle_config(toggles),
        },
        ConfigCommandInput::Save => ConfigCommandResult::SaveRequested,
        ConfigCommandInput::Restore => ConfigCommandResult::RestoreRequested,
        ConfigCommandInput::SetRule { rule, enable } => {
            let mut next = toggles;
            match crate::simplifier_setup_toggle::set_simplifier_toggle_rule(
                &mut next, &rule, enable,
            ) {
                Ok(()) => ConfigCommandResult::ApplyToggleConfig {
                    toggles: simplifier_toggle_state_from_runtime(next),
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

fn simplifier_toggle_state_from_runtime(toggles: SimplifierToggleConfig) -> SimplifierToggleState {
    SimplifierToggleState {
        distribute: toggles.distribute,
        expand_binomials: toggles.expand_binomials,
        distribute_constants: toggles.distribute_constants,
        factor_difference_squares: toggles.factor_difference_squares,
        root_denesting: toggles.root_denesting,
        trig_double_angle: toggles.trig_double_angle,
        trig_angle_sum: toggles.trig_angle_sum,
        log_split_exponents: toggles.log_split_exponents,
        rationalize_denominator: toggles.rationalize_denominator,
        canonicalize_trig_square: toggles.canonicalize_trig_square,
        auto_factor: toggles.auto_factor,
    }
}
