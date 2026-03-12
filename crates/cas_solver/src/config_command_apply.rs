use crate::config_command_eval::evaluate_config_command;
use cas_solver_core::config_command_types::ConfigCommandResult;
pub use cas_solver_core::config_runtime::{ConfigCommandApplyContext, ConfigCommandApplyOutput};

/// Evaluate and apply `config` command effects against generic mutable config state.
pub fn evaluate_and_apply_config_command<C>(line: &str, context: &mut C) -> ConfigCommandApplyOutput
where
    C: ConfigCommandApplyContext,
{
    match evaluate_config_command(line, context.current_toggles()) {
        ConfigCommandResult::ShowList { message } => ConfigCommandApplyOutput {
            message,
            sync_simplifier: false,
        },
        ConfigCommandResult::SaveRequested => match context.save() {
            Ok(()) => ConfigCommandApplyOutput {
                message: "Configuration saved.".to_string(),
                sync_simplifier: false,
            },
            Err(err) => ConfigCommandApplyOutput {
                message: format!("Failed to save configuration: {err}"),
                sync_simplifier: false,
            },
        },
        ConfigCommandResult::RestoreRequested => {
            context.restore_defaults();
            ConfigCommandApplyOutput {
                message: "Configuration restored to defaults.".to_string(),
                sync_simplifier: true,
            }
        }
        ConfigCommandResult::ApplyToggleConfig { toggles, message } => {
            context.apply_toggles(toggles);
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
