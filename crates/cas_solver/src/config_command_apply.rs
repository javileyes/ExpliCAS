use crate::{evaluate_config_command, ConfigCommandResult, SimplifierToggleConfig};

/// Applied result for `config ...` command against mutable config state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigCommandApplyOutput {
    pub message: String,
    pub sync_simplifier: bool,
}

/// Context abstraction for applying config command effects.
pub trait ConfigCommandApplyContext {
    fn current_toggles(&self) -> SimplifierToggleConfig;
    fn save(&mut self) -> Result<(), String>;
    fn restore_defaults(&mut self);
    fn apply_toggles(&mut self, toggles: SimplifierToggleConfig);
}

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
