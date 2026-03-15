use crate::config_command_eval::evaluate_config_command;
use cas_api_models::{ConfigCommandResult, SimplifierToggleState};
pub use cas_solver_core::config_runtime::{ConfigCommandApplyContext, ConfigCommandApplyOutput};
use cas_solver_core::simplifier_config::SimplifierToggleConfig;

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
            context.apply_toggles(runtime_toggle_config_from_state(toggles));
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

fn runtime_toggle_config_from_state(toggles: SimplifierToggleState) -> SimplifierToggleConfig {
    SimplifierToggleConfig {
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
