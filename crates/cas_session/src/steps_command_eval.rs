use crate::steps_command_format::{
    format_steps_collection_set_message, format_steps_current_message,
    format_steps_display_set_message, format_steps_unknown_mode_message,
};
use crate::steps_command_parse::parse_steps_command_input;
use crate::steps_command_types::{
    StepsCommandApplyEffects, StepsCommandInput, StepsCommandResult, StepsCommandState,
    StepsDisplayMode,
};

/// Evaluate a `steps` command into state changes + message.
pub fn evaluate_steps_command_input(line: &str, state: StepsCommandState) -> StepsCommandResult {
    match parse_steps_command_input(line) {
        StepsCommandInput::ShowCurrent => StepsCommandResult::ShowCurrent {
            message: format_steps_current_message(state.steps_mode, state.display_mode),
        },
        StepsCommandInput::SetCollectionMode(mode) => {
            let display = match mode {
                cas_solver::StepsMode::On => Some(StepsDisplayMode::Normal),
                cas_solver::StepsMode::Off => Some(StepsDisplayMode::None),
                cas_solver::StepsMode::Compact => None,
            };
            StepsCommandResult::Update {
                set_steps_mode: Some(mode),
                set_display_mode: display,
                message: format_steps_collection_set_message(mode).to_string(),
            }
        }
        StepsCommandInput::SetDisplayMode(mode) => {
            let steps_mode = match mode {
                StepsDisplayMode::Verbose
                | StepsDisplayMode::Succinct
                | StepsDisplayMode::Normal => Some(cas_solver::StepsMode::On),
                StepsDisplayMode::None => None,
            };
            StepsCommandResult::Update {
                set_steps_mode: steps_mode,
                set_display_mode: Some(mode),
                message: format_steps_display_set_message(mode).to_string(),
            }
        }
        StepsCommandInput::UnknownMode(mode) => StepsCommandResult::Invalid {
            message: format_steps_unknown_mode_message(&mode),
        },
    }
}

/// Apply `steps` update fields into eval options and return external effects.
pub fn apply_steps_command_update(
    set_steps_mode: Option<cas_solver::StepsMode>,
    set_display_mode: Option<StepsDisplayMode>,
    eval_options: &mut cas_solver::EvalOptions,
) -> StepsCommandApplyEffects {
    let mut changed_steps_mode = None;
    if let Some(mode) = set_steps_mode {
        if eval_options.steps_mode != mode {
            eval_options.steps_mode = mode;
            changed_steps_mode = Some(mode);
        }
    }

    StepsCommandApplyEffects {
        set_steps_mode: changed_steps_mode,
        set_display_mode,
    }
}
