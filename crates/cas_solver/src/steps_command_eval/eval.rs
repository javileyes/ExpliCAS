use crate::steps_command_format::{
    format_steps_collection_set_message, format_steps_current_message,
    format_steps_display_set_message, format_steps_unknown_mode_message,
};
use crate::steps_command_parse::parse_steps_command_input;
use cas_api_models::{
    EvalStepsMode, StepsCommandInput, StepsCommandResult, StepsCommandState, StepsDisplayMode,
};

/// Evaluate a `steps` command into state changes + message.
pub fn evaluate_steps_command_input(line: &str, state: StepsCommandState) -> StepsCommandResult {
    match parse_steps_command_input(line) {
        StepsCommandInput::ShowCurrent => StepsCommandResult::ShowCurrent {
            message: format_steps_current_message(
                steps_mode_from_eval(state.steps_mode),
                state.display_mode,
            ),
        },
        StepsCommandInput::SetCollectionMode(mode) => {
            let display = match mode {
                EvalStepsMode::On => Some(StepsDisplayMode::Normal),
                EvalStepsMode::Off => Some(StepsDisplayMode::None),
                EvalStepsMode::Compact => None,
            };
            StepsCommandResult::Update {
                set_steps_mode: Some(mode),
                set_display_mode: display,
                message: format_steps_collection_set_message(steps_mode_from_eval(mode))
                    .to_string(),
            }
        }
        StepsCommandInput::SetDisplayMode(mode) => {
            let steps_mode = match mode {
                StepsDisplayMode::Verbose
                | StepsDisplayMode::Succinct
                | StepsDisplayMode::Normal => Some(EvalStepsMode::On),
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

fn steps_mode_from_eval(mode: EvalStepsMode) -> crate::StepsMode {
    match mode {
        EvalStepsMode::On => crate::StepsMode::On,
        EvalStepsMode::Off => crate::StepsMode::Off,
        EvalStepsMode::Compact => crate::StepsMode::Compact,
    }
}
