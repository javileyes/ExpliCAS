use crate::steps_command_format::{
    format_steps_collection_set_message, format_steps_current_message,
    format_steps_display_set_message, format_steps_unknown_mode_message,
};
use crate::steps_command_parse::parse_steps_command_input;
use crate::steps_command_types::{
    StepsCommandInput, StepsCommandResult, StepsCommandState, StepsDisplayMode,
};

/// Evaluate a `steps` command into state changes + message.
pub fn evaluate_steps_command_input(line: &str, state: StepsCommandState) -> StepsCommandResult {
    match parse_steps_command_input(line) {
        StepsCommandInput::ShowCurrent => StepsCommandResult::ShowCurrent {
            message: format_steps_current_message(state.steps_mode, state.display_mode),
        },
        StepsCommandInput::SetCollectionMode(mode) => {
            let display = match mode {
                crate::StepsMode::On => Some(StepsDisplayMode::Normal),
                crate::StepsMode::Off => Some(StepsDisplayMode::None),
                crate::StepsMode::Compact => None,
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
                | StepsDisplayMode::Normal => Some(crate::StepsMode::On),
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
