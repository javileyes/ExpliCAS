use crate::{
    format_set_help_text, format_set_option_value, SetCommandInput, SetCommandResult,
    SetCommandState,
};

pub use crate::set_command_parse::parse_set_command_input;

/// Evaluate a `set ...` command into a read/update decision.
pub fn evaluate_set_command_input(line: &str, state: SetCommandState) -> SetCommandResult {
    match parse_set_command_input(line) {
        SetCommandInput::ShowAll => SetCommandResult::ShowHelp {
            message: format_set_help_text(state),
        },
        SetCommandInput::ShowOption(option) => SetCommandResult::ShowValue {
            message: format_set_option_value(option, state),
        },
        SetCommandInput::SetOption { option, value } => {
            crate::set_command_options::evaluate_set_option(option, value, state)
        }
    }
}
