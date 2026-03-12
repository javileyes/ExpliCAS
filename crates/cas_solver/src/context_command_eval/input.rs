use crate::context_command_format::{
    format_context_current_message, format_context_set_message, format_context_unknown_message,
};
use crate::context_command_parse::parse_context_command_input;
use cas_solver_core::context_command_types::{ContextCommandInput, ContextCommandResult};

/// Evaluate a `context` command into mode changes + message.
pub fn evaluate_context_command_input(
    line: &str,
    current_mode: crate::ContextMode,
) -> ContextCommandResult {
    match parse_context_command_input(line) {
        ContextCommandInput::ShowCurrent => ContextCommandResult::ShowCurrent {
            message: format_context_current_message(current_mode),
        },
        ContextCommandInput::SetMode(mode) => ContextCommandResult::SetMode {
            mode,
            message: format_context_set_message(mode),
        },
        ContextCommandInput::UnknownMode(mode) => ContextCommandResult::Invalid {
            message: format_context_unknown_message(&mode),
        },
    }
}
