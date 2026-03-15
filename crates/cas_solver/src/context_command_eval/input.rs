use crate::context_command_format::{
    format_context_current_message, format_context_set_message, format_context_unknown_message,
};
use crate::context_command_parse::parse_context_command_input;
use cas_api_models::{ContextCommandInput, ContextCommandResult, EvalContextMode};

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
            message: format_context_set_message(context_mode_from_eval(mode)),
        },
        ContextCommandInput::UnknownMode(mode) => ContextCommandResult::Invalid {
            message: format_context_unknown_message(&mode),
        },
    }
}

fn context_mode_from_eval(mode: EvalContextMode) -> crate::ContextMode {
    match mode {
        EvalContextMode::Auto => crate::ContextMode::Auto,
        EvalContextMode::Standard => crate::ContextMode::Standard,
        EvalContextMode::Solve => crate::ContextMode::Solve,
        EvalContextMode::Integrate => crate::ContextMode::IntegratePrep,
    }
}
