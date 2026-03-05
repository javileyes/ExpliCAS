use crate::context_command_format::{
    format_context_current_message, format_context_set_message, format_context_unknown_message,
};
use crate::context_command_parse::parse_context_command_input;
use crate::context_command_types::{
    ContextCommandApplyOutput, ContextCommandInput, ContextCommandResult,
};

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

/// Apply context mode into eval options, returning whether mode changed.
pub fn apply_context_mode_to_options(
    mode: crate::ContextMode,
    eval_options: &mut crate::EvalOptions,
) -> bool {
    if eval_options.shared.context_mode == mode {
        return false;
    }
    eval_options.shared.context_mode = mode;
    true
}

/// Evaluate and apply a `context` command directly to runtime options.
pub fn evaluate_and_apply_context_command(
    line: &str,
    eval_options: &mut crate::EvalOptions,
) -> ContextCommandApplyOutput {
    match evaluate_context_command_input(line, eval_options.shared.context_mode) {
        ContextCommandResult::ShowCurrent { message } => ContextCommandApplyOutput {
            message,
            rebuild_simplifier: false,
        },
        ContextCommandResult::SetMode { mode, message } => ContextCommandApplyOutput {
            message,
            rebuild_simplifier: apply_context_mode_to_options(mode, eval_options),
        },
        ContextCommandResult::Invalid { message } => ContextCommandApplyOutput {
            message,
            rebuild_simplifier: false,
        },
    }
}
