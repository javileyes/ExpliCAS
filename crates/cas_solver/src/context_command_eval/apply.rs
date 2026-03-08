use crate::context_command_types::{ContextCommandApplyOutput, ContextCommandResult};

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
    match super::input::evaluate_context_command_input(line, eval_options.shared.context_mode) {
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
