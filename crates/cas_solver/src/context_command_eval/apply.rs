use cas_api_models::{ContextCommandApplyOutput, ContextCommandResult, EvalContextMode};

/// Apply context mode into eval options, returning whether mode changed.
pub fn apply_context_mode_to_options(
    mode: EvalContextMode,
    eval_options: &mut crate::EvalOptions,
) -> bool {
    let runtime_mode = context_mode_from_eval(mode);
    if eval_options.shared.context_mode == runtime_mode {
        return false;
    }
    eval_options.shared.context_mode = runtime_mode;
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

fn context_mode_from_eval(mode: EvalContextMode) -> crate::ContextMode {
    match mode {
        EvalContextMode::Auto => crate::ContextMode::Auto,
        EvalContextMode::Standard => crate::ContextMode::Standard,
        EvalContextMode::Solve => crate::ContextMode::Solve,
        EvalContextMode::Integrate => crate::ContextMode::IntegratePrep,
    }
}
