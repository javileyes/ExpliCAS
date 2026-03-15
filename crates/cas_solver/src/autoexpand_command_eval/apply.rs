mod eval;
mod runtime;

use cas_api_models::{
    AutoexpandCommandApplyOutput, AutoexpandCommandResult, AutoexpandCommandState, EvalExpandPolicy,
};

/// Evaluate an `autoexpand` command into policy changes + message.
pub fn evaluate_autoexpand_command_input(
    line: &str,
    state: AutoexpandCommandState,
) -> AutoexpandCommandResult {
    eval::evaluate_autoexpand_command_input(line, state)
}

/// Apply autoexpand policy into eval options, returning whether policy changed.
pub fn apply_autoexpand_policy_to_options(
    policy: EvalExpandPolicy,
    eval_options: &mut crate::EvalOptions,
) -> bool {
    runtime::apply_autoexpand_policy_to_options(policy, eval_options)
}

/// Evaluate and apply an `autoexpand` command directly to runtime options.
pub fn evaluate_and_apply_autoexpand_command(
    line: &str,
    eval_options: &mut crate::EvalOptions,
) -> AutoexpandCommandApplyOutput {
    runtime::evaluate_and_apply_autoexpand_command(line, eval_options)
}
