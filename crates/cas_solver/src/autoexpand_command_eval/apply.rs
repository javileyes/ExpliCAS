use crate::autoexpand_command_format::{
    format_autoexpand_current_message, format_autoexpand_set_message,
    format_autoexpand_unknown_mode_message,
};
use crate::autoexpand_command_parse::parse_autoexpand_command_input;
use crate::autoexpand_command_types::{
    AutoexpandCommandApplyOutput, AutoexpandCommandResult, AutoexpandCommandState,
};

use super::state::autoexpand_budget_view_from_options;

/// Evaluate an `autoexpand` command into policy changes + message.
pub fn evaluate_autoexpand_command_input(
    line: &str,
    state: AutoexpandCommandState,
) -> AutoexpandCommandResult {
    match parse_autoexpand_command_input(line) {
        crate::AutoexpandCommandInput::ShowCurrent => AutoexpandCommandResult::ShowCurrent {
            message: format_autoexpand_current_message(state.policy, state.budget),
        },
        crate::AutoexpandCommandInput::SetPolicy(policy) => AutoexpandCommandResult::SetPolicy {
            policy,
            message: format_autoexpand_set_message(policy, state.budget),
        },
        crate::AutoexpandCommandInput::UnknownMode(mode) => AutoexpandCommandResult::Invalid {
            message: format_autoexpand_unknown_mode_message(&mode),
        },
    }
}

/// Apply autoexpand policy into eval options, returning whether policy changed.
pub fn apply_autoexpand_policy_to_options(
    policy: crate::ExpandPolicy,
    eval_options: &mut crate::EvalOptions,
) -> bool {
    if eval_options.shared.expand_policy == policy {
        return false;
    }
    eval_options.shared.expand_policy = policy;
    true
}

/// Evaluate and apply an `autoexpand` command directly to runtime options.
pub fn evaluate_and_apply_autoexpand_command(
    line: &str,
    eval_options: &mut crate::EvalOptions,
) -> AutoexpandCommandApplyOutput {
    let state = AutoexpandCommandState {
        policy: eval_options.shared.expand_policy,
        budget: autoexpand_budget_view_from_options(eval_options),
    };
    match evaluate_autoexpand_command_input(line, state) {
        AutoexpandCommandResult::ShowCurrent { message } => AutoexpandCommandApplyOutput {
            message,
            rebuild_simplifier: false,
        },
        AutoexpandCommandResult::SetPolicy { policy, message } => AutoexpandCommandApplyOutput {
            message,
            rebuild_simplifier: apply_autoexpand_policy_to_options(policy, eval_options),
        },
        AutoexpandCommandResult::Invalid { message } => AutoexpandCommandApplyOutput {
            message,
            rebuild_simplifier: false,
        },
    }
}
