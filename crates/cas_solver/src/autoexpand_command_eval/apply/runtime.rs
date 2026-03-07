use crate::autoexpand_command_types::{
    AutoexpandCommandApplyOutput, AutoexpandCommandResult, AutoexpandCommandState,
};

use super::super::state::autoexpand_budget_view_from_options;

pub(super) fn apply_autoexpand_policy_to_options(
    policy: crate::ExpandPolicy,
    eval_options: &mut crate::EvalOptions,
) -> bool {
    if eval_options.shared.expand_policy == policy {
        return false;
    }
    eval_options.shared.expand_policy = policy;
    true
}

pub(super) fn evaluate_and_apply_autoexpand_command(
    line: &str,
    eval_options: &mut crate::EvalOptions,
) -> AutoexpandCommandApplyOutput {
    let state = AutoexpandCommandState {
        policy: eval_options.shared.expand_policy,
        budget: autoexpand_budget_view_from_options(eval_options),
    };
    match super::eval::evaluate_autoexpand_command_input(line, state) {
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
