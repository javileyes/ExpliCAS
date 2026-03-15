use cas_api_models::{
    AutoexpandCommandApplyOutput, AutoexpandCommandResult, AutoexpandCommandState, EvalExpandPolicy,
};

use super::super::state::autoexpand_budget_view_from_options;

pub(super) fn apply_autoexpand_policy_to_options(
    policy: EvalExpandPolicy,
    eval_options: &mut crate::EvalOptions,
) -> bool {
    let runtime_policy = expand_policy_from_eval(policy);
    if eval_options.shared.expand_policy == runtime_policy {
        return false;
    }
    eval_options.shared.expand_policy = runtime_policy;
    true
}

pub(super) fn evaluate_and_apply_autoexpand_command(
    line: &str,
    eval_options: &mut crate::EvalOptions,
) -> AutoexpandCommandApplyOutput {
    let state = AutoexpandCommandState {
        policy: eval_expand_policy_from_runtime(eval_options.shared.expand_policy),
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

fn expand_policy_from_eval(policy: EvalExpandPolicy) -> crate::ExpandPolicy {
    match policy {
        EvalExpandPolicy::Auto => crate::ExpandPolicy::Auto,
        EvalExpandPolicy::Off => crate::ExpandPolicy::Off,
    }
}

fn eval_expand_policy_from_runtime(policy: crate::ExpandPolicy) -> EvalExpandPolicy {
    match policy {
        crate::ExpandPolicy::Auto => EvalExpandPolicy::Auto,
        crate::ExpandPolicy::Off => EvalExpandPolicy::Off,
    }
}
