use crate::autoexpand_command_format::{
    format_autoexpand_current_message, format_autoexpand_set_message,
    format_autoexpand_unknown_mode_message,
};
use crate::autoexpand_command_parse::parse_autoexpand_command_input;
use crate::autoexpand_command_types::{AutoexpandCommandResult, AutoexpandCommandState};

pub(super) fn evaluate_autoexpand_command_input(
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
