use crate::{apply_set_command_plan_on_repl_core, set_command_state_for_repl_core};
use crate::{ReplCore, ReplSetCommandOutput, ReplSetMessageKind, SetDisplayMode};

/// Evaluate and apply a full `set ...` command on REPL runtime.
pub fn evaluate_set_command_on_repl_core(
    line: &str,
    core: &mut ReplCore,
    display_mode: SetDisplayMode,
) -> ReplSetCommandOutput {
    let state = set_command_state_for_repl_core(core, display_mode);
    match crate::evaluate_set_command_input(line, state) {
        crate::SetCommandResult::ShowHelp { message } => ReplSetCommandOutput {
            message_kind: ReplSetMessageKind::Output,
            message,
            set_display_mode: None,
        },
        crate::SetCommandResult::ShowValue { message } => ReplSetCommandOutput {
            message_kind: ReplSetMessageKind::Info,
            message,
            set_display_mode: None,
        },
        crate::SetCommandResult::Invalid { message } => ReplSetCommandOutput {
            message_kind: ReplSetMessageKind::Info,
            message,
            set_display_mode: None,
        },
        crate::SetCommandResult::Apply { plan } => {
            let effects = apply_set_command_plan_on_repl_core(core, &plan);
            ReplSetCommandOutput {
                message_kind: ReplSetMessageKind::Info,
                message: plan.message,
                set_display_mode: effects.set_display_mode,
            }
        }
    }
}
