use crate::{
    evaluate_set_command_input, SetCommandApplyEffects, SetCommandPlan, SetCommandResult,
    SetCommandState, SetDisplayMode,
};
use cas_solver_core::repl_set_types::{ReplSetCommandOutput, ReplSetMessageKind};

/// Runtime context needed by `set` command orchestration.
pub trait ReplSetRuntimeContext {
    fn set_command_state(&self, display_mode: SetDisplayMode) -> SetCommandState;
    fn apply_set_command_plan(&mut self, plan: &SetCommandPlan) -> SetCommandApplyEffects;
}

/// Build `set` command state snapshot from runtime context.
pub fn set_command_state_for_runtime<C: ReplSetRuntimeContext>(
    context: &C,
    display_mode: SetDisplayMode,
) -> SetCommandState {
    context.set_command_state(display_mode)
}

/// Apply a `set` mutation plan directly on runtime context.
pub fn apply_set_command_plan_on_runtime<C: ReplSetRuntimeContext>(
    context: &mut C,
    plan: &SetCommandPlan,
) -> SetCommandApplyEffects {
    context.apply_set_command_plan(plan)
}

/// Evaluate and apply a full `set ...` command on runtime context.
pub fn evaluate_set_command_on_runtime<C: ReplSetRuntimeContext>(
    line: &str,
    context: &mut C,
    display_mode: SetDisplayMode,
) -> ReplSetCommandOutput {
    let state = set_command_state_for_runtime(context, display_mode);
    match evaluate_set_command_input(line, state) {
        SetCommandResult::ShowHelp { message } => ReplSetCommandOutput {
            message_kind: ReplSetMessageKind::Output,
            message,
            set_display_mode: None,
        },
        SetCommandResult::ShowValue { message } => ReplSetCommandOutput {
            message_kind: ReplSetMessageKind::Info,
            message,
            set_display_mode: None,
        },
        SetCommandResult::Invalid { message } => ReplSetCommandOutput {
            message_kind: ReplSetMessageKind::Info,
            message,
            set_display_mode: None,
        },
        SetCommandResult::Apply { plan } => {
            let effects = apply_set_command_plan_on_runtime(context, &plan);
            ReplSetCommandOutput {
                message_kind: ReplSetMessageKind::Info,
                message: plan.message,
                set_display_mode: effects.set_display_mode,
            }
        }
    }
}
