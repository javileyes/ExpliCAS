//! Runtime adapters for `set` command over REPL core state.

use crate::{ReplCore, SetCommandApplyEffects, SetCommandPlan, SetCommandState, SetDisplayMode};

/// Build `set` command state snapshot from REPL runtime.
pub fn set_command_state_for_repl_core(
    core: &ReplCore,
    display_mode: SetDisplayMode,
) -> SetCommandState {
    SetCommandState {
        transform: core.simplify_options().enable_transform,
        rationalize: core.simplify_options().rationalize.auto_level,
        heuristic_poly: core.simplify_options().shared.heuristic_poly,
        autoexpand_binomials: core.simplify_options().shared.autoexpand_binomials,
        steps_mode: core.eval_options().steps_mode,
        display_mode,
        max_rewrites: core.simplify_options().budgets.max_total_rewrites,
        debug_mode: core.debug_mode(),
    }
}

/// Apply a `set` mutation plan directly on REPL runtime.
pub fn apply_set_command_plan_on_repl_core(
    core: &mut ReplCore,
    plan: &SetCommandPlan,
) -> SetCommandApplyEffects {
    let mut debug_mode = core.debug_mode();
    let effects = core.with_simplify_and_eval_options_mut(|simplify_options, eval_options| {
        crate::apply_set_command_plan(plan, simplify_options, eval_options, &mut debug_mode)
    });
    core.set_debug_mode(debug_mode);

    if let Some(mode) = effects.set_steps_mode {
        core.simplifier_mut().set_steps_mode(mode);
    }

    effects
}

/// Message kind for REPL `set` command output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplSetMessageKind {
    Output,
    Info,
}

/// Fully-evaluated REPL `set` command output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplSetCommandOutput {
    pub message_kind: ReplSetMessageKind,
    pub message: String,
    pub set_display_mode: Option<SetDisplayMode>,
}

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

#[cfg(test)]
mod tests {
    #[test]
    fn set_command_state_for_repl_core_reads_runtime() {
        let core = crate::ReplCore::new();
        let state = super::set_command_state_for_repl_core(&core, crate::SetDisplayMode::Normal);
        assert_eq!(state.display_mode, crate::SetDisplayMode::Normal);
    }

    #[test]
    fn apply_set_command_plan_on_repl_core_applies_steps_mode() {
        let mut core = crate::ReplCore::new();
        let state = super::set_command_state_for_repl_core(&core, crate::SetDisplayMode::Normal);
        let plan = match crate::evaluate_set_command_input("set steps compact", state) {
            crate::SetCommandResult::Apply { plan } => plan,
            other => panic!("unexpected result: {other:?}"),
        };
        let effects = super::apply_set_command_plan_on_repl_core(&mut core, &plan);
        assert_eq!(effects.set_steps_mode, Some(cas_solver::StepsMode::Compact));
        assert_eq!(
            core.eval_options().steps_mode,
            cas_solver::StepsMode::Compact
        );
    }

    #[test]
    fn evaluate_set_command_on_repl_core_applies_and_reports_display_mode() {
        let mut core = crate::ReplCore::new();
        let out = super::evaluate_set_command_on_repl_core(
            "set steps verbose",
            &mut core,
            crate::SetDisplayMode::Normal,
        );
        assert_eq!(out.message_kind, super::ReplSetMessageKind::Info);
        assert_eq!(out.set_display_mode, Some(crate::SetDisplayMode::Verbose));
    }
}
