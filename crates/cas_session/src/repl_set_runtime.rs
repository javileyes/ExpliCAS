//! Runtime adapters for `set` command over REPL core state.

pub use crate::repl_set_apply::{
    apply_set_command_plan_on_repl_core, set_command_state_for_repl_core,
};
pub use crate::repl_set_eval::evaluate_set_command_on_repl_core;
pub use crate::repl_set_types::{ReplSetCommandOutput, ReplSetMessageKind};

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
