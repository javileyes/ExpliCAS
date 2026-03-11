#[cfg(test)]
mod tests {
    use crate::solver_exports::{
        apply_set_command_plan_on_repl_core, evaluate_set_command_input,
        evaluate_set_command_on_repl_core, set_command_state_for_repl_core, ReplSetMessageKind,
        SetCommandResult, SetDisplayMode,
    };
    use cas_solver_core::eval_option_axes::StepsMode;

    #[test]
    fn set_command_state_for_repl_core_reads_runtime() {
        let core = crate::ReplCore::new();
        let state = set_command_state_for_repl_core(&core, SetDisplayMode::Normal);
        assert_eq!(state.display_mode, SetDisplayMode::Normal);
    }

    #[test]
    fn apply_set_command_plan_on_repl_core_applies_steps_mode() {
        let mut core = crate::ReplCore::new();
        let state = set_command_state_for_repl_core(&core, SetDisplayMode::Normal);
        let plan = match evaluate_set_command_input("set steps compact", state) {
            SetCommandResult::Apply { plan } => plan,
            other => panic!("unexpected result: {other:?}"),
        };
        let effects = apply_set_command_plan_on_repl_core(&mut core, &plan);
        assert_eq!(effects.set_steps_mode, Some(StepsMode::Compact));
        assert_eq!(core.eval_options().steps_mode, StepsMode::Compact);
    }

    #[test]
    fn evaluate_set_command_on_repl_core_applies_and_reports_display_mode() {
        let mut core = crate::ReplCore::new();
        let out = evaluate_set_command_on_repl_core(
            "set steps verbose",
            &mut core,
            SetDisplayMode::Normal,
        );
        assert_eq!(out.message_kind, ReplSetMessageKind::Info);
        assert_eq!(out.set_display_mode, Some(SetDisplayMode::Verbose));
    }
}
