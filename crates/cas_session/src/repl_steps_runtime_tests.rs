#[cfg(test)]
mod tests {
    use crate::solver_exports::{
        apply_steps_command_update_on_repl_core, steps_command_state_for_repl_core,
        StepsDisplayMode,
    };
    use crate::StepsMode;

    #[test]
    fn steps_command_state_for_repl_core_reads_state() {
        let core = crate::ReplCore::new();
        let state = steps_command_state_for_repl_core(&core, StepsDisplayMode::Normal);
        assert_eq!(state.display_mode, StepsDisplayMode::Normal);
    }

    #[test]
    fn apply_steps_command_update_on_repl_core_updates_engine_and_options() {
        let mut core = crate::ReplCore::new();
        let effects = apply_steps_command_update_on_repl_core(
            &mut core,
            Some(StepsMode::Compact),
            Some(StepsDisplayMode::Succinct),
        );
        assert_eq!(effects.set_steps_mode, Some(StepsMode::Compact));
        assert_eq!(effects.set_display_mode, Some(StepsDisplayMode::Succinct));
        assert_eq!(core.eval_options().steps_mode, StepsMode::Compact);
    }
}
