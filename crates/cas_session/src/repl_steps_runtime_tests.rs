#[cfg(test)]
mod tests {
    use crate::repl_steps_runtime::{
        apply_steps_command_update_on_repl_core, steps_command_state_for_repl_core,
    };

    #[test]
    fn steps_command_state_for_repl_core_reads_state() {
        let core = crate::ReplCore::new();
        let state = steps_command_state_for_repl_core(&core, crate::StepsDisplayMode::Normal);
        assert_eq!(state.display_mode, crate::StepsDisplayMode::Normal);
    }

    #[test]
    fn apply_steps_command_update_on_repl_core_updates_engine_and_options() {
        let mut core = crate::ReplCore::new();
        let effects = apply_steps_command_update_on_repl_core(
            &mut core,
            Some(crate::StepsMode::Compact),
            Some(crate::StepsDisplayMode::Succinct),
        );
        assert_eq!(effects.set_steps_mode, Some(crate::StepsMode::Compact));
        assert_eq!(
            effects.set_display_mode,
            Some(crate::StepsDisplayMode::Succinct)
        );
        assert_eq!(core.eval_options().steps_mode, crate::StepsMode::Compact);
    }
}
