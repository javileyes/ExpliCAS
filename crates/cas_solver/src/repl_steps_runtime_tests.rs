#[cfg(test)]
mod tests {
    use crate::repl_steps_runtime::{
        apply_steps_command_update_on_runtime, steps_command_state_for_runtime,
        ReplStepsRuntimeContext,
    };
    use crate::{
        apply_steps_command_update, EvalOptions, Simplifier, StepsCommandApplyEffects,
        StepsDisplayMode, StepsMode,
    };

    struct MockReplStepsRuntime {
        eval_options: EvalOptions,
        simplifier: Simplifier,
    }

    impl MockReplStepsRuntime {
        fn new() -> Self {
            Self {
                eval_options: EvalOptions::default(),
                simplifier: Simplifier::with_default_rules(),
            }
        }
    }

    impl ReplStepsRuntimeContext for MockReplStepsRuntime {
        fn steps_mode_current(&self) -> StepsMode {
            self.eval_options.steps_mode
        }

        fn apply_steps_effects_to_eval_options(
            &mut self,
            set_steps_mode: Option<StepsMode>,
            set_display_mode: Option<StepsDisplayMode>,
        ) -> StepsCommandApplyEffects {
            apply_steps_command_update(set_steps_mode, set_display_mode, &mut self.eval_options)
        }

        fn set_simplifier_steps_mode(&mut self, mode: StepsMode) {
            self.simplifier.set_steps_mode(mode);
        }
    }

    #[test]
    fn steps_command_state_for_runtime_reads_mode() {
        let runtime = MockReplStepsRuntime::new();
        let state = steps_command_state_for_runtime(&runtime, StepsDisplayMode::Normal);
        assert_eq!(state.steps_mode, StepsMode::On);
    }

    #[test]
    fn apply_steps_command_update_on_runtime_updates_modes() {
        let mut runtime = MockReplStepsRuntime::new();
        let effects = apply_steps_command_update_on_runtime(
            &mut runtime,
            Some(StepsMode::Off),
            Some(StepsDisplayMode::Succinct),
        );
        assert_eq!(effects.set_steps_mode, Some(StepsMode::Off));
        assert_eq!(effects.set_display_mode, Some(StepsDisplayMode::Succinct));
        assert_eq!(runtime.eval_options.steps_mode, StepsMode::Off);
    }
}
