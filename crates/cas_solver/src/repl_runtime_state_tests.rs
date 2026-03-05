#[cfg(test)]
mod tests {
    use crate::{
        build_repl_prompt_on_runtime, clear_repl_profile_cache_on_runtime,
        eval_options_from_runtime, reset_repl_runtime_state_on_runtime, EvalOptions, PipelineStats,
        ReplRuntimeStateContext,
    };

    struct MockReplRuntimeState {
        cleared_state: bool,
        debug_mode: bool,
        health_enabled: bool,
        last_stats: Option<PipelineStats>,
        last_health_report: Option<String>,
        profile_cache_cleared: bool,
        eval_options: EvalOptions,
    }

    impl MockReplRuntimeState {
        fn new() -> Self {
            Self {
                cleared_state: false,
                debug_mode: true,
                health_enabled: true,
                last_stats: Some(PipelineStats::default()),
                last_health_report: Some("cached".to_string()),
                profile_cache_cleared: false,
                eval_options: EvalOptions::default(),
            }
        }
    }

    impl ReplRuntimeStateContext for MockReplRuntimeState {
        fn clear_state(&mut self) {
            self.cleared_state = true;
        }

        fn set_debug_mode(&mut self, value: bool) {
            self.debug_mode = value;
        }

        fn set_last_stats(&mut self, value: Option<PipelineStats>) {
            self.last_stats = value;
        }

        fn set_health_enabled(&mut self, value: bool) {
            self.health_enabled = value;
        }

        fn set_last_health_report(&mut self, value: Option<String>) {
            self.last_health_report = value;
        }

        fn clear_profile_cache(&mut self) {
            self.profile_cache_cleared = true;
        }

        fn eval_options(&self) -> &EvalOptions {
            &self.eval_options
        }
    }

    #[test]
    fn reset_repl_runtime_state_on_runtime_resets_transient_state() {
        let mut runtime = MockReplRuntimeState::new();
        reset_repl_runtime_state_on_runtime(&mut runtime);
        assert!(runtime.cleared_state);
        assert!(!runtime.debug_mode);
        assert!(!runtime.health_enabled);
        assert!(runtime.last_stats.is_none());
        assert!(runtime.last_health_report.is_none());
    }

    #[test]
    fn build_repl_prompt_on_runtime_uses_eval_options() {
        let runtime = MockReplRuntimeState::new();
        let prompt = build_repl_prompt_on_runtime(&runtime);
        assert_eq!(prompt, "> ");
    }

    #[test]
    fn eval_options_from_runtime_clones_state() {
        let runtime = MockReplRuntimeState::new();
        let options = eval_options_from_runtime(&runtime);
        assert_eq!(options.steps_mode, EvalOptions::default().steps_mode);
    }

    #[test]
    fn clear_repl_profile_cache_on_runtime_marks_cache_cleared() {
        let mut runtime = MockReplRuntimeState::new();
        clear_repl_profile_cache_on_runtime(&mut runtime);
        assert!(runtime.profile_cache_cleared);
    }
}
