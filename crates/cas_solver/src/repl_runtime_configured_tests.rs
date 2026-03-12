#[cfg(test)]
mod tests {
    use crate::runtime::{EvalOptions, PipelineStats, Simplifier};
    use crate::session_api::runtime::{
        build_runtime_with_config, reset_runtime_full_with_config, reset_runtime_with_config,
        ReplConfiguredRuntimeContext, ReplRuntimeStateContext, ReplSimplifierRuntimeContext,
    };

    #[derive(Clone, Copy)]
    struct MockConfig {
        sync_id: u8,
    }

    struct MockRuntime {
        simplifier: Simplifier,
        state_cleared: bool,
        debug_mode: bool,
        health_enabled: bool,
        last_stats: Option<PipelineStats>,
        last_health_report: Option<String>,
        profile_cache_cleared: bool,
        eval_options: EvalOptions,
    }

    impl MockRuntime {
        fn from_simplifier(simplifier: Simplifier) -> Self {
            Self {
                simplifier,
                state_cleared: false,
                debug_mode: true,
                health_enabled: true,
                last_stats: Some(PipelineStats::default()),
                last_health_report: Some("cached".to_string()),
                profile_cache_cleared: false,
                eval_options: EvalOptions::default(),
            }
        }
    }

    impl ReplRuntimeStateContext for MockRuntime {
        fn clear_state(&mut self) {
            self.state_cleared = true;
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

    impl ReplSimplifierRuntimeContext for MockRuntime {
        fn simplifier_mut(&mut self) -> &mut Simplifier {
            &mut self.simplifier
        }
    }

    impl ReplConfiguredRuntimeContext for MockRuntime {
        fn replace_simplifier(&mut self, simplifier: Simplifier) {
            self.simplifier = simplifier;
        }
    }

    #[test]
    fn build_runtime_with_config_runs_sync_hook() {
        let config = MockConfig { sync_id: 7 };
        let runtime = build_runtime_with_config(
            &config,
            |_cfg| Simplifier::with_default_rules(),
            MockRuntime::from_simplifier,
            |simplifier, cfg| {
                simplifier.debug_mode = cfg.sync_id % 2 == 1;
            },
        );
        assert!(runtime.simplifier.debug_mode);
    }

    #[test]
    fn reset_runtime_with_config_replaces_simplifier_and_resets_state() {
        let config = MockConfig { sync_id: 5 };
        let mut runtime = MockRuntime::from_simplifier(Simplifier::with_default_rules());
        runtime.simplifier.debug_mode = false;
        reset_runtime_with_config(
            &mut runtime,
            &config,
            |_cfg| Simplifier::with_default_rules(),
            |simplifier, cfg| {
                simplifier.debug_mode = cfg.sync_id % 2 == 1;
            },
        );

        assert!(runtime.simplifier.debug_mode);
        assert!(runtime.state_cleared);
        assert!(!runtime.debug_mode);
        assert!(!runtime.health_enabled);
        assert!(runtime.last_stats.is_none());
        assert!(runtime.last_health_report.is_none());
    }

    #[test]
    fn reset_runtime_full_with_config_also_clears_profile_cache() {
        let config = MockConfig { sync_id: 2 };
        let mut runtime = MockRuntime::from_simplifier(Simplifier::with_default_rules());
        reset_runtime_full_with_config(
            &mut runtime,
            &config,
            |_cfg| Simplifier::with_default_rules(),
            |simplifier, cfg| {
                simplifier.debug_mode = cfg.sync_id % 2 == 1;
            },
        );
        assert!(runtime.profile_cache_cleared);
        assert!(!runtime.simplifier.debug_mode);
    }
}
