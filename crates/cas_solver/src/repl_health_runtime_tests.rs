#[cfg(test)]
mod tests {
    use crate::{
        evaluate_health_command_message_on_runtime, update_health_report_on_runtime, PipelineStats,
        ReplHealthRuntimeContext, Simplifier,
    };

    struct MockReplHealthRuntime {
        simplifier: Simplifier,
        health_enabled: bool,
        last_stats: Option<PipelineStats>,
        last_health_report: Option<String>,
    }

    impl MockReplHealthRuntime {
        fn new() -> Self {
            Self {
                simplifier: Simplifier::with_default_rules(),
                health_enabled: false,
                last_stats: None,
                last_health_report: None,
            }
        }
    }

    impl ReplHealthRuntimeContext for MockReplHealthRuntime {
        fn simplifier(&self) -> &Simplifier {
            &self.simplifier
        }

        fn simplifier_mut(&mut self) -> &mut Simplifier {
            &mut self.simplifier
        }

        fn health_enabled(&self) -> bool {
            self.health_enabled
        }

        fn set_health_enabled(&mut self, value: bool) {
            self.health_enabled = value;
        }

        fn last_stats(&self) -> Option<&PipelineStats> {
            self.last_stats.as_ref()
        }

        fn last_health_report(&self) -> Option<&str> {
            self.last_health_report.as_deref()
        }

        fn clear_last_health_report(&mut self) {
            self.last_health_report = None;
        }

        fn set_last_health_report(&mut self, value: Option<String>) {
            self.last_health_report = value;
        }
    }

    #[test]
    fn update_health_report_on_runtime_sets_report_only_when_enabled() {
        let mut runtime = MockReplHealthRuntime::new();
        update_health_report_on_runtime(&mut runtime);
        assert!(runtime.last_health_report.is_none());

        runtime.health_enabled = true;
        update_health_report_on_runtime(&mut runtime);
        assert!(runtime.last_health_report.is_some());
    }

    #[test]
    fn evaluate_health_command_message_on_runtime_applies_enabled_flag() {
        let mut runtime = MockReplHealthRuntime::new();
        let output =
            evaluate_health_command_message_on_runtime(&mut runtime, "health on").expect("output");
        assert!(output.contains("ENABLED"));
        assert!(runtime.health_enabled);
    }

    #[test]
    fn evaluate_health_command_message_on_runtime_clear_resets_report() {
        let mut runtime = MockReplHealthRuntime::new();
        runtime.last_health_report = Some("report".to_string());
        let output = evaluate_health_command_message_on_runtime(&mut runtime, "health clear")
            .expect("output");
        assert!(output.contains("cleared"));
        assert!(runtime.last_health_report.is_none());
    }
}
