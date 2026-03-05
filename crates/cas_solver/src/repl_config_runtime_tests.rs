#[cfg(test)]
mod tests {
    use crate::{
        evaluate_and_apply_config_command_on_runtime, ConfigCommandApplyOutput,
        ReplSimplifierRuntimeContext, Simplifier,
    };

    #[derive(Default)]
    struct MockConfig {
        sync: bool,
    }

    struct MockReplRuntime {
        simplifier: Simplifier,
    }

    impl MockReplRuntime {
        fn new() -> Self {
            Self {
                simplifier: Simplifier::with_default_rules(),
            }
        }
    }

    impl ReplSimplifierRuntimeContext for MockReplRuntime {
        fn simplifier_mut(&mut self) -> &mut Simplifier {
            &mut self.simplifier
        }
    }

    #[test]
    fn evaluate_and_apply_config_command_on_runtime_syncs_when_requested() {
        let mut config = MockConfig { sync: true };
        let mut runtime = MockReplRuntime::new();
        let mut sync_called = false;

        let message = evaluate_and_apply_config_command_on_runtime(
            "config status",
            &mut config,
            &mut runtime,
            |_line, cfg| ConfigCommandApplyOutput {
                message: "ok".to_string(),
                sync_simplifier: cfg.sync,
            },
            |_simplifier, _cfg| sync_called = true,
        );

        assert_eq!(message, "ok");
        assert!(sync_called);
    }

    #[test]
    fn evaluate_and_apply_config_command_on_runtime_skips_sync_when_not_requested() {
        let mut config = MockConfig { sync: false };
        let mut runtime = MockReplRuntime::new();
        let mut sync_called = false;

        let message = evaluate_and_apply_config_command_on_runtime(
            "config status",
            &mut config,
            &mut runtime,
            |_line, cfg| ConfigCommandApplyOutput {
                message: "ok".to_string(),
                sync_simplifier: cfg.sync,
            },
            |_simplifier, _cfg| sync_called = true,
        );

        assert_eq!(message, "ok");
        assert!(!sync_called);
    }
}
