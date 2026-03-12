#[cfg(test)]
mod tests {
    use crate::runtime::{EvalOptions, ExpandPolicy, Simplifier, SimplifyOptions};
    use crate::session_api::runtime::{
        apply_autoexpand_command_on_repl_core, apply_context_command_on_repl_core,
        apply_semantics_command_on_repl_core, evaluate_autoexpand_command_on_repl_core,
        evaluate_autoexpand_command_with_config_sync_on_runtime,
        evaluate_context_command_on_repl_core,
        evaluate_context_command_with_config_sync_on_runtime,
        evaluate_semantics_command_on_repl_core,
        evaluate_semantics_command_with_config_sync_on_runtime, ReplSemanticsRuntimeContext,
    };

    struct MockReplSemanticsRuntime {
        eval_options: EvalOptions,
        simplify_options: SimplifyOptions,
        simplifier: Simplifier,
    }

    impl MockReplSemanticsRuntime {
        fn new() -> Self {
            let eval_options = EvalOptions::default();
            let simplify_options = eval_options.to_simplify_options();
            Self {
                eval_options,
                simplify_options,
                simplifier: Simplifier::with_default_rules(),
            }
        }
    }

    impl ReplSemanticsRuntimeContext for MockReplSemanticsRuntime {
        fn eval_options_mut(&mut self) -> &mut EvalOptions {
            &mut self.eval_options
        }

        fn with_simplify_and_eval_options_mut<R>(
            &mut self,
            f: impl FnOnce(&mut SimplifyOptions, &mut EvalOptions) -> R,
        ) -> R {
            f(&mut self.simplify_options, &mut self.eval_options)
        }

        fn rebuild_simplifier_from_profile(&mut self) {
            self.simplifier = Simplifier::with_profile(&self.eval_options);
        }
    }

    #[test]
    fn apply_context_command_on_runtime_updates_mode() {
        let mut runtime = MockReplSemanticsRuntime::new();
        let out = apply_context_command_on_repl_core("context solve", &mut runtime);
        assert!(out.rebuilt_simplifier);
        assert_eq!(
            runtime.eval_options.shared.context_mode,
            crate::ContextMode::Solve
        );
    }

    #[test]
    fn apply_autoexpand_command_on_runtime_updates_policy() {
        let mut runtime = MockReplSemanticsRuntime::new();
        let out = apply_autoexpand_command_on_repl_core("autoexpand on", &mut runtime);
        assert!(out.rebuilt_simplifier);
        assert_eq!(
            runtime.eval_options.shared.expand_policy,
            ExpandPolicy::Auto
        );
    }

    #[test]
    fn apply_semantics_command_on_runtime_updates_domain() {
        let mut runtime = MockReplSemanticsRuntime::new();
        let out = apply_semantics_command_on_repl_core("semantics set domain assume", &mut runtime);
        assert!(out.sync_simplifier);
        assert_eq!(
            runtime.simplify_options.shared.semantics.domain_mode,
            crate::DomainMode::Assume
        );
    }

    #[test]
    fn evaluate_context_command_on_runtime_runs_rebuild_hook() {
        let mut runtime = MockReplSemanticsRuntime::new();
        let mut hook_called = false;
        let message =
            evaluate_context_command_on_repl_core("context solve", &mut runtime, |_runtime| {
                hook_called = true
            });
        assert!(message.contains("Context"));
        assert!(hook_called);
    }

    #[test]
    fn evaluate_autoexpand_command_on_runtime_runs_rebuild_hook() {
        let mut runtime = MockReplSemanticsRuntime::new();
        let mut hook_called = false;
        let message =
            evaluate_autoexpand_command_on_repl_core("autoexpand on", &mut runtime, |_runtime| {
                hook_called = true
            });
        assert!(message.contains("Auto-expand"));
        assert!(hook_called);
    }

    #[test]
    fn evaluate_semantics_command_on_runtime_runs_rebuild_hook() {
        let mut runtime = MockReplSemanticsRuntime::new();
        let mut hook_called = false;
        let message = evaluate_semantics_command_on_repl_core(
            "semantics set domain assume",
            &mut runtime,
            |_runtime| hook_called = true,
        );
        assert!(!message.trim().is_empty());
        assert!(hook_called);
    }

    #[test]
    fn evaluate_context_command_with_config_sync_on_runtime_runs_sync_hook() {
        #[derive(Clone, Copy)]
        struct MockConfig {
            sync_enabled: bool,
        }
        let config = MockConfig { sync_enabled: true };
        let mut runtime = MockReplSemanticsRuntime::new();
        let mut synced = false;
        let message = evaluate_context_command_with_config_sync_on_runtime(
            "context solve",
            &mut runtime,
            &config,
            |_runtime, cfg| synced = cfg.sync_enabled,
        );
        assert!(message.contains("Context"));
        assert!(synced);
    }

    #[test]
    fn evaluate_autoexpand_command_with_config_sync_on_runtime_runs_sync_hook() {
        #[derive(Clone, Copy)]
        struct MockConfig {
            sync_enabled: bool,
        }
        let config = MockConfig { sync_enabled: true };
        let mut runtime = MockReplSemanticsRuntime::new();
        let mut synced = false;
        let message = evaluate_autoexpand_command_with_config_sync_on_runtime(
            "autoexpand on",
            &mut runtime,
            &config,
            |_runtime, cfg| synced = cfg.sync_enabled,
        );
        assert!(message.contains("Auto-expand"));
        assert!(synced);
    }

    #[test]
    fn evaluate_semantics_command_with_config_sync_on_runtime_runs_sync_hook() {
        #[derive(Clone, Copy)]
        struct MockConfig {
            sync_enabled: bool,
        }
        let config = MockConfig { sync_enabled: true };
        let mut runtime = MockReplSemanticsRuntime::new();
        let mut synced = false;
        let message = evaluate_semantics_command_with_config_sync_on_runtime(
            "semantics set domain assume",
            &mut runtime,
            &config,
            |_runtime, cfg| synced = cfg.sync_enabled,
        );
        assert!(!message.trim().is_empty());
        assert!(synced);
    }
}
