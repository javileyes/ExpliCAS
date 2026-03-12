#[cfg(test)]
mod tests {
    use crate::session_api::runtime::{
        evaluate_eval_command_output, evaluate_eval_command_render_plan_on_repl_core,
        evaluate_expand_command_render_plan_on_repl_core, profile_cache_len_on_repl_core,
        ReplEvalRuntimeContext,
    };
    use crate::session_api::types::{EvalCommandError, EvalCommandOutput};
    use crate::Engine;

    struct MockReplEvalRuntime {
        engine: Engine,
        state: cas_session_core::eval::StatelessEvalSession<
            crate::EvalOptions,
            crate::DomainMode,
            crate::RequiredItem,
            crate::Step,
            crate::Diagnostics,
        >,
        debug_mode: bool,
    }

    impl MockReplEvalRuntime {
        fn new() -> Self {
            Self {
                engine: Engine::new(),
                state: cas_session_core::eval::StatelessEvalSession::new(
                    crate::EvalOptions::default(),
                ),
                debug_mode: false,
            }
        }
    }

    impl ReplEvalRuntimeContext for MockReplEvalRuntime {
        fn debug_mode(&self) -> bool {
            self.debug_mode
        }

        fn evaluate_eval_command_output(
            &mut self,
            line: &str,
            debug_mode: bool,
        ) -> Result<EvalCommandOutput, EvalCommandError> {
            evaluate_eval_command_output(&mut self.engine, &mut self.state, line, debug_mode)
        }

        fn profile_cache_len(&self) -> usize {
            self.engine.profile_cache_len()
        }
    }

    #[test]
    fn evaluate_eval_command_render_plan_on_runtime_returns_result() {
        let mut runtime = MockReplEvalRuntime::new();
        let plan = evaluate_eval_command_render_plan_on_repl_core(&mut runtime, "2+2", false)
            .expect("plan");
        assert!(plan.result_message.is_some());
    }

    #[test]
    fn evaluate_expand_command_render_plan_on_runtime_handles_expand() {
        let mut runtime = MockReplEvalRuntime::new();
        let plan =
            evaluate_expand_command_render_plan_on_repl_core(&mut runtime, "expand (x+1)^2", false)
                .expect("plan");
        assert!(plan.result_message.is_some());
    }

    #[test]
    fn profile_cache_len_on_runtime_reads_value() {
        let runtime = MockReplEvalRuntime::new();
        assert_eq!(profile_cache_len_on_repl_core(&runtime), 0);
    }
}
