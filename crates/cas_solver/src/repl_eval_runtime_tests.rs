#[cfg(test)]
mod tests {
    use crate::session_api::eval::{
        evaluate_collect_command_render_plan_on_repl_core, evaluate_eval_command_output,
        evaluate_eval_command_render_plan_on_repl_core,
        evaluate_expand_command_render_plan_on_repl_core, profile_cache_len_on_repl_core,
        ReplEvalRuntimeContext,
    };
    use crate::session_api::eval::{EvalCommandError, EvalCommandOutput};
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
    fn evaluate_collect_command_render_plan_on_runtime_handles_collect() {
        let mut runtime = MockReplEvalRuntime::new();
        let plan = evaluate_collect_command_render_plan_on_repl_core(
            &mut runtime,
            "collect a*x + b*x + c, x",
            false,
        )
        .expect("plan");
        let rendered = plan.result_message.as_ref().expect("result").text.as_str();
        assert!(rendered.contains("(a + b)"));
        assert!(rendered.contains("x"));
    }

    #[test]
    fn profile_cache_len_on_runtime_reads_value() {
        let runtime = MockReplEvalRuntime::new();
        assert_eq!(profile_cache_len_on_repl_core(&runtime), 0);
    }

    #[test]
    fn preordered_steps_keep_global_snapshots_for_fraction_sum_pythagorean_chain() {
        let mut runtime = MockReplEvalRuntime::new();
        let output = runtime
            .evaluate_eval_command_output("1/(1 + sin(x)) + 1/(1 - sin(x))", false)
            .expect("eval output");

        let steps = output.steps.as_slice();
        let ctx = &runtime.engine.simplifier.context;
        let diff_squares = steps
            .iter()
            .find(|step| step.rule_name == "Difference of Squares")
            .expect("difference of squares step");
        let pythagorean = steps
            .iter()
            .find(|step| step.rule_name == "Pythagorean Factor Form")
            .expect("pythagorean step");

        let diff_after = cas_formatter::clean_display_string(&format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: ctx,
                id: diff_squares.global_after.expect("global after")
            }
        ));
        let pyth_before = cas_formatter::clean_display_string(&format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: ctx,
                id: pythagorean.global_before.expect("global before")
            }
        ));

        assert_eq!(diff_after, "2 / (1 - sin(x)^2)");
        assert_eq!(pyth_before, "2 / (1 - sin(x)^2)");
    }
}
