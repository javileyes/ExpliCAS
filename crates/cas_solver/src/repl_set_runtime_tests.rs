#[cfg(test)]
mod tests {
    use crate::repl_set_runtime::{
        apply_set_command_plan_on_runtime, evaluate_set_command_on_runtime,
        set_command_state_for_runtime, ReplSetRuntimeContext,
    };
    use crate::{
        apply_set_command_plan, evaluate_set_command_input, EvalOptions, SetCommandApplyEffects,
        SetCommandPlan, SetCommandState, SetDisplayMode, Simplifier, SimplifyOptions,
    };

    struct MockReplSetRuntime {
        simplify_options: SimplifyOptions,
        eval_options: EvalOptions,
        debug_mode: bool,
        simplifier: Simplifier,
    }

    impl MockReplSetRuntime {
        fn new() -> Self {
            Self {
                simplify_options: SimplifyOptions::default(),
                eval_options: EvalOptions::default(),
                debug_mode: false,
                simplifier: Simplifier::with_default_rules(),
            }
        }
    }

    impl ReplSetRuntimeContext for MockReplSetRuntime {
        fn set_command_state(&self, display_mode: SetDisplayMode) -> SetCommandState {
            SetCommandState {
                transform: self.simplify_options.enable_transform,
                rationalize: self.simplify_options.rationalize.auto_level,
                heuristic_poly: self.simplify_options.shared.heuristic_poly,
                autoexpand_binomials: self.simplify_options.shared.autoexpand_binomials,
                steps_mode: self.eval_options.steps_mode,
                display_mode,
                max_rewrites: self.simplify_options.budgets.max_total_rewrites,
                debug_mode: self.debug_mode,
            }
        }

        fn apply_set_command_plan(&mut self, plan: &SetCommandPlan) -> SetCommandApplyEffects {
            let mut debug_mode = self.debug_mode;
            let effects = apply_set_command_plan(
                plan,
                &mut self.simplify_options,
                &mut self.eval_options,
                &mut debug_mode,
            );
            self.debug_mode = debug_mode;
            if let Some(mode) = effects.set_steps_mode {
                self.simplifier.set_steps_mode(mode);
            }
            effects
        }
    }

    #[test]
    fn set_command_state_for_runtime_reads_context() {
        let runtime = MockReplSetRuntime::new();
        let state = set_command_state_for_runtime(&runtime, SetDisplayMode::Normal);
        assert_eq!(state.steps_mode, runtime.eval_options.steps_mode);
    }

    #[test]
    fn apply_set_command_plan_on_runtime_updates_steps_mode() {
        let mut runtime = MockReplSetRuntime::new();
        let state = set_command_state_for_runtime(&runtime, SetDisplayMode::Normal);
        let result = evaluate_set_command_input("set steps off", state);
        let plan = match result {
            crate::SetCommandResult::Apply { plan } => plan,
            other => panic!("expected apply plan, got {:?}", other),
        };
        let effects = apply_set_command_plan_on_runtime(&mut runtime, &plan);
        assert_eq!(effects.set_steps_mode, Some(crate::StepsMode::Off));
    }

    #[test]
    fn evaluate_set_command_on_runtime_applies_display_mode() {
        let mut runtime = MockReplSetRuntime::new();
        let out = evaluate_set_command_on_runtime(
            "set steps verbose",
            &mut runtime,
            SetDisplayMode::Normal,
        );
        assert_eq!(out.set_display_mode, Some(SetDisplayMode::Verbose));
    }
}
