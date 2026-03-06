//! Runtime trait adapters for `ReplCore`.

use crate::{ReplCore, SessionState};
use cas_solver::{
    Engine, EvalCommandError, EvalCommandOutput, EvalOptions, PipelineStats,
    ReplConfiguredRuntimeContext, ReplEngineRuntimeContext, ReplEvalRuntimeContext,
    ReplHealthRuntimeContext, ReplRuntimeStateContext, ReplSemanticsRuntimeContext,
    ReplSessionEngineRuntimeContext, ReplSessionRuntimeContext,
    ReplSessionSimplifierRuntimeContext, ReplSessionStateMutRuntimeContext,
    ReplSessionViewRuntimeContext, ReplSetRuntimeContext, ReplSimplifierRuntimeContext,
    ReplSolveRuntimeContext, ReplStepsRuntimeContext, SetCommandApplyEffects, SetCommandPlan,
    SetCommandState, SetDisplayMode, Simplifier, SimplifyOptions, StepsCommandApplyEffects,
    StepsDisplayMode, StepsMode,
};

impl ReplEvalRuntimeContext for ReplCore {
    fn debug_mode(&self) -> bool {
        ReplCore::debug_mode(self)
    }

    fn evaluate_eval_command_output(
        &mut self,
        line: &str,
        debug_mode: bool,
    ) -> Result<EvalCommandOutput, EvalCommandError> {
        self.with_engine_and_state(|engine, state| {
            crate::evaluate_eval_command_output(engine, state, line, debug_mode)
        })
    }

    fn profile_cache_len(&self) -> usize {
        ReplCore::profile_cache_len(self)
    }
}

impl ReplRuntimeStateContext for ReplCore {
    fn clear_state(&mut self) {
        ReplCore::clear_state(self);
    }

    fn set_debug_mode(&mut self, value: bool) {
        ReplCore::set_debug_mode(self, value);
    }

    fn set_last_stats(&mut self, value: Option<PipelineStats>) {
        ReplCore::set_last_stats(self, value);
    }

    fn set_health_enabled(&mut self, value: bool) {
        ReplCore::set_health_enabled(self, value);
    }

    fn set_last_health_report(&mut self, value: Option<String>) {
        ReplCore::set_last_health_report(self, value);
    }

    fn clear_profile_cache(&mut self) {
        ReplCore::clear_profile_cache(self);
    }

    fn eval_options(&self) -> &EvalOptions {
        ReplCore::eval_options(self)
    }
}

impl ReplSimplifierRuntimeContext for ReplCore {
    fn simplifier_mut(&mut self) -> &mut Simplifier {
        ReplCore::simplifier_mut(self)
    }
}

impl ReplConfiguredRuntimeContext for ReplCore {
    fn replace_simplifier(&mut self, simplifier: Simplifier) {
        ReplCore::set_simplifier(self, simplifier);
    }
}

impl ReplHealthRuntimeContext for ReplCore {
    fn simplifier(&self) -> &Simplifier {
        ReplCore::simplifier(self)
    }

    fn simplifier_mut(&mut self) -> &mut Simplifier {
        ReplCore::simplifier_mut(self)
    }

    fn health_enabled(&self) -> bool {
        ReplCore::health_enabled(self)
    }

    fn set_health_enabled(&mut self, value: bool) {
        ReplCore::set_health_enabled(self, value);
    }

    fn last_stats(&self) -> Option<&PipelineStats> {
        ReplCore::last_stats(self)
    }

    fn last_health_report(&self) -> Option<&str> {
        ReplCore::last_health_report(self)
    }

    fn clear_last_health_report(&mut self) {
        ReplCore::clear_last_health_report(self);
    }

    fn set_last_health_report(&mut self, value: Option<String>) {
        ReplCore::set_last_health_report(self, value);
    }
}

impl ReplSemanticsRuntimeContext for ReplCore {
    fn eval_options_mut(&mut self) -> &mut EvalOptions {
        ReplCore::eval_options_mut(self)
    }

    fn with_simplify_and_eval_options_mut<R>(
        &mut self,
        f: impl FnOnce(&mut SimplifyOptions, &mut EvalOptions) -> R,
    ) -> R {
        ReplCore::with_simplify_and_eval_options_mut(self, f)
    }

    fn rebuild_simplifier_from_profile(&mut self) {
        ReplCore::rebuild_simplifier_from_profile(self);
    }
}

impl ReplSessionRuntimeContext for ReplCore {
    type State = SessionState;

    fn state(&self) -> &Self::State {
        ReplCore::state(self)
    }
}

impl ReplSessionViewRuntimeContext for ReplCore {
    fn simplifier_context(&self) -> &cas_ast::Context {
        &ReplCore::simplifier(self).context
    }
}

impl ReplSessionStateMutRuntimeContext for ReplCore {
    fn state_mut(&mut self) -> &mut Self::State {
        ReplCore::state_mut(self)
    }
}

impl ReplSessionSimplifierRuntimeContext for ReplCore {
    fn with_state_and_simplifier_mut<R>(
        &mut self,
        f: impl FnOnce(&mut Self::State, &mut Simplifier) -> R,
    ) -> R {
        ReplCore::with_state_and_simplifier_mut(self, f)
    }
}

impl ReplSessionEngineRuntimeContext for ReplCore {
    fn with_engine_and_state<R>(
        &mut self,
        f: impl FnOnce(&mut Engine, &mut Self::State) -> R,
    ) -> R {
        ReplCore::with_engine_and_state(self, f)
    }
}

impl ReplEngineRuntimeContext for ReplCore {
    type Engine = Engine;

    fn with_engine_mut<R>(&mut self, f: impl FnOnce(&mut Self::Engine) -> R) -> R {
        ReplCore::with_engine_and_state(self, |engine, _state| f(engine))
    }
}

impl ReplSolveRuntimeContext for ReplCore {
    fn debug_mode(&self) -> bool {
        ReplCore::debug_mode(self)
    }
}

impl ReplSetRuntimeContext for ReplCore {
    fn set_command_state(&self, display_mode: SetDisplayMode) -> SetCommandState {
        SetCommandState {
            transform: self.simplify_options().enable_transform,
            rationalize: self.simplify_options().rationalize.auto_level,
            heuristic_poly: self.simplify_options().shared.heuristic_poly,
            autoexpand_binomials: self.simplify_options().shared.autoexpand_binomials,
            steps_mode: self.eval_options().steps_mode,
            display_mode,
            max_rewrites: self.simplify_options().budgets.max_total_rewrites,
            debug_mode: self.debug_mode(),
        }
    }

    fn apply_set_command_plan(&mut self, plan: &SetCommandPlan) -> SetCommandApplyEffects {
        let mut debug_mode = self.debug_mode();
        let effects = self.with_simplify_and_eval_options_mut(|simplify_options, eval_options| {
            crate::apply_set_command_plan(plan, simplify_options, eval_options, &mut debug_mode)
        });
        self.set_debug_mode(debug_mode);

        if let Some(mode) = effects.set_steps_mode {
            self.simplifier_mut().set_steps_mode(mode);
        }

        effects
    }
}

impl ReplStepsRuntimeContext for ReplCore {
    fn steps_mode_current(&self) -> StepsMode {
        self.eval_options().steps_mode
    }

    fn apply_steps_effects_to_eval_options(
        &mut self,
        set_steps_mode: Option<StepsMode>,
        set_display_mode: Option<StepsDisplayMode>,
    ) -> StepsCommandApplyEffects {
        crate::apply_steps_command_update(set_steps_mode, set_display_mode, self.eval_options_mut())
    }

    fn set_simplifier_steps_mode(&mut self, mode: StepsMode) {
        self.simplifier_mut().set_steps_mode(mode);
    }
}
