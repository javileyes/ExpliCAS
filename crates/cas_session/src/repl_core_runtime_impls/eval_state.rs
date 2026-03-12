use crate::repl_core::ReplCore;
use cas_solver::runtime::Simplifier;
use cas_solver::session_api::runtime::{
    evaluate_eval_command_output, ReplConfiguredRuntimeContext, ReplEvalRuntimeContext,
    ReplRuntimeStateContext, ReplSimplifierRuntimeContext,
};
use cas_solver::session_api::types::{EvalCommandError, EvalCommandOutput};
use cas_solver_core::eval_options::EvalOptions;
use cas_solver_core::phase_stats::PipelineStats;

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
            evaluate_eval_command_output(engine, state, line, debug_mode)
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
