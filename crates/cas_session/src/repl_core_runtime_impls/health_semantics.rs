use crate::ReplCore;
use cas_solver::{
    runtime::Simplifier, ReplHealthRuntimeContext, ReplSemanticsRuntimeContext,
    ReplSolveRuntimeContext,
};
use cas_solver_core::eval_options::EvalOptions;
use cas_solver_core::phase_stats::PipelineStats;
use cas_solver_core::simplify_options::SimplifyOptions;

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

impl ReplSolveRuntimeContext for ReplCore {
    fn debug_mode(&self) -> bool {
        ReplCore::debug_mode(self)
    }
}
