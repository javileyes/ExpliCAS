use crate::{EvalOptions, PipelineStats};

/// Runtime context for generic REPL state helpers.
pub trait ReplRuntimeStateContext {
    fn clear_state(&mut self);
    fn set_debug_mode(&mut self, value: bool);
    fn set_last_stats(&mut self, value: Option<PipelineStats>);
    fn set_health_enabled(&mut self, value: bool);
    fn set_last_health_report(&mut self, value: Option<String>);
    fn clear_profile_cache(&mut self);
    fn eval_options(&self) -> &EvalOptions;
}

/// Reset non-config runtime state while preserving the current simplifier/profile setup.
pub fn reset_repl_runtime_state_on_runtime<C: ReplRuntimeStateContext>(context: &mut C) {
    context.clear_state();
    context.set_debug_mode(false);
    context.set_last_stats(None);
    context.set_health_enabled(false);
    context.set_last_health_report(None);
}

/// Build REPL prompt text from current runtime state.
pub fn build_repl_prompt_on_runtime<C: ReplRuntimeStateContext>(context: &C) -> String {
    crate::prompt_display::build_prompt_from_eval_options(context.eval_options())
}

/// Clone current eval options from runtime state.
pub fn eval_options_from_runtime<C: ReplRuntimeStateContext>(context: &C) -> EvalOptions {
    context.eval_options().clone()
}

/// Clear engine profile cache for the active runtime.
pub fn clear_repl_profile_cache_on_runtime<C: ReplRuntimeStateContext>(context: &mut C) {
    context.clear_profile_cache();
}
