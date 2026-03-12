use crate::command_api::eval::{EvalCommandError, EvalCommandOutput};

/// Runtime context needed for REPL eval/expand render-plan orchestration.
pub trait ReplEvalRuntimeContext {
    fn debug_mode(&self) -> bool;
    fn evaluate_eval_command_output(
        &mut self,
        line: &str,
        debug_mode: bool,
    ) -> Result<EvalCommandOutput, EvalCommandError>;
    fn profile_cache_len(&self) -> usize;
}

/// Return profile cache size for the current runtime engine.
pub fn profile_cache_len_on_runtime<C: ReplEvalRuntimeContext>(context: &C) -> usize {
    context.profile_cache_len()
}
