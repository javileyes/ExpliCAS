use crate::{
    build_eval_command_render_plan, evaluate_expand_wrapped_expression, EvalCommandError,
    EvalCommandOutput, EvalCommandRenderPlan,
};

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

/// Evaluate REPL expression and return a frontend-agnostic render plan.
pub fn evaluate_eval_command_render_plan_on_runtime<C: ReplEvalRuntimeContext>(
    context: &mut C,
    line: &str,
    verbosity_is_none: bool,
) -> Result<EvalCommandRenderPlan, String> {
    let debug_mode = context.debug_mode();
    let output = context
        .evaluate_eval_command_output(line, debug_mode)
        .map_err(|error| match error {
            EvalCommandError::Parse(parse_error) => crate::render_parse_error(line, &parse_error),
            EvalCommandError::Eval(message) => message,
        })?;

    Ok(build_eval_command_render_plan(output, verbosity_is_none))
}

/// Evaluate `expand ...` and return a frontend-agnostic render plan.
pub fn evaluate_expand_command_render_plan_on_runtime<C: ReplEvalRuntimeContext>(
    context: &mut C,
    line: &str,
    verbosity_is_none: bool,
) -> Result<EvalCommandRenderPlan, String> {
    let wrapped = evaluate_expand_wrapped_expression(line)?;
    evaluate_eval_command_render_plan_on_runtime(context, &wrapped, verbosity_is_none)
}

/// Return profile cache size for the current runtime engine.
pub fn profile_cache_len_on_runtime<C: ReplEvalRuntimeContext>(context: &C) -> usize {
    context.profile_cache_len()
}
