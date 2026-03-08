use crate::{build_eval_command_render_plan, EvalCommandError, EvalCommandRenderPlan};

use super::ReplEvalRuntimeContext;

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
