use crate::{evaluate_expand_wrapped_expression, EvalCommandRenderPlan};

use super::{evaluate_eval_command_render_plan_on_runtime, ReplEvalRuntimeContext};

/// Evaluate `expand ...` and return a frontend-agnostic render plan.
pub fn evaluate_expand_command_render_plan_on_runtime<C: ReplEvalRuntimeContext>(
    context: &mut C,
    line: &str,
    verbosity_is_none: bool,
) -> Result<EvalCommandRenderPlan, String> {
    let wrapped = evaluate_expand_wrapped_expression(line)?;
    evaluate_eval_command_render_plan_on_runtime(context, &wrapped, verbosity_is_none)
}
