use crate::command_api::eval::EvalCommandRenderPlan;
use crate::evaluate_collect_wrapped_expression;

use super::{evaluate_eval_command_render_plan_on_runtime, ReplEvalRuntimeContext};

/// Evaluate `collect ...` and return a frontend-agnostic render plan.
pub fn evaluate_collect_command_render_plan_on_runtime<C: ReplEvalRuntimeContext>(
    context: &mut C,
    line: &str,
    verbosity_is_none: bool,
) -> Result<EvalCommandRenderPlan, String> {
    let wrapped = evaluate_collect_wrapped_expression(line)?;
    evaluate_eval_command_render_plan_on_runtime(context, &wrapped, verbosity_is_none)
}
