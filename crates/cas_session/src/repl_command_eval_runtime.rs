//! REPL eval/expand render-plan adapters extracted from `repl_command_runtime`.

use crate::ReplCore;

/// Evaluate REPL expression and return a frontend-agnostic render plan.
pub fn evaluate_eval_command_render_plan_on_repl_core(
    core: &mut ReplCore,
    line: &str,
    verbosity_is_none: bool,
) -> Result<crate::EvalCommandRenderPlan, String> {
    let debug_mode = core.debug_mode();
    let out = core
        .with_engine_and_state(|engine, state| {
            crate::evaluate_eval_command_output(engine, state, line, debug_mode)
        })
        .map_err(|error| match error {
            crate::EvalCommandError::Parse(parse_error) => {
                crate::render_parse_error(line, &parse_error)
            }
            crate::EvalCommandError::Eval(message) => message,
        })?;
    Ok(crate::build_eval_command_render_plan(
        out,
        verbosity_is_none,
    ))
}

/// Evaluate `expand ...` and return a frontend-agnostic render plan.
pub fn evaluate_expand_command_render_plan_on_repl_core(
    core: &mut ReplCore,
    line: &str,
    verbosity_is_none: bool,
) -> Result<crate::EvalCommandRenderPlan, String> {
    let wrapped = crate::evaluate_expand_wrapped_expression(line)?;
    evaluate_eval_command_render_plan_on_repl_core(core, &wrapped, verbosity_is_none)
}

/// Return profile cache size for the current REPL core engine.
pub fn profile_cache_len_on_repl_core(core: &ReplCore) -> usize {
    core.profile_cache_len()
}
