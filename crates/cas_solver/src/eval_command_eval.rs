mod execute;
mod output;
mod prepare;

use crate::command_api::eval::{EvalCommandError, EvalCommandOutput};

/// Evaluate full REPL `eval` input and prepare display payload.
pub fn evaluate_eval_command_output<S>(
    engine: &mut crate::Engine,
    session: &mut S,
    line: &str,
    debug_mode: bool,
) -> Result<EvalCommandOutput, EvalCommandError>
where
    S: crate::SolverEvalSession,
{
    // Relative session refs (`#-1` = última celda) become absolute `#N`
    // before parsing — same normalization as the wire path (prepare.rs).
    let line = cas_session_core::resolve::rewrite_relative_session_refs(
        line,
        &session.entry_ids_in_order(),
    )
    .map_err(EvalCommandError::Eval)?;
    let (style_signals, req) = prepare::build_eval_request(engine, &line)?;
    let eval_view = execute::execute_eval_request(engine, session, req)?;
    Ok(output::build_eval_command_output(
        &mut engine.simplifier.context,
        session.options().clone(),
        eval_view,
        style_signals,
        debug_mode,
    ))
}
