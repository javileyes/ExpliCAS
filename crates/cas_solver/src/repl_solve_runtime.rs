mod display_mode;
mod full_simplify;
mod solve;

use crate::SetDisplayMode;

/// Runtime context needed by solve/full-simplify REPL command adapters.
pub trait ReplSolveRuntimeContext:
    crate::ReplSessionSimplifierRuntimeContext<State: crate::SolverEvalSession>
{
    fn debug_mode(&self) -> bool;
}

/// Evaluate `solve ...` invocation against runtime simplifier/session state.
pub fn evaluate_solve_command_message_on_runtime<C: ReplSolveRuntimeContext>(
    context: &mut C,
    line: &str,
    display_mode: SetDisplayMode,
) -> Result<String, String> {
    solve::evaluate_solve_command_message_on_runtime(context, line, display_mode)
}

/// Evaluate `simplify ...` invocation against runtime simplifier/session state.
pub fn evaluate_full_simplify_command_lines_on_runtime<C: ReplSolveRuntimeContext>(
    context: &mut C,
    line: &str,
    display_mode: SetDisplayMode,
) -> Result<Vec<String>, String> {
    full_simplify::evaluate_full_simplify_command_lines_on_runtime(context, line, display_mode)
}
