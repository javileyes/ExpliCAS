use crate::SetDisplayMode;

use super::ReplSimplifierRuntimeContext;

/// Evaluate `equiv ...` against runtime simplifier.
pub fn evaluate_equiv_invocation_message_on_runtime<C: ReplSimplifierRuntimeContext>(
    context: &mut C,
    line: &str,
) -> Result<String, String> {
    crate::evaluate_equiv_invocation_message(context.simplifier_mut(), line)
}

/// Evaluate `subst ...` against runtime simplifier.
pub fn evaluate_substitute_invocation_user_message_on_runtime<C: ReplSimplifierRuntimeContext>(
    context: &mut C,
    line: &str,
    display_mode: SetDisplayMode,
) -> Result<String, String> {
    crate::evaluate_substitute_invocation_user_message(context.simplifier_mut(), line, display_mode)
}

/// Evaluate `rationalize ...` against runtime simplifier.
pub fn evaluate_rationalize_command_lines_on_runtime<C: ReplSimplifierRuntimeContext>(
    context: &mut C,
    line: &str,
) -> Result<Vec<String>, String> {
    crate::evaluate_rationalize_command_lines(context.simplifier_mut(), line)
}
