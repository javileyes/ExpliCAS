use crate::SetDisplayMode;

use super::ReplSimplifierRuntimeContext;

/// Evaluate `equiv ...` against runtime simplifier. The comparison runs under
/// the SESSION's value domain: the comparator's internal `simplify()` honors
/// the simplifier's sticky value domain, so arming it here (save/restore) is
/// what carries `semantics set value complex` into `equiv` (ficha S5-002).
pub fn evaluate_equiv_invocation_message_on_runtime<C: ReplSimplifierRuntimeContext>(
    context: &mut C,
    line: &str,
) -> Result<String, String> {
    let value_domain = context.session_value_domain();
    let simplifier = context.simplifier_mut();
    let previous = simplifier.set_sticky_value_domain(value_domain);
    let result = crate::evaluate_equiv_invocation_message(simplifier, line);
    simplifier.set_sticky_value_domain(previous);
    result
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
