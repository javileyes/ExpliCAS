use crate::VisualizeCommandOutput;

use super::ReplSimplifierRuntimeContext;

/// Evaluate `weierstrass ...` invocation against runtime simplifier.
pub fn evaluate_weierstrass_invocation_message_on_runtime<C: ReplSimplifierRuntimeContext>(
    context: &mut C,
    line: &str,
) -> Result<String, String> {
    crate::evaluate_weierstrass_invocation_message(context.simplifier_mut(), line)
}

/// Evaluate `telescope ...` using runtime context.
pub fn evaluate_telescope_invocation_message_on_runtime<C: ReplSimplifierRuntimeContext>(
    context: &mut C,
    line: &str,
) -> Result<String, String> {
    crate::evaluate_telescope_invocation_message(&mut context.simplifier_mut().context, line)
}

/// Evaluate `expand_log ...` using runtime context.
pub fn evaluate_expand_log_invocation_message_on_runtime<C: ReplSimplifierRuntimeContext>(
    context: &mut C,
    line: &str,
) -> Result<String, String> {
    crate::evaluate_expand_log_invocation_message(&mut context.simplifier_mut().context, line)
}

/// Evaluate `solve_system ...` using runtime context.
pub fn evaluate_linear_system_command_message_on_runtime<C: ReplSimplifierRuntimeContext>(
    context: &mut C,
    line: &str,
) -> String {
    crate::evaluate_linear_system_command_message(&mut context.simplifier_mut().context, line)
}

/// Evaluate `visualize ...` using runtime context.
pub fn evaluate_visualize_invocation_output_on_runtime<C: ReplSimplifierRuntimeContext>(
    context: &mut C,
    line: &str,
) -> Result<VisualizeCommandOutput, String> {
    crate::evaluate_visualize_invocation_output(&mut context.simplifier_mut().context, line)
}

/// Evaluate `explain ...` using runtime context.
pub fn evaluate_explain_invocation_message_on_runtime<C: ReplSimplifierRuntimeContext>(
    context: &mut C,
    line: &str,
) -> Result<String, String> {
    crate::evaluate_explain_invocation_message(&mut context.simplifier_mut().context, line)
}
