//! Context/analysis REPL adapters extracted from `repl_command_runtime`.

use crate::{ReplCore, VisualizeCommandOutput};

/// Evaluate `weierstrass ...` invocation against REPL simplifier.
pub fn evaluate_weierstrass_invocation_message_on_repl_core(
    core: &mut ReplCore,
    line: &str,
) -> Result<String, String> {
    crate::evaluate_weierstrass_invocation_message(core.simplifier_mut(), line)
}

/// Evaluate `telescope ...` using REPL context.
pub fn evaluate_telescope_invocation_message_on_repl_core(
    core: &mut ReplCore,
    line: &str,
) -> Result<String, String> {
    crate::evaluate_telescope_invocation_message(&mut core.simplifier_mut().context, line)
}

/// Evaluate `expand_log ...` using REPL context.
pub fn evaluate_expand_log_invocation_message_on_repl_core(
    core: &mut ReplCore,
    line: &str,
) -> Result<String, String> {
    crate::evaluate_expand_log_invocation_message(&mut core.simplifier_mut().context, line)
}

/// Evaluate `solve_system ...` using REPL context.
pub fn evaluate_linear_system_command_message_on_repl_core(
    core: &mut ReplCore,
    line: &str,
) -> String {
    crate::evaluate_linear_system_command_message(&mut core.simplifier_mut().context, line)
}

/// Evaluate `visualize ...` using REPL context.
pub fn evaluate_visualize_invocation_output_on_repl_core(
    core: &mut ReplCore,
    line: &str,
) -> Result<VisualizeCommandOutput, String> {
    crate::evaluate_visualize_invocation_output(&mut core.simplifier_mut().context, line)
}

/// Evaluate `explain ...` using REPL context.
pub fn evaluate_explain_invocation_message_on_repl_core(
    core: &mut ReplCore,
    line: &str,
) -> Result<String, String> {
    crate::evaluate_explain_invocation_message(&mut core.simplifier_mut().context, line)
}
