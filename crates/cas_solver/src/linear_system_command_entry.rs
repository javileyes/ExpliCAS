//! Orchestration for `solve_system` command rendering.

use cas_ast::Context;

mod execute;

/// Evaluate full `solve_system ...` invocation and return CLI-ready message lines.
pub fn evaluate_linear_system_command_message(ctx: &mut Context, line: &str) -> String {
    execute::evaluate_linear_system_command_message(ctx, line)
}

/// Simplifier-backed variant (REPL runtime): nonlinear 2×2 systems get the
/// S2 substitution composition, keeping REPL/wire parity.
pub fn evaluate_linear_system_command_message_with_simplifier(
    simplifier: &mut crate::Simplifier,
    line: &str,
) -> String {
    execute::evaluate_linear_system_command_message_with_simplifier(simplifier, line)
}
