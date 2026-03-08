//! Orchestration for `solve_system` command rendering.

use cas_ast::Context;

mod execute;

/// Evaluate full `solve_system ...` invocation and return CLI-ready message lines.
pub fn evaluate_linear_system_command_message(ctx: &mut Context, line: &str) -> String {
    execute::evaluate_linear_system_command_message(ctx, line)
}
