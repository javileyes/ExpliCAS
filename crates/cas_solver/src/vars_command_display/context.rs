use cas_solver_core::session_runtime::BindingOverviewEntry;

use super::render::evaluate_vars_command_lines;

/// Evaluate `vars` command lines using an explicit AST context.
pub fn evaluate_vars_command_lines_with_context(
    bindings: &[BindingOverviewEntry],
    context: &cas_ast::Context,
) -> Vec<String> {
    evaluate_vars_command_lines(bindings, |id| {
        format!("{}", cas_formatter::DisplayExpr { context, id })
    })
}
