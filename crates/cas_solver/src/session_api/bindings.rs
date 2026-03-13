//! Assignment and bindings-facing API for session/repl consumers.

use crate::vars_command_display::{
    evaluate_vars_command_lines as render_vars_command_lines,
    evaluate_vars_command_lines_with_context as render_vars_command_lines_with_context,
};

pub use crate::assignment_apply::apply_assignment_with_context as apply_assignment;
pub use crate::assignment_command::format_assignment_command_output_message;
pub use crate::assignment_command_runtime::evaluate_assignment_command_message_with_context as evaluate_assignment_command_message_with_simplifier;
pub use crate::assignment_command_runtime::evaluate_assignment_command_with_context as evaluate_assignment_command;
pub use crate::assignment_command_runtime::evaluate_let_assignment_command_message_with_context as evaluate_let_assignment_command_message_with_simplifier;
pub use crate::assignment_command_runtime::evaluate_let_assignment_command_with_context as evaluate_let_assignment_command;
pub use crate::assignment_format::{
    format_assignment_error_message, format_assignment_success_message,
    format_let_assignment_parse_error_message,
};
pub use crate::assignment_parse::{let_assignment_usage_message, parse_let_assignment_input};
pub use crate::bindings_command::{binding_overview_entries, clear_bindings_command};
pub use crate::bindings_format::{
    format_binding_overview_lines, format_clear_bindings_result_lines, vars_empty_message,
};
pub use crate::repl_session_runtime::evaluate_assignment_command_message_on_runtime as evaluate_assignment_command_message_on_repl_core;
pub use crate::repl_session_runtime::evaluate_clear_command_lines_on_runtime as evaluate_clear_command_lines_on_repl_core;
pub use crate::repl_session_runtime::evaluate_let_assignment_command_message_on_runtime as evaluate_let_assignment_command_message_on_repl_core;
pub use crate::repl_session_runtime::evaluate_vars_command_message_on_runtime as evaluate_vars_command_message_on_repl_core;
pub use cas_solver_core::assignment_command_types::{
    AssignmentCommandOutput, AssignmentError, LetAssignmentParseError, ParsedLetAssignment,
};
pub use cas_solver_core::session_runtime::{BindingOverviewEntry, ClearBindingsResult};

/// Evaluate `clear` command and return output lines.
pub fn evaluate_clear_command_lines<C: crate::bindings_command::BindingsContext>(
    context: &mut C,
    input: &str,
) -> Vec<String> {
    let result = clear_bindings_command(context, input);
    format_clear_bindings_result_lines(&result)
}

/// Evaluate `vars` command lines using an expression renderer callback.
pub fn evaluate_vars_command_lines<C: crate::bindings_command::BindingsContext, F>(
    context: &C,
    render_expr: F,
) -> Vec<String>
where
    F: FnMut(cas_ast::ExprId) -> String,
{
    let bindings = binding_overview_entries(context);
    render_vars_command_lines(&bindings, render_expr)
}

/// Evaluate `vars` command lines using an explicit AST context.
pub fn evaluate_vars_command_lines_with_context<C: crate::bindings_command::BindingsContext>(
    context: &C,
    ast_context: &cas_ast::Context,
) -> Vec<String> {
    let bindings = binding_overview_entries(context);
    render_vars_command_lines_with_context(&bindings, ast_context)
}
