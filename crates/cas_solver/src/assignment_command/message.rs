use cas_ast::ExprId;
use cas_solver_core::assignment_command_types::AssignmentCommandOutput;

use crate::{format_assignment_success_message, AssignmentError};

use super::eval::{evaluate_assignment_command_with, evaluate_let_assignment_command_with};

/// Format assignment output payload once caller rendered the expression.
pub fn format_assignment_command_output_message(
    output: &AssignmentCommandOutput,
    rendered_expr: &str,
) -> String {
    format_assignment_success_message(&output.name, rendered_expr, output.lazy)
}

/// Evaluate assignment command pieces and return formatted user-facing message.
#[allow(dead_code)]
pub fn evaluate_assignment_command_message_with<F, R>(
    name: &str,
    expr_str: &str,
    lazy: bool,
    apply_assignment: F,
    mut render_expr: R,
) -> Result<String, String>
where
    F: FnMut(&str, &str, bool) -> Result<ExprId, AssignmentError>,
    R: FnMut(ExprId) -> String,
{
    let output = evaluate_assignment_command_with(name, expr_str, lazy, apply_assignment)?;
    let rendered = render_expr(output.expr);
    Ok(format_assignment_command_output_message(&output, &rendered))
}

/// Evaluate `let ...` command tail and return formatted user-facing message.
#[allow(dead_code)]
pub fn evaluate_let_assignment_command_message_with<F, R>(
    input: &str,
    apply_assignment: F,
    mut render_expr: R,
) -> Result<String, String>
where
    F: FnMut(&str, &str, bool) -> Result<ExprId, AssignmentError>,
    R: FnMut(ExprId) -> String,
{
    let output = evaluate_let_assignment_command_with(input, apply_assignment)?;
    let rendered = render_expr(output.expr);
    Ok(format_assignment_command_output_message(&output, &rendered))
}
