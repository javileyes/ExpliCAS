use cas_ast::ExprId;

use crate::{
    format_assignment_error_message, format_assignment_success_message,
    format_let_assignment_parse_error_message, parse_let_assignment_input, AssignmentError,
};

/// Successful output payload for assignment-style commands (`let`, `:=`, direct assign).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssignmentCommandOutput {
    pub name: String,
    pub expr: ExprId,
    pub lazy: bool,
}

/// Evaluate assignment command pieces and return a typed output payload.
pub fn evaluate_assignment_command_with<F>(
    name: &str,
    expr_str: &str,
    lazy: bool,
    mut apply_assignment: F,
) -> Result<AssignmentCommandOutput, String>
where
    F: FnMut(&str, &str, bool) -> Result<ExprId, AssignmentError>,
{
    match apply_assignment(name, expr_str, lazy) {
        Ok(expr) => Ok(AssignmentCommandOutput {
            name: name.to_string(),
            expr,
            lazy,
        }),
        Err(error) => Err(format_assignment_error_message(&error)),
    }
}

/// Evaluate `let ...` command tail and return assignment output payload.
pub fn evaluate_let_assignment_command_with<F>(
    input: &str,
    apply_assignment: F,
) -> Result<AssignmentCommandOutput, String>
where
    F: FnMut(&str, &str, bool) -> Result<ExprId, AssignmentError>,
{
    let parsed = parse_let_assignment_input(input)
        .map_err(|error| format_let_assignment_parse_error_message(&error))?;
    evaluate_assignment_command_with(parsed.name, parsed.expr, parsed.lazy, apply_assignment)
}

/// Format assignment output payload once caller rendered the expression.
pub fn format_assignment_command_output_message(
    output: &AssignmentCommandOutput,
    rendered_expr: &str,
) -> String {
    format_assignment_success_message(&output.name, rendered_expr, output.lazy)
}

/// Evaluate assignment command pieces and return formatted user-facing message.
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
