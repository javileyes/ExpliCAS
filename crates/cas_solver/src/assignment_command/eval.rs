use cas_ast::ExprId;

use crate::{
    format_assignment_error_message, format_let_assignment_parse_error_message,
    parse_let_assignment_input, AssignmentError,
};
use cas_api_models::AssignmentCommandOutput;

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
