use crate::{
    apply_assignment, format_assignment_error_message, format_assignment_success_message,
    format_let_assignment_parse_error_message, parse_let_assignment_input, SessionState,
};

/// Successful output payload for assignment-style commands (`let`, `:=`, direct assign).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssignmentCommandOutput {
    pub name: String,
    pub expr: cas_ast::ExprId,
    pub lazy: bool,
}

/// Evaluate assignment command pieces and return a typed output payload.
pub fn evaluate_assignment_command(
    state: &mut SessionState,
    simplifier: &mut cas_solver::Simplifier,
    name: &str,
    expr_str: &str,
    lazy: bool,
) -> Result<AssignmentCommandOutput, String> {
    match apply_assignment(state, simplifier, name, expr_str, lazy) {
        Ok(expr) => Ok(AssignmentCommandOutput {
            name: name.to_string(),
            expr,
            lazy,
        }),
        Err(error) => Err(format_assignment_error_message(&error)),
    }
}

/// Evaluate `let ...` command tail and return assignment output payload.
pub fn evaluate_let_assignment_command(
    state: &mut SessionState,
    simplifier: &mut cas_solver::Simplifier,
    input: &str,
) -> Result<AssignmentCommandOutput, String> {
    let parsed = parse_let_assignment_input(input)
        .map_err(|error| format_let_assignment_parse_error_message(&error))?;
    evaluate_assignment_command(state, simplifier, parsed.name, parsed.expr, parsed.lazy)
}

/// Evaluate assignment command pieces and return formatted user-facing message.
pub fn evaluate_assignment_command_message_with_simplifier(
    state: &mut SessionState,
    simplifier: &mut cas_solver::Simplifier,
    name: &str,
    expr_str: &str,
    lazy: bool,
) -> Result<String, String> {
    let output = evaluate_assignment_command(state, simplifier, name, expr_str, lazy)?;
    let rendered = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: output.expr
        }
    );
    Ok(format_assignment_command_output_message(&output, &rendered))
}

/// Evaluate `let ...` command tail and return formatted user-facing message.
pub fn evaluate_let_assignment_command_message_with_simplifier(
    state: &mut SessionState,
    simplifier: &mut cas_solver::Simplifier,
    input: &str,
) -> Result<String, String> {
    let output = evaluate_let_assignment_command(state, simplifier, input)?;
    let rendered = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: output.expr
        }
    );
    Ok(format_assignment_command_output_message(&output, &rendered))
}

/// Format assignment output payload once caller rendered the expression.
pub fn format_assignment_command_output_message(
    output: &AssignmentCommandOutput,
    rendered_expr: &str,
) -> String {
    format_assignment_success_message(&output.name, rendered_expr, output.lazy)
}
