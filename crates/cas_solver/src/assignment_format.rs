use crate::assignment_parse::let_assignment_usage_message;
use crate::assignment_parse_types::LetAssignmentParseError;
use cas_api_models::AssignmentError;

/// Format a `let` parse error as a user-facing message.
pub fn format_let_assignment_parse_error_message(error: &LetAssignmentParseError) -> String {
    match error {
        LetAssignmentParseError::MissingAssignmentOperator => let_assignment_usage_message().into(),
    }
}

/// Format assignment execution errors as user-facing messages.
pub fn format_assignment_error_message(error: &AssignmentError) -> String {
    match error {
        AssignmentError::EmptyName => "Error: Variable name cannot be empty".to_string(),
        AssignmentError::InvalidNameStart => {
            "Error: Variable name must start with a letter or underscore".to_string()
        }
        AssignmentError::ReservedName(name) => {
            format!(
                "Error: '{}' is a reserved name and cannot be assigned",
                name
            )
        }
        AssignmentError::Parse(e) => format!("Parse error: {}", e),
    }
}

/// Format assignment success message.
pub fn format_assignment_success_message(name: &str, rendered_expr: &str, lazy: bool) -> String {
    if lazy {
        format!("{} := {}", name, rendered_expr)
    } else {
        format!("{} = {}", name, rendered_expr)
    }
}
