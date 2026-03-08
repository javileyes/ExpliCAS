/// Format visualize command errors into user-facing messages.
pub fn format_visualize_command_error_message(error: &crate::VisualizeEvalError) -> String {
    match error {
        crate::VisualizeEvalError::Parse(message) => format!("Parse error: {message}"),
    }
}
