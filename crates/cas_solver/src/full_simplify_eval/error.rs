/// Full-simplify evaluation error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FullSimplifyEvalError {
    Parse(String),
    Resolve(String),
}

/// Format full-simplify evaluation errors for user-facing output.
pub fn format_full_simplify_eval_error_message(error: &FullSimplifyEvalError) -> String {
    match error {
        FullSimplifyEvalError::Parse(message) => format!("Error: {message}"),
        FullSimplifyEvalError::Resolve(message) => format!("Error resolving variables: {message}"),
    }
}
