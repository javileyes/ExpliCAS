/// Format explain command errors into user-facing messages.
pub fn format_explain_command_error_message(error: &crate::ExplainCommandEvalError) -> String {
    match error {
        crate::ExplainCommandEvalError::Parse(e) => format!("Parse error: {e}"),
        crate::ExplainCommandEvalError::ExpectedFunctionCall => {
            "Explain mode currently only supports function calls\n\
             Try: explain gcd(48, 18)"
                .to_string()
        }
        crate::ExplainCommandEvalError::UnsupportedFunction(name) => format!(
            "Explain mode not yet implemented for function '{}'\n\
             Currently supported: gcd",
            name
        ),
        crate::ExplainCommandEvalError::InvalidArity {
            function, expected, ..
        } => {
            if function == "gcd" && *expected == 2 {
                "Usage: explain gcd(a, b)".to_string()
            } else {
                format!("Invalid arity for '{function}'")
            }
        }
    }
}
