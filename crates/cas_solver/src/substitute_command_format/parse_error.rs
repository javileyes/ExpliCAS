use crate::substitute_command_parse::substitute_usage_message;
use crate::substitute_command_types::SubstituteParseError;

/// Format substitute parse errors into user-facing messages.
pub fn format_substitute_parse_error_message(error: &SubstituteParseError) -> String {
    match error {
        SubstituteParseError::InvalidArity => substitute_usage_message().to_string(),
        SubstituteParseError::Expression(e) => format!("Error parsing expression: {e}"),
        SubstituteParseError::Target(e) => format!("Error parsing target: {e}"),
        SubstituteParseError::Replacement(e) => format!("Error parsing replacement: {e}"),
    }
}
