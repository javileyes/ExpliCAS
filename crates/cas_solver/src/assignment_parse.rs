use crate::assignment_types::{LetAssignmentParseError, ParsedLetAssignment};

/// Usage message for `let` assignment command.
pub fn let_assignment_usage_message() -> &'static str {
    "Usage: let <name> = <expr>   (eager - evaluates)\n\
                        let <name> := <expr>  (lazy - stores formula)\n\
                 Example: let a = expand((1+x)^3)"
}

/// Parse `let` tail input:
/// - `name := expr` -> lazy
/// - `name = expr` -> eager
pub fn parse_let_assignment_input(
    rest: &str,
) -> Result<ParsedLetAssignment<'_>, LetAssignmentParseError> {
    if let Some(idx) = rest.find(":=") {
        Ok(ParsedLetAssignment {
            name: rest[..idx].trim(),
            expr: rest[idx + 2..].trim(),
            lazy: true,
        })
    } else if let Some(eq_idx) = rest.find('=') {
        Ok(ParsedLetAssignment {
            name: rest[..eq_idx].trim(),
            expr: rest[eq_idx + 1..].trim(),
            lazy: false,
        })
    } else {
        Err(LetAssignmentParseError::MissingAssignmentOperator)
    }
}
