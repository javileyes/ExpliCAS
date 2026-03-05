use cas_ast::ExprId;

use crate::substitute_command_types::SubstituteParseError;

const SUBSTITUTE_USAGE_MESSAGE: &str = "Usage: subst <expr>, <target>, <replacement>\n\n\
                     Examples:\n\
                       subst x^2 + x, x, 3              -> 12\n\
                       subst x^4 + x^2 + 1, x^2, y      -> y^2 + y + 1\n\
                       subst x^3, x^2, y                -> y*x";

pub(crate) fn substitute_usage_message() -> &'static str {
    SUBSTITUTE_USAGE_MESSAGE
}

pub(crate) fn split_by_comma_ignoring_parens(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut balance = 0;
    let mut start = 0;

    for (i, c) in s.char_indices() {
        match c {
            '(' | '[' | '{' => balance += 1,
            ')' | ']' | '}' => balance -= 1,
            ',' if balance == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }

    parts.push(&s[start..]);
    parts
}

/// Parse REPL-like substitute arguments (`expr, target, replacement`) into ids.
pub(crate) fn parse_substitute_args(
    ctx: &mut cas_ast::Context,
    input: &str,
) -> Result<(ExprId, ExprId, ExprId), SubstituteParseError> {
    let parts = split_by_comma_ignoring_parens(input);
    if parts.len() != 3 {
        return Err(SubstituteParseError::InvalidArity);
    }

    let expr_str = parts[0].trim();
    let target_str = parts[1].trim();
    let replacement_str = parts[2].trim();

    let expr = cas_parser::parse(expr_str, ctx)
        .map_err(|e| SubstituteParseError::Expression(e.to_string()))?;
    let target = cas_parser::parse(target_str, ctx)
        .map_err(|e| SubstituteParseError::Target(e.to_string()))?;
    let replacement = cas_parser::parse(replacement_str, ctx)
        .map_err(|e| SubstituteParseError::Replacement(e.to_string()))?;

    Ok((expr, target, replacement))
}
