use cas_ast::{Context, ExprId};

use super::SubstituteParseError;

#[cfg_attr(not(test), allow(dead_code))]
fn split_by_comma_ignoring_parens(s: &str) -> Vec<&str> {
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
#[cfg_attr(not(test), allow(dead_code))]
pub fn parse_substitute_args(
    ctx: &mut Context,
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
