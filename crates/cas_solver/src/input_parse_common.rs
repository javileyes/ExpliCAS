//! Shared parsing helpers used across session command handlers.

/// Split string by delimiter, ignoring delimiters inside parentheses.
pub fn rsplit_ignoring_parens(s: &str, delimiter: char) -> Option<(&str, &str)> {
    let mut balance = 0;
    let mut split_idx = None;

    for (i, c) in s.char_indices().rev() {
        if c == ')' {
            balance += 1;
        } else if c == '(' {
            balance -= 1;
        } else if c == delimiter && balance == 0 {
            split_idx = Some(i);
            break;
        }
    }

    split_idx.map(|idx| (&s[..idx], &s[idx + 1..]))
}

/// Parse statement input, allowing `#N` session-ref shorthand as expression.
pub fn parse_statement_or_session_ref(
    ctx: &mut cas_ast::Context,
    input: &str,
) -> Result<cas_parser::Statement, String> {
    if input.starts_with('#') && input[1..].chars().all(char::is_numeric) {
        Ok(cas_parser::Statement::Expression(ctx.var(input)))
    } else {
        cas_parser::parse_statement(input, ctx).map_err(|e| e.to_string())
    }
}

/// Convert parser statement into a single expression id.
pub fn statement_to_expr_id(
    ctx: &mut cas_ast::Context,
    stmt: cas_parser::Statement,
) -> cas_ast::ExprId {
    match stmt {
        cas_parser::Statement::Equation(eq) => ctx.call("Equal", vec![eq.lhs, eq.rhs]),
        cas_parser::Statement::Expression(expr) => expr,
    }
}
