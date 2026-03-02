use cas_ast::Context;

/// Parse statement input, allowing `#N` session-ref shorthand as expression.
pub(crate) fn parse_statement_or_session_ref(
    ctx: &mut Context,
    input: &str,
) -> Result<cas_parser::Statement, String> {
    if input.starts_with('#') && input[1..].chars().all(char::is_numeric) {
        Ok(cas_parser::Statement::Expression(ctx.var(input)))
    } else {
        cas_parser::parse_statement(input, ctx).map_err(|e| e.to_string())
    }
}

/// Split string by delimiter, ignoring delimiters inside parentheses.
pub(crate) fn rsplit_ignoring_parens(s: &str, delimiter: char) -> Option<(&str, &str)> {
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

#[cfg(test)]
mod tests {
    use super::{parse_statement_or_session_ref, rsplit_ignoring_parens};

    #[test]
    fn parse_statement_or_session_ref_accepts_session_id() {
        let mut ctx = cas_ast::Context::new();
        let stmt = parse_statement_or_session_ref(&mut ctx, "#3").expect("parse");
        match stmt {
            cas_parser::Statement::Expression(_) => {}
            _ => panic!("expected expression"),
        }
    }

    #[test]
    fn rsplit_ignoring_parens_skips_inner_delimiters() {
        let split = rsplit_ignoring_parens("f(a,b),x", ',').expect("split");
        assert_eq!(split.0, "f(a,b)");
        assert_eq!(split.1, "x");
    }
}
