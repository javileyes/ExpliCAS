use cas_ast::{Context, ExprId};

/// Parse errors for `<expr1>, <expr2>` command shapes (e.g. `equiv`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseExprPairError {
    MissingDelimiter,
    FirstArg(String),
    SecondArg(String),
}

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

fn statement_to_expr_id(ctx: &mut Context, stmt: cas_parser::Statement) -> ExprId {
    match stmt {
        cas_parser::Statement::Equation(eq) => ctx.call("Equal", vec![eq.lhs, eq.rhs]),
        cas_parser::Statement::Expression(expr) => expr,
    }
}

fn parse_expr_or_equation_as_expr(ctx: &mut Context, input: &str) -> Result<ExprId, String> {
    let stmt = parse_statement_or_session_ref(ctx, input)?;
    Ok(statement_to_expr_id(ctx, stmt))
}

/// Parse `<expr1>, <expr2>` input into expression ids, preserving equation-as-expression semantics.
pub fn parse_expr_pair(
    ctx: &mut Context,
    input: &str,
) -> Result<(ExprId, ExprId), ParseExprPairError> {
    let (left, right) =
        rsplit_ignoring_parens(input, ',').ok_or(ParseExprPairError::MissingDelimiter)?;
    let left = left.trim();
    let right = right.trim();

    let first = parse_expr_or_equation_as_expr(ctx, left).map_err(ParseExprPairError::FirstArg)?;
    let second =
        parse_expr_or_equation_as_expr(ctx, right).map_err(ParseExprPairError::SecondArg)?;
    Ok((first, second))
}

#[cfg(test)]
mod tests {
    use super::{
        parse_expr_pair, parse_statement_or_session_ref, rsplit_ignoring_parens, ParseExprPairError,
    };

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

    #[test]
    fn parse_expr_pair_requires_delimiter() {
        let mut ctx = cas_ast::Context::new();
        let err = parse_expr_pair(&mut ctx, "x + 1").expect_err("missing delimiter");
        assert_eq!(err, ParseExprPairError::MissingDelimiter);
    }

    #[test]
    fn parse_expr_pair_parses_equations_as_exprs() {
        let mut ctx = cas_ast::Context::new();
        let (lhs, rhs) = parse_expr_pair(&mut ctx, "x + 1 = 2, x").expect("pair");
        assert!(ctx.is_call_named(lhs, "Equal"));
        assert_eq!(cas_formatter::render_expr(&ctx, rhs), "x");
    }
}
