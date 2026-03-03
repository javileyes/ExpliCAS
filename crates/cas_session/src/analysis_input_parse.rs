//! Parsing helpers for analysis-style pair inputs (`<expr1>, <expr2>`).

use cas_ast::{Context, ExprId};

/// Parse errors for `<expr1>, <expr2>` command shapes (e.g. `equiv`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseExprPairError {
    MissingDelimiter,
    FirstArg(String),
    SecondArg(String),
}

fn parse_expr_or_equation_as_expr(ctx: &mut Context, input: &str) -> Result<ExprId, String> {
    let stmt = crate::input_parse_common::parse_statement_or_session_ref(ctx, input)?;
    Ok(crate::input_parse_common::statement_to_expr_id(ctx, stmt))
}

/// Parse `<expr1>, <expr2>` input into expression ids.
pub fn parse_expr_pair(
    ctx: &mut Context,
    input: &str,
) -> Result<(ExprId, ExprId), ParseExprPairError> {
    let (left, right) = crate::input_parse_common::rsplit_ignoring_parens(input, ',')
        .ok_or(ParseExprPairError::MissingDelimiter)?;
    let left = left.trim();
    let right = right.trim();

    let first = parse_expr_or_equation_as_expr(ctx, left).map_err(ParseExprPairError::FirstArg)?;
    let second =
        parse_expr_or_equation_as_expr(ctx, right).map_err(ParseExprPairError::SecondArg)?;
    Ok((first, second))
}

#[cfg(test)]
mod tests {
    use super::{parse_expr_pair, ParseExprPairError};

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
