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
