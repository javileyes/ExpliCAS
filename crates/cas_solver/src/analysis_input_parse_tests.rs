#[cfg(test)]
mod tests {
    use crate::{parse_expr_pair, ParseExprPairError};

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
