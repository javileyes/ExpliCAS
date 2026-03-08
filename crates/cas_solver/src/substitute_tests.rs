#[cfg(test)]
mod tests {
    use crate::substitute::{
        detect_substitute_strategy, evaluate_substitute_and_simplify, parse_substitute_args,
        substitute_auto_with_strategy, SubstituteOptions, SubstituteParseError, SubstituteStrategy,
    };

    #[test]
    fn detect_substitute_strategy_variable_target() {
        let mut ctx = cas_ast::Context::new();
        let target = cas_parser::parse("x", &mut ctx).expect("target parse");
        assert_eq!(
            detect_substitute_strategy(&ctx, target),
            SubstituteStrategy::Variable
        );
    }

    #[test]
    fn detect_substitute_strategy_expression_target() {
        let mut ctx = cas_ast::Context::new();
        let target = cas_parser::parse("x^2", &mut ctx).expect("target parse");
        assert_eq!(
            detect_substitute_strategy(&ctx, target),
            SubstituteStrategy::PowerAware
        );
    }

    #[test]
    fn substitute_auto_with_strategy_uses_variable_path() {
        let mut ctx = cas_ast::Context::new();
        let expr = cas_parser::parse("x + 1", &mut ctx).expect("expr parse");
        let target = cas_parser::parse("x", &mut ctx).expect("target parse");
        let replacement = cas_parser::parse("3", &mut ctx).expect("replacement parse");

        let (result, strategy) = substitute_auto_with_strategy(
            &mut ctx,
            expr,
            target,
            replacement,
            SubstituteOptions::default(),
        );
        assert_eq!(strategy, SubstituteStrategy::Variable);
        let out = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: result
            }
        );
        assert!(out.contains('3'));
    }

    #[test]
    fn parse_substitute_args_invalid_arity() {
        let mut ctx = cas_ast::Context::new();
        let err = parse_substitute_args(&mut ctx, "x + 1, x").expect_err("invalid arity");
        assert_eq!(err, SubstituteParseError::InvalidArity);
    }

    #[test]
    fn evaluate_substitute_and_simplify_runs() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let out = evaluate_substitute_and_simplify(
            &mut simplifier,
            "x^2 + x, x, 3",
            SubstituteOptions::default(),
        )
        .expect("evaluate");
        let rendered = cas_formatter::render_expr(&simplifier.context, out.simplified_expr);
        assert_eq!(rendered, "12");
    }
}
