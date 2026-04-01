#[cfg(test)]
mod tests {
    use crate::VisualizeEvalError;
    use crate::{
        evaluate_derive_command_lines_with_resolver, evaluate_equiv_command_lines,
        evaluate_equiv_command_message, evaluate_equiv_invocation_message,
        evaluate_explain_command_lines, evaluate_explain_command_message,
        evaluate_explain_invocation_message, evaluate_visualize_command_dot,
        evaluate_visualize_command_output, evaluate_visualize_invocation_output,
        extract_derive_command_tail,
    };

    #[test]
    fn evaluate_equiv_command_lines_true() {
        let mut simplifier = crate::Simplifier::new();
        let lines = evaluate_equiv_command_lines(&mut simplifier, "x+1, 1+x")
            .expect("equiv should evaluate");
        assert!(lines.iter().any(|line| line.contains("True")));
    }

    #[test]
    fn evaluate_visualize_command_dot_parse_error() {
        let mut ctx = cas_ast::Context::new();
        let err = evaluate_visualize_command_dot(&mut ctx, "x+").expect_err("expected parse");
        assert!(matches!(err, VisualizeEvalError::Parse(_)));
    }

    #[test]
    fn evaluate_visualize_command_output_sets_file_and_hints() {
        let mut ctx = cas_ast::Context::new();
        let out =
            evaluate_visualize_command_output(&mut ctx, "x+1").expect("visualize should evaluate");
        assert_eq!(out.file_name, "ast.dot");
        assert!(out.dot_source.contains("digraph"));
        assert_eq!(out.hint_lines.len(), 2);
        assert!(out.hint_lines[0].contains("dot -Tsvg"));
    }

    #[test]
    fn evaluate_explain_command_lines_contains_result() {
        let mut ctx = cas_ast::Context::new();
        let lines =
            evaluate_explain_command_lines(&mut ctx, "gcd(8, 6)").expect("explain should evaluate");
        assert!(lines.iter().any(|line| line.starts_with("Result:")));
    }

    #[test]
    fn evaluate_equiv_command_message_joins_lines() {
        let mut simplifier = crate::Simplifier::new();
        let message = evaluate_equiv_command_message(&mut simplifier, "x+1,1+x")
            .expect("equiv should evaluate");
        assert!(message.contains("True"));
    }

    #[test]
    fn evaluate_equiv_invocation_message_formats_parse_error() {
        let mut simplifier = crate::Simplifier::new();
        let message =
            evaluate_equiv_invocation_message(&mut simplifier, "equiv x+1").expect_err("parse");
        assert!(message.contains("equiv"));
    }

    #[test]
    fn extract_derive_command_tail_trims_prefix() {
        assert_eq!(extract_derive_command_tail("derive x+x,2*x"), "x+x,2*x");
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_default_simplify_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive x + x, 2*x",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Target:") && line.contains('x')));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains('x')));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_factored_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive x^2 - 1, (x - 1)*(x + 1)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Target:") && line.contains("x")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("(x")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_collected_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive a*x + b*x + c, (a + b)*x + c",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("collect")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("a + b")));
    }

    #[test]
    fn evaluate_derive_command_lines_reports_equivalent_but_unsupported_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive a*x + b*x + c, x*(a + b + c/x)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines.iter().any(|line| line.contains(
            "Equivalent, but the second expression is not a supported simplification target yet."
        )));
        assert!(lines.iter().any(|line| line.contains("Equivalence: True")));
    }

    #[test]
    fn evaluate_derive_command_lines_reports_non_equivalent_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive x + 1, x + 2",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines.iter().any(
            |line| line.contains("Derive unavailable: the two expressions are not equivalent.")
        ));
    }

    #[test]
    fn evaluate_explain_command_message_contains_result() {
        let mut ctx = cas_ast::Context::new();
        let message = evaluate_explain_command_message(&mut ctx, "gcd(8, 6)")
            .expect("explain should evaluate");
        assert!(message.contains("Result:"));
    }

    #[test]
    fn evaluate_explain_invocation_message_contains_result() {
        let mut ctx = cas_ast::Context::new();
        let message = evaluate_explain_invocation_message(&mut ctx, "explain gcd(8, 6)")
            .expect("explain should evaluate");
        assert!(message.contains("Result:"));
    }

    #[test]
    fn evaluate_visualize_invocation_output_sets_file() {
        let mut ctx = cas_ast::Context::new();
        let out = evaluate_visualize_invocation_output(&mut ctx, "visualize x+1")
            .expect("visualize should evaluate");
        assert_eq!(out.file_name, "ast.dot");
    }
}
