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

    fn normalized_inline_math(value: &str) -> String {
        value.replace('·', "*").replace(' ', "")
    }

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
    fn evaluate_derive_command_lines_reaches_perfect_square_factored_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive x^2 + 2*x + 1, (x + 1)^2",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("factor")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("(x + 1)^(2)")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_symbolic_perfect_square_factored_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive a^2 + 2*a*b + b^2, (a + b)^2",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(
            lines
                .iter()
                .any(|line| line.starts_with("Strategy:") && line.contains("factor")),
            "expected factor strategy, got: {lines:?}"
        );
        assert!(
            lines
                .iter()
                .any(|line| line.starts_with("Result:") && line.contains("(a + b)^(2)")),
            "expected symbolic perfect-square factorization in derive output, got: {lines:?}"
        );
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_symbolic_binomial_cube_factored_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive a^3 + 3*a^2*b + 3*a*b^2 + b^3, (a+b)^3",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(
            lines
                .iter()
                .any(|line| line.starts_with("Strategy:") && line.contains("factor")),
            "expected factor strategy, got: {lines:?}"
        );
        assert!(
            lines
                .iter()
                .any(|line| line.starts_with("Result:") && line.contains("(a + b)^(3)")),
            "expected symbolic cubic factorization in derive output, got: {lines:?}"
        );
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_negative_symbolic_binomial_cube_factored_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive a^3 - 3*a^2*b + 3*a*b^2 - b^3, (a-b)^3",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(
            lines
                .iter()
                .any(|line| line.starts_with("Strategy:") && line.contains("factor")),
            "expected factor strategy, got: {lines:?}"
        );
        assert!(
            lines
                .iter()
                .any(|line| line.starts_with("Result:") && line.contains("(a - b)^(3)")),
            "expected negative symbolic cubic factorization in derive output, got: {lines:?}"
        );
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_geometric_difference_factored_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive x^6 - 1, (x-1)*(x^5 + x^4 + x^3 + x^2 + x + 1)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(
            lines
                .iter()
                .any(|line| line.starts_with("Strategy:") && line.contains("factor")),
            "expected factor strategy, got: {lines:?}"
        );
        assert!(
            lines.iter().any(|line| {
                line.starts_with("Result:") && line.contains("x^(5)") && line.contains("(x - 1)")
            }),
            "expected geometric difference factorization in derive output, got: {lines:?}"
        );
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_negative_perfect_square_factored_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive a^2 - 2*a*b + b^2, (a - b)^2",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(
            lines
                .iter()
                .any(|line| line.starts_with("Strategy:") && line.contains("factor")),
            "expected factor strategy, got: {lines:?}"
        );
        assert!(
            lines
                .iter()
                .any(|line| line.starts_with("Result:") && line.contains("(a - b)^(2)")),
            "expected negative perfect-square factorization in derive output, got: {lines:?}"
        );
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_sophie_germain_factored_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive x^4 + 4*y^4, (x^2 - 2*x*y + 2*y^2)*(x^2 + 2*x*y + 2*y^2)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(
            lines
                .iter()
                .any(|line| line.starts_with("Strategy:") && line.contains("factor")),
            "expected factor strategy, got: {lines:?}"
        );
        assert!(
            lines.iter().any(|line| {
                line.starts_with("Result:")
                    && line.contains("x^(2)")
                    && line.contains("2 * y^(2)")
                    && line.contains("2 * x * y")
            }),
            "expected Sophie Germain factorization in derive output, got: {lines:?}"
        );
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_sophie_germain_expanded_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (x^2 - 2*x*y + 2*y^2)*(x^2 + 2*x*y + 2*y^2), x^4 + 4*y^4",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(
            lines
                .iter()
                .any(|line| line.starts_with("Strategy:") && line.contains("expand")),
            "expected expand strategy, got: {lines:?}"
        );
        assert!(
            lines.iter().any(|line| {
                line.starts_with("Result:") && line.contains("x^(4)") && line.contains("4 * y^(4)")
            }),
            "expected Sophie Germain expansion in derive output, got: {lines:?}"
        );
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_difference_of_cubes_expanded_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (a-b)*(a^2+a*b+b^2), a^3-b^3",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(
            lines
                .iter()
                .any(|line| line.starts_with("Strategy:") && line.contains("expand")),
            "expected expand strategy, got: {lines:?}"
        );
        assert!(
            lines
                .iter()
                .any(|line| line.starts_with("Result:") && line.contains("a^(3) - b^(3)")),
            "expected difference-of-cubes expansion in derive output, got: {lines:?}"
        );
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_sum_of_cubes_expanded_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (a+b)*(a^2-a*b+b^2), a^3+b^3",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(
            lines
                .iter()
                .any(|line| line.starts_with("Strategy:") && line.contains("expand")),
            "expected expand strategy, got: {lines:?}"
        );
        assert!(
            lines
                .iter()
                .any(|line| line.starts_with("Result:") && line.contains("a^(3) + b^(3)")),
            "expected sum-of-cubes expansion in derive output, got: {lines:?}"
        );
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_alternating_cubic_vandermonde_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive a^3*(b-c) + b^3*(c-a) + c^3*(a-b), (a-b)*(a-c)*(b-c)*(a+b+c)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(
            lines
                .iter()
                .any(|line| line.starts_with("Strategy:") && line.contains("factor")),
            "expected factor strategy, got: {lines:?}"
        );
        assert!(
            lines.iter().any(|line| {
                line.starts_with("Result:")
                    && line.contains("a + b + c")
                    && line.contains("a - b")
                    && line.contains("a - c")
                    && line.contains("b - c")
            }),
            "expected alternating cubic Vandermonde factorization in derive output, got: {lines:?}"
        );
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
    fn evaluate_derive_command_lines_reaches_expanded_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (x + 1)^2, x^2 + 2*x + 1",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("x")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_symbolic_binomial_expanded_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (a + b)^2, a^2 + 2*a*b + b^2",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("2")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_symbolic_binomial_cube_expanded_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (a + b)^3, a^3 + 3*a^2*b + 3*a*b^2 + b^3",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(
            lines
                .iter()
                .any(|line| line.starts_with("Strategy:") && line.contains("expand")),
            "expected expand strategy, got: {lines:?}"
        );
        assert!(
            lines.iter().any(|line| {
                line.starts_with("Result:")
                    && line.contains("a^(3)")
                    && line.contains("b^(3)")
                    && line.contains("3 * a * b^(2)")
            }),
            "expected symbolic cubic expansion in derive output, got: {lines:?}"
        );
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_negative_symbolic_binomial_expanded_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (a - b)^2, a^2 - 2*a*b + b^2",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(
            lines
                .iter()
                .any(|line| line.starts_with("Strategy:") && line.contains("expand")),
            "expected expand strategy, got: {lines:?}"
        );
        assert!(
            lines.iter().any(|line| {
                line.starts_with("Result:")
                    && line.contains("a^(2)")
                    && line.contains("b^(2)")
                    && line.contains("- 2 * a * b")
            }),
            "expected negative binomial expansion in derive output, got: {lines:?}"
        );
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_negative_symbolic_binomial_cube_expanded_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (a - b)^3, a^3 - 3*a^2*b + 3*a*b^2 - b^3",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(
            lines
                .iter()
                .any(|line| line.starts_with("Strategy:") && line.contains("expand")),
            "expected expand strategy, got: {lines:?}"
        );
        assert!(
            lines.iter().any(|line| {
                line.starts_with("Result:")
                    && line.contains("a^(3)")
                    && line.contains("b^(3)")
                    && line.contains("- 3 * b * a^(2)")
            }),
            "expected negative symbolic cubic expansion in derive output, got: {lines:?}"
        );
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_log_expanded_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive ln(x^2*y), ln(y) + 2*ln(abs(x))",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand_log")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("ln")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_even_power_log_abs_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive ln(x^2), 2*ln(abs(x))",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("simplify")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Factor Perfect Square in Logarithm")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:") && line.contains("2") && line.contains("ln(")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_log_expanded_target_preserving_powers() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive ln(x^3*y^2), ln(x^3) + ln(y^2)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand_log")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("ln(") && line.contains("y")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_general_base_log_power_expanded_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive log(2, x^3), 3*log(2, x)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("simplify")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Evaluate Logarithms")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("3") && line.contains("log")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_even_power_log_abs_contracted_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 2*ln(abs(x)), ln(x^2)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("simplify")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("ln(")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_general_base_log_power_contracted_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 3*log(2, x), log(2, x^3)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("simplify")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("log(")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_log_contracted_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive ln(x) + ln(y), ln(x*y)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("contract logs")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("ln")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_log_contracted_target_with_powers() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive ln(x^3) + ln(y^2), ln(x^3*y^2)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("contract logs")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("ln")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_scaled_log_sum_contracted_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 3*ln(x) + 2*ln(abs(y)), ln(x^3*y^2)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("contract logs")));
        assert!(lines.iter().any(|line| line.contains("Log Contraction")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("ln")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_scaled_general_base_log_difference_contracted_target()
    {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 3*log(2, x) - 2*log(2, y), log(2, x^3/y^2)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("contract logs")));
        assert!(lines.iter().any(|line| line.contains("Log Contraction")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("log(")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_trig_expanded_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sin(2*x), 2*sin(x)*cos(x)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand trig")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("sin")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_cos_double_angle_sin_squared_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive cos(2*x), 1 - 2*sin(x)^2",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand trig")));
        assert!(lines.iter().any(|line| line.trim_start().starts_with("1.")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:") && line.contains("1 - 2") && line.contains("sin")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_cos_double_angle_cos_squared_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive cos(2*x), 2*cos(x)^2 - 1",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand trig")));
        assert!(lines.iter().any(|line| line.trim_start().starts_with("1.")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:") && line.contains("2 * cos") && line.contains("- 1")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_trig_contracted_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (sin(2*x))/(cos(2*x)), tan(2*x)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("contract trig")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("tan")));
    }

    #[test]
    fn evaluate_derive_command_lines_expands_secant_to_reciprocal_directly() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sec(x), 1/cos(x)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand trig")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Reciprocal Trig Identity")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("1 / cos")));
    }

    #[test]
    fn evaluate_derive_command_lines_contracts_reciprocal_cosine_to_secant_directly() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1/cos(x), sec(x)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("contract trig")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Reciprocal Trig Identity")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("sec(x)")));
    }

    #[test]
    fn evaluate_derive_command_lines_contracts_cosine_over_sine_to_cotangent_directly() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive cos(x)/sin(x), cot(x)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("contract trig")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Reciprocal Trig Identity")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("cot(x)")));
    }

    #[test]
    fn evaluate_derive_command_lines_contracts_reciprocal_trig_product_to_one_directly() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive tan(x)*cot(x), 1",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("simplify")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Reciprocal Product Identity")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("1")));
    }

    #[test]
    fn evaluate_derive_command_lines_contracts_sec_tan_pythagorean_to_one_directly() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sec(x)^2 - tan(x)^2, 1",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("simplify")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Reciprocal Pythagorean Identity")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("1")));
    }

    #[test]
    fn evaluate_derive_command_lines_contracts_csc_cot_pythagorean_to_one_directly() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive csc(x)^2 - cot(x)^2, 1",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("simplify")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Reciprocal Pythagorean Identity")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("1")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_double_angle_contracted_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 2*sin(x)*cos(x), sin(2*x)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("contract trig")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("sin(2")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_cos_double_angle_contracted_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1 - 2*sin(x)^2, cos(2*x)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("contract trig")));
        assert!(lines.iter().any(|line| line.trim_start().starts_with("1.")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("cos(2")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_half_angle_tangent_target_directly() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (1-cos(2*x))/sin(2*x), tan(x)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("contract trig")));
        assert!(lines.iter().any(|line| line.trim_start().starts_with("1.")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("tan(x)")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_half_angle_tangent_alt_target_directly() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sin(2*x)/(1+cos(2*x)), tan(x)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("contract trig")));
        assert!(lines.iter().any(|line| line.trim_start().starts_with("1.")));
        assert!(lines.iter().any(|line| line.contains("tan(x)")));
    }

    #[test]
    fn evaluate_derive_command_lines_expands_to_half_angle_tangent_target_directly() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive tan(x), (1-cos(2*x))/sin(2*x)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand trig")));
        assert!(lines.iter().any(|line| line.trim_start().starts_with("1.")));
        assert!(lines
            .iter()
            .any(|line| line.contains("(1 - cos(2") && line.contains("sin(2")));
    }

    #[test]
    fn evaluate_derive_command_lines_expands_to_half_angle_tangent_alt_target_directly() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive tan(x), sin(2*x)/(1+cos(2*x))",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand trig")));
        assert!(lines.iter().any(|line| line.trim_start().starts_with("1.")));
        assert!(lines
            .iter()
            .any(|line| line.contains("sin(2") && line.contains("cos(2")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_half_angle_sin_square_target_directly() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sin(x)^2, (1-cos(2*x))/2",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand trig")));
        assert!(lines.iter().any(|line| line.trim_start().starts_with("1.")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:") && line.contains("cos(2") && line.contains("/ 2")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_half_angle_cos_square_contraction_directly() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (1+cos(2*x))/2, cos(x)^2",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("contract trig")));
        assert!(lines.iter().any(|line| line.trim_start().starts_with("1.")));
        assert!(lines
            .iter()
            .any(|line| line.contains("cos(x)") && (line.contains("^2") || line.contains("²"))));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_rationalized_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1/(sqrt(x)-1), (sqrt(x)+1)/(x-1)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("rationalize")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("sqrt")));
        assert!(
            !lines.iter().any(|line| line.contains("x + 1^(2)")),
            "derive rationalize CLI output should not corrupt the retargeted after expression, got: {lines:?}"
        );
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_rationalized_difference_zero_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1 / (sqrt(x) - 1) - (sqrt(x) + 1) / (x - 1), 0",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("simplify")));
        assert!(lines.iter().any(|line| {
            line.contains("Rationalize Linear Sqrt Denominator")
                || line.contains("Subtraction Self-Cancel")
        }));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.trim_end().ends_with("0")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_radical_notable_quotient_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (x^(3/2)-1)/(sqrt(x)-1), sqrt(x)+x+1",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("rationalize")));
        assert!(lines.iter().any(|line| {
            line.contains("Rationalize Linear Sqrt Denominator")
                || line.contains("Polynomial division with opaque substitution")
        }));
        assert!(lines
            .iter()
            .any(|line| line.contains("Cancel Reciprocal Exponents")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("sqrt(x) + x + 1")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_perfect_square_root_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sqrt(x^2 + 2*x + 1), abs(x+1)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("simplify")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Sqrt Perfect Square")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("|x + 1|")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_fraction_expanded_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (a+b)/x, a/x + b/x",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand fraction")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("/ x")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_fraction_expanded_target_after_term_cancellation() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (x+y)/(x*y), 1/x + 1/y",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand fraction")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("1 /")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_fraction_part_combination_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1 + a/d + b/d, 1 + (a+b)/d",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("combine fraction")));
        assert!(
            lines
                .iter()
                .any(|line| line
                    .contains("Combine fractions that already share the same denominator"))
        );
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:") && line.contains("/ d") && line.contains("a + b")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_like_terms_with_zero_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 2*x + 3*x + 0, 5*x",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("simplify")));
        assert!(lines.iter().any(|line| line.contains("Combine Like Terms")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("5")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_numeric_common_factor_fraction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (2*x)/(4*x), 1/2",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("simplify")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Simplify Nested Fraction")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:") && (line.contains("1 / 2") || line.contains("1/2"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_monomial_common_factor_fraction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (6*x^2)/(3*x), 2*x",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("simplify")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Simplify Nested Fraction")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains('2')
                && (line.contains('x') || line.contains("x"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_nested_fraction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1/(1/x + 1/y), (x*y)/(x+y)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("simplify")));
        assert!(lines.iter().any(|line| {
            line.contains("Add Fractions") || line.contains("Simplify Complex Fraction")
        }));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("(x + y)")
                && (line.contains("x·y /") || line.contains("x * y /") || line.contains("x·y/"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_small_polynomial_product_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (x - 1)*(x^5 + x^4 + x^3 + x^2 + x + 1), x^6 - 1",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("x^")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_fraction_decomposed_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (x+1)/(x-1), 1 + 2/(x-1)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("split fraction")));
        assert!(lines
            .iter()
            .any(|line| { line.contains("Split a fraction into a whole part plus remainder") }));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && (line.contains("2 / (x - 1) + 1") || line.contains("1 + 2 / (x - 1)"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_consecutive_telescoping_fraction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1/(n*(n+1)), 1/n - 1/(n+1)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand fraction")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Telescoping Fraction Split")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("1 / n")
                && line.contains("1 / (n + 1)")
                && line.contains('-')
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_gap_two_telescoping_fraction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1/(n*(n+2)), 1/2*(1/n - 1/(n+2))",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand fraction")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Telescoping Fraction Split")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("1 / 2")
                && line.contains("1 / n")
                && line.contains("1 / (n + 2)")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_gap_three_telescoping_fraction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1/(n*(n+3)), 1/3*(1/n - 1/(n+3))",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand fraction")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Telescoping Fraction Split")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("1 / 3")
                && line.contains("1 / n")
                && line.contains("1 / (n + 3)")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_negative_consecutive_telescoping_fraction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1/(n*(n-1)), 1/(n-1) - 1/n",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand fraction")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Telescoping Fraction Split")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("1 / (n - 1)")
                && line.contains("1 / n")
                && line.contains('-')
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_negative_gap_two_telescoping_fraction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1/(n*(n-2)), 1/2*(1/(n-2) - 1/n)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand fraction")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Telescoping Fraction Split")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("1 / 2")
                && line.contains("1 / (n - 2)")
                && line.contains("1 / n")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_affine_gap_two_telescoping_fraction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1/((2*n+1)*(2*n+3)), 1/2*(1/(2*n+1) - 1/(2*n+3))",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand fraction")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Telescoping Fraction Split")));
        assert!(lines.iter().any(|line| {
            let normalized = normalized_inline_math(line);
            line.starts_with("Result:")
                && normalized.contains("1/2")
                && normalized.contains("2*n+1")
                && normalized.contains("2*n+3")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_affine_shifted_telescoping_fraction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1/((2*n-1)*(2*n+1)), 1/2*(1/(2*n-1) - 1/(2*n+1))",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand fraction")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Telescoping Fraction Split")));
        assert!(lines.iter().any(|line| {
            let normalized = normalized_inline_math(line);
            line.starts_with("Result:")
                && normalized.contains("1/2")
                && normalized.contains("2*n-1")
                && normalized.contains("2*n+1")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_affine_coeff_three_telescoping_fraction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1/((3*n+2)*(3*n+5)), 1/3*(1/(3*n+2) - 1/(3*n+5))",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand fraction")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Telescoping Fraction Split")));
        assert!(lines.iter().any(|line| {
            let normalized = normalized_inline_math(line);
            line.starts_with("Result:")
                && normalized.contains("1/3")
                && normalized.contains("3*n+2")
                && normalized.contains("3*n+5")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_affine_coeff_three_shifted_telescoping_fraction_target(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1/((3*n-1)*(3*n+2)), 1/3*(1/(3*n-1) - 1/(3*n+2))",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand fraction")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Telescoping Fraction Split")));
        assert!(lines.iter().any(|line| {
            let normalized = normalized_inline_math(line);
            line.starts_with("Result:")
                && normalized.contains("1/3")
                && normalized.contains("3*n-1")
                && normalized.contains("3*n+2")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_affine_symbolic_coeff_telescoping_fraction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1/((a*n+2)*(a*n+5)), 1/3*(1/(a*n+2) - 1/(a*n+5))",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand fraction")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Telescoping Fraction Split")));
        assert!(lines.iter().any(|line| {
            let normalized = normalized_inline_math(line);
            line.starts_with("Result:")
                && normalized.contains("1/3")
                && normalized.contains("a*n+2")
                && normalized.contains("a*n+5")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_affine_symbolic_coeff_shifted_telescoping_fraction_target(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1/((a*n-1)*(a*n+2)), 1/3*(1/(a*n-1) - 1/(a*n+2))",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand fraction")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Telescoping Fraction Split")));
        assert!(lines.iter().any(|line| {
            let normalized = normalized_inline_math(line);
            line.starts_with("Result:")
                && normalized.contains("1/3")
                && normalized.contains("a*n-1")
                && normalized.contains("a*n+2")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_symbolic_shift_gap_telescoping_fraction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1/((n+a)*(n+b)), 1/(b-a)*(1/(n+a) - 1/(n+b))",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand fraction")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Telescoping Fraction Split")));
        assert!(lines.iter().any(|line| {
            let normalized = normalized_inline_math(line);
            line.starts_with("Result:")
                && normalized.contains("b-a")
                && normalized.contains("a+n")
                && normalized.contains("b+n")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_affine_symbolic_shift_gap_telescoping_fraction_target()
    {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1/((a*n+b)*(a*n+c)), 1/(c-b)*(1/(a*n+b) - 1/(a*n+c))",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand fraction")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Telescoping Fraction Split")));
        assert!(lines.iter().any(|line| {
            let normalized = normalized_inline_math(line);
            line.starts_with("Result:")
                && normalized.contains("1/(c-b)")
                && normalized.contains("a*n+b")
                && normalized.contains("a*n+c")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_consecutive_telescoping_fraction_combined_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1/n - 1/(n+1), 1/(n*(n+1))",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("combine fraction")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Telescoping Fraction Combine")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("1 /")
                && line.contains("n")
                && line.contains("n + 1")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_gap_two_telescoping_fraction_combined_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1/2*(1/n - 1/(n+2)), 1/(n*(n+2))",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("combine fraction")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Telescoping Fraction Combine")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("1 /")
                && line.contains("n")
                && line.contains("n + 2")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_gap_three_telescoping_fraction_combined_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1/3*(1/n - 1/(n+3)), 1/(n*(n+3))",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("combine fraction")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Telescoping Fraction Combine")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("1 /")
                && line.contains("n")
                && line.contains("n + 3")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_negative_consecutive_telescoping_fraction_combined_target(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1/(n-1) - 1/n, 1/(n*(n-1))",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("combine fraction")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Telescoping Fraction Combine")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("1 /")
                && line.contains("n")
                && line.contains("n - 1")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_negative_gap_two_telescoping_fraction_combined_target()
    {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1/2*(1/(n-2) - 1/n), 1/(n*(n-2))",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("combine fraction")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Telescoping Fraction Combine")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("1 /")
                && line.contains("n")
                && line.contains("n - 2")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_affine_gap_two_telescoping_fraction_combined_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1/2*(1/(2*n+1) - 1/(2*n+3)), 1/((2*n+1)*(2*n+3))",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("combine fraction")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Telescoping Fraction Combine")));
        assert!(lines.iter().any(|line| {
            let normalized = normalized_inline_math(line);
            line.starts_with("Result:")
                && normalized.contains("2*n+1")
                && normalized.contains("2*n+3")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_affine_shifted_telescoping_fraction_combined_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1/2*(1/(2*n-1) - 1/(2*n+1)), 1/((2*n-1)*(2*n+1))",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("combine fraction")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Telescoping Fraction Combine")));
        assert!(lines.iter().any(|line| {
            let normalized = normalized_inline_math(line);
            line.starts_with("Result:")
                && normalized.contains("2*n-1")
                && normalized.contains("2*n+1")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_affine_coeff_three_telescoping_fraction_combined_target(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1/3*(1/(3*n+2) - 1/(3*n+5)), 1/((3*n+2)*(3*n+5))",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("combine fraction")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Telescoping Fraction Combine")));
        assert!(lines.iter().any(|line| {
            let normalized = normalized_inline_math(line);
            line.starts_with("Result:")
                && normalized.contains("3*n+2")
                && normalized.contains("3*n+5")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_affine_coeff_three_shifted_telescoping_fraction_combined_target(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1/3*(1/(3*n-1) - 1/(3*n+2)), 1/((3*n-1)*(3*n+2))",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("combine fraction")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Telescoping Fraction Combine")));
        assert!(lines.iter().any(|line| {
            let normalized = normalized_inline_math(line);
            line.starts_with("Result:")
                && normalized.contains("3*n-1")
                && normalized.contains("3*n+2")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_affine_symbolic_coeff_telescoping_fraction_combined_target(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1/3*(1/(a*n+2) - 1/(a*n+5)), 1/((a*n+2)*(a*n+5))",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("combine fraction")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Telescoping Fraction Combine")));
        assert!(lines.iter().any(|line| {
            let normalized = normalized_inline_math(line);
            line.starts_with("Result:")
                && normalized.contains("a*n+2")
                && normalized.contains("a*n+5")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_affine_symbolic_coeff_shifted_telescoping_fraction_combined_target(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1/3*(1/(a*n-1) - 1/(a*n+2)), 1/((a*n-1)*(a*n+2))",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("combine fraction")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Telescoping Fraction Combine")));
        assert!(lines.iter().any(|line| {
            let normalized = normalized_inline_math(line);
            line.starts_with("Result:")
                && normalized.contains("a*n-1")
                && normalized.contains("a*n+2")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_symbolic_shift_gap_telescoping_fraction_combined_target(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1/(b-a)*(1/(n+a) - 1/(n+b)), 1/((n+a)*(n+b))",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("combine fraction")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Telescoping Fraction Combine")));
        assert!(lines.iter().any(|line| {
            let normalized = normalized_inline_math(line);
            line.starts_with("Result:") && normalized.contains("a+n") && normalized.contains("b+n")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_affine_symbolic_shift_gap_telescoping_fraction_combined_target(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1/(c-b)*(1/(a*n+b) - 1/(a*n+c)), 1/((a*n+b)*(a*n+c))",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("combine fraction")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Telescoping Fraction Combine")));
        assert!(lines.iter().any(|line| {
            let normalized = normalized_inline_math(line);
            line.starts_with("Result:")
                && normalized.contains("a*n+b")
                && normalized.contains("a*n+c")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_morrie_telescoping_target_directly() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive cos(x)*cos(2*x)*cos(4*x), sin(8*x)/(8*sin(x))",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("integrate prep")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Cos Product Telescoping")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("sin(8")
                && line.contains("8")
                && line.contains("sin(x)")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_scaled_morrie_telescoping_target_directly() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive cos(3*x)*cos(6*x), sin(12*x)/(4*sin(3*x))",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("integrate prep")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Cos Product Telescoping")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("sin(12")
                && line.contains("4")
                && line.contains("sin(3")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_dirichlet_kernel_target_directly() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1 + 2*cos(x) + 2*cos(2*x), sin(5*x/2)/sin(x/2)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("integrate prep")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Dirichlet Kernel Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:") && line.contains("sin(5") && line.contains("sin(x / 2)")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_longer_dirichlet_kernel_target_directly() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1 + 2*cos(x) + 2*cos(2*x) + 2*cos(3*x), sin(7*x/2)/sin(x/2)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("integrate prep")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Dirichlet Kernel Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:") && line.contains("sin(7") && line.contains("sin(x / 2)")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_scaled_dirichlet_kernel_target_directly() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1 + 2*cos(3*x) + 2*cos(6*x), sin(15*x/2)/sin(3*x/2)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("integrate prep")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Dirichlet Kernel Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:") && line.contains("sin(15") && line.contains("sin(3")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_longer_scaled_dirichlet_kernel_target_directly() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1 + 2*cos(2*x) + 2*cos(4*x) + 2*cos(6*x), sin(7*x)/sin(x)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("integrate prep")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Dirichlet Kernel Identity")));
        assert!(lines.iter().any(|line| line.starts_with("Result:")
            && line.contains("sin(7")
            && line.contains("sin(x)")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_finite_telescoping_product_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive product((k+1)/k, k, 1, n), n+1",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("simplify")));
        assert!(lines.iter().any(|line| line.contains("Finite Product")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("n + 1")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_shifted_finite_telescoping_product_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive product((k+2)/(k+1), k, 1, n), (n+2)/2",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("simplify")));
        assert!(lines.iter().any(|line| line.contains("Finite Product")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:") && line.contains("n + 2") && line.contains("/ 2")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_finite_telescoping_sum_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sum(1/(k*(k+1)), k, 1, n), 1 - 1/(n+1)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("simplify")));
        assert!(lines.iter().any(|line| line.contains("Finite Summation")));
        assert!(lines
            .iter()
            .any(|line| { line.starts_with("Result:") && line.contains("1 - 1 / (n + 1)") }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_shifted_finite_telescoping_sum_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sum(1/((k+2)*(k+3)), k, 1, n), 1/3 - 1/(n+3)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("simplify")));
        assert!(lines.iter().any(|line| line.contains("Finite Summation")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:") && line.contains("1 / 3") && line.contains("n + 3")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_log_change_of_base_chain_contraction() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive log(b,a)*log(a,c), log(b,c)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("contract logs")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("log(b, c)")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_log_change_of_base_chain_expansion() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive log(b,c), log(b,a)*log(a,c)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("simplify")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:") && line.contains("log(a, c)") && line.contains("log(b, a)")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_three_link_log_change_of_base_chain_contraction() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive log(a,b)*log(b,c)*log(c,d), log(a,d)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("contract logs")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("log(a, d)")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_three_link_log_change_of_base_chain_expansion() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive log(a,d), log(a,b)*log(b,c)*log(c,d)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("simplify")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("log(a, b)")
                && line.contains("log(b, c)")
                && line.contains("log(c, d)")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_fraction_combined_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1 + 2/(x-1), (x+1)/(x-1)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("combine fraction")));
        assert!(lines
            .iter()
            .any(|line| { line.contains("Combine the whole part with the remaining fraction") }));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && (line.contains("(1 + x) / (x - 1)") || line.contains("(x + 1) / (x - 1)"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_same_denominator_fraction_sum_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive a/x + b/x, (a+b)/x",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("combine fraction")));
        assert!(
            lines
                .iter()
                .any(|line| line
                    .contains("Combine fractions that already share the same denominator"))
        );
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && (line.contains("(a + b) / x") || line.contains("(b + a) / x"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_general_fraction_sum_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1/x + 1/y, (x+y)/(x*y)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("combine fraction")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Combine two fractions into a single denominator")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && (line.contains("(x + y) / (x * y)")
                    || line.contains("(y + x) / (x * y)")
                    || line.contains("(x + y) / (y * x)"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_same_denominator_fraction_difference_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive a/x - b/x, (a-b)/x",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("combine fraction")));
        assert!(lines.iter().any(|line| line
            .contains("Combine fractions with the same denominator into one subtraction")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && (line.contains("(a - b) / x") || line.contains("(-b + a) / x"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_general_fraction_difference_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1/x - 1/y, (y-x)/(x*y)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("combine fraction")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Subtract two fractions into a single denominator")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && (line.contains("(y - x) / (x * y)")
                    || line.contains("(-x + y) / (x * y)")
                    || line.contains("(y - x) / (y * x)"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_three_fraction_same_denominator_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive a/d + b/d + c/d, (a+b+c)/d",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("combine fraction")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("/ d")
                && line.contains('a')
                && line.contains('b')
                && line.contains('c')
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_combines_same_denominator_fraction_subset_with_passthrough() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1 + a/d + b/d, 1 + (a+b)/d",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("combine fraction")));
        assert!(lines.iter().any(|line| {
            line.contains("Combine fractions that already share the same denominator")
        }));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && (line.contains("1 + (a + b) / d") || line.contains("(a + b) / d + 1"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_combines_three_same_denominator_fractions_with_passthrough() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1 + a/d + b/d + c/d, 1 + (a+b+c)/d",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("combine fraction")));
        assert!(lines.iter().any(|line| {
            line.contains("Combine fractions that already share the same denominator")
        }));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && (line.contains("1 + (a + b + c) / d") || line.contains("(a + b + c) / d + 1"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_three_fraction_distribution_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (a+b+c)/d, a/d + b/d + c/d",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand fraction")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("a / d")
                && line.contains("b / d")
                && line.contains("c / d")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_expands_fraction_subset_with_passthrough() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1 + (a+b+c)/d, 1 + a/d + b/d + c/d",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand fraction")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && (line.contains("1 + a / d + b / d + c / d")
                    || line.contains("a / d + b / d + c / d + 1"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_common_factor_sum_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive a*b + a*c, a*(b+c)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("factor")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && (line.contains("a * (b + c)") || line.contains("(b + c) * a"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_common_factor_difference_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive a*b - a*c, a*(b-c)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("factor")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && (line.contains("a * (b - c)") || line.contains("(b - c) * a"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_common_factor_sum_expansion_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive a*(b+c), a*b + a*c",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("a * b")
                && line.contains("a * c")
                && line.contains('+')
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_common_factor_difference_expansion_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive a*(b-c), a*b - a*c",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("a * b")
                && line.contains("a * c")
                && line.contains('-')
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_three_term_common_factor_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive a*x + b*x + c*x, x*(a+b+c)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("factor")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && (line.contains("x * (a + b + c)") || line.contains("(a + b + c) * x"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_three_term_common_factor_expansion_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive x*(a-b-c), a*x - b*x - c*x",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("a * x")
                && line.contains("b * x")
                && line.contains("c * x")
                && line.contains('-')
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_mirrored_difference_of_squares_fraction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (a^2-b^2)/(a+b), a-b",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("simplify")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("a - b")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_negative_perfect_square_fraction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (x^2-2*x+1)/(x-1), x-1",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(
            lines
                .iter()
                .any(|line| line.starts_with("Strategy:") && line.contains("simplify")),
            "expected simplify strategy, got: {lines:?}"
        );
        assert!(
            lines
                .iter()
                .any(|line| line.contains("Cancel common factor")),
            "expected exact cancellation step, got: {lines:?}"
        );
        assert!(
            lines
                .iter()
                .any(|line| line.starts_with("Result:") && line.contains("x - 1")),
            "expected x - 1 result, got: {lines:?}"
        );
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_symbolic_negative_perfect_square_fraction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (a^2-2*a*b+b^2)/(a-b), a-b",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(
            lines
                .iter()
                .any(|line| line.starts_with("Strategy:") && line.contains("simplify")),
            "expected simplify strategy, got: {lines:?}"
        );
        assert!(
            lines
                .iter()
                .any(|line| line.contains("Cancel common factor")),
            "expected exact symbolic cancellation step, got: {lines:?}"
        );
        assert!(
            lines
                .iter()
                .any(|line| line.starts_with("Result:") && line.contains("a - b")),
            "expected a - b result, got: {lines:?}"
        );
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_difference_of_cubes_fraction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (a^3-b^3)/(a-b), a^2+a*b+b^2",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("simplify")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Cancel Sum/Difference of Cubes Fraction")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains('a')
                && line.contains('b')
                && !line.contains('/')
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_sum_of_cubes_fraction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (a^3+b^3)/(a+b), a^2-a*b+b^2",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("simplify")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Cancel Sum/Difference of Cubes Fraction")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains('a')
                && line.contains('b')
                && line.contains('-')
                && !line.contains('/')
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_difference_of_cubes_factor_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive a^3-b^3, (a-b)*(a^2+a*b+b^2)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("factor")));
        assert!(lines.iter().any(|line| line.contains("Factorization")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains('a')
                && line.contains('b')
                && line.contains('(')
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_sum_of_cubes_factor_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive a^3+b^3, (a+b)*(a^2-a*b+b^2)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("factor")));
        assert!(lines.iter().any(|line| line.contains("Factorization")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains('a')
                && line.contains('b')
                && line.contains('(')
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_pythagorean_identity_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sin(x)^2 + cos(x)^2, 1",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("simplify")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Pythagorean Chain Identity")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains('1')));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_pythagorean_factor_form_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1 - sin(x)^2, cos(x)^2",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("simplify")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Pythagorean Factor Form")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("cos")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_reverse_pythagorean_factor_form_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sin(x)^2, 1-cos(x)^2",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("simplify")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Pythagorean Factor Form")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("1 - cos")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_sec_squared_contraction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1 + tan(x)^2, sec(x)^2",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("contract trig")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Recognize Secant Squared")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("sec")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_sec_squared_expansion_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sec(x)^2, 1 + tan(x)^2",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand trig")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Expand Secant Squared")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("tan")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_inverse_tan_identity_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive arctan(3) + arctan(1/3), pi/2",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("simplify")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Inverse Tan Relations")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:") && line.contains("pi") && line.contains('/')
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_term_and_fraction_subtraction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive a - b/a, (a^2-b)/a",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("combine fraction")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Put the term and the fraction over the same denominator")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains('b')
                && (line.contains("/ a") || line.contains(")/a") || line.contains("/a"))
                && (line.contains("a^2")
                    || line.contains("a²")
                    || line.contains("a^(2)")
                    || line.contains("a * a")
                    || line.contains("a² -")
                    || line.contains("a^2 -"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_odd_half_power_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive x^(3/2), abs(x)*sqrt(x)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines.iter().any(|line| {
            line.starts_with("Strategy:") && line.contains("expand odd half power")
        }));
        assert!(lines
            .iter()
            .any(|line| line.contains("Rewrite an odd half-integer power using a square root")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && (line.contains("sqrt") || line.contains("√"))
                && (line.contains("|x|") || line.contains("abs"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_odd_half_power_target_after_simplify() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sqrt(x^3), abs(x)*sqrt(x)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines.iter().any(|line| {
            line.starts_with("Strategy:") && line.contains("expand odd half power")
        }));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && (line.contains("sqrt") || line.contains("√"))
                && (line.contains("|x|") || line.contains("abs"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_same_base_fractional_power_merge_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive x^(1/2)*x^(2/3), x^(7/6)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| { line.starts_with("Strategy:") && line.contains("combine powers") }));
        assert!(lines
            .iter()
            .any(|line| line.contains("Combine powers with same base (n-ary)")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("7 / 6")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_mixed_root_and_power_merge_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sqrt(x)*x^(2/3), x^(7/6)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| { line.starts_with("Strategy:") && line.contains("combine powers") }));
        assert!(lines.iter().any(|line| line.contains("sqrt(x) = x^(1/2)")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Combine powers with same base (n-ary)")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("7 / 6")));
    }

    #[test]
    fn evaluate_derive_command_lines_treats_root_and_fractional_power_targets_as_same_form() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive x^(1/2), sqrt(x)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines.iter().any(|line| line.contains("Already at target.")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("sqrt")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_factor_with_division_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive a*x + b*x + c, x*(a + b + c/x)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines.iter().any(|line| {
            line.starts_with("Strategy:") && line.contains("factor out with division")
        }));
        assert!(lines.iter().any(|line| line.contains("x ≠ 0")));
        assert!(lines
            .iter()
            .any(|line| { line.starts_with("Result:") && line.contains("c / x") }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_product_to_sum_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 2*sin(x)*cos(y), sin(x+y) + sin(x-y)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand trig")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Product-to-Sum Identity")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("sin(x + y) + sin(x - y)")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_cos_sin_product_to_sum_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 2*cos(x)*sin(y), sin(x+y) - sin(x-y)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand trig")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Product-to-Sum Identity")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("sin(x + y) - sin(x - y)")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_general_sine_sum_to_product_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sin(5*x)+sin(x), 2*sin(3*x)*cos(2*x)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand trig")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Sum-to-Product Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("sin(3")
                && line.contains("cos(2")
                && line.contains("2")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_general_sine_difference_to_product_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sin(5*x)-sin(x), 2*cos(3*x)*sin(2*x)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand trig")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Sum-to-Product Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("sin(2")
                && line.contains("cos(3")
                && line.contains("2")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_general_cosine_sum_to_product_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive cos(5*x)+cos(x), 2*cos(3*x)*cos(2*x)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand trig")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Sum-to-Product Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("cos(3")
                && line.contains("cos(2")
                && line.contains("2")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_general_cosine_difference_to_product_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive cos(5*x)-cos(x), -2*sin(3*x)*sin(2*x)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand trig")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Sum-to-Product Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("sin(3")
                && line.contains("sin(2")
                && line.contains("-2")
        }));
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
