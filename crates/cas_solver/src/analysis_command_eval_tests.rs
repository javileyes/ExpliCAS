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

    fn derive_lines(input: &str) -> Vec<String> {
        let mut simplifier = crate::Simplifier::with_default_rules();
        evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            input,
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate")
    }

    fn assert_derive_strategy(lines: &[String], strategy: &str) {
        assert!(
            lines
                .iter()
                .any(|line| line.starts_with("Strategy:") && line.contains(strategy)),
            "expected strategy `{strategy}`, got: {lines:?}"
        );
    }

    fn assert_result_contains_all(lines: &[String], fragments: &[&str]) {
        let result = lines
            .iter()
            .find(|line| line.starts_with("Result:"))
            .expect("expected result line");
        let normalized = normalized_inline_math(result);
        for fragment in fragments {
            let needle = normalized_inline_math(fragment);
            assert!(
                normalized.contains(&needle),
                "expected result `{result}` to contain fragment `{fragment}`"
            );
        }
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
            "derive a^2 - b^2, (a - b)*(a + b)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Target:") && line.contains("a")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:") && line.contains("(a + b)") && line.contains("(a - b)")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_sign_distributed_factored_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive x^2 - y^2, -(y-x)*(x+y)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("factor")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:") && line.contains("-(y - x)") && line.contains("(x + y)")
        }));
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

    type SolvePrepEvalCase = (&'static str, &'static [&'static str]);

    fn assert_tabulated_solve_prep_eval_cases(cases: &[SolvePrepEvalCase]) {
        for (input, fragments) in cases {
            let lines = derive_lines(input);
            assert_derive_strategy(&lines, "solve prep");
            assert_result_contains_all(&lines, fragments);
        }
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_tabulated_solve_prep_monic_targets() {
        let cases: &[SolvePrepEvalCase] = &[
            (
                "derive x^2 + 6*x + 5, (x+3)^2 - 4",
                &["(x+3)^(2)", "-4"][..],
            ),
            (
                "derive x^2 + 2*b*x + c, (x+b)^2 + c - b^2",
                &["(b+x)^(2)", "c-b^(2)"][..],
            ),
            (
                "derive x^2 + 3*x + 1, (x+3/2)^2 - 5/4",
                &["(3/2+x)^(2)", "-5/4"][..],
            ),
        ];

        assert_tabulated_solve_prep_eval_cases(cases);
        for (input, _fragments) in cases {
            let lines = derive_lines(input);
            assert!(lines.iter().any(|line| line.trim() == "Subpasos:"));
            assert!(lines
                .iter()
                .any(|line| line.contains("Añadir y restar el cuadrado del semicoeficiente")));
            assert!(lines
                .iter()
                .any(|line| line.contains("Agrupar el trinomio como cuadrado perfecto")));
        }
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_tabulated_solve_prep_symbolic_positive_targets() {
        assert_tabulated_solve_prep_eval_cases(&[(
            "derive a*x^2 + b*x + c, a*(x + b/(2*a))^2 + c - b^2/(4*a)",
            &["b/(2*a)", "^(2)", "c-b^(2)/(4*a)"][..],
        )]);
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_tabulated_solve_prep_negative_linear_targets() {
        assert_tabulated_solve_prep_eval_cases(&[(
            "derive a*x^2 - b*x + c, a*(x - b/(2*a))^2 + c - b^2/(4*a)",
            &["(x-b/(2*a))^(2)", "c-b^(2)/(4*a)"][..],
        )]);
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_tabulated_solve_prep_negative_leading_targets() {
        assert_tabulated_solve_prep_eval_cases(&[(
            "derive -x^2 + b*x + c, -(x - b/2)^2 + c + b^2/4",
            &["(x-b/2)^(2)", "b^(2)/4", "-("][..],
        )]);
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_tabulated_solve_prep_fractional_targets() {
        assert_tabulated_solve_prep_eval_cases(&[(
            "derive (a/2)*x^2 + b*x + c, (a/2)*(x + b/a)^2 + c - b^2/(2*a)",
            &["b/a", "^(2)", "b^(2)/(2*a)"][..],
        )]);
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
            lines
                .iter()
                .any(|line| line.contains("Sophie Germain Identity")),
            "expected Sophie Germain identity step in derive output, got: {lines:?}"
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
    fn evaluate_derive_command_lines_reaches_signed_symbolic_trinomial_square_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (a-b+c)^2, a^2+b^2+c^2-2*a*b+2*a*c-2*b*c",
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
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("a^(2)")
                && line.contains("b^(2)")
                && line.contains("c^(2)")
                && line.contains("- 2 * a * b")
                && line.contains("+ 2 * a * c")
                && line.contains("- 2 * b * c")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_signed_symbolic_trinomial_square_alt_order_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (a+b-c)^2, a^2+b^2+c^2+2*a*b-2*a*c-2*b*c",
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
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("a^(2)")
                && line.contains("b^(2)")
                && line.contains("c^(2)")
                && line.contains("+ 2 * a * b")
                && line.contains("- 2 * a * c")
                && line.contains("- 2 * b * c")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_signed_xyz_trinomial_square_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (x-y+z)^2, x^2+y^2+z^2-2*x*y+2*x*z-2*y*z",
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
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("x^(2)")
                && line.contains("y^(2)")
                && line.contains("z^(2)")
                && line.contains("- 2 * x * y")
                && line.contains("+ 2 * x * z")
                && line.contains("- 2 * y * z")
        }));
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
    fn evaluate_derive_command_lines_reaches_tabulated_collect_targets() {
        let collect_cases = [
            (
                "derive a*x + b*x + c, (a + b)*x + c",
                &["a+b", "x", "+c"][..],
            ),
            (
                "derive a*y + b*y + c, (a + b)*y + c",
                &["a+b", "y", "+c"][..],
            ),
            (
                "derive a*x^2 + b*x + c*x^2 + d*x + e*x^2 + f, (a + c + e)*x^2 + (b + d)*x + f",
                &["a+c+e", "x^(2)", "b+d", "+f"][..],
            ),
            (
                "derive x*y + x*z + w, x*(y + z) + w",
                &["x", "y+z", "+w"][..],
            ),
            (
                "derive a*x*y + b*x*y + c, (a + b)*x*y + c",
                &["a+b", "x", "y", "+c"][..],
            ),
            (
                "derive a*x*y + b*x*y + c*x*z + d*x*z + e, (a + b)*x*y + (c + d)*x*z + e",
                &["a+b", "x", "y", "c+d", "z", "+e"][..],
            ),
        ];

        for (input, fragments) in collect_cases {
            let lines = derive_lines(input);
            assert_derive_strategy(&lines, "collect");
            assert_result_contains_all(&lines, fragments);
        }

        let lines = derive_lines("derive x^2 + 2*x + 1, x*(x + 2) + 1");
        assert_derive_strategy(&lines, "combine like terms");
        assert_result_contains_all(&lines, &["x", "x+2", "+1"]);
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
    fn evaluate_derive_command_lines_prefers_expand_for_binomial_expansion_with_cancellation() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (a+b)^2 - a^2 - 2*a*b, b^2",
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
            .any(|line| line.starts_with("Result:") && line.contains("b^(2)")));
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
    fn evaluate_derive_command_lines_reaches_tabulated_trinomial_square_expanded_targets() {
        let cases = [
            (
                "derive (a + b + c)^2, a^2 + b^2 + c^2 + 2*a*b + 2*a*c + 2*b*c",
                &[
                    "a^(2)",
                    "b^(2)",
                    "c^(2)",
                    "2 * a * b",
                    "2 * a * c",
                    "2 * b * c",
                ][..],
            ),
            (
                "derive (a - b + c)^2, a^2 + b^2 + c^2 - 2*a*b + 2*a*c - 2*b*c",
                &[
                    "a^(2)",
                    "b^(2)",
                    "c^(2)",
                    "- 2 * a * b",
                    "2 * a * c",
                    "- 2 * b * c",
                ][..],
            ),
        ];

        for (input, fragments) in cases {
            let lines = derive_lines(input);
            assert_derive_strategy(&lines, "expand");
            assert_result_contains_all(&lines, fragments);
        }
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_tabulated_log_expanded_targets() {
        let cases = [
            ("derive ln(x*y), ln(x) + ln(y)", "expand_log", &["ln(x)", "ln(y)"][..]),
            ("derive ln(x/y), ln(x) - ln(y)", "expand_log", &["ln(x)", "ln(y)"][..]),
            (
                "derive ln((x*y)/z), ln(x) + ln(y) - ln(z)",
                "expand_log",
                &["ln(x)", "ln(y)", "ln(z)"][..],
            ),
            (
                "derive ln((x^2*y)/(z*t)), 2*ln(abs(x)) + ln(y) - ln(z) - ln(t)",
                "expand_log",
                &["2 * ln(|x|)", "ln(y)", "ln(z)", "ln(t)"][..],
            ),
            (
                "derive log(b, (x*y)/z), log(b, x) + log(b, y) - log(b, z)",
                "expand_log",
                &["log(b, x)", "log(b, y)", "log(b, z)"][..],
            ),
            (
                "derive log(b, (x^2*y^3)/(z^2*t)), 2*log(b, x) + 3*log(b, y) - 2*log(b, z) - log(b, t)",
                "expand_log",
                &["2 * log(b, x)", "3 * log(b, y)", "2 * log(b, z)", "log(b, t)"][..],
            ),
            ("derive ln(x^2), 2*ln(abs(x))", "expand_log", &["2", "ln("][..]),
            ("derive log(b, x^3), 3*log(b, x)", "expand_log", &["3", "log("][..]),
            (
                "derive ln((x*y)^2), ln(x^2)+ln(y^2)",
                "expand_log",
                &["ln(x^(2))", "ln(y^(2))"][..],
            ),
            (
                "derive 2*ln(abs(x*y)), 2*ln(abs(x))+2*ln(abs(y))",
                "expand_log",
                &["2 * ln(|x|)", "2 * ln(|y|)"][..],
            ),
            (
                "derive log(b,(x*y)^2), 2*log(b,x)+2*log(b,y)",
                "expand_log",
                &["2 * log(b, x)", "2 * log(b, y)"][..],
            ),
            (
                "derive log(b,c), log(b,a)*log(a,c)",
                "simplify",
                &["log(a, c)", "log(b, a)"][..],
            ),
        ];

        for (input, strategy, fragments) in cases {
            let lines = derive_lines(input);
            assert_derive_strategy(&lines, strategy);
            assert_result_contains_all(&lines, fragments);
        }
    }

    #[test]
    fn evaluate_derive_command_lines_factors_log_argument_before_expanding() {
        let lines = derive_lines("derive log(x^2-y^2), log(x-y)+log(x+y)");

        assert_derive_strategy(&lines, "expand_log");
        assert!(
            lines.iter().any(|line| line.contains("Factorization")),
            "expected factorization step in derive output, got: {lines:?}"
        );
        assert!(
            lines.iter().any(|line| line.contains("expand_log")),
            "expected expand_log step in derive output, got: {lines:?}"
        );
        assert_result_contains_all(&lines, &["ln(x - y)", "ln(x + y)"]);
    }

    #[test]
    fn evaluate_derive_command_lines_factors_log_quotient_argument_before_expanding() {
        let lines = derive_lines("derive log((x^2-y^2)/(u*v)), log(x-y)+log(x+y)-log(u)-log(v)");

        assert_derive_strategy(&lines, "expand_log");
        assert!(
            lines.iter().any(|line| line.contains("Factorization")),
            "expected factorization step in derive output, got: {lines:?}"
        );
        assert!(
            lines.iter().any(|line| line.contains("expand_log")),
            "expected expand_log step in derive output, got: {lines:?}"
        );
        assert_result_contains_all(&lines, &["ln(x - y)", "ln(x + y)", "ln(u)", "ln(v)"]);
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_tabulated_log_contracted_targets() {
        let cases = [
            ("derive ln(x) + ln(y), ln(x*y)", "contract logs", &["ln("][..]),
            ("derive ln(x) - ln(y), ln(x/y)", "contract logs", &["ln("][..]),
            (
                "derive ln(x) + ln(y) - ln(z), ln((x*y)/z)",
                "contract logs",
                &["ln(", "x", "y", "z"][..],
            ),
            (
                "derive 2*ln(abs(x)) + ln(y) - ln(z) - ln(t), ln((x^2*y)/(z*t))",
                "contract logs",
                &["ln(", "x^(2)", "y", "z", "t"][..],
            ),
            (
                "derive 3*ln(x) + 2*ln(abs(y)), ln(x^3*y^2)",
                "contract logs",
                &["ln("][..],
            ),
            (
                "derive 3*ln(x) - 2*ln(y), ln(x^3/y^2)",
                "contract logs",
                &["ln("][..],
            ),
            (
                "derive log(2, x) - log(2, y), log(2, x/y)",
                "contract logs",
                &["log("][..],
            ),
            (
                "derive 2*log(b, x) + 3*log(b, y) - 2*log(b, z) - log(b, t), log(b, (x^2*y^3)/(z^2*t))",
                "contract logs",
                &["log(", "x^(2)", "y^(3)", "z^(2)", "t"][..],
            ),
            (
                "derive 3*log(2, x) - 2*log(2, y), log(2, x^3/y^2)",
                "contract logs",
                &["log("][..],
            ),
            (
                "derive log(b,a)*log(a,c), log(b,c)",
                "contract logs",
                &["log(b, c)"][..],
            ),
        ];

        for (input, strategy, fragments) in cases {
            let lines = derive_lines(input);
            assert_derive_strategy(&lines, strategy);
            assert_result_contains_all(&lines, fragments);
        }
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_grouped_log_contraction_targets_directly() {
        let cases = [
            (
                "derive ln(x^2)+ln(y^2), ln((x*y)^2)",
                &["ln((x * y)^(2))"][..],
            ),
            (
                "derive 2*ln(abs(x))+2*ln(abs(y)), 2*ln(abs(x*y))",
                &["2 * ln(|x * y|)"][..],
            ),
            (
                "derive 2*log(b,x)+2*log(b,y), log(b,(x*y)^2)",
                &["log(b, (x * y)^(2))"][..],
            ),
        ];

        for (input, fragments) in cases {
            let lines = derive_lines(input);
            assert_derive_strategy(&lines, "contract logs");
            assert_result_contains_all(&lines, fragments);
        }
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_grouped_log_contraction_targets_with_passthrough_directly(
    ) {
        let cases = [
            (
                "derive ln(x^2)+ln(y^2)+a, ln((x*y)^2)+a",
                &["ln((x * y)^(2))", "+ a"][..],
            ),
            (
                "derive 2*ln(abs(x))+2*ln(abs(y))+a, 2*ln(abs(x*y))+a",
                &["2 * ln(|x * y|)", "+ a"][..],
            ),
            (
                "derive 2*log(b,x)+2*log(b,y)+a, log(b,(x*y)^2)+a",
                &["log(b, (x * y)^(2))", "+ a"][..],
            ),
        ];

        for (input, fragments) in cases {
            let lines = derive_lines(input);
            assert_derive_strategy(&lines, "contract logs");
            assert_result_contains_all(&lines, fragments);
        }
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_grouped_log_expansion_targets_with_passthrough_directly(
    ) {
        let cases = [
            (
                "derive ln((x*y)^2)+a, ln(x^2)+ln(y^2)+a",
                &["ln(x^(2))", "ln(y^(2))", "+ a"][..],
            ),
            (
                "derive 2*ln(abs(x*y))+a, 2*ln(abs(x))+2*ln(abs(y))+a",
                &["2 * ln(|x|)", "2 * ln(|y|)", "+ a"][..],
            ),
            (
                "derive log(b,(x*y)^2)+a, 2*log(b,x)+2*log(b,y)+a",
                &["2 * log(b, x)", "2 * log(b, y)", "+ a"][..],
            ),
        ];

        for (input, fragments) in cases {
            let lines = derive_lines(input);
            assert_derive_strategy(&lines, "expand_log");
            assert_result_contains_all(&lines, fragments);
        }
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
    fn evaluate_derive_command_lines_reaches_tabulated_reciprocal_and_quotient_trig_expanded_targets(
    ) {
        let cases = [
            (
                "derive tan(2*x), (sin(2*x))/(cos(2*x))",
                &["sin(2*x)", "cos(2*x)"][..],
            ),
            ("derive csc(x), 1/sin(x)", &["1 / sin(x)"][..]),
        ];

        for (input, fragments) in cases {
            let lines = derive_lines(input);
            assert_derive_strategy(&lines, "expand trig");
            assert_result_contains_all(&lines, fragments);
        }
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
    fn evaluate_derive_command_lines_reaches_tabulated_scaled_trig_contracted_targets() {
        let cases = [
            ("derive 1/cos(a*x), sec(a*x)", &["sec(a*x)"][..]),
            ("derive 1/sin(a*x), csc(a*x)", &["csc(a*x)"][..]),
            ("derive cos(a*x)/sin(a*x), cot(a*x)", &["cot(a*x)"][..]),
        ];

        for (input, fragments) in cases {
            let lines = derive_lines(input);
            assert_derive_strategy(&lines, "contract trig");
            assert_result_contains_all(&lines, fragments);
        }
    }

    #[test]
    fn evaluate_derive_command_lines_contracts_trig_quotient_after_arg_simplify_directly() {
        let lines = derive_lines("derive sin(2*x)/cos(x+x), tan(2*x)");

        assert_derive_strategy(&lines, "contract trig");
        assert_result_contains_all(&lines, &["tan(2*x)"]);
    }

    #[test]
    fn evaluate_derive_command_lines_expands_double_angle_after_arg_simplify_directly() {
        let lines = derive_lines("derive sin(x+x), 2*sin(x)*cos(x)");

        assert_derive_strategy(&lines, "expand trig");
        assert_result_contains_all(&lines, &["2 * sin(x) * cos(x)"]);
    }

    #[test]
    fn evaluate_derive_command_lines_contracts_cos_diff_over_sin_diff_directly() {
        let lines = derive_lines("derive (cos(x)-cos(3*x))/(sin(3*x)-sin(x)), tan(2*x)");

        assert_derive_strategy(&lines, "contract trig");
        assert_result_contains_all(&lines, &["tan(2*x)"]);
        assert!(lines.iter().any(|line| line.contains("Trig Quotient")));
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
            .any(|line| { line.starts_with("Strategy:") && line.contains("rewrite trigs") }));
        assert!(lines
            .iter()
            .any(|line| line.contains("Reciprocal Product Identity")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("1")));
    }

    #[test]
    fn evaluate_derive_command_lines_contracts_reciprocal_trig_product_to_one_with_passthrough_directly(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive tan(x)*cot(x)+a, 1+a",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| { line.starts_with("Strategy:") && line.contains("rewrite trigs") }));
        assert!(lines
            .iter()
            .any(|line| line.contains("Reciprocal Product Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:") && (line.contains("1 + a") || line.contains("a + 1"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_tabulated_reciprocal_pythagorean_identity_targets() {
        let cases = ["derive sec(x)^2 - tan(x)^2, 1"];

        for command in cases {
            let mut simplifier = crate::Simplifier::with_default_rules();
            let lines = evaluate_derive_command_lines_with_resolver(
                &mut simplifier,
                command,
                crate::FullSimplifyDisplayMode::Normal,
                crate::SimplifyOptions::default(),
                |_ctx, expr| Ok(expr),
            )
            .expect("derive should evaluate");

            assert!(lines
                .iter()
                .any(|line| { line.starts_with("Strategy:") && line.contains("rewrite trigs") }));
            assert!(lines
                .iter()
                .any(|line| line.contains("Reciprocal Pythagorean Identity")));
            assert!(lines
                .iter()
                .any(|line| line.starts_with("Result:") && line.contains("1")));
        }
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
    fn evaluate_derive_command_lines_reaches_scaled_half_angle_tangent_target_directly_without_trailing_noise(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (1-cos(2*a*x))/sin(2*a*x), tan(a*x)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("contract trig")));
        assert_eq!(
            lines
                .iter()
                .filter(|line| line.trim_start().starts_with("1."))
                .count(),
            1
        );
        assert!(!lines.iter().any(|line| line.contains("Tan to Sin/Cos")));
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
    fn evaluate_derive_command_lines_reaches_scaled_half_angle_tangent_alt_target_directly_without_trailing_noise(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sin(2*a*x)/(1+cos(2*a*x)), tan(a*x)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("contract trig")));
        assert_eq!(
            lines
                .iter()
                .filter(|line| line.trim_start().starts_with("1."))
                .count(),
            1
        );
        assert!(!lines.iter().any(|line| line.contains("Tan to Sin/Cos")));
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

    type RationalizedEvalCase = (&'static str, &'static str, &'static [&'static str]);

    fn assert_tabulated_rationalized_eval_cases(cases: &[RationalizedEvalCase]) {
        for (input, strategy, fragments) in cases {
            let lines = derive_lines(input);
            assert_derive_strategy(&lines, strategy);
            assert_result_contains_all(&lines, fragments);
            assert!(
                !lines.iter().any(|line| line.contains("x + 1^(2)")),
                "derive rationalize CLI output should not corrupt the retargeted after expression, got: {lines:?}"
            );
        }
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_tabulated_rationalized_numeric_targets() {
        assert_tabulated_rationalized_eval_cases(&[
            (
                "derive 1/(sqrt(x)+1), (sqrt(x)-1)/(x-1)",
                "rationalize",
                &["sqrt", "x-1"][..],
            ),
            (
                "derive 1/(sqrt(x)-2), (sqrt(x)+2)/(x-4)",
                "rationalize",
                &["sqrt", "x-4"][..],
            ),
            (
                "derive 1/(1+x^(1/3)), (1-x^(1/3)+x^(2/3))/(1+x)",
                "rationalize",
                &["x^(1 / 3)", "x^(2 / 3)", "x + 1"][..],
            ),
        ]);
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_representative_rationalized_zero_target() {
        let lines = derive_lines("derive 1 / (sqrt(x) - 1) - (sqrt(x) + 1) / (x - 1), 0");

        assert_derive_strategy(&lines, "rationalize");
        assert!(lines.iter().any(|line| {
            line.contains("Rationalize Linear Sqrt Denominator")
                || line.contains("Subtraction Self-Cancel")
        }));
        assert_result_contains_all(&lines, &["0"]);
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_radical_notable_quotient_directly() {
        let lines = derive_lines("derive (x^(3/2)-1)/(sqrt(x)-1), sqrt(x)+x+1");

        assert_derive_strategy(&lines, "rationalize");
        assert!(
            lines
                .iter()
                .filter(|line| line.trim_start().starts_with("1. "))
                .count()
                == 1
                && !lines
                    .iter()
                    .any(|line| line.trim_start().starts_with("2. ")),
            "expected one visible rationalize step, got: {lines:?}"
        );
        assert!(lines.iter().any(|line| {
            line.contains("Rationalize Linear Sqrt Denominator")
                || line.contains("Polynomial division with opaque substitution")
                || line.contains("Reconocer un cociente notable")
        }));
        assert_result_contains_all(&lines, &["sqrt(x)", "x+1"]);
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_perfect_square_root_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sqrt(a^2 + 2*a*b + b^2), abs(a+b)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| { line.starts_with("Strategy:") && line.contains("rewrite radicals") }));
        assert!(lines
            .iter()
            .any(|line| line.contains("Sqrt Perfect Square")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("|a + b|")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_perfect_square_root_target_with_passthrough() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sqrt(a^2 + 2*a*b + b^2)+c, abs(a+b)+c",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("rewrite radicals")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Sqrt Perfect Square")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("|a + b| + c")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_nested_radical_denesting_target_as_radical_rewrite() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sqrt(6 + 2*sqrt(5)), sqrt(5)+1",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("rewrite radicals")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Sqrt Perfect Square")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Abs Of Sum Of Squares")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("sqrt(5) + 1")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_hyperbolic_double_angle_target_with_passthrough() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 2*sinh(x)*cosh(x)+a, sinh(2*x)+a",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("rewrite hyperbolics")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Hyperbolic Double-Angle Identity")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("sinh(2 * x) + a")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_hyperbolic_pythagorean_target_with_passthrough() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive cosh(x)^2-sinh(x)^2+a, 1+a",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("rewrite hyperbolics")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Hyperbolic Pythagorean Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:") && (line.contains("1 + a") || line.contains("a + 1"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_tabulated_fraction_expanded_targets() {
        let cases = [
            ("derive (a+b)/x, a/x + b/x", &["a / x", "b / x"][..]),
            (
                "derive (a*x+b)/(c*x), a/c + b/(c*x)",
                &["a / c", "b / (c * x)"][..],
            ),
            (
                "derive (a*x*y+b*x*z+c*y*z+d*x*y*z)/(x*y*z), a/z + b/y + c/x + d",
                &["a / z", "b / y", "c / x", " + d"][..],
            ),
        ];

        for (input, fragments) in cases {
            let lines = derive_lines(input);
            assert_derive_strategy(&lines, "expand fraction");
            assert_result_contains_all(&lines, fragments);
        }
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_fraction_part_combination_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive c + a/d + b/d, c + (a+b)/d",
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
                && line.contains("/ d")
                && line.contains("a + b")
                && line.contains("c")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_expands_three_same_denominator_fractions_with_passthrough() {
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
                && line.contains("1")
                && line.contains("a / d")
                && line.contains("b / d")
                && line.contains("c / d")
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
            .any(|line| line.starts_with("Strategy:") && line.contains("combine like terms")));
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
            .any(|line| line.starts_with("Strategy:") && line.contains("cancel fraction")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Cancel common factor")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:") && (line.contains("1 / 2") || line.contains("1/2"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_monomial_common_factor_fraction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (a*x^2)/(b*x), (a*x)/b",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("cancel fraction")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Cancel common factor")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("a")
                && line.contains("x")
                && line.contains("/ b")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_tabulated_additive_nested_fraction_targets() {
        let cases: &[(&str, &[&str])] = &[
            (
                "derive 1/(1/a + 1/b), (a*b)/(a+b)",
                &["(a + b)", "a", "b", "/"],
            ),
            (
                "derive 1/(1/a + 1/b + 1/c), (a*b*c)/(a*b + a*c + b*c)",
                &["a * b * c /", "a * b + a * c + b * c"],
            ),
            (
                "derive 1/(1/(x+y) + 1/z), z*(x+y)/(x+y+z)",
                &["z * (x + y) /", "x + y + z"],
            ),
            (
                "derive (1/a + 1/b)/(1/t), t*(a+b)/(a*b)",
                &["t * (a + b) /", "a * b"],
            ),
            (
                "derive 1/(1/x + 1/(y+z)), x*(y+z)/(x+y+z)",
                &["x * (y + z) /", "x + y + z"],
            ),
            (
                "derive 1/(1/(x+y) + 1/(z+t)), ((x+y)*(z+t))/(x+y+z+t)",
                &["(t + z) * (x + y) /", "t + x + y + z"],
            ),
            (
                "derive (1/(a+b) + 1/(c+d))/(1/t), t*(a+b+c+d)/((a+b)*(c+d))",
                &["t * (a + b + c + d) /", "(a + b) * (c + d)"],
            ),
        ];

        for (command, expected_result_fragments) in cases {
            let mut simplifier = crate::Simplifier::with_default_rules();
            let lines = evaluate_derive_command_lines_with_resolver(
                &mut simplifier,
                command,
                crate::FullSimplifyDisplayMode::Normal,
                crate::SimplifyOptions::default(),
                |_ctx, expr| Ok(expr),
            )
            .expect("derive should evaluate");

            assert!(lines
                .iter()
                .any(|line| { line.starts_with("Strategy:") && line.contains("nested fraction") }));
            assert!(lines.iter().any(|line| {
                line.contains("Add Fractions")
                    || line.contains("Simplify Complex Fraction")
                    || line.contains("Simplify Nested Fraction")
            }));

            let result_line = lines
                .iter()
                .find(|line| line.starts_with("Result:"))
                .expect("expected result line");
            for fragment in *expected_result_fragments {
                assert!(
                    result_line.contains(fragment),
                    "expected result `{result_line}` to contain fragment `{fragment}` for `{command}`"
                );
            }
        }
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_tabulated_reverse_nested_fraction_targets() {
        let cases: &[(&str, &[&str])] = &[
            ("derive a*d/(b*d+c), a/(b + c/d)", &["a /", "c / d + b"]),
            ("derive (a*c+b)/(c*d), (a + b/c)/d", &["b / c + a", "/ d"]),
        ];

        for (command, expected_result_fragments) in cases {
            let mut simplifier = crate::Simplifier::with_default_rules();
            let lines = evaluate_derive_command_lines_with_resolver(
                &mut simplifier,
                command,
                crate::FullSimplifyDisplayMode::Normal,
                crate::SimplifyOptions::default(),
                |_ctx, expr| Ok(expr),
            )
            .expect("derive should evaluate");

            assert!(lines
                .iter()
                .any(|line| line.starts_with("Strategy:") && line.contains("simplify")));

            let result_line = lines
                .iter()
                .find(|line| line.starts_with("Result:"))
                .expect("expected result line");
            for fragment in *expected_result_fragments {
                assert!(
                    result_line.contains(fragment),
                    "expected result `{result_line}` to contain fragment `{fragment}` for `{command}`"
                );
            }
        }
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_tabulated_structural_nested_fraction_targets() {
        let cases: &[(&str, &[&str])] = &[
            ("derive 1/(x + y/z), z/(x*z+y)", &["z /", "x * z + y"]),
            ("derive a/(b + c/d), a*d/(b*d+c)", &["a * d /", "b * d + c"]),
            (
                "derive (a + b/c)/d, (a*c+b)/(c*d)",
                &["(a * c + b) /", "c * d"],
            ),
            (
                "derive 1/(a + b/(c+d)), (c+d)/(a*(c+d)+b)",
                &["(c + d) /", "a *"],
            ),
            ("derive (a+b)/(c + d/e), (a*e+b*e)/(c*e+d)", &["/", "d + e"]),
            (
                "derive (a + b/(c+d))/e, (a*(c+d)+b)/(e*(c+d))",
                &["/", "e *"],
            ),
        ];

        for (command, expected_result_fragments) in cases {
            let mut simplifier = crate::Simplifier::with_default_rules();
            let lines = evaluate_derive_command_lines_with_resolver(
                &mut simplifier,
                command,
                crate::FullSimplifyDisplayMode::Normal,
                crate::SimplifyOptions::default(),
                |_ctx, expr| Ok(expr),
            )
            .expect("derive should evaluate");

            assert!(lines
                .iter()
                .any(|line| line.starts_with("Strategy:") && line.contains("nested fraction")));

            let result_line = lines
                .iter()
                .find(|line| line.starts_with("Result:"))
                .expect("expected result line");
            for fragment in *expected_result_fragments {
                assert!(
                    result_line.contains(fragment),
                    "expected result `{result_line}` to contain fragment `{fragment}` for `{command}`"
                );
            }
        }
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_tabulated_reverse_structural_nested_fraction_targets()
    {
        let cases: &[(&str, &[&str])] = &[
            ("derive z/(x*z+y), 1/(x + y/z)", &["1 /", "y / z + x"]),
            ("derive (a*c+b)/(c*d), (a + b/c)/d", &["(b / c + a) / d"]),
            (
                "derive (c+d)/(a*(c+d)+b), 1/(a + b/(c+d))",
                &["1 /", "b / (c + d) + a"],
            ),
        ];

        for (command, expected_result_fragments) in cases {
            let mut simplifier = crate::Simplifier::with_default_rules();
            let lines = evaluate_derive_command_lines_with_resolver(
                &mut simplifier,
                command,
                crate::FullSimplifyDisplayMode::Normal,
                crate::SimplifyOptions::default(),
                |_ctx, expr| Ok(expr),
            )
            .expect("derive should evaluate");

            assert!(lines
                .iter()
                .any(|line| line.starts_with("Strategy:") && line.contains("simplify")));

            let result_line = lines
                .iter()
                .find(|line| line.starts_with("Result:"))
                .expect("expected result line");
            for fragment in *expected_result_fragments {
                assert!(
                    result_line.contains(fragment),
                    "expected result `{result_line}` to contain fragment `{fragment}` for `{command}`"
                );
            }
        }
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_tabulated_numeric_polynomial_product_targets() {
        let cases: &[(&str, &[&str])] =
            &[("derive (x^4+1)*(x^8-x^4+1), x^12+1", &["x^(12)", "+ 1"])];

        for (command, expected_result_fragments) in cases {
            let mut simplifier = crate::Simplifier::with_default_rules();
            let lines = evaluate_derive_command_lines_with_resolver(
                &mut simplifier,
                command,
                crate::FullSimplifyDisplayMode::Normal,
                crate::SimplifyOptions::default(),
                |_ctx, expr| Ok(expr),
            )
            .expect("derive should evaluate");

            assert!(lines
                .iter()
                .any(|line| line.starts_with("Strategy:") && line.contains("expand")));

            let result_line = lines
                .iter()
                .find(|line| line.starts_with("Result:"))
                .expect("expected result line");
            for fragment in *expected_result_fragments {
                assert!(
                    result_line.contains(fragment),
                    "expected result `{result_line}` to contain fragment `{fragment}` for `{command}`"
                );
            }
        }
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_tabulated_symbolic_polynomial_product_targets() {
        let cases: &[(&str, &[&str])] = &[
            (
                "derive (x^2+a^2)*(x^4-a^2*x^2+a^4), x^6+a^6",
                &["a^(6)", "x^(6)"],
            ),
            (
                "derive (x^2-a^2)*(x^4+a^2*x^2+a^4), x^6-a^6",
                &["x^(6) - a^(6)"],
            ),
        ];

        for (command, expected_result_fragments) in cases {
            let mut simplifier = crate::Simplifier::with_default_rules();
            let lines = evaluate_derive_command_lines_with_resolver(
                &mut simplifier,
                command,
                crate::FullSimplifyDisplayMode::Normal,
                crate::SimplifyOptions::default(),
                |_ctx, expr| Ok(expr),
            )
            .expect("derive should evaluate");

            assert!(lines
                .iter()
                .any(|line| line.starts_with("Strategy:") && line.contains("expand")));

            let result_line = lines
                .iter()
                .find(|line| line.starts_with("Result:"))
                .expect("expected result line");
            for fragment in *expected_result_fragments {
                assert!(
                    result_line.contains(fragment),
                    "expected result `{result_line}` to contain fragment `{fragment}` for `{command}`"
                );
            }
        }
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_symbolic_trinomial_square_expansion_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (a+b+c)^2, a^2 + b^2 + c^2 + 2*a*b + 2*a*c + 2*b*c",
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
                && line.contains("a^(2)")
                && line.contains("b^(2)")
                && line.contains("c^(2)")
                && line.contains("2 * a * b")
                && line.contains("2 * a * c")
                && line.contains("2 * b * c")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_xyz_trinomial_square_expansion_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (x+y+z)^2, x^2 + y^2 + z^2 + 2*x*y + 2*x*z + 2*y*z",
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
                && line.contains("x^(2)")
                && line.contains("y^(2)")
                && line.contains("z^(2)")
                && line.contains("2 * x * y")
                && line.contains("2 * x * z")
                && line.contains("2 * y * z")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_symbolic_trinomial_cube_expansion_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (a+b+c)^3, a^3 + b^3 + c^3 + 3*a^2*b + 3*a^2*c + 3*a*b^2 + 6*a*b*c + 3*a*c^2 + 3*b^2*c + 3*b*c^2",
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
                && line.contains("a^(3)")
                && line.contains("b^(3)")
                && line.contains("c^(3)")
                && line.contains("6 * a * b * c")
        }));
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
    fn evaluate_derive_command_lines_reaches_general_fraction_decomposed_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (3*x+1)/(x-1), 3 + 4/(x-1)",
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
                && (line.contains("4 / (x - 1) + 3") || line.contains("3 + 4 / (x - 1)"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_scaled_linear_fraction_decomposed_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (4*x+7)/(2*x+1), 2 + 5/(2*x+1)",
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
                && (line.contains("5 / (2 * x + 1) + 2") || line.contains("2 + 5 / (2 * x + 1)"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_symbolic_fraction_decomposed_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (a*x+b)/(x-1), a + (a+b)/(x-1)",
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
                && line.contains("(x - 1)")
                && line.contains("a + b")
                && line.contains("+ a")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_symbolic_fraction_decomposed_plus_one_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (a*x+b)/(x+1), a + (b-a)/(x+1)",
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
                && line.contains("(x + 1)")
                && line.contains("b - a")
                && line.contains("+ a")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_symbolic_fraction_decomposed_plus_one_target_in_y() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (a*y+b)/(y+1), a + (b-a)/(y+1)",
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
                && line.contains("(y + 1)")
                && line.contains("b - a")
                && line.contains("+ a")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_symbolic_fraction_decomposed_general_shift_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (a*x+b)/(x+c), a + (b-a*c)/(x+c)",
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
                && line.contains("c + x")
                && line.contains("b - a")
                && line.contains("+ a")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_symbolic_fraction_decomposed_general_shift_target_in_y(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (a*y+b)/(y+c), a + (b-a*c)/(y+c)",
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
                && line.contains("c + y")
                && line.contains("b - a")
                && line.contains("+ a")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_alt_scaled_linear_fraction_decomposed_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (6*x+5)/(3*x+1), 2 + 3/(3*x+1)",
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
                && (line.contains("3 / (3 * x + 1) + 2") || line.contains("2 + 3 / (3 * x + 1)"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_scaled_symbolic_fraction_decomposed_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (a*x+b)/(c*x+d), a/c + (b-a*d/c)/(c*x+d)",
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
            .any(|line| line.contains("Split a fraction into a whole part plus remainder")));
        assert!(lines.iter().any(|line| line.contains("c ≠ 0")));
        assert!(lines.iter().any(|line| line.contains("c * x + d ≠ 0")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("a / c")
                && line.contains("(c * x + d)")
                && line.contains("b - a * d / c")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_scaled_symbolic_fraction_decomposed_target_in_y() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (a*y+b)/(c*y+d), a/c + (b-a*d/c)/(c*y+d)",
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
            .any(|line| line.contains("Split a fraction into a whole part plus remainder")));
        assert!(lines.iter().any(|line| line.contains("c ≠ 0")));
        assert!(lines.iter().any(|line| line.contains("c * y + d ≠ 0")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("a / c")
                && line.contains("(c * y + d)")
                && line.contains("b - a * d / c")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_half_plus_remainder_scaled_fraction_decomposed_target()
    {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (x+a)/(d+2*x), 1/2 + (a-d/2)/(d+2*x)",
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
            .any(|line| line.contains("Split a fraction into a whole part plus remainder")));
        assert!(lines.iter().any(|line| line.contains("d + 2 * x ≠ 0")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("1 / 2")
                && line.contains("(d + 2 * x)")
                && line.contains("a - d / 2")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_negative_scaled_symbolic_fraction_decomposed_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (a*x+b)/(d-c*x), -a/c + (b+a*d/c)/(d-c*x)",
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
            .any(|line| line.contains("Split a fraction into a whole part plus remainder")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("-a / c")
                && line.contains("(d - c * x)")
                && (line.contains("b + a * d / c") || line.contains("a * d / c + b"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_tabulated_monic_fraction_decomposed_targets() {
        let cases = ["derive (x+a)/(x+b), 1 + (a-b)/(x+b)"];

        for command in cases {
            let mut simplifier = crate::Simplifier::with_default_rules();
            let lines = evaluate_derive_command_lines_with_resolver(
                &mut simplifier,
                command,
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
                .any(|line| line.contains("Split a fraction into a whole part plus remainder")));
            let result_line = lines
                .iter()
                .find(|line| line.starts_with("Result:"))
                .expect("expected derive result line");
            match command {
                "derive (x+a)/(x+b), 1 + (a-b)/(x+b)" => assert!(
                    (result_line.contains("(a - b) / (x + b)")
                        || result_line.contains("(a - b) / (b + x)"))
                        && (result_line.contains("+ 1") || result_line.contains("1 +")),
                    "unexpected result line for `{command}`: {result_line}"
                ),
                "derive (x+a)/(c*x+d), 1/c + (a-d/c)/(c*x+d)" => {
                    assert!(lines.iter().any(|line| line.contains("c ≠ 0")));
                    assert!(lines.iter().any(|line| line.contains("c * x + d ≠ 0")));
                    assert!(
                        result_line.contains("1 / c")
                            && result_line.contains("(c * x + d)")
                            && result_line.contains("a - d / c"),
                        "unexpected result line for `{command}`: {result_line}"
                    );
                }
                "derive (y+a)/(c*y+d), 1/c + (a-d/c)/(c*y+d)" => {
                    assert!(lines.iter().any(|line| line.contains("c ≠ 0")));
                    assert!(lines.iter().any(|line| line.contains("c * y + d ≠ 0")));
                    assert!(
                        result_line.contains("1 / c")
                            && result_line.contains("(c * y + d)")
                            && result_line.contains("a - d / c"),
                        "unexpected result line for `{command}`: {result_line}"
                    );
                }
                "derive (x+a)/(d-c*x), -1/c + (a+d/c)/(d-c*x)" => assert!(
                    result_line.contains("-1 / c")
                        && result_line.contains("(d - c * x)")
                        && (result_line.contains("a + d / c") || result_line.contains("d / c + a")),
                    "unexpected result line for `{command}`: {result_line}"
                ),
                _ => unreachable!("unhandled monic fraction decomposition case"),
            }
        }
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_fraction_expansion_with_common_scalar_denominator() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (a*x+b)/(c*x), a/c + b/(c*x)",
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
            .any(|line| line.contains("Distribute a sum over the common denominator")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:") && line.contains("a / c") && line.contains("b / (c * x)")
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
        assert!(lines.iter().any(|line| line.trim() == "Subpasos:"));
        assert!(lines
            .iter()
            .any(|line| line.contains("Introducir el numerador telescópico")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Separar sobre el denominador común")));
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
        assert!(lines.iter().any(|line| line.trim() == "Subpasos:"));
        assert!(lines
            .iter()
            .any(|line| line.contains("Llevar las fracciones al denominador común")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Simplificar el numerador telescópico")));
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
    fn evaluate_derive_command_lines_reaches_unfactored_difference_squares_telescoping_fraction_target(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1/(x^2-1), 1/2*(1/(x-1) - 1/(x+1))",
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
                && normalized.contains("x-1")
                && normalized.contains("x+1")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_unfactored_shifted_quadratic_telescoping_fraction_combined_target(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1/(x+1) - 1/(x+2), 1/(x^2+3*x+2)",
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
        assert!(lines.iter().any(|line| line.starts_with("Result:")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_symbolic_unfactored_difference_squares_telescoping_fraction_target(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1/(x^2-a^2), 1/(2*a)*(1/(x-a) - 1/(x+a))",
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
        assert!(lines.iter().any(|line| line.starts_with("Result:")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_symbolic_unfactored_difference_squares_telescoping_fraction_combined_target(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1/(2*a)*(1/(x-a) - 1/(x+a)), 1/(x^2-a^2)",
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
        assert!(lines.iter().any(|line| line.starts_with("Result:")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_tabulated_morrie_telescoping_targets() {
        let cases: &[(&str, &[&str])] = &[
            (
                "derive cos(x)*cos(2*x)*cos(4*x), sin(8*x)/(8*sin(x))",
                &["sin(8", "8", "sin(x)"],
            ),
            (
                "derive cos(3*x)*cos(6*x), sin(12*x)/(4*sin(3*x))",
                &["sin(12", "4", "sin(3"],
            ),
            (
                "derive cos(u)*cos(2*u)*cos(4*u)*cos(8*u), sin(16*u)/(16*sin(u))",
                &["sin(16", "16", "sin(u)"],
            ),
            (
                "derive cos(a*x)*cos(2*a*x), sin(4*a*x)/(4*sin(a*x))",
                &["sin(4", "sin(a * x)"],
            ),
            (
                "derive cos(a*x)*cos(2*a*x)*cos(4*a*x), sin(8*a*x)/(8*sin(a*x))",
                &["sin(8", "sin(a * x)"],
            ),
            (
                "derive sin(8*x)/(8*sin(x)), cos(x)*cos(2*x)*cos(4*x)",
                &["cos(x)", "cos(4 * x)"],
            ),
            (
                "derive sin(12*x)/(4*sin(3*x)), cos(3*x)*cos(6*x)",
                &["cos(3 * x)", "cos(6 * x)"],
            ),
            (
                "derive sin(16*u)/(16*sin(u)), cos(u)*cos(2*u)*cos(4*u)*cos(8*u)",
                &["cos(u)", "cos(8 * u)"],
            ),
            (
                "derive sin(8*a*x)/(8*sin(a*x)), cos(a*x)*cos(2*a*x)*cos(4*a*x)",
                &["cos(a * x)", "cos(4 * a * x)"],
            ),
        ];

        for (command, expected_result_fragments) in cases {
            let mut simplifier = crate::Simplifier::with_default_rules();
            let lines = evaluate_derive_command_lines_with_resolver(
                &mut simplifier,
                command,
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

            let result_line = lines
                .iter()
                .find(|line| line.starts_with("Result:"))
                .expect("expected result line");
            for fragment in *expected_result_fragments {
                assert!(
                    result_line.contains(fragment),
                    "expected result `{result_line}` to contain fragment `{fragment}` for `{command}`"
                );
            }
        }
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_tabulated_dirichlet_kernel_targets() {
        let cases: &[(&str, &[&str])] = &[
            (
                "derive 1 + 2*cos(x) + 2*cos(2*x), sin(5*x/2)/sin(x/2)",
                &["sin(5", "sin(x / 2)"],
            ),
            (
                "derive 1 + 2*cos(x) + 2*cos(2*x) + 2*cos(3*x), sin(7*x/2)/sin(x/2)",
                &["sin(7", "sin(x / 2)"],
            ),
            (
                "derive 1 + 2*cos(3*x) + 2*cos(6*x), sin(15*x/2)/sin(3*x/2)",
                &["sin(15", "sin(3"],
            ),
            (
                "derive 1 + 2*cos(2*x) + 2*cos(4*x) + 2*cos(6*x), sin(7*x)/sin(x)",
                &["sin(7", "sin(x)"],
            ),
            (
                "derive 1 + 2*cos(u) + 2*cos(2*u) + 2*cos(3*u) + 2*cos(4*u), sin(9*u/2)/sin(u/2)",
                &["sin(9", "sin(u / 2)"],
            ),
            (
                "derive 1 + 2*cos(a*x) + 2*cos(2*a*x), sin(5*a*x/2)/sin(a*x/2)",
                &["sin(5", "sin(a * x / 2)"],
            ),
            (
                "derive 1 + 2*cos(a*x) + 2*cos(2*a*x) + 2*cos(3*a*x), sin(7*a*x/2)/sin(a*x/2)",
                &["sin(7", "sin(a * x / 2)"],
            ),
            (
                "derive sin(5*x/2)/sin(x/2), 1 + 2*cos(x) + 2*cos(2*x)",
                &["cos(x)", "cos(2 * x)"],
            ),
            (
                "derive sin(7*x/2)/sin(x/2), 1 + 2*cos(x) + 2*cos(2*x) + 2*cos(3*x)",
                &["cos(2 * x)", "cos(3 * x)"],
            ),
            (
                "derive sin(5*a*x/2)/sin(a*x/2), 1 + 2*cos(a*x) + 2*cos(2*a*x)",
                &["cos(a * x)", "cos(2 * a * x)"],
            ),
            (
                "derive sin(7*a*x/2)/sin(a*x/2), 1 + 2*cos(a*x) + 2*cos(2*a*x) + 2*cos(3*a*x)",
                &["cos(2 * a * x)", "cos(3 * a * x)"],
            ),
        ];

        for (command, expected_result_fragments) in cases {
            let mut simplifier = crate::Simplifier::with_default_rules();
            let lines = evaluate_derive_command_lines_with_resolver(
                &mut simplifier,
                command,
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

            let result_line = lines
                .iter()
                .find(|line| line.starts_with("Result:"))
                .expect("expected result line");
            for fragment in *expected_result_fragments {
                assert!(
                    result_line.contains(fragment),
                    "expected result `{result_line}` to contain fragment `{fragment}` for `{command}`"
                );
            }
        }
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
            .any(|line| line.starts_with("Strategy:") && line.contains("finite sums/products")));
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
            .any(|line| line.starts_with("Strategy:") && line.contains("finite sums/products")));
        assert!(lines.iter().any(|line| line.contains("Finite Product")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:") && line.contains("n + 2") && line.contains("/ 2")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_symbolic_shift_finite_telescoping_product_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive product((k+a+1)/(k+a), k, 1, n), (n+a+1)/(a+1)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("finite sums/products")));
        assert!(lines.iter().any(|line| line.contains("Finite Product")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && (line.contains("a + n + 1") || line.contains("n + a + 1"))
                && line.contains("a + 1")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_symbolic_lower_finite_telescoping_product_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive product((k+a+1)/(k+a), k, m, n), (n+a+1)/(m+a)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("finite sums/products")));
        assert!(lines.iter().any(|line| line.contains("Finite Product")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && (line.contains("a + n + 1") || line.contains("n + a + 1"))
                && (line.contains("a + m") || line.contains("m + a"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_affine_symbolic_shift_finite_telescoping_product_target(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive product((a*k+b+a)/(a*k+b), k, 1, n), (a*n+a+b)/(a+b)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("finite sums/products")));
        assert!(lines.iter().any(|line| line.contains("Finite Product")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("a")
                && line.contains("n")
                && line.contains("b")
                && line.contains("/")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_affine_symbolic_shift_symbolic_lower_finite_telescoping_product_target(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive product((a*k+b+a)/(a*k+b), k, m, n), (a*n+a+b)/(a*m+b)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("finite sums/products")));
        assert!(lines.iter().any(|line| line.contains("Finite Product")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && (line.contains("a·n + a + b") || line.contains("a * n + a + b"))
                && (line.contains("a·m + b") || line.contains("a * m + b"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_affine_symbolic_shifted_finite_telescoping_product_target(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive product((a*k+b+2*a)/(a*k+b+a), k, 1, n), (a*n+b+2*a)/(2*a+b)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("finite sums/products")));
        assert!(lines.iter().any(|line| line.contains("Finite Product")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("a")
                && line.contains("n")
                && line.contains("b")
                && line.contains("2")
                && line.contains("/")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_affine_symbolic_shifted_symbolic_lower_finite_telescoping_product_target(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive product((a*k+b+2*a)/(a*k+b+a), k, m, n), (a*n+b+2*a)/(a*m+a+b)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("finite sums/products")));
        assert!(lines.iter().any(|line| line.contains("Finite Product")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && (line.contains("a·n + b + 2·a")
                    || line.contains("a·n + 2·a + b")
                    || line.contains("a * n + b + 2 * a")
                    || line.contains("a * n + 2 * a + b"))
                && (line.contains("a·m + a + b")
                    || line.contains("a·m + b + a")
                    || line.contains("a * m + a + b")
                    || line.contains("a * m + b + a"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_affine_symbolic_arbitrary_shift_finite_telescoping_product_target(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive product((a*k+b+c+a)/(a*k+b+c), k, 1, n), (a*n+a+b+c)/(a+b+c)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("finite sums/products")));
        assert!(lines.iter().any(|line| line.contains("Finite Product")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("a")
                && line.contains("n")
                && line.contains("b")
                && line.contains("c")
                && line.contains("/")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_affine_symbolic_arbitrary_shift_symbolic_lower_finite_telescoping_product_target(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive product((a*k+b+c+a)/(a*k+b+c), k, m, n), (a*n+a+b+c)/(a*m+b+c)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("finite sums/products")));
        assert!(lines.iter().any(|line| line.contains("Finite Product")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && (line.contains("a·n + a + b + c") || line.contains("a * n + a + b + c"))
                && (line.contains("a·m + b + c") || line.contains("a * m + b + c"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_factorized_finite_telescoping_product_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive product(1 - 1/k^2, k, 2, n), (n+1)/(2*n)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("finite sums/products")));
        assert!(lines.iter().any(|line| line.contains("Finite Product")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("n + 1")
                && line.contains("2")
                && line.contains("n")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_shifted_start_factorized_finite_telescoping_product_target(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive product(1 - 1/k^2, k, 3, n), 2*(n+1)/(3*n)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("finite sums/products")));
        assert!(lines.iter().any(|line| line.contains("Finite Product")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("2")
                && line.contains("n + 1")
                && line.contains("3")
                && line.contains("n")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_shifted_base_factorized_finite_telescoping_product_target(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive product(1 - 1/(k+2)^2, k, 1, n), 2*(n+3)/(3*(n+2))",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("finite sums/products")));
        assert!(lines.iter().any(|line| line.contains("Finite Product")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("2")
                && line.contains("n + 3")
                && line.contains("3")
                && line.contains("n + 2")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_symbolic_shift_base_factorized_finite_telescoping_product_target(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive product(1 - 1/(k+a)^2, k, 1, n), a*(n+a+1)/((a+1)*(n+a))",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("finite sums/products")));
        assert!(lines.iter().any(|line| line.contains("Finite Product")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("a")
                && (line.contains("a + n + 1") || line.contains("n + a + 1"))
                && line.contains("a + 1")
                && (line.contains("a + n") || line.contains("n + a"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_symbolic_lower_shifted_base_factorized_finite_telescoping_product_target(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive product(1 - 1/(k+2)^2, k, m, n), ((m+1)*(n+3))/((m+2)*(n+2))",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("finite sums/products")));
        assert!(lines.iter().any(|line| line.contains("Finite Product")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("m + 1")
                && line.contains("n + 3")
                && line.contains("m + 2")
                && line.contains("n + 2")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_symbolic_lower_symbolic_shift_base_factorized_finite_telescoping_product_target(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive product(1 - 1/(k+a)^2, k, m, n), ((m+a-1)*(n+a+1))/((m+a)*(n+a))",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("finite sums/products")));
        assert!(lines.iter().any(|line| line.contains("Finite Product")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && (line.contains("a + m - 1") || line.contains("m + a - 1"))
                && (line.contains("a + n + 1") || line.contains("n + a + 1"))
                && (line.contains("a + m") || line.contains("m + a"))
                && (line.contains("a + n") || line.contains("n + a"))
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
            .any(|line| line.starts_with("Strategy:") && line.contains("finite sums/products")));
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
            .any(|line| line.starts_with("Strategy:") && line.contains("finite sums/products")));
        assert!(lines.iter().any(|line| line.contains("Finite Summation")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:") && line.contains("1 / 3") && line.contains("n + 3")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_symbolic_shift_finite_telescoping_sum_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sum(1/((k+a)*(k+a+1)), k, 1, n), 1/(a+1) - 1/(n+a+1)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("finite sums/products")));
        assert!(lines.iter().any(|line| line.contains("Finite Summation")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("a + 1")
                && (line.contains("a + n + 1") || line.contains("n + a + 1"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_symbolic_lower_finite_telescoping_sum_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sum(1/((k+a)*(k+a+1)), k, m, n), 1/(m+a) - 1/(n+a+1)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("finite sums/products")));
        assert!(lines.iter().any(|line| line.contains("Finite Summation")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && (line.contains("a + m") || line.contains("m + a"))
                && (line.contains("a + n + 1") || line.contains("n + a + 1"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_affine_symbolic_shift_finite_telescoping_sum_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sum(1/((a*k+b)*(a*k+b+a)), k, 1, n), 1/a*(1/(a+b) - 1/(a*n+b+a))",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("finite sums/products")));
        assert!(lines.iter().any(|line| line.contains("Finite Summation")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && (line.contains("1 / a") || line.contains(" / a"))
                && line.contains("a + b")
                && (line.contains("a·n + a + b")
                    || line.contains("a·n + b + a")
                    || line.contains("a + a·n + b")
                    || line.contains("a * (n + 1) + b")
                    || line.contains("a * n + a + b")
                    || line.contains("a * n + b + a"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_affine_symbolic_shift_symbolic_lower_finite_telescoping_sum_target(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sum(1/((a*k+b)*(a*k+b+a)), k, m, n), 1/a*(1/(a*m+b) - 1/(a*n+a+b))",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("finite sums/products")));
        assert!(lines.iter().any(|line| line.contains("Finite Summation")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && (line.contains("1 / a") || line.contains(" / a"))
                && (line.contains("a·m + b") || line.contains("a * m + b"))
                && (line.contains("a·n + a + b")
                    || line.contains("a·n + b + a")
                    || line.contains("a * n + a + b")
                    || line.contains("a * n + b + a"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_affine_symbolic_shifted_finite_telescoping_sum_target()
    {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sum(1/((a*k+b+a)*(a*k+b+2*a)), k, 1, n), 1/a*(1/(2*a+b) - 1/(a*n+2*a+b))",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("finite sums/products")));
        assert!(lines.iter().any(|line| line.contains("Finite Summation")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && (line.contains("1 / a") || line.contains(" / a"))
                && (line.contains("b + 2·a")
                    || line.contains("b + 2 * a")
                    || line.contains("2·a + b")
                    || line.contains("2 * a + b"))
                && (line.contains("a·n + b + 2·a")
                    || line.contains("a·n + 2·a + b")
                    || line.contains("a * n + b + 2 * a")
                    || line.contains("a * n + 2 * a + b"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_affine_symbolic_shifted_symbolic_lower_finite_telescoping_sum_target(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sum(1/((a*k+b+a)*(a*k+b+2*a)), k, m, n), 1/a*(1/(a*m+a+b) - 1/(a*n+2*a+b))",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("finite sums/products")));
        assert!(lines.iter().any(|line| line.contains("Finite Summation")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && (line.contains("1 / a") || line.contains(" / a"))
                && (line.contains("a·m + a + b")
                    || line.contains("a·m + b + a")
                    || line.contains("a * m + a + b")
                    || line.contains("a * m + b + a"))
                && (line.contains("a·n + b + 2·a")
                    || line.contains("a·n + 2·a + b")
                    || line.contains("a * n + b + 2 * a")
                    || line.contains("a * n + 2 * a + b"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_affine_symbolic_arbitrary_shift_finite_telescoping_sum_target(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sum(1/((a*k+b+c)*(a*k+b+c+a)), k, 1, n), 1/a*(1/(a+b+c) - 1/(a*n+a+b+c))",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("finite sums/products")));
        assert!(lines.iter().any(|line| line.contains("Finite Summation")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && (line.contains("1 / a") || line.contains(" / a"))
                && (line.contains("a + b + c")
                    || line.contains("a + c + b")
                    || line.contains("b + c + a")
                    || line.contains("c + b + a"))
                && (line.contains("a·n + a + b + c")
                    || line.contains("a·n + a + c + b")
                    || line.contains("a * n + a + b + c")
                    || line.contains("a * n + a + c + b"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_affine_symbolic_arbitrary_shift_symbolic_lower_finite_telescoping_sum_target(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sum(1/((a*k+b+c)*(a*k+b+c+a)), k, m, n), 1/a*(1/(a*m+b+c) - 1/(a*n+a+b+c))",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("finite sums/products")));
        assert!(lines.iter().any(|line| line.contains("Finite Summation")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && (line.contains("1 / a") || line.contains(" / a"))
                && (line.contains("a·m + b + c")
                    || line.contains("a·m + c + b")
                    || line.contains("a * m + b + c")
                    || line.contains("a * m + c + b"))
                && (line.contains("a·n + a + b + c")
                    || line.contains("a·n + a + c + b")
                    || line.contains("a * n + a + b + c")
                    || line.contains("a * n + a + c + b"))
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
    fn evaluate_derive_command_lines_reaches_symbolic_whole_plus_remainder_fraction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive a + (b-a*c)/(x+c), (a*x+b)/(x+c)",
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
            .any(|line| line.contains("Combine the whole part with the remaining fraction")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && (line.contains("(a * x + b) / (x + c)")
                    || line.contains("(a * x + b) / (c + x)"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_scaled_symbolic_whole_plus_remainder_fraction_target()
    {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive a/c + (b-a*d/c)/(c*x+d), (a*x+b)/(c*x+d)",
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
            .any(|line| line.contains("Combine the whole part with the remaining fraction")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:") && line.contains("(a * x + b) / (c * x + d)")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_half_plus_remainder_scaled_fraction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1/2 + (a-d/2)/(d+2*x), (x+a)/(d+2*x)",
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
            .any(|line| line.contains("Combine the whole part with the remaining fraction")));
        assert!(lines.iter().any(|line| line.contains("d + 2 * x ≠ 0")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && (line.contains("(a + x) / (d + 2 * x)")
                    || line.contains("(x + a) / (d + 2 * x)"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_negative_scaled_symbolic_whole_plus_remainder_fraction_target(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive -a/c + (b+a*d/c)/(d-c*x), (a*x+b)/(d-c*x)",
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
            .any(|line| line.contains("Combine the whole part with the remaining fraction")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:") && line.contains("(a * x + b) / (d - c * x)")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_tabulated_monic_fraction_combined_targets() {
        let cases = [(
            "derive 1/c + (a-d/c)/(c*x+d), (x+a)/(c*x+d)",
            &["c * x + d ≠ 0", "(x + a) / (c * x + d)"][..],
        )];

        for (command, required_fragments) in cases {
            let mut simplifier = crate::Simplifier::with_default_rules();
            let lines = evaluate_derive_command_lines_with_resolver(
                &mut simplifier,
                command,
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
                .any(|line| line.contains("Combine the whole part with the remaining fraction")));
            let result_line = lines
                .iter()
                .find(|line| line.starts_with("Result:"))
                .expect("expected derive result line");
            for fragment in required_fragments {
                assert!(
                    lines.iter().any(|line| line.contains(fragment))
                        || result_line.contains(fragment)
                        || result_line.contains(&fragment.replace("(x + a)", "(a + x)"))
                        || result_line.contains(&fragment.replace("(y + a)", "(a + y)")),
                    "missing fragment `{fragment}` in derive output for `{command}`"
                );
            }
        }
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
            "derive a/x + b/y, (a*y+b*x)/(x*y)",
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
                && (line.contains("(a * y + b * x) / (x * y)")
                    || line.contains("(b * x + a * y) / (x * y)")
                    || line.contains("(a * y + b * x) / (y * x)"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_symbolic_same_denominator_fraction_sum_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive a/(x+y) + b/(x+y), (a+b)/(x+y)",
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
        assert!(lines.iter().any(|line| line.contains("x + y ≠ 0")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && (line.contains("(a + b) / (x + y)") || line.contains("(b + a) / (x + y)"))
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
    fn evaluate_derive_command_lines_reaches_symbolic_same_denominator_fraction_difference_target()
    {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive a/(x+1) - b/(x+1), (a-b)/(x+1)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("combine fraction")));
        assert!(lines.iter().any(|line| {
            line.contains("Combine fractions with the same denominator into one subtraction")
        }));
        assert!(lines.iter().any(|line| line.contains("x + 1 ≠ 0")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && (line.contains("(a - b) / (x + 1)") || line.contains("(-b + a) / (x + 1)"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_general_fraction_difference_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive a/x - b/y, (a*y-b*x)/(x*y)",
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
                && (line.contains("(a * y - b * x) / (x * y)")
                    || line.contains("(-b * x + a * y) / (x * y)")
                    || line.contains("(a * y - b * x) / (y * x)"))
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
    fn evaluate_derive_command_lines_combines_symbolic_same_denominator_fraction_subset_with_passthrough(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive a/(x+y) + b/(x+y) + c, (a+b)/(x+y) + c",
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
        assert!(lines.iter().any(|line| line.contains("x + y ≠ 0")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && (line.contains("(a + b) / (x + y) + c")
                    || line.contains("c + (a + b) / (x + y)"))
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
    fn evaluate_derive_command_lines_reaches_tabulated_common_factor_targets() {
        let cases = [("derive a*b + a*c, a*(b+c)", &["a", "b + c"][..])];

        for (input, fragments) in cases {
            let lines = derive_lines(input);
            assert_derive_strategy(&lines, "factor");
            assert_result_contains_all(&lines, fragments);
        }
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
            .any(|line| line.starts_with("Strategy:") && line.contains("cancel fraction")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("a - b")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_tabulated_perfect_square_fraction_targets() {
        let cases = [
            ("derive (a^2+2*a*b+b^2)/(a+b), a+b", &["a + b"][..]),
            ("derive (x^2-2*x+1)/(x-1), x-1", &["x - 1"][..]),
            ("derive (a^2-2*a*b+b^2)/(a-b), a-b", &["a - b"][..]),
        ];

        for (input, fragments) in cases {
            let lines = derive_lines(input);
            assert_derive_strategy(&lines, "cancel fraction");
            assert!(
                lines
                    .iter()
                    .any(|line| line.contains("Cancel common factor")),
                "expected exact cancellation step, got: {lines:?}"
            );
            assert_result_contains_all(&lines, fragments);
        }
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
            .any(|line| line.starts_with("Strategy:") && line.contains("cancel fraction")));
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
    fn evaluate_derive_command_lines_reaches_difference_of_squares_fraction_target_with_passthrough(
    ) {
        let lines = derive_lines("derive (a^2-b^2)/(a-b)+c, a+b+c");
        assert_derive_strategy(&lines, "cancel fraction");
        assert!(
            lines
                .iter()
                .any(|line| line.contains("Pre-order Difference of Squares Cancel")),
            "expected direct fraction-cancel step, got: {lines:?}"
        );
        assert_result_contains_all(&lines, &["a + b + c"]);
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_difference_of_cubes_fraction_target_with_passthrough()
    {
        let lines = derive_lines("derive (a^3-b^3)/(a-b)+c, a^2+a*b+b^2+c");
        assert_derive_strategy(&lines, "cancel fraction");
        assert!(
            lines
                .iter()
                .any(|line| line.contains("Cancel Sum/Difference of Cubes Fraction")),
            "expected direct fraction-cancel step, got: {lines:?}"
        );
        assert_result_contains_all(&lines, &["a^(2)", "a * b", "b^(2)", "c"]);
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
            .any(|line| line.starts_with("Strategy:") && line.contains("cancel fraction")));
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
    fn evaluate_derive_command_lines_reaches_tabulated_symbolic_sixth_power_factor_targets() {
        let cases = [("derive x^6-a^6, (x^2-a^2)*(x^4+a^2*x^2+a^4)", 'x')];

        for (command, variable) in cases {
            let mut simplifier = crate::Simplifier::with_default_rules();
            let lines = evaluate_derive_command_lines_with_resolver(
                &mut simplifier,
                command,
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
                    && line.contains(variable)
                    && line.contains('a')
                    && line.contains('(')
            }));
        }
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
            .any(|line| line.starts_with("Strategy:") && line.contains("rewrite trigs")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Pythagorean Chain Identity")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains('1')));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_tabulated_pythagorean_factor_form_targets() {
        let cases = [
            ("derive 1 - sin(x)^2, cos(x)^2", &["cos(x)^(2)"][..]),
            ("derive 1 - cos(x)^2, sin(x)^2", &["sin(x)^(2)"][..]),
            ("derive sin(x)^2, 1-cos(x)^2", &["1-cos(x)^(2)"][..]),
            ("derive cos(x)^2, 1-sin(x)^2", &["1-sin(x)^(2)"][..]),
        ];

        for (input, fragments) in cases {
            let lines = derive_lines(input);
            assert_derive_strategy(&lines, "rewrite trigs");
            assert!(
                lines
                    .iter()
                    .any(|line| line.contains("Pythagorean Factor Form")),
                "expected pythagorean factor-form rewrite, got: {lines:?}"
            );
            assert_result_contains_all(&lines, fragments);
        }
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
    fn evaluate_derive_command_lines_reaches_scaled_sec_squared_contraction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1 + tan(a*x)^2, sec(a*x)^2",
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
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_scaled_csc_squared_contraction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 1 + cot(a*x)^2, csc(a*x)^2",
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
            .any(|line| line.contains("Recognize Cosecant Squared")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_scaled_tangent_quotient_contraction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sin(2*a*x)/cos(2*a*x), tan(2*a*x)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("contract trig")));
        assert!(lines.iter().any(|line| line.contains("Trig Quotient")));
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
    fn evaluate_derive_command_lines_reaches_tabulated_inverse_tan_identity_targets() {
        let cases = [
            ("derive arctan(a) + arctan(1/a), pi/2", &["pi", "/ 2"][..]),
            ("derive atan(3) + (atan(1/3) - pi/2), 0", &["0"][..]),
        ];

        for (command, result_fragments) in cases {
            let mut simplifier = crate::Simplifier::with_default_rules();
            let lines = evaluate_derive_command_lines_with_resolver(
                &mut simplifier,
                command,
                crate::FullSimplifyDisplayMode::Normal,
                crate::SimplifyOptions::default(),
                |_ctx, expr| Ok(expr),
            )
            .expect("derive should evaluate");

            assert!(lines.iter().any(|line| {
                line.starts_with("Strategy:") && line.contains("rewrite inverse trigs")
            }));
            assert!(lines
                .iter()
                .any(|line| line.contains("Inverse Tan Relations")));
            let result_line = lines
                .iter()
                .find(|line| line.starts_with("Result:"))
                .expect("expected derive result line");
            for fragment in result_fragments {
                assert!(
                    result_line.contains(fragment),
                    "missing fragment `{fragment}` in `{result_line}` for `{command}`"
                );
            }
        }
    }

    #[test]
    fn evaluate_derive_command_lines_explains_safe_arcsin_arctan_composition() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive asin(x/sqrt(x^2 + 1)), arctan(x)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("rewrite inverse trigs")));
        assert!(lines.iter().any(|line| line.trim() == "Subpasos:"));
        assert!(lines.iter().any(|line| line.contains("sin(arctan(x))")));
        assert!(lines
            .iter()
            .any(|line| line.contains("arcsin(sin(arctan(x)))")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("arctan(x)")));
    }

    #[test]
    fn evaluate_derive_command_lines_explains_direct_log_change_of_base() {
        for (command, strategy, expected_lines) in [
            (
                "derive log(2, x), ln(x)/ln(2)",
                "expand_log",
                &[
                    "Poner el argumento en el numerador",
                    "x -> ln(x)",
                    "Poner la base en el denominador",
                    "2 -> ln(2)",
                    "Formar el cociente de cambio de base",
                ][..],
            ),
            (
                "derive ln(x)/ln(2), log(2, x)",
                "contract logs",
                &[
                    "Leer el argumento desde el numerador",
                    "ln(x) -> argumento x",
                    "Leer la base desde el denominador",
                    "ln(2) -> base 2",
                    "Reconstruir el logaritmo de base indicada",
                ][..],
            ),
        ] {
            let mut simplifier = crate::Simplifier::with_default_rules();
            let lines = evaluate_derive_command_lines_with_resolver(
                &mut simplifier,
                command,
                crate::FullSimplifyDisplayMode::Normal,
                crate::SimplifyOptions::default(),
                |_ctx, expr| Ok(expr),
            )
            .expect("derive should evaluate");

            assert!(lines
                .iter()
                .any(|line| { line.starts_with("Strategy:") && line.contains(strategy) }));
            assert!(lines.iter().any(|line| line.trim() == "Subpasos:"));
            for expected in expected_lines {
                assert!(
                    lines.iter().any(|line| line.contains(expected)),
                    "missing `{expected}` in CLI output for `{command}`:\n{}",
                    lines.join("\n")
                );
            }
        }
    }

    #[test]
    fn evaluate_derive_command_lines_prefers_direct_consecutive_factorial_ratio_rewrite() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (n+1)!/n!, n+1",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| { line.starts_with("Strategy:") && line.contains("rewrite factorials") }));
        assert!(lines
            .iter()
            .any(|line| line.contains("Consecutive Factorial Ratio")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("n + 1")));
    }

    #[test]
    fn evaluate_derive_command_lines_prefers_direct_consecutive_factorial_ratio_rewrite_with_passthrough(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (n+1)!/n!+a, n+1+a",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| { line.starts_with("Strategy:") && line.contains("rewrite factorials") }));
        assert!(lines
            .iter()
            .any(|line| line.contains("Consecutive Factorial Ratio")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && (line.contains("n + 1 + a") || line.contains("a + n + 1"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_prefers_direct_gap_two_factorial_ratio_rewrite() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (n+1)!/(n-1)!, n*(n+1)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| { line.starts_with("Strategy:") && line.contains("rewrite factorials") }));
        assert!(lines
            .iter()
            .any(|line| line.contains("Consecutive Factorial Ratio")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("n * (n + 1)")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_tabulated_like_term_simplification_targets() {
        let cases = [
            ("derive x + x, 2*x", &["2 * x"][..]),
            ("derive 2*x + 3*x + 0, 5*x", &["5 * x"][..]),
        ];

        for (command, result_fragments) in cases {
            let mut simplifier = crate::Simplifier::with_default_rules();
            let lines = evaluate_derive_command_lines_with_resolver(
                &mut simplifier,
                command,
                crate::FullSimplifyDisplayMode::Normal,
                crate::SimplifyOptions::default(),
                |_ctx, expr| Ok(expr),
            )
            .expect("derive should evaluate");

            assert!(lines
                .iter()
                .any(|line| line.starts_with("Strategy:") && line.contains("combine like terms")));
            let result_line = lines
                .iter()
                .find(|line| line.starts_with("Result:"))
                .expect("expected derive result line");
            for fragment in result_fragments {
                assert!(
                    result_line.contains(fragment),
                    "missing fragment `{fragment}` in `{result_line}` for `{command}`"
                );
            }
        }
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
    fn evaluate_derive_command_lines_reaches_odd_half_power_target_after_simplify_with_passthrough()
    {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sqrt(x^3)+a, abs(x)*sqrt(x)+a",
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
                && line.contains('a')
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_higher_odd_half_power_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive x^(5/2), abs(x)^2*sqrt(x)",
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
    fn evaluate_derive_command_lines_reaches_higher_odd_half_power_target_after_simplify() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sqrt(x^5), abs(x)^2*sqrt(x)",
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
    fn evaluate_derive_command_lines_reaches_higher_odd_half_power_nonnegative_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sqrt(x^5), x^2*sqrt(x)",
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
                && line.contains('x')
                && !line.contains("|x|")
                && !line.contains("abs")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_higher_odd_half_power_nonnegative_target_with_passthrough(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sqrt(x^7)+a, x^3*sqrt(x)+a",
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
                && line.contains('a')
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_higher_odd_half_power_target_in_y() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive y^(7/2), abs(y)^3*sqrt(y)",
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
                && (line.contains("|y|") || line.contains("abs"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_tabulated_direct_odd_half_power_targets() {
        let cases = [
            ("derive x^(5/2), abs(x)^2*sqrt(x)", "|x|"),
            ("derive x^(11/2), abs(x)^5*sqrt(x)", "|x|"),
            ("derive y^(7/2), abs(y)^3*sqrt(y)", "|y|"),
            ("derive y^(13/2), abs(y)^6*sqrt(y)", "|y|"),
            ("derive x^(15/2), abs(x)^7*sqrt(x)", "|x|"),
        ];

        for (input, abs_hint) in cases {
            let mut simplifier = crate::Simplifier::with_default_rules();
            let lines = evaluate_derive_command_lines_with_resolver(
                &mut simplifier,
                input,
                crate::FullSimplifyDisplayMode::Normal,
                crate::SimplifyOptions::default(),
                |_ctx, expr| Ok(expr),
            )
            .expect("derive should evaluate");

            assert!(
                lines.iter().any(|line| {
                    line.starts_with("Strategy:") && line.contains("expand odd half power")
                }),
                "expected odd-half-power strategy for {input}; got: {lines:?}"
            );
            assert!(lines.iter().any(|line| {
                line.starts_with("Result:")
                    && (line.contains("sqrt") || line.contains("√"))
                    && (line.contains(abs_hint) || line.contains("abs"))
            }));
        }
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_tabulated_simplify_then_odd_half_power_targets() {
        let cases = [
            ("derive sqrt(x^5), abs(x)^2*sqrt(x)", "|x|"),
            ("derive sqrt(y^9), abs(y)^4*sqrt(y)", "|y|"),
        ];

        for (input, abs_hint) in cases {
            let mut simplifier = crate::Simplifier::with_default_rules();
            let lines = evaluate_derive_command_lines_with_resolver(
                &mut simplifier,
                input,
                crate::FullSimplifyDisplayMode::Normal,
                crate::SimplifyOptions::default(),
                |_ctx, expr| Ok(expr),
            )
            .expect("derive should evaluate");

            assert!(
                lines.iter().any(|line| {
                    line.starts_with("Strategy:") && line.contains("expand odd half power")
                }),
                "expected odd-half-power strategy for {input}; got: {lines:?}"
            );
            assert!(lines.iter().any(|line| {
                line.starts_with("Result:")
                    && (line.contains("sqrt") || line.contains("√"))
                    && (line.contains(abs_hint) || line.contains("abs"))
            }));
        }
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_sparse_octic_factor_with_division_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive a*x^8 + b*x^4 + c*x + d, x*(a*x^7 + b*x^3 + c + d/x)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines.iter().any(|line| {
            line.starts_with("Strategy:") && line.contains("factor out with division")
        }));
        assert!(lines.iter().any(|line| line.contains("x ≠ 0")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("x")
                && line.contains("d / x")
                && line.contains("a")
                && (line.contains("x^7") || line.contains("x^(7)"))
                && line.contains("b")
                && (line.contains("x^3") || line.contains("x^(3)"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_sparse_octic_factor_with_division_target_in_y() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive a*y^8 + b*y^4 + c*y + d, y*(a*y^7 + b*y^3 + c + d/y)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines.iter().any(|line| {
            line.starts_with("Strategy:") && line.contains("factor out with division")
        }));
        assert!(lines.iter().any(|line| line.contains("y ≠ 0")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("y")
                && line.contains("d / y")
                && line.contains("a")
                && (line.contains("y^7") || line.contains("y^(7)"))
                && line.contains("b")
                && (line.contains("y^3") || line.contains("y^(3)"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_mixed_nonic_factor_with_division_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive a*x^9 + b*x^6 + c*x^2 + d, x*(a*x^8 + b*x^5 + c*x + d/x)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines.iter().any(|line| {
            line.starts_with("Strategy:") && line.contains("factor out with division")
        }));
        assert!(lines.iter().any(|line| line.contains("x ≠ 0")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("x")
                && line.contains("d / x")
                && line.contains("a")
                && (line.contains("x^8") || line.contains("x^(8)"))
                && line.contains("b")
                && (line.contains("x^5") || line.contains("x^(5)"))
                && line.contains("c")
                && (line.contains("· x") || line.contains("·x") || line.contains("* x"))
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_tabulated_direct_power_merge_targets() {
        let cases = [
            ("derive x^(1/2)*x^(2/3), x^(7/6)", &["7/6"][..]),
            ("derive x^(3/4)*x^(1/4), x", &["Result:", "x"][..]),
            ("derive x*x^(1/3), x^(4/3)", &["4/3"][..]),
            ("derive x^a*x^b, x^(a+b)", &["a+b"][..]),
            ("derive x*x^a, x^(a+1)", &["a+1"][..]),
            ("derive x^a*x^b*x^c*x^d, x^(a+b+c+d)", &["a+b+c+d"][..]),
        ];

        for (input, result_fragments) in cases {
            let lines = derive_lines(input);
            assert_derive_strategy(&lines, "combine powers");
            assert!(
                lines
                    .iter()
                    .any(|line| line.contains("Combine powers with same base (n-ary)")),
                "expected n-ary merge rule in {lines:?}"
            );
            assert_result_contains_all(&lines, result_fragments);
        }
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_tabulated_root_power_merge_targets() {
        let cases = [
            ("derive sqrt(x)*x^(1/3), x^(5/6)", &["5/6"][..]),
            ("derive sqrt(x)*x^a, x^(a+1/2)", &["1/2+a"][..]),
        ];

        for (input, result_fragments) in cases {
            let lines = derive_lines(input);
            assert_derive_strategy(&lines, "combine powers");
            assert!(lines.iter().any(|line| {
                line.contains("Sumar exponentes de la misma base")
                    || line.contains("Combine powers with same base (n-ary)")
            }));
            assert!(
                lines
                    .iter()
                    .filter(|line| line.trim_start().starts_with("1. "))
                    .count()
                    == 1
                    && !lines.iter().any(|line| line.trim_start().starts_with("2. ")),
                "expected root+power merge to collapse into one visible combine-powers step in {lines:?}"
            );
            assert!(
                !lines.iter().any(|line| {
                    line.contains("Reescribir la raíz como potencia fraccionaria")
                        || line.contains("sqrt(x) = x^(1/2)")
                }),
                "expected root canonicalization to stay internal to the combine-powers step in {lines:?}"
            );
            assert!(!lines.iter().any(|line| line.contains("Product of Powers")));
            assert_result_contains_all(&lines, result_fragments);
        }
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
    fn evaluate_derive_command_lines_reaches_tabulated_factor_with_division_targets() {
        let cases = [(
            "derive a*x + b*x + c, x*(a + b + c/x)",
            "x ≠ 0",
            &["c/x"][..],
        )];

        for (input, require_fragment, result_fragments) in cases {
            let lines = derive_lines(input);
            assert_derive_strategy(&lines, "factor out with division");
            assert!(
                lines.iter().any(|line| line.contains(require_fragment)),
                "expected requirement `{require_fragment}`, got: {lines:?}"
            );
            assert_result_contains_all(&lines, result_fragments);
        }
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
    fn evaluate_derive_command_lines_reaches_general_xy_sine_sum_to_product_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sin(x)+sin(y), 2*sin((x+y)/2)*cos((x-y)/2)",
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
                && line.contains("sin((x + y) / 2)")
                && line.contains("cos((x - y) / 2)")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_general_xy_cosine_sum_to_product_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive cos(x)+cos(y), 2*cos((x+y)/2)*cos((x-y)/2)",
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
                && line.contains("cos((x + y) / 2)")
                && line.contains("cos((x - y) / 2)")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_general_xy_cosine_difference_sum_to_product_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive cos(x)-cos(y), -2*sin((x+y)/2)*sin((x-y)/2)",
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
                && line.contains("sin((x + y) / 2)")
                && line.contains("sin((x - y) / 2)")
                && line.contains("-2")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_phase_shift_contraction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sin(x)+cos(x), sqrt(2)*sin(x+pi/4)",
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
            .any(|line| line.contains("Phase Shift Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("sqrt(2)")
                && line.contains("sin(")
                && line.contains("pi / 4 + x")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_phase_shift_expansion_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sqrt(2)*sin(x+pi/4), sin(x)+cos(x)",
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
            .any(|line| line.contains("Phase Shift Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:") && line.contains("sin(x)") && line.contains("+ cos(x)")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_general_phase_shift_contraction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 3*sin(x)+4*cos(x), 5*sin(x+arctan(4/3))",
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
            .any(|line| line.contains("Phase Shift Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("5")
                && line.contains("sin(")
                && line.contains("arctan(4 / 3)")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_phase_shift_term_normalization_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sqrt(2)*sin(x+pi/4), sqrt(2)*cos(x-pi/4)",
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
            .any(|line| line.contains("Phase Shift Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("sqrt(2)")
                && line.contains("cos(")
                && line.contains("x - pi / 4")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_repeated_phase_shift_contraction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sin(x)+cos(x)+sin(y)+cos(y), sqrt(2)*sin(x+pi/4)+sqrt(2)*sin(y+pi/4)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("contract trig")));
        assert_eq!(
            lines
                .iter()
                .filter(|line| line.contains("Phase Shift Identity"))
                .count(),
            2
        );
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("sqrt(2)")
                && line.contains("pi / 4 + x")
                && line.contains("pi / 4 + y")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_sine_fourth_power_reduction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sin(x)^4, (3-4*cos(2*x)+cos(4*x))/8",
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
            .any(|line| line.contains("Power Reduction Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("cos(4")
                && line.contains("cos(2")
                && line.contains("/ 8")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_cosine_fourth_power_reduction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive cos(x)^4, (3+4*cos(2*x)+cos(4*x))/8",
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
            .any(|line| line.contains("Power Reduction Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("cos(4")
                && line.contains("cos(2")
                && line.contains("/ 8")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_sine_cosine_square_product_reduction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sin(x)^2*cos(x)^2, (1-cos(4*x))/8",
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
            .any(|line| line.contains("Power Reduction Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:") && line.contains("cos(4") && line.contains("/ 8")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_sine_sixth_power_reduction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sin(x)^6, (10-15*cos(2*x)+6*cos(4*x)-cos(6*x))/32",
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
            .any(|line| line.contains("Power Reduction Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("cos(6")
                && line.contains("cos(4")
                && line.contains("cos(2")
                && line.contains("/ 32")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_cosine_sixth_power_reduction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive cos(x)^6, (10+15*cos(2*x)+6*cos(4*x)+cos(6*x))/32",
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
            .any(|line| line.contains("Power Reduction Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("cos(6")
                && line.contains("cos(4")
                && line.contains("cos(2")
                && line.contains("/ 32")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_sine_eighth_power_reduction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sin(x)^8, (35-56*cos(2*x)+28*cos(4*x)-8*cos(6*x)+cos(8*x))/128",
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
            .any(|line| line.contains("Power Reduction Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("cos(8")
                && line.contains("cos(6")
                && line.contains("cos(4")
                && line.contains("cos(2")
                && line.contains("/ 128")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_cosine_eighth_power_reduction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive cos(x)^8, (35+56*cos(2*x)+28*cos(4*x)+8*cos(6*x)+cos(8*x))/128",
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
            .any(|line| line.contains("Power Reduction Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("cos(8")
                && line.contains("cos(6")
                && line.contains("cos(4")
                && line.contains("cos(2")
                && line.contains("/ 128")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_sine_tenth_power_reduction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sin(x)^10, (126-210*cos(2*x)+120*cos(4*x)-45*cos(6*x)+10*cos(8*x)-cos(10*x))/512",
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
            .any(|line| line.contains("Power Reduction Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("cos(10")
                && line.contains("cos(8")
                && line.contains("cos(6")
                && line.contains("cos(4")
                && line.contains("cos(2")
                && line.contains("/ 512")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_cosine_tenth_power_reduction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive cos(x)^10, (126+210*cos(2*x)+120*cos(4*x)+45*cos(6*x)+10*cos(8*x)+cos(10*x))/512",
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
            .any(|line| line.contains("Power Reduction Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("cos(10")
                && line.contains("cos(8")
                && line.contains("cos(6")
                && line.contains("cos(4")
                && line.contains("cos(2")
                && line.contains("/ 512")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_sine_twelfth_power_reduction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sin(x)^12, (462-792*cos(2*x)+495*cos(4*x)-220*cos(6*x)+66*cos(8*x)-12*cos(10*x)+cos(12*x))/2048",
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
            .any(|line| line.contains("Power Reduction Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("cos(12")
                && line.contains("cos(10")
                && line.contains("cos(8")
                && line.contains("cos(6")
                && line.contains("cos(4")
                && line.contains("cos(2")
                && line.contains("/ 2048")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_cosine_twelfth_power_reduction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive cos(x)^12, (462+792*cos(2*x)+495*cos(4*x)+220*cos(6*x)+66*cos(8*x)+12*cos(10*x)+cos(12*x))/2048",
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
            .any(|line| line.contains("Power Reduction Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("cos(12")
                && line.contains("cos(10")
                && line.contains("cos(8")
                && line.contains("cos(6")
                && line.contains("cos(4")
                && line.contains("cos(2")
                && line.contains("/ 2048")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_sine_fourteenth_power_reduction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sin(x)^14, (1716-3003*cos(2*x)+2002*cos(4*x)-1001*cos(6*x)+364*cos(8*x)-91*cos(10*x)+14*cos(12*x)-cos(14*x))/8192",
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
            .any(|line| line.contains("Power Reduction Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("cos(14")
                && line.contains("cos(12")
                && line.contains("cos(10")
                && line.contains("cos(8")
                && line.contains("cos(6")
                && line.contains("cos(4")
                && line.contains("cos(2")
                && line.contains("/ 8192")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_cosine_fourteenth_power_reduction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive cos(x)^14, (1716+3003*cos(2*x)+2002*cos(4*x)+1001*cos(6*x)+364*cos(8*x)+91*cos(10*x)+14*cos(12*x)+cos(14*x))/8192",
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
            .any(|line| line.contains("Power Reduction Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("cos(14")
                && line.contains("cos(12")
                && line.contains("cos(10")
                && line.contains("cos(8")
                && line.contains("cos(6")
                && line.contains("cos(4")
                && line.contains("cos(2")
                && line.contains("/ 8192")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_sine_sixteenth_power_reduction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sin(x)^16, (6435-11440*cos(2*x)+8008*cos(4*x)-4368*cos(6*x)+1820*cos(8*x)-560*cos(10*x)+120*cos(12*x)-16*cos(14*x)+cos(16*x))/32768",
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
            .any(|line| line.contains("Power Reduction Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("cos(16")
                && line.contains("cos(14")
                && line.contains("cos(12")
                && line.contains("cos(10")
                && line.contains("cos(8")
                && line.contains("cos(6")
                && line.contains("cos(4")
                && line.contains("cos(2")
                && line.contains("/ 32768")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_cosine_sixteenth_power_reduction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive cos(x)^16, (6435+11440*cos(2*x)+8008*cos(4*x)+4368*cos(6*x)+1820*cos(8*x)+560*cos(10*x)+120*cos(12*x)+16*cos(14*x)+cos(16*x))/32768",
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
            .any(|line| line.contains("Power Reduction Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("cos(16")
                && line.contains("cos(14")
                && line.contains("cos(12")
                && line.contains("cos(10")
                && line.contains("cos(8")
                && line.contains("cos(6")
                && line.contains("cos(4")
                && line.contains("cos(2")
                && line.contains("/ 32768")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_sine_eighteenth_power_reduction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sin(x)^18, (24310-43758*cos(2*x)+31824*cos(4*x)-18564*cos(6*x)+8568*cos(8*x)-3060*cos(10*x)+816*cos(12*x)-153*cos(14*x)+18*cos(16*x)-cos(18*x))/131072",
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
            .any(|line| line.contains("Power Reduction Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("cos(18")
                && line.contains("cos(16")
                && line.contains("cos(14")
                && line.contains("cos(12")
                && line.contains("cos(10")
                && line.contains("cos(8")
                && line.contains("cos(6")
                && line.contains("cos(4")
                && line.contains("cos(2")
                && line.contains("/ 131072")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_cosine_eighteenth_power_reduction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive cos(x)^18, (24310+43758*cos(2*x)+31824*cos(4*x)+18564*cos(6*x)+8568*cos(8*x)+3060*cos(10*x)+816*cos(12*x)+153*cos(14*x)+18*cos(16*x)+cos(18*x))/131072",
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
            .any(|line| line.contains("Power Reduction Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("cos(18")
                && line.contains("cos(16")
                && line.contains("cos(14")
                && line.contains("cos(12")
                && line.contains("cos(10")
                && line.contains("cos(8")
                && line.contains("cos(6")
                && line.contains("cos(4")
                && line.contains("cos(2")
                && line.contains("/ 131072")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_sine_twentieth_power_reduction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sin(x)^20, (92378-167960*cos(2*x)+125970*cos(4*x)-77520*cos(6*x)+38760*cos(8*x)-15504*cos(10*x)+4845*cos(12*x)-1140*cos(14*x)+190*cos(16*x)-20*cos(18*x)+cos(20*x))/524288",
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
            .any(|line| line.contains("Power Reduction Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("cos(20")
                && line.contains("cos(18")
                && line.contains("cos(16")
                && line.contains("cos(14")
                && line.contains("cos(12")
                && line.contains("cos(10")
                && line.contains("cos(8")
                && line.contains("cos(6")
                && line.contains("cos(4")
                && line.contains("cos(2")
                && line.contains("/ 524288")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_cosine_twentieth_power_reduction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive cos(x)^20, (92378+167960*cos(2*x)+125970*cos(4*x)+77520*cos(6*x)+38760*cos(8*x)+15504*cos(10*x)+4845*cos(12*x)+1140*cos(14*x)+190*cos(16*x)+20*cos(18*x)+cos(20*x))/524288",
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
            .any(|line| line.contains("Power Reduction Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("cos(20")
                && line.contains("cos(18")
                && line.contains("cos(16")
                && line.contains("cos(14")
                && line.contains("cos(12")
                && line.contains("cos(10")
                && line.contains("cos(8")
                && line.contains("cos(6")
                && line.contains("cos(4")
                && line.contains("cos(2")
                && line.contains("/ 524288")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_sine_twenty_second_power_reduction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sin(x)^22, (352716-646646*cos(2*x)+497420*cos(4*x)-319770*cos(6*x)+170544*cos(8*x)-74613*cos(10*x)+26334*cos(12*x)-7315*cos(14*x)+1540*cos(16*x)-231*cos(18*x)+22*cos(20*x)-cos(22*x))/2097152",
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
            .any(|line| line.contains("Power Reduction Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("cos(22")
                && line.contains("cos(20")
                && line.contains("cos(18")
                && line.contains("cos(16")
                && line.contains("cos(14")
                && line.contains("cos(12")
                && line.contains("cos(10")
                && line.contains("cos(8")
                && line.contains("cos(6")
                && line.contains("cos(4")
                && line.contains("cos(2")
                && line.contains("/ 2097152")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_cosine_twenty_second_power_reduction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive cos(x)^22, (352716+646646*cos(2*x)+497420*cos(4*x)+319770*cos(6*x)+170544*cos(8*x)+74613*cos(10*x)+26334*cos(12*x)+7315*cos(14*x)+1540*cos(16*x)+231*cos(18*x)+22*cos(20*x)+cos(22*x))/2097152",
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
            .any(|line| line.contains("Power Reduction Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("cos(22")
                && line.contains("cos(20")
                && line.contains("cos(18")
                && line.contains("cos(16")
                && line.contains("cos(14")
                && line.contains("cos(12")
                && line.contains("cos(10")
                && line.contains("cos(8")
                && line.contains("cos(6")
                && line.contains("cos(4")
                && line.contains("cos(2")
                && line.contains("/ 2097152")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_sine_twenty_fourth_power_reduction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sin(x)^24, (1352078-2496144*cos(2*x)+1961256*cos(4*x)-1307504*cos(6*x)+735471*cos(8*x)-346104*cos(10*x)+134596*cos(12*x)-42504*cos(14*x)+10626*cos(16*x)-2024*cos(18*x)+276*cos(20*x)-24*cos(22*x)+cos(24*x))/8388608",
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
            .any(|line| line.contains("Power Reduction Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("cos(24")
                && line.contains("cos(22")
                && line.contains("cos(20")
                && line.contains("cos(18")
                && line.contains("cos(16")
                && line.contains("cos(14")
                && line.contains("cos(12")
                && line.contains("cos(10")
                && line.contains("cos(8")
                && line.contains("cos(6")
                && line.contains("cos(4")
                && line.contains("cos(2")
                && line.contains("/ 8388608")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_cosine_twenty_fourth_power_reduction_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive cos(x)^24, (1352078+2496144*cos(2*x)+1961256*cos(4*x)+1307504*cos(6*x)+735471*cos(8*x)+346104*cos(10*x)+134596*cos(12*x)+42504*cos(14*x)+10626*cos(16*x)+2024*cos(18*x)+276*cos(20*x)+24*cos(22*x)+cos(24*x))/8388608",
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
            .any(|line| line.contains("Power Reduction Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("cos(24")
                && line.contains("cos(22")
                && line.contains("cos(20")
                && line.contains("cos(18")
                && line.contains("cos(16")
                && line.contains("cos(14")
                && line.contains("cos(12")
                && line.contains("cos(10")
                && line.contains("cos(8")
                && line.contains("cos(6")
                && line.contains("cos(4")
                && line.contains("cos(2")
                && line.contains("/ 8388608")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_sine_plus_cosine_square_identity_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (sin(x)+cos(x))^2, 1+sin(2*x)",
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
            .any(|line| line.contains("Trig Square Identity")));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result:") && line.contains("sin(2")));
    }

    #[test]
    fn evaluate_derive_command_lines_reaches_sine_minus_cosine_square_identity_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive (sin(x)-cos(x))^2, 1-sin(2*x)",
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
            .any(|line| line.contains("Trig Square Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:") && line.contains("1 -") && line.contains("sin(2")
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
    fn evaluate_derive_command_lines_planner_bridges_hyperbolic_sum_to_exponential_definition() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sinh(2*x)*cosh(x)+cosh(2*x)*sinh(x), (e^(3*x)-e^(-3*x))/2",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("planner")));
        assert!(
            lines
                .iter()
                .filter(|line| {
                    let trimmed = line.trim_start();
                    let Some(prefix) = trimmed.split_whitespace().next() else {
                        return false;
                    };
                    prefix.ends_with('.')
                        && prefix[..prefix.len().saturating_sub(1)]
                            .chars()
                            .all(|ch| ch.is_ascii_digit())
                })
                .count()
                >= 2,
            "expected planner path to expose at least two visible steps; lines={lines:?}"
        );
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("e^(3")
                && line.contains("e^(-3")
                && line.contains("/ 2")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_prefers_expand_for_hyperbolic_sinh_difference() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sinh(x-y), sinh(x)*cosh(y)-sinh(y)*cosh(x)",
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
            .any(|line| line.contains("Hyperbolic Angle Sum/Difference Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("sinh(x)")
                && line.contains("cosh(y)")
                && line.contains("-")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_prefers_expand_for_hyperbolic_sinh_sum() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sinh(x+y), sinh(x)*cosh(y)+sinh(y)*cosh(x)",
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
            .any(|line| line.contains("Hyperbolic Angle Sum/Difference Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("sinh(x)")
                && line.contains("cosh(y)")
                && line.contains("+")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_planner_chains_hyperbolic_sum_to_split_exponential_products() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sinh(2*x)*cosh(x)+cosh(2*x)*sinh(x), (e^x*e^(2*x)-e^(-x)*e^(-2*x))/2",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("planner")));
        assert!(
            lines
                .iter()
                .filter(|line| {
                    let trimmed = line.trim_start();
                    let Some(prefix) = trimmed.split_whitespace().next() else {
                        return false;
                    };
                    prefix.ends_with('.')
                        && prefix[..prefix.len().saturating_sub(1)]
                            .chars()
                            .all(|ch| ch.is_ascii_digit())
                })
                .count()
                >= 2,
            "expected planner path to expose at least two visible steps; lines={lines:?}"
        );
        assert!(lines.iter().any(|line| line.contains("e^")));
    }

    #[test]
    fn evaluate_derive_command_lines_prefers_direct_scaled_cosh_from_scaled_argument_exponentials()
    {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive exp(2*x)+exp(-2*x), 2*cosh(2*x)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| { line.starts_with("Strategy:") && line.contains("rewrite hyperbolics") }));
        assert!(lines
            .iter()
            .any(|line| line.contains("Hyperbolic Exponential Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:") && line.contains("2") && line.contains("cosh(2")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_prefers_direct_scaled_cosh_to_exponential_sum() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 2*cosh(2*x), exp(2*x)+exp(-2*x)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("rewrite hyperbolics")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Hyperbolic Exponential Identity")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:") && line.contains("e^(-2") && line.contains("e^(2")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_uses_named_exponential_sum_difference_with_passthrough() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive exp(x+y)+a, exp(x)*exp(y)+a",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("rewrite exponentials")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Exponential Sum/Difference Identity")));
    }

    #[test]
    fn evaluate_derive_command_lines_uses_named_exponential_contraction_with_passthrough() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive exp(x)*exp(y)+a, exp(x+y)+a",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("rewrite exponentials")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Exponential Sum/Difference Identity")));
    }

    #[test]
    fn evaluate_derive_command_lines_expands_trig_sum_to_triple_angle_polynomial_directly() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sin(2*x)*cos(x)+cos(2*x)*sin(x), 3*sin(x)-4*sin(x)^3",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand trig")));
        assert!(
            lines
                .iter()
                .filter(|line| {
                    let trimmed = line.trim_start();
                    let Some(prefix) = trimmed.split_whitespace().next() else {
                        return false;
                    };
                    prefix.ends_with('.')
                        && prefix[..prefix.len().saturating_sub(1)]
                            .chars()
                            .all(|ch| ch.is_ascii_digit())
                })
                .count()
                >= 2,
            "expected direct expand path to expose at least two visible steps; lines={lines:?}"
        );
        assert!(lines
            .iter()
            .any(|line| line.contains("Angle Sum/Diff Identity")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Triple Angle Expansion")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:")
                && line.contains("3")
                && line.contains("sin(")
                && line.contains("- 4")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_prefers_direct_product_to_sum_identity() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 2*sin(2*x)*cos(x), sin(3*x)+sin(x)",
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
        assert!(!lines
            .iter()
            .any(|line| line.contains("Linear Angle Simplification")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:") && line.contains("sin(3") && line.contains("sin(x)")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_prefers_direct_cosine_product_to_sum_identity() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 2*cos(2*x)*cos(x), cos(3*x)+cos(x)",
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
        assert!(!lines
            .iter()
            .any(|line| line.contains("Linear Angle Simplification")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:") && line.contains("cos(3") && line.contains("cos(x)")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_prefers_direct_sine_product_to_sum_difference_identity() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 2*sin(2*x)*sin(x), cos(x)-cos(3*x)",
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
        assert!(!lines
            .iter()
            .any(|line| line.contains("Linear Angle Simplification")));
        assert!(lines.iter().any(|line| {
            line.starts_with("Result:") && line.contains("cos(3") && line.contains("cos(x)")
        }));
    }

    #[test]
    fn evaluate_derive_command_lines_prefers_trig_expand_targeted_additive_triple_angle_bridge() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 2*sin(2*x)*sin(x), cos(x)-4*cos(x)^3+3*cos(x)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand trig")));
        assert_eq!(
            lines
                .iter()
                .filter(|line| {
                    let trimmed = line.trim_start();
                    let Some(prefix) = trimmed.split_whitespace().next() else {
                        return false;
                    };
                    prefix.ends_with('.')
                        && prefix[..prefix.len().saturating_sub(1)]
                            .chars()
                            .all(|ch| ch.is_ascii_digit())
                })
                .count(),
            2,
            "expected trig-expand path to expose exactly two visible steps; lines={lines:?}"
        );
        assert!(lines
            .iter()
            .any(|line| line.contains("Product-to-Sum Identity")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Triple Angle Expansion")));
    }

    #[test]
    fn evaluate_derive_command_lines_prefers_trig_expand_combined_additive_triple_angle_bridge_for_cosine_sum(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 2*cos(2*x)*cos(x), 4*cos(x)^3-2*cos(x)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand trig")));
        assert_eq!(
            lines
                .iter()
                .filter(|line| {
                    let trimmed = line.trim_start();
                    let Some(prefix) = trimmed.split_whitespace().next() else {
                        return false;
                    };
                    prefix.ends_with('.')
                        && prefix[..prefix.len().saturating_sub(1)]
                            .chars()
                            .all(|ch| ch.is_ascii_digit())
                })
                .count(),
            2,
            "expected trig-expand path to expose exactly two visible steps; lines={lines:?}"
        );
        assert!(lines
            .iter()
            .any(|line| line.contains("Product-to-Sum Identity")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Triple Angle Expansion")));
    }

    #[test]
    fn evaluate_derive_command_lines_prefers_trig_expand_combined_additive_triple_angle_bridge_for_cosine_difference(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 2*sin(2*x)*sin(x), 4*cos(x)-4*cos(x)^3",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand trig")));
        assert_eq!(
            lines
                .iter()
                .filter(|line| {
                    let trimmed = line.trim_start();
                    let Some(prefix) = trimmed.split_whitespace().next() else {
                        return false;
                    };
                    prefix.ends_with('.')
                        && prefix[..prefix.len().saturating_sub(1)]
                            .chars()
                            .all(|ch| ch.is_ascii_digit())
                })
                .count(),
            2,
            "expected trig-expand path to expose exactly two visible steps; lines={lines:?}"
        );
        assert!(lines
            .iter()
            .any(|line| line.contains("Product-to-Sum Identity")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Triple Angle Expansion")));
    }

    #[test]
    fn evaluate_derive_command_lines_prefers_trig_expand_combined_additive_triple_angle_bridge_for_sine_difference_mixed_polynomial(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 2*cos(2*x)*sin(x), 4*cos(x)^2*sin(x)-2*sin(x)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand trig")));
        assert_eq!(
            lines
                .iter()
                .filter(|line| {
                    let trimmed = line.trim_start();
                    let Some(prefix) = trimmed.split_whitespace().next() else {
                        return false;
                    };
                    prefix.ends_with('.')
                        && prefix[..prefix.len().saturating_sub(1)]
                            .chars()
                            .all(|ch| ch.is_ascii_digit())
                })
                .count(),
            2,
            "expected trig-expand path to expose exactly two visible steps; lines={lines:?}"
        );
        assert!(lines
            .iter()
            .any(|line| line.contains("Product-to-Sum Identity")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Triple Angle Expansion")));
    }

    #[test]
    fn evaluate_derive_command_lines_prefers_trig_expand_product_to_sum_triple_angle_with_passthrough(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 2*sin(2*x)*sin(x)+a, 4*cos(x)-4*cos(x)^3+a",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand trig")));
        assert_eq!(
            lines
                .iter()
                .filter(|line| {
                    let trimmed = line.trim_start();
                    let Some(prefix) = trimmed.split_whitespace().next() else {
                        return false;
                    };
                    prefix.ends_with('.')
                        && prefix[..prefix.len().saturating_sub(1)]
                            .chars()
                            .all(|ch| ch.is_ascii_digit())
                })
                .count(),
            2,
            "expected trig-expand passthrough path to expose exactly two visible steps; lines={lines:?}"
        );
        assert!(lines
            .iter()
            .any(|line| line.contains("Product-to-Sum Identity")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Triple Angle Expansion")));
    }

    #[test]
    fn evaluate_derive_command_lines_prefers_trig_expand_mixed_product_to_sum_triple_angle_with_passthrough(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 2*cos(2*x)*sin(x)+a, 4*cos(x)^2*sin(x)-2*sin(x)+a",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand trig")));
        assert_eq!(
            lines
                .iter()
                .filter(|line| {
                    let trimmed = line.trim_start();
                    let Some(prefix) = trimmed.split_whitespace().next() else {
                        return false;
                    };
                    prefix.ends_with('.')
                        && prefix[..prefix.len().saturating_sub(1)]
                            .chars()
                            .all(|ch| ch.is_ascii_digit())
                })
                .count(),
            2,
            "expected trig-expand passthrough path to expose exactly two visible steps; lines={lines:?}"
        );
        assert!(lines
            .iter()
            .any(|line| line.contains("Product-to-Sum Identity")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Triple Angle Expansion")));
    }

    #[test]
    fn evaluate_derive_command_lines_prefers_direct_mixed_cosine_difference_double_angle_expansion()
    {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 2*sin(2*x)*sin(x), 4*sin(x)^2*cos(x)",
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
            .any(|line| line.contains("Double Angle Expansion")));
    }

    #[test]
    fn evaluate_derive_command_lines_prefers_expand_for_hyperbolic_product_to_sum_triple_angle_polynomial(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 2*sinh(2*x)*cosh(x), 4*sinh(x)+4*sinh(x)^3",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand")));
        assert_eq!(
            lines
                .iter()
                .filter(|line| {
                    let trimmed = line.trim_start();
                    let Some(prefix) = trimmed.split_whitespace().next() else {
                        return false;
                    };
                    prefix.ends_with('.')
                        && prefix[..prefix.len().saturating_sub(1)]
                            .chars()
                            .all(|ch| ch.is_ascii_digit())
                })
                .count(),
            1,
            "expected direct expand path to expose exactly one visible step; lines={lines:?}"
        );
        assert!(lines
            .iter()
            .any(|line| line.contains("Hyperbolic Product-to-Sum Identity")));
    }

    #[test]
    fn evaluate_derive_command_lines_prefers_direct_hyperbolic_product_to_sum_expansion_over_planner(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 2*sinh(2*x)*cosh(x), sinh(3*x)+sinh(x)",
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
            .any(|line| line.contains("Hyperbolic Product-to-Sum Identity")));
    }

    #[test]
    fn evaluate_derive_command_lines_prefers_direct_hyperbolic_sum_to_product_contraction_over_planner(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sinh(3*x)-sinh(x), 2*cosh(2*x)*sinh(x)",
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
            .any(|line| line.contains("Hyperbolic Product-to-Sum Identity")));
    }

    #[test]
    fn evaluate_derive_command_lines_prefers_direct_exact_hyperbolic_sum_to_product_xy_over_planner(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive sinh(x)+sinh(y), 2*sinh((x+y)/2)*cosh((x-y)/2)",
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
            .any(|line| line.contains("Hyperbolic Product-to-Sum Identity")));
    }

    #[test]
    fn evaluate_derive_command_lines_prefers_expand_for_hyperbolic_cosh_product_to_sum_triple_angle_polynomial(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 2*sinh(2*x)*sinh(x), 4*cosh(x)^3-4*cosh(x)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand")));
        assert_eq!(
            lines
                .iter()
                .filter(|line| {
                    let trimmed = line.trim_start();
                    let Some(prefix) = trimmed.split_whitespace().next() else {
                        return false;
                    };
                    prefix.ends_with('.')
                        && prefix[..prefix.len().saturating_sub(1)]
                            .chars()
                            .all(|ch| ch.is_ascii_digit())
                })
                .count(),
            2,
            "expected expand path to expose exactly two visible steps; lines={lines:?}"
        );
        assert!(lines
            .iter()
            .any(|line| line.contains("Hyperbolic Product-to-Sum Identity")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Hyperbolic Triple-Angle Identity")));
    }

    #[test]
    fn evaluate_derive_command_lines_prefers_expand_for_hyperbolic_product_to_sum_polynomial_with_passthrough_term(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 2*sinh(2*x)*sinh(x)+a, 4*cosh(x)^3-4*cosh(x)+a",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| line.starts_with("Strategy:") && line.contains("expand")));
        assert_eq!(
            lines
                .iter()
                .filter(|line| {
                    let trimmed = line.trim_start();
                    let Some(prefix) = trimmed.split_whitespace().next() else {
                        return false;
                    };
                    prefix.ends_with('.')
                        && prefix[..prefix.len().saturating_sub(1)]
                            .chars()
                            .all(|ch| ch.is_ascii_digit())
                })
                .count(),
            2,
            "expected expand path to expose exactly two visible steps; lines={lines:?}"
        );
        assert!(lines
            .iter()
            .any(|line| line.contains("Hyperbolic Product-to-Sum Identity")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Hyperbolic Triple-Angle Identity")));
    }

    #[test]
    fn evaluate_derive_command_lines_prefers_direct_hyperbolic_cosh_double_angle_expansion_to_sinh_cubic_polynomial(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 2*cosh(2*x)*sinh(x), 2*sinh(x)+4*sinh(x)^3",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| { line.starts_with("Strategy:") && line.contains("rewrite hyperbolics") }));
        assert!(lines
            .iter()
            .any(|line| line.contains("Hyperbolic Double-Angle Identity")));
    }

    #[test]
    fn evaluate_derive_command_lines_prefers_direct_hyperbolic_cosh_double_angle_expansion_to_sinh_mixed_polynomial(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 2*cosh(2*x)*sinh(x), 4*cosh(x)^2*sinh(x)-2*sinh(x)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| { line.starts_with("Strategy:") && line.contains("rewrite hyperbolics") }));
        assert!(lines
            .iter()
            .any(|line| line.contains("Hyperbolic Double-Angle Identity")));
    }

    #[test]
    fn evaluate_derive_command_lines_prefers_direct_hyperbolic_cosh_double_angle_contraction_from_sinh_mixed_polynomial(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 4*cosh(x)^2*sinh(x)-2*sinh(x), 2*cosh(2*x)*sinh(x)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| { line.starts_with("Strategy:") && line.contains("rewrite hyperbolics") }));
        assert!(lines
            .iter()
            .any(|line| line.contains("Hyperbolic Double-Angle Identity")));
    }

    #[test]
    fn evaluate_derive_command_lines_prefers_direct_hyperbolic_cosh_double_angle_contraction_from_cosh_mixed_polynomial(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 2*cosh(x)+4*sinh(x)^2*cosh(x), 2*cosh(2*x)*cosh(x)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| { line.starts_with("Strategy:") && line.contains("rewrite hyperbolics") }));
        assert!(lines
            .iter()
            .any(|line| line.contains("Hyperbolic Double-Angle Identity")));
    }

    #[test]
    fn evaluate_derive_command_lines_prefers_direct_hyperbolic_cosh_double_angle_expansion_to_cosh_mixed_polynomial(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 2*cosh(2*x)*cosh(x), 2*cosh(x)+4*sinh(x)^2*cosh(x)",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("derive should evaluate");

        assert!(lines
            .iter()
            .any(|line| { line.starts_with("Strategy:") && line.contains("rewrite hyperbolics") }));
        assert!(lines
            .iter()
            .any(|line| line.contains("Hyperbolic Double-Angle Identity")));
    }

    #[test]
    fn evaluate_derive_command_lines_uses_named_double_angle_for_mixed_trig_product() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 4*sin(x)^2*cos(x), 2*sin(2*x)*sin(x)",
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
            .any(|line| line.contains("Double Angle Expansion")));
    }

    #[test]
    fn evaluate_derive_command_lines_uses_named_double_angle_for_forward_mixed_trig_product() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive 2*sin(2*x)*sin(x), 4*sin(x)^2*cos(x)",
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
            .any(|line| line.contains("Double Angle Expansion")));
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
    fn evaluate_derive_command_lines_reaches_symbolic_general_phase_shift_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_derive_command_lines_with_resolver(
            &mut simplifier,
            "derive a*sin(x)+b*cos(x), sqrt(a^2+b^2)*sin(x+arctan(b/a))",
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
            .any(|line| line.contains("Phase Shift Identity")));
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
