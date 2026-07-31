use super::*;

#[test]
fn reciprocal_sqrt_product_diff_keeps_post_calculus_root_fraction_presentation() {
    let cases = [
        (
            "diff(1/sqrt(x), x)",
            "-1 / (2 * x * sqrt(x))",
            "diff(1/sqrt(x), x) + 1/(2*x*sqrt(x))",
        ),
        (
            "diff(1/(sqrt(x)*(x+1)), x)",
            "-(3 * x + 1) / (2 * x * sqrt(x) * (x + 1)^2)",
            "diff(1/(sqrt(x)*(x+1)), x) + (3*x+1)/(2*x*sqrt(x)*(x+1)^2)",
        ),
        (
            "diff(1/(sqrt(x)*(sqrt(x)+1)), x)",
            "-(2 * sqrt(x) + 1) / (2 * x * sqrt(x) * (sqrt(x) + 1)^2)",
            "diff(1/(sqrt(x)*(sqrt(x)+1)), x) + (2*sqrt(x)+1)/(2*x*sqrt(x)*(sqrt(x)+1)^2)",
        ),
        (
            "diff(1/(sqrt(x)*(2*x+2)), x)",
            "-(3 * x + 1) / (4 * x * sqrt(x) * (x + 1)^2)",
            "diff(1/(sqrt(x)*(2*x+2)), x) + (3*x+1)/(4*x*sqrt(x)*(x+1)^2)",
        ),
        (
            "diff(1/(2*sqrt(x)*(x+1)), x)",
            "-(3 * x + 1) / (4 * x * sqrt(x) * (x + 1)^2)",
            "diff(1/(2*sqrt(x)*(x+1)), x) + (3*x+1)/(4*x*sqrt(x)*(x+1)^2)",
        ),
    ];

    for (input, expected, residual_input) in cases {
        let mut engine = Engine::new();
        let mut state = SessionState::new();
        state.options_mut().steps_mode = StepsMode::On;
        let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

        let output = engine
            .eval(
                &mut state,
                EvalRequest {
                    raw_input: input.to_string(),
                    parsed,
                    action: EvalAction::Simplify,
                    auto_store: false,
                },
            )
            .expect("eval failed");

        let result = match output.result {
            EvalResult::Expr(expr) => format!(
                "{}",
                DisplayExpr {
                    context: &engine.simplifier.context,
                    id: expr,
                }
            ),
            other => panic!("expected expression result, got {other:?}"),
        };
        assert_eq!(result, expected, "input: {input}");
        assert!(
            !result.contains("x^(-") && !result.contains("x^(1/2)"),
            "post-calculus presentation should use sqrt notation instead of half-power noise: {result}"
        );
        assert!(
            !result.contains("(2 * x + 2)^2"),
            "post-calculus presentation should keep denominator content factored outside the squared polynomial: {result}"
        );

        let required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();
        assert_eq!(
            required,
            vec!["x > 0".to_string()],
            "input: {input}, unexpected required_conditions: {required:?}"
        );
        assert!(
            !output.steps.iter().any(|step| {
                matches!(
                    step.rule_name.as_str(),
                    "Rationalize Product Denominator"
                        | "Pull Constant From Fraction"
                        | "Normalize Negation in Product"
                )
            }),
            "reciprocal-root presentation should not expose a rationalize/power round trip; steps: {:?}",
            output
                .steps
                .iter()
                .map(|step| step.rule_name.as_str())
                .collect::<Vec<_>>()
        );

        let parsed_residual =
            parse(residual_input, &mut engine.simplifier.context).expect("parse residual");
        let residual_output = engine
            .eval(
                &mut state,
                EvalRequest {
                    raw_input: residual_input.to_string(),
                    parsed: parsed_residual,
                    action: EvalAction::Simplify,
                    auto_store: false,
                },
            )
            .expect("eval residual");
        let residual = match residual_output.result {
            EvalResult::Expr(expr) => format!(
                "{}",
                DisplayExpr {
                    context: &engine.simplifier.context,
                    id: expr,
                }
            ),
            other => panic!("expected residual expression result, got {other:?}"),
        };
        assert_eq!(residual, "0", "residual did not collapse for {input}");
    }
}
#[test]
fn reciprocal_sqrt_product_diff_handles_nonzero_negative_root_shift_compactly() {
    let cases = [
        (
            "diff(1/(sqrt(x)*(sqrt(x)-1)), x)",
            "-(2 * sqrt(x) - 1) / (2 * x * sqrt(x) * (sqrt(x) - 1)^2)",
        ),
        (
            "diff(arctan(1/(sqrt(x)*(sqrt(x)-1))), x)",
            "-(2 * sqrt(x) - 1) / (2 * sqrt(x) * (x * (sqrt(x) - 1)^2 + 1))",
        ),
    ];

    for (input, expected) in cases {
        let mut engine = Engine::new();
        let mut state = SessionState::new();
        state.options_mut().steps_mode = StepsMode::On;
        let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

        let output = engine
            .eval(
                &mut state,
                EvalRequest {
                    raw_input: input.to_string(),
                    parsed,
                    action: EvalAction::Simplify,
                    auto_store: false,
                },
            )
            .expect("eval failed");

        let result = match output.result {
            EvalResult::Expr(expr) => format!(
                "{}",
                DisplayExpr {
                    context: &engine.simplifier.context,
                    id: expr,
                }
            ),
            other => panic!("expected expression result, got {other:?}"),
        };
        assert_eq!(result, expected, "input: {input}");
        assert!(
            !result.contains("x^(-") && !result.contains("x^(1/2)"),
            "post-calculus presentation should use sqrt notation instead of half-power noise: {result}"
        );
        assert!(
            !output.steps.iter().any(|step| {
                matches!(
                    step.rule_name.as_str(),
                    "Expandir la expresión"
                        | "Rationalize Product Denominator"
                        | "Pull Constant From Fraction"
                        | "Normalize Negation in Product"
                )
            }),
            "negative shifted-root route should not expose expansion/rationalize cleanup; steps: {:?}",
            output
                .steps
                .iter()
                .map(|step| step.rule_name.as_str())
                .collect::<Vec<_>>()
        );

        let required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();
        assert_eq!(
            required,
            vec!["x > 0".to_string(), "sqrt(x) - 1 ≠ 0".to_string()],
            "input: {input}, unexpected required_conditions: {required:?}"
        );
    }
}
#[test]
fn reciprocal_sqrt_product_negative_shift_residual_bridges_half_power_denominator() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let input = "diff(1/(sqrt(x)*(sqrt(x)-1)), x) + (2*sqrt(x)-1)/(2*x*sqrt(x)*(sqrt(x)-1)^2)";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let output = engine
        .eval(
            &mut state,
            EvalRequest {
                raw_input: input.to_string(),
                parsed,
                action: EvalAction::Simplify,
                auto_store: false,
            },
        )
        .expect("eval failed");

    let result = match output.result {
        EvalResult::Expr(expr) => format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: expr,
            }
        ),
        other => panic!("expected expression result, got {other:?}"),
    };

    assert_eq!(result, "0");
    assert!(
        output
            .steps
            .iter()
            .any(|step| step.rule_name == "Cancel Opposite Fractions"),
        "expected shifted reciprocal-sqrt residual to close through fraction cancellation, got: {:?}",
        output
            .steps
            .iter()
            .map(|step| step.rule_name.as_str())
            .collect::<Vec<_>>()
    );
    assert!(
        output.domain_warnings.is_empty(),
        "warnings: {:?}",
        output.domain_warnings
    );

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert_eq!(
        required,
        vec!["x > 0".to_string(), "sqrt(x) - 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn sqrt_negative_even_power_diff_uses_essential_domain_guard() {
    for (input, expected_result, expected_required) in [
        (
            "diff(sqrt(x^-2), x)",
            "-1 / (x * |x|)",
            vec!["x ≠ 0".to_string()],
        ),
        (
            "diff(sqrt((x+1)^-2), x)",
            "-1 / (|x + 1| * (x + 1))",
            vec!["x ≠ -1".to_string()],
        ),
    ] {
        let mut engine = Engine::new();
        let mut state = SessionState::new();
        let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

        let output = engine
            .eval(
                &mut state,
                EvalRequest {
                    raw_input: input.to_string(),
                    parsed,
                    action: EvalAction::Simplify,
                    auto_store: false,
                },
            )
            .expect("eval failed");
        let result = match output.result {
            EvalResult::Expr(expr) => format!(
                "{}",
                DisplayExpr {
                    context: &engine.simplifier.context,
                    id: expr,
                }
            ),
            other => panic!("expected expression result, got {other:?}"),
        };

        assert_eq!(result, expected_result, "input: {input}");

        let required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();

        assert_eq!(
            required, expected_required,
            "input: {input}, unexpected required_conditions: {required:?}"
        );
    }
}
#[test]
fn log_sqrt_negative_even_power_diff_preserves_essential_domain_guard() {
    for (input, expected_result, expected_required) in [
        (
            "diff(ln(sqrt(x^-2)), x)",
            "-1 / x",
            vec!["x ≠ 0".to_string()],
        ),
        (
            "diff(ln(sqrt((x+1)^-2)), x)",
            "-1 / (x + 1)",
            vec!["x ≠ -1".to_string()],
        ),
    ] {
        let mut engine = Engine::new();
        let mut state = SessionState::new();
        let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

        let output = engine
            .eval(
                &mut state,
                EvalRequest {
                    raw_input: input.to_string(),
                    parsed,
                    action: EvalAction::Simplify,
                    auto_store: false,
                },
            )
            .expect("eval failed");
        let result = match output.result {
            EvalResult::Expr(expr) => format!(
                "{}",
                DisplayExpr {
                    context: &engine.simplifier.context,
                    id: expr,
                }
            ),
            other => panic!("expected expression result, got {other:?}"),
        };

        assert_eq!(result, expected_result, "input: {input}");

        let required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();

        assert_eq!(
            required, expected_required,
            "input: {input}, unexpected required_conditions: {required:?}"
        );
    }
}
#[test]
fn square_root_diff_evaluates_with_positive_domain_condition() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(sqrt(x), x)";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    let result = match output.result {
        EvalResult::Expr(expr) => format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: expr,
            }
        ),
        other => panic!("expected expression result, got {other:?}"),
    };

    assert_eq!(result, "1 / (2 * sqrt(x))");

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert_eq!(
        required,
        vec!["x > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn square_root_quotient_diff_uses_compact_post_calculus_presentation() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let input = "diff(sqrt(x)/(x+1), x)";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    let result_expr = match output.result {
        EvalResult::Expr(expr) => expr,
        other => panic!("expected expression result, got {other:?}"),
    };
    let result = format!(
        "{}",
        DisplayExpr {
            context: &engine.simplifier.context,
            id: result_expr,
        }
    );

    assert_eq!(result, "(1 - x) / (2 * sqrt(x) * (x + 1)^2)");
    assert!(
        !result.contains("^(-1/2)"),
        "presentation should use a sqrt denominator, got: {result}"
    );
    assert!(
        result.contains("(x + 1)^2"),
        "presentation should preserve the compact squared denominator: {result}"
    );

    let expected =
        parse("(1-x)/(2*sqrt(x)*(x+1)^2)", &mut engine.simplifier.context).expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "post-calculus presentation must stay equivalent to the compact quotient derivative, got: {result}"
    );

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert_eq!(
        required,
        vec!["x > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert!(
        output
            .steps
            .iter()
            .any(|step| step.rule_name == "Symbolic Differentiation"),
        "expected the derivative to keep the ordinary symbolic differentiation trace"
    );
    assert!(
        output
            .steps
            .iter()
            .any(|step| step.rule_name == "Present calculus result in compact form"),
        "expected a visible post-calculus presentation step"
    );

    for residual_input in [
        "diff(sqrt(x)/(x+1), x) - (1-x)/(2*sqrt(x)*(x+1)^2)",
        "diff(sqrt(x)/(x^2+1), x) - (1-3*x^2)/(2*sqrt(x)*(x^2+1)^2)",
    ] {
        let mut engine = Engine::new();
        let mut state = SessionState::new();
        state.options_mut().steps_mode = StepsMode::On;
        let parsed = parse(residual_input, &mut engine.simplifier.context).expect("parse residual");
        let output = engine
            .eval(
                &mut state,
                EvalRequest {
                    raw_input: residual_input.to_string(),
                    parsed,
                    action: EvalAction::Simplify,
                    auto_store: false,
                },
            )
            .expect("eval residual");
        let residual = match output.result {
            EvalResult::Expr(expr) => format!(
                "{}",
                DisplayExpr {
                    context: &engine.simplifier.context,
                    id: expr,
                }
            ),
            other => panic!("expected expression result, got {other:?}"),
        };
        assert_eq!(
            residual, "0",
            "sqrt quotient diff residual did not collapse for {residual_input}"
        );
    }
}
#[test]
fn log_over_sqrt_diff_uses_compact_post_calculus_root_denominator_presentation() {
    let cases = [
        (
            "diff(ln(x)/sqrt(x), x)",
            "(2 - ln(x)) / (2 * x * sqrt(x))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(2*ln(x)/sqrt(x), x)",
            "(2 - ln(x)) / (x * sqrt(x))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(ln(x)/(2*sqrt(x)), x)",
            "(2 - ln(x)) / (4 * x * sqrt(x))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(ln(x)/(a*sqrt(x)), x)",
            "(2 - ln(x)) / (2 * a * x * sqrt(x))",
            vec!["a ≠ 0".to_string(), "x > 0".to_string()],
        ),
        (
            "diff((a*ln(x))/sqrt(x), x)",
            "a * (2 - ln(x)) / (2 * x * sqrt(x))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(-ln(x)/sqrt(x), x)",
            "-(2 - ln(x)) / (2 * x * sqrt(x))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(ln(x+1)/sqrt(x+1), x)",
            "(2 - ln(x + 1)) / (2 * (x + 1) * sqrt(x + 1))",
            vec!["x > -1".to_string()],
        ),
        (
            "diff(ln(x+1)/(2*sqrt(x+1)), x)",
            "(2 - ln(x + 1)) / (4 * (x + 1) * sqrt(x + 1))",
            vec!["x > -1".to_string()],
        ),
        (
            "diff(ln(x+1)/(a*sqrt(x+1)), x)",
            "(2 - ln(x + 1)) / (2 * a * (x + 1) * sqrt(x + 1))",
            vec!["a ≠ 0".to_string(), "x > -1".to_string()],
        ),
        (
            "diff(ln(2*x+1)/sqrt(2*x+1), x)",
            "(2 - ln(2 * x + 1)) / ((2 * x + 1) * sqrt(2 * x + 1))",
            vec!["x > -1/2".to_string()],
        ),
        (
            "diff(ln(2*x+1)/(2*sqrt(2*x+1)), x)",
            "(2 - ln(2 * x + 1)) / (2 * (2 * x + 1) * sqrt(2 * x + 1))",
            vec!["x > -1/2".to_string()],
        ),
        (
            "diff(ln(x^2+1)/sqrt(x^2+1), x)",
            "x * (2 - ln(x^2 + 1)) / ((x^2 + 1) * sqrt(x^2 + 1))",
            Vec::new(),
        ),
        (
            "diff((3*ln(x^2+1))/sqrt(x^2+1), x)",
            "3 * x * (2 - ln(x^2 + 1)) / ((x^2 + 1) * sqrt(x^2 + 1))",
            Vec::new(),
        ),
    ];

    for (input, expected, expected_required) in cases {
        let mut engine = Engine::new();
        let mut state = SessionState::new();
        state.options_mut().steps_mode = StepsMode::On;
        let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

        let output = engine
            .eval(
                &mut state,
                EvalRequest {
                    raw_input: input.to_string(),
                    parsed,
                    action: EvalAction::Simplify,
                    auto_store: false,
                },
            )
            .expect("eval failed");
        let result_expr = match output.result {
            EvalResult::Expr(expr) => expr,
            other => panic!("expected expression result, got {other:?}"),
        };
        let result = format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result_expr,
            }
        );

        assert_eq!(result, expected, "input: {input}");
        assert!(
            !result.contains("^(-3/2)") && !result.contains("^(1/2)"),
            "post-calculus presentation should avoid half-power notation: {result}"
        );

        let expected_expr =
            parse(expected, &mut engine.simplifier.context).expect("parse expected");
        assert!(
            engine.simplifier.are_equivalent(result_expr, expected_expr),
            "post-calculus presentation must remain equivalent, got: {result}"
        );

        let required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();
        assert_eq!(
            required, expected_required,
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert!(
            output
                .steps
                .iter()
                .any(|step| step.rule_name == "Symbolic Differentiation"),
            "expected the derivative to keep the ordinary symbolic differentiation trace"
        );
        assert!(
            output
                .steps
                .iter()
                .any(|step| step.rule_name == "Present calculus result in compact form"),
            "expected a visible post-calculus presentation step"
        );
    }
}
#[test]
fn sqrt_over_log_symbolic_denominator_scale_diff_avoids_depth_overflow_route() {
    let cases = [
        (
            "diff(sqrt(x)/(a*ln(x)), x)",
            "(ln(x) - 2) / (2 * a * ln(x)^2 * sqrt(x))",
            vec![
                "a ≠ 0".to_string(),
                "x ≠ 1".to_string(),
                "x > 0".to_string(),
            ],
        ),
        (
            "diff(sqrt(x+1)/(a*ln(x+1)), x)",
            "(ln(x + 1) - 2) / (2 * a * ln(x + 1)^2 * sqrt(x + 1))",
            vec![
                "a ≠ 0".to_string(),
                "x ≠ 0".to_string(),
                "x > -1".to_string(),
            ],
        ),
    ];

    for (input, expected, expected_required) in cases {
        let mut engine = Engine::new();
        let mut state = SessionState::new();
        state.options_mut().steps_mode = StepsMode::On;
        let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

        let output = engine
            .eval(
                &mut state,
                EvalRequest {
                    raw_input: input.to_string(),
                    parsed,
                    action: EvalAction::Simplify,
                    auto_store: false,
                },
            )
            .expect("eval failed");
        let result_expr = match output.result {
            EvalResult::Expr(expr) => expr,
            other => panic!("expected expression result, got {other:?}"),
        };
        let result = format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result_expr,
            }
        );

        assert_eq!(result, expected, "input: {input}");
        assert!(
            !result.contains("^(-1/2)") && !result.contains("^(1/2)"),
            "symbolic denominator scale presentation should avoid half-power notation: {result}"
        );

        let expected_expr =
            parse(expected, &mut engine.simplifier.context).expect("parse expected");
        assert!(
            engine.simplifier.are_equivalent(result_expr, expected_expr),
            "post-calculus presentation must remain equivalent, got: {result}"
        );

        let required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();
        assert_eq!(
            required, expected_required,
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert!(
            output
                .steps
                .iter()
                .any(|step| step.rule_name == "Symbolic Differentiation"),
            "expected the derivative to keep the ordinary symbolic differentiation trace"
        );
    }
}
#[test]
fn shifted_square_root_quotient_diff_residual_uses_compact_presentation() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let input = "diff(sqrt(x+1)/(x+2), x)";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let output = engine
        .eval(
            &mut state,
            EvalRequest {
                raw_input: input.to_string(),
                parsed,
                action: EvalAction::Simplify,
                auto_store: false,
            },
        )
        .expect("eval shifted sqrt quotient");
    let result_expr = match output.result {
        EvalResult::Expr(expr) => expr,
        other => panic!("expected expression result, got {other:?}"),
    };
    let result = format!(
        "{}",
        DisplayExpr {
            context: &engine.simplifier.context,
            id: result_expr,
        }
    );

    assert_eq!(result, "-x / (2 * sqrt(x + 1) * (x + 2)^2)");
    assert!(
        output
            .steps
            .iter()
            .any(|step| step.rule_name == "Symbolic Differentiation"),
        "expected the derivative to keep the ordinary symbolic differentiation trace"
    );
    assert!(
        output
            .steps
            .iter()
            .any(|step| step.rule_name == "Present calculus result in compact form"),
        "expected a visible shifted post-calculus presentation step"
    );

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();
    assert_eq!(
        required,
        vec!["x > -1".to_string()],
        "unexpected required_conditions: {required:?}"
    );

    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let residual_input = "diff(sqrt(x+1)/(x+2), x) - (-x)/(2*sqrt(x+1)*(x+2)^2)";
    let parsed = parse(residual_input, &mut engine.simplifier.context).expect("parse residual");
    let output = engine
        .eval(
            &mut state,
            EvalRequest {
                raw_input: residual_input.to_string(),
                parsed,
                action: EvalAction::Simplify,
                auto_store: false,
            },
        )
        .expect("eval residual");
    let residual = match output.result {
        EvalResult::Expr(expr) => format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: expr,
            }
        ),
        other => panic!("expected expression result, got {other:?}"),
    };
    assert_eq!(residual, "0");
}
#[test]
fn affine_square_root_quotient_diff_residual_accepts_rationalized_target() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let input = "diff(sqrt(2*x+1)/(x+2), x)";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let output = engine
        .eval(
            &mut state,
            EvalRequest {
                raw_input: input.to_string(),
                parsed,
                action: EvalAction::Simplify,
                auto_store: false,
            },
        )
        .expect("eval affine sqrt quotient");
    let result = match output.result {
        EvalResult::Expr(expr) => format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: expr,
            }
        ),
        other => panic!("expected expression result, got {other:?}"),
    };
    assert_eq!(result, "(1 - x) / (sqrt(2 * x + 1) * (x + 2)^2)");

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();
    assert_eq!(
        required,
        vec!["x > -1/2".to_string()],
        "unexpected required_conditions: {required:?}"
    );

    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let residual_input = "diff(sqrt(2*x+1)/(x+2), x) - ((1-x)*sqrt(2*x+1))/((2*x+1)*(x+2)^2)";
    let parsed = parse(residual_input, &mut engine.simplifier.context).expect("parse residual");
    let output = engine
        .eval(
            &mut state,
            EvalRequest {
                raw_input: residual_input.to_string(),
                parsed,
                action: EvalAction::Simplify,
                auto_store: false,
            },
        )
        .expect("eval rationalized residual");
    let residual = match output.result {
        EvalResult::Expr(expr) => format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: expr,
            }
        ),
        other => panic!("expected expression result, got {other:?}"),
    };
    assert_eq!(residual, "0");
    assert!(
        output
            .steps
            .iter()
            .any(|step| step.rule_name == "Post-calculus residual simplification"),
        "expected a visible residual simplification step"
    );
}
#[test]
fn scaled_square_root_quotient_diff_uses_compact_post_calculus_presentation() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let input = "diff(2*sqrt(x)/(x+1), x)";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    let result_expr = match output.result {
        EvalResult::Expr(expr) => expr,
        other => panic!("expected expression result, got {other:?}"),
    };
    let result = format!(
        "{}",
        DisplayExpr {
            context: &engine.simplifier.context,
            id: result_expr,
        }
    );

    assert_eq!(result, "(1 - x) / (sqrt(x) * (x + 1)^2)");
    assert!(
        !result.contains("^(-1/2)"),
        "scaled presentation should use a sqrt denominator, got: {result}"
    );
    assert!(
        result.contains("(x + 1)^2"),
        "scaled presentation should preserve the compact squared denominator: {result}"
    );

    let expected =
        parse("(1-x)/(sqrt(x)*(x+1)^2)", &mut engine.simplifier.context).expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "scaled post-calculus presentation must stay equivalent, got: {result}"
    );

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert_eq!(
        required,
        vec!["x > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert!(
        output
            .steps
            .iter()
            .any(|step| step.rule_name == "Symbolic Differentiation"),
        "expected the derivative to keep the ordinary symbolic differentiation trace"
    );
    assert!(
        output
            .steps
            .iter()
            .any(|step| step.rule_name == "Present calculus result in compact form"),
        "expected a visible scaled post-calculus presentation step"
    );
}
#[test]
fn polynomial_times_square_root_diff_uses_compact_post_calculus_presentation() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let input = "diff((x+1)*sqrt(x), x)";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    let result_expr = match output.result {
        EvalResult::Expr(expr) => expr,
        other => panic!("expected expression result, got {other:?}"),
    };
    let result = format!(
        "{}",
        DisplayExpr {
            context: &engine.simplifier.context,
            id: result_expr,
        }
    );

    assert_eq!(result, "(3 * x + 1) / (2 * sqrt(x))");
    assert!(
        !result.contains("^(-1/2)") && !result.contains("x^(1/2)"),
        "product presentation should avoid fractional-power internals, got: {result}"
    );

    let expected =
        parse("(3*x+1)/(2*sqrt(x))", &mut engine.simplifier.context).expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "post-calculus product presentation must stay equivalent, got: {result}"
    );

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert_eq!(
        required,
        vec!["x > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert!(
        output
            .steps
            .iter()
            .any(|step| step.rule_name == "Symbolic Differentiation"),
        "expected the derivative to keep the ordinary symbolic differentiation trace"
    );
    assert!(
        output
            .steps
            .iter()
            .any(|step| step.rule_name == "Present calculus result in compact form"),
        "expected a visible polynomial-root product presentation step"
    );
}
#[test]
fn shifted_polynomial_times_square_root_diff_preserves_shifted_domain_condition() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let input = "diff((x+1)*sqrt(x+2), x)";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    let result_expr = match output.result {
        EvalResult::Expr(expr) => expr,
        other => panic!("expected expression result, got {other:?}"),
    };
    let result = format!(
        "{}",
        DisplayExpr {
            context: &engine.simplifier.context,
            id: result_expr,
        }
    );

    assert_eq!(result, "(3 * x + 5) / (2 * sqrt(x + 2))");
    assert!(
        !result.contains("^(-1/2)") && !result.contains("^(1/2)"),
        "shifted product presentation should avoid fractional-power internals, got: {result}"
    );

    let expected =
        parse("(3*x+5)/(2*sqrt(x+2))", &mut engine.simplifier.context).expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "shifted product post-calculus presentation must stay equivalent, got: {result}"
    );

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert_eq!(
        required,
        vec!["x > -2".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert!(
        output
            .steps
            .iter()
            .any(|step| step.rule_name == "Symbolic Differentiation"),
        "expected the derivative to keep the ordinary symbolic differentiation trace"
    );
    assert!(
        output
            .steps
            .iter()
            .any(|step| step.rule_name == "Present calculus result in compact form"),
        "expected a visible shifted polynomial-root product presentation step"
    );
}
#[test]
fn ln_sqrt_plus_matching_polynomial_diff_avoids_removable_domain_pole() {
    let input = "diff(ln(sqrt(x)+x), x)";
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    let result_expr = match output.result {
        EvalResult::Expr(expr) => expr,
        other => panic!("expected expression result, got {other:?}"),
    };
    let result = format!(
        "{}",
        DisplayExpr {
            context: &engine.simplifier.context,
            id: result_expr,
        }
    );

    assert_eq!(result, "1 / x - 1 / (2 * x * (sqrt(x) + 1))");
    assert!(
        !result.contains("x^(3/2)") && !result.contains("(x - 1"),
        "presentation should avoid expanded removable-pole artifacts: {result}"
    );

    let expected = parse(
        "(1+2*sqrt(x))/(2*x*(sqrt(x)+1))",
        &mut engine.simplifier.context,
    )
    .expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "post-calculus presentation must stay equivalent for {input}, got: {result}"
    );

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert_eq!(
        required,
        vec!["x > 0".to_string()],
        "unexpected required_conditions for {input}: {required:?}"
    );
}
#[test]
fn ln_sqrt_minus_matching_polynomial_diff_exposes_open_unit_domain() {
    let input = "diff(ln(sqrt(x)-x), x)";
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    let result_expr = match output.result {
        EvalResult::Expr(expr) => expr,
        other => panic!("expected expression result, got {other:?}"),
    };
    let result = format!(
        "{}",
        DisplayExpr {
            context: &engine.simplifier.context,
            id: result_expr,
        }
    );

    assert_eq!(result, "1 / x - 1 / (2 * x * (1 - sqrt(x)))");
    assert!(
        !result.contains("x^(-1/2)") && !result.contains("sqrt(x) - x > 0"),
        "presentation should avoid reciprocal-power and raw sqrt-gap artifacts: {result}"
    );

    let expected = parse(
        "(1-2*sqrt(x))/(2*x*(1-sqrt(x)))",
        &mut engine.simplifier.context,
    )
    .expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "post-calculus presentation must stay equivalent for {input}, got: {result}"
    );

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert_eq!(
        required,
        vec!["x < 1".to_string(), "x > 0".to_string()],
        "unexpected required_conditions for {input}: {required:?}"
    );
}
#[test]
fn reciprocal_positive_shifted_sqrt_diff_avoids_rationalized_domain_hole() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let input = "diff(1/(sqrt(3-2*x)+1), x)";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    let result_expr = match output.result {
        EvalResult::Expr(expr) => expr,
        other => panic!("expected expression result, got {other:?}"),
    };
    let result = format!(
        "{}",
        DisplayExpr {
            context: &engine.simplifier.context,
            id: result_expr,
        }
    );

    assert!(
        !result.contains("(2 - 2 * x)^2"),
        "diff should avoid the rationalized denominator with a removable hole: {result}"
    );
    assert_eq!(result, "1 / (sqrt(3 - 2 * x) * (sqrt(3 - 2 * x) + 1)^2)");
    assert!(
        !result.contains("(3 - 2 * x)^(1/2) /"),
        "post-calculus presentation should keep reciprocal-root form: {result}"
    );
    let expected = parse(
        "1/(sqrt(3-2*x)*(sqrt(3-2*x)+1)^2)",
        &mut engine.simplifier.context,
    )
    .expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "expected compact reciprocal shifted-root derivative, got: {result}"
    );

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert_eq!(
        required,
        vec!["x < 3/2".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert!(
        output
            .steps
            .iter()
            .all(|step| step.rule_name.as_str() != "Racionalizar el denominador"),
        "positive shifted sqrt reciprocal should take the direct diff route"
    );
}
#[test]
fn reciprocal_positive_shifted_sqrt_diff_keeps_nonunit_scale_and_shift_compact() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let input = "diff(2/(sqrt(3-2*x)+2), x)";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    let result_expr = match output.result {
        EvalResult::Expr(expr) => expr,
        other => panic!("expected expression result, got {other:?}"),
    };
    let result = format!(
        "{}",
        DisplayExpr {
            context: &engine.simplifier.context,
            id: result_expr,
        }
    );

    assert_eq!(result, "2 / (sqrt(3 - 2 * x) * (sqrt(3 - 2 * x) + 2)^2)");
    assert!(
        !result.contains("2 * x + 1"),
        "diff should avoid a rationalized denominator with an artificial hole: {result}"
    );
    assert!(
        !result.contains("(3 - 2 * x)^(1/2) /"),
        "post-calculus presentation should keep reciprocal-root form: {result}"
    );
    let expected = parse(
        "2/(sqrt(3-2*x)*(sqrt(3-2*x)+2)^2)",
        &mut engine.simplifier.context,
    )
    .expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "expected compact reciprocal shifted-root derivative, got: {result}"
    );

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert_eq!(
        required,
        vec!["x < 3/2".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert!(
        output
            .steps
            .iter()
            .all(|step| step.rule_name.as_str() != "Racionalizar el denominador"),
        "nonunit positive shifted sqrt reciprocal should take the direct diff route"
    );
}
#[test]
fn reciprocal_positive_shifted_sqrt_diff_handles_commuted_shift_and_chain_sign() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let input = "diff(3/(2+sqrt(2*x+5)), x)";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    let result_expr = match output.result {
        EvalResult::Expr(expr) => expr,
        other => panic!("expected expression result, got {other:?}"),
    };
    let result = format!(
        "{}",
        DisplayExpr {
            context: &engine.simplifier.context,
            id: result_expr,
        }
    );

    assert_eq!(result, "-3 / (sqrt(2 * x + 5) * (sqrt(2 * x + 5) + 2)^2)");
    assert!(
        !result.contains("2 + sqrt"),
        "post-calculus presentation should canonicalize the shifted root denominator: {result}"
    );
    assert!(
        !result.contains("(2 * x + 5)^(1/2) /"),
        "post-calculus presentation should keep reciprocal-root form: {result}"
    );
    let expected = parse(
        "-3/(sqrt(2*x+5)*(sqrt(2*x+5)+2)^2)",
        &mut engine.simplifier.context,
    )
    .expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "expected compact reciprocal shifted-root derivative, got: {result}"
    );

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert_eq!(
        required,
        vec!["x > -5/2".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert!(
        output
            .steps
            .iter()
            .all(|step| step.rule_name.as_str() != "Racionalizar el denominador"),
        "commuted positive shifted sqrt reciprocal should take the direct diff route"
    );
}
#[test]
fn sqrt_over_positive_shifted_sqrt_diff_avoids_rationalized_domain_hole() {
    let cases = [
        (
            "diff(sqrt(x)/(sqrt(x)+1), x)",
            "1 / (2 * sqrt(x) * (sqrt(x) + 1)^2)",
            "1/(2*sqrt(x)*(sqrt(x)+1)^2)",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(2*sqrt(3-2*x)/(sqrt(3-2*x)+2), x)",
            "-4 / (sqrt(3 - 2 * x) * (sqrt(3 - 2 * x) + 2)^2)",
            "-4/(sqrt(3-2*x)*(sqrt(3-2*x)+2)^2)",
            vec!["x < 3/2".to_string()],
        ),
    ];

    for (input, expected_display, expected_expr, expected_conditions) in cases {
        let mut engine = Engine::new();
        let mut state = SessionState::new();
        state.options_mut().steps_mode = StepsMode::On;
        let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

        let output = engine
            .eval(
                &mut state,
                EvalRequest {
                    raw_input: input.to_string(),
                    parsed,
                    action: EvalAction::Simplify,
                    auto_store: false,
                },
            )
            .expect("eval failed");

        let result_expr = match output.result {
            EvalResult::Expr(expr) => expr,
            other => panic!("expected expression result, got {other:?}"),
        };
        let result = format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result_expr,
            }
        );

        assert_eq!(result, expected_display, "input: {input}");
        assert!(
            !result.contains("x - 1") && !result.contains("2 * x + 1"),
            "diff should avoid rationalized denominator artifacts: {result}"
        );
        assert!(
            !result.contains("^(-1/2)") && !result.contains(")^(1/2) /"),
            "post-calculus presentation should keep reciprocal-root form: {result}"
        );

        let expected =
            parse(expected_expr, &mut engine.simplifier.context).expect("parse expected");
        assert!(
            engine.simplifier.are_equivalent(result_expr, expected),
            "expected compact shifted-root quotient derivative for {input}, got: {result}"
        );

        let required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();

        assert_eq!(
            required, expected_conditions,
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert_no_redundant_post_calculus_presentation_round_trip(&output.steps);
    }
}
#[test]
fn sqrt_over_positive_shifted_sqrt_diff_residual_collapses() {
    let cases = [
        (
            "diff(sqrt(x)/(sqrt(x)+1), x) - 1/(2*sqrt(x)*(sqrt(x)+1)^2)",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(2*sqrt(3-2*x)/(sqrt(3-2*x)+2), x) - (-4/(sqrt(3-2*x)*(sqrt(3-2*x)+2)^2))",
            vec!["x < 3/2".to_string()],
        ),
        (
            "diff(2*sqrt(3-2*x)/(sqrt(3-2*x)+2), x) + 4/(sqrt(3-2*x)*(sqrt(3-2*x)+2)^2)",
            vec!["x < 3/2".to_string()],
        ),
        (
            "diff(sqrt(x)/(sqrt(x)+1), x) + (-1/(2*sqrt(x)*(sqrt(x)+1)^2))",
            vec!["x > 0".to_string()],
        ),
    ];

    for (input, expected_conditions) in cases {
        let mut engine = Engine::new();
        let mut state = SessionState::new();
        state.options_mut().steps_mode = StepsMode::On;
        let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

        let output = engine
            .eval(
                &mut state,
                EvalRequest {
                    raw_input: input.to_string(),
                    parsed,
                    action: EvalAction::Simplify,
                    auto_store: false,
                },
            )
            .expect("eval failed");

        let result = match output.result {
            EvalResult::Expr(expr) => format!(
                "{}",
                DisplayExpr {
                    context: &engine.simplifier.context,
                    id: expr,
                }
            ),
            other => panic!("expected expression result, got {other:?}"),
        };

        assert_eq!(result, "0", "input: {input}");

        let required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();

        assert_eq!(
            required, expected_conditions,
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert!(
            output
                .steps
                .iter()
                .any(|step| step.rule_name == "Post-calculus residual simplification"),
            "input: {input}, expected direct post-calculus residual simplification step"
        );
    }
}
#[test]
fn affine_linear_over_sqrt_x_diff_uses_post_calculus_root_denominator_presentation() {
    for (input, expected_render, residual_input, expected_required) in [
        (
            "diff((x+1)/sqrt(x), x)",
            "(x - 1) / (2 * x * sqrt(x))",
            "diff((x+1)/sqrt(x), x) - (x-1)/(2*x*sqrt(x))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff((x+2)/sqrt(x), x)",
            "(x - 2) / (2 * x * sqrt(x))",
            "diff((x+2)/sqrt(x), x) - (x-2)/(2*x*sqrt(x))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff((2*x+1)/sqrt(x), x)",
            "(2 * x - 1) / (2 * x * sqrt(x))",
            "diff((2*x+1)/sqrt(x), x) - (2*x-1)/(2*x*sqrt(x))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff((3*x+2)/sqrt(x), x)",
            "(3 * x - 2) / (2 * x * sqrt(x))",
            "diff((3*x+2)/sqrt(x), x) - (3*x-2)/(2*x*sqrt(x))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff((x^2+1)/sqrt(x), x)",
            "(3 * x^2 - 1) / (2 * x * sqrt(x))",
            "diff((x^2+1)/sqrt(x), x) - (3*x^2-1)/(2*x*sqrt(x))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff((x^3+1)/sqrt(x), x)",
            "(5 * x^3 - 1) / (2 * x * sqrt(x))",
            "diff((x^3+1)/sqrt(x), x) - (5*x^3-1)/(2*x*sqrt(x))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff((x+1)/sqrt(x+2), x)",
            "(x + 3) / (2 * (x + 2) * sqrt(x + 2))",
            "diff((x+1)/sqrt(x+2), x) - (x+3)/(2*(x+2)*sqrt(x+2))",
            vec!["x > -2".to_string()],
        ),
        (
            "diff((2*x+1)/sqrt(3*x+2), x)",
            "(6 * x + 5) / (2 * (3 * x + 2) * sqrt(3 * x + 2))",
            "diff((2*x+1)/sqrt(3*x+2), x) - (6*x+5)/(2*(3*x+2)*sqrt(3*x+2))",
            vec!["x > -2/3".to_string()],
        ),
        (
            "diff((x^2+1)/sqrt(x+1), x)",
            "(3 * x^2 + 4 * x - 1) / (2 * (x + 1) * sqrt(x + 1))",
            "diff((x^2+1)/sqrt(x+1), x) - (3*x^2+4*x-1)/(2*(x+1)*sqrt(x+1))",
            vec!["x > -1".to_string()],
        ),
        (
            "diff((x^3+1)/sqrt(x+1), x)",
            "(5 * x^2 + x - 1) / (2 * sqrt(x + 1))",
            "diff((x^3+1)/sqrt(x+1), x) - (5*x^2+x-1)/(2*sqrt(x+1))",
            vec!["x > -1".to_string()],
        ),
        (
            "diff((2*x^2+x+1)/sqrt(3*x+2), x)",
            "(18 * x^2 + 19 * x + 1) / (2 * (3 * x + 2) * sqrt(3 * x + 2))",
            "diff((2*x^2+x+1)/sqrt(3*x+2), x) - (18*x^2+19*x+1)/(2*(3*x+2)*sqrt(3*x+2))",
            vec!["x > -2/3".to_string()],
        ),
    ] {
        let mut engine = Engine::new();
        let mut state = SessionState::new();
        state.options_mut().steps_mode = StepsMode::On;
        let parsed = parse(input, &mut engine.simplifier.context).expect("parse input");

        let output = engine
            .eval(
                &mut state,
                EvalRequest {
                    raw_input: input.to_string(),
                    parsed,
                    action: EvalAction::Simplify,
                    auto_store: false,
                },
            )
            .expect("eval failed");
        let result_expr = match output.result {
            EvalResult::Expr(expr) => expr,
            other => panic!("expected expression result, got {other:?}"),
        };
        let result = format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result_expr,
            }
        );
        assert_eq!(result, expected_render, "input: {input}");
        assert!(
            !result.contains("^(-1/2)") && !result.contains("^(-3/2)"),
            "presentation should hide reciprocal-power internals for {input}: {result}"
        );

        let required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();
        assert_eq!(
            required, expected_required,
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert!(
            output
                .steps
                .iter()
                .any(|step| step.rule_name == "Present calculus result in compact form"),
            "expected compact post-calculus presentation step for {input}"
        );

        let mut residual_engine = Engine::new();
        let mut residual_state = SessionState::new();
        let residual =
            parse(residual_input, &mut residual_engine.simplifier.context).expect("parse residual");
        let residual_output = residual_engine
            .eval(
                &mut residual_state,
                EvalRequest {
                    raw_input: residual_input.to_string(),
                    parsed: residual,
                    action: EvalAction::Simplify,
                    auto_store: false,
                },
            )
            .expect("eval residual");
        let residual_expr = match residual_output.result {
            EvalResult::Expr(expr) => expr,
            other => panic!("expected residual expression result, got {other:?}"),
        };
        let residual_result = format!(
            "{}",
            DisplayExpr {
                context: &residual_engine.simplifier.context,
                id: residual_expr,
            }
        );
        assert_eq!(
            residual_result, "0",
            "residual did not collapse for {input}"
        );
    }
}
#[test]
fn square_root_affine_diff_evaluates_with_positive_radicand_condition() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(sqrt(x+1), x)";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    let result = match output.result {
        EvalResult::Expr(expr) => format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: expr,
            }
        ),
        other => panic!("expected expression result, got {other:?}"),
    };

    assert_eq!(result, "1 / (2 * sqrt(x + 1))");

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert_eq!(
        required,
        vec!["x > -1".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn square_root_quadratic_diff_evaluates_without_redundant_domain_condition() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(sqrt(x^2+1), x)";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    let result = match output.result {
        EvalResult::Expr(expr) => format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: expr,
            }
        ),
        other => panic!("expected expression result, got {other:?}"),
    };

    assert_eq!(result, "x / sqrt(x^2 + 1)");

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn square_root_convex_quadratic_diff_displays_exterior_domain_interval() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(sqrt(x^2-1), x)";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    let result = match output.result {
        EvalResult::Expr(expr) => format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: expr,
            }
        ),
        other => panic!("expected expression result, got {other:?}"),
    };

    assert_eq!(result, "x / sqrt(x^2 - 1)");

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert_eq!(
        required,
        vec!["x < -1 or x > 1".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn square_root_convex_quadratic_diff_displays_surd_exterior_domain_interval() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(sqrt(x^2-2), x)";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    let result = match output.result {
        EvalResult::Expr(expr) => format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: expr,
            }
        ),
        other => panic!("expected expression result, got {other:?}"),
    };

    assert_eq!(result, "x / sqrt(x^2 - 2)");

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert_eq!(
        required,
        vec!["x < -sqrt(2) or x > sqrt(2)".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn square_root_shifted_convex_quadratic_diff_displays_surd_exterior_domain_interval() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(sqrt(x^2+2*x-1), x)";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    let result = match output.result {
        EvalResult::Expr(expr) => format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: expr,
            }
        ),
        other => panic!("expected expression result, got {other:?}"),
    };

    assert_eq!(result, "(x + 1) / sqrt(x^2 + 2 * x - 1)");

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert_eq!(
        required,
        vec!["x < -1 - sqrt(2) or x > -1 + sqrt(2)".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn elementary_sqrt_chain_rule_diff_uses_explicit_root_denominator_presentation() {
    for (input, expected_render, expected_required) in [
        (
            "diff(exp(sqrt(x)), x)",
            "e^sqrt(x) / (2 * sqrt(x))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(exp(sqrt(x+1)), x)",
            "e^sqrt(x + 1) / (2 * sqrt(x + 1))",
            vec!["x > -1".to_string()],
        ),
        (
            "diff(sin(sqrt(x)), x)",
            "cos(sqrt(x)) / (2 * sqrt(x))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(cos(sqrt(x)), x)",
            "-sin(sqrt(x)) / (2 * sqrt(x))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(tan(sqrt(x)), x)",
            "1 / (2 * sqrt(x) * cos(sqrt(x))^2)",
            vec!["x > 0".to_string(), "cos(sqrt(x)) ≠ 0".to_string()],
        ),
        (
            "diff(cot(sqrt(x)), x)",
            "-1 / (2 * sqrt(x) * sin(sqrt(x))^2)",
            vec!["x > 0".to_string(), "sin(sqrt(x)) ≠ 0".to_string()],
        ),
        (
            "diff(sec(sqrt(x)), x)",
            "sec(sqrt(x)) * tan(sqrt(x)) / (2 * sqrt(x))",
            vec!["x > 0".to_string(), "cos(sqrt(x)) ≠ 0".to_string()],
        ),
        (
            "diff(-sec(sqrt(x)), x)",
            "-sec(sqrt(x)) * tan(sqrt(x)) / (2 * sqrt(x))",
            vec!["x > 0".to_string(), "cos(sqrt(x)) ≠ 0".to_string()],
        ),
        (
            "diff(1/cos(sqrt(x)), x)",
            "sec(sqrt(x)) * tan(sqrt(x)) / (2 * sqrt(x))",
            vec!["x > 0".to_string(), "cos(sqrt(x)) ≠ 0".to_string()],
        ),
        (
            "diff(sec(sqrt(x+1)), x)",
            "sec(sqrt(x + 1)) * tan(sqrt(x + 1)) / (2 * sqrt(x + 1))",
            vec!["x > -1".to_string(), "cos(sqrt(x + 1)) ≠ 0".to_string()],
        ),
        (
            "diff(csc(sqrt(x)), x)",
            "-csc(sqrt(x)) * cot(sqrt(x)) / (2 * sqrt(x))",
            vec!["x > 0".to_string(), "sin(sqrt(x)) ≠ 0".to_string()],
        ),
        (
            "diff(-csc(sqrt(x)), x)",
            "csc(sqrt(x)) * cot(sqrt(x)) / (2 * sqrt(x))",
            vec!["x > 0".to_string(), "sin(sqrt(x)) ≠ 0".to_string()],
        ),
        (
            "diff(1/sin(sqrt(x)), x)",
            "-csc(sqrt(x)) * cot(sqrt(x)) / (2 * sqrt(x))",
            vec!["x > 0".to_string(), "sin(sqrt(x)) ≠ 0".to_string()],
        ),
        (
            "diff(ln(sec(sqrt(x))), x)",
            "tan(sqrt(x)) / (2 * sqrt(x))",
            vec!["cos(sqrt(x)) > 0".to_string(), "x > 0".to_string()],
        ),
        (
            "diff(ln(csc(sqrt(x))), x)",
            "-cot(sqrt(x)) / (2 * sqrt(x))",
            vec!["sin(sqrt(x)) > 0".to_string(), "x > 0".to_string()],
        ),
        (
            "diff(sinh(sqrt(x)), x)",
            "cosh(sqrt(x)) / (2 * sqrt(x))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(cosh(sqrt(x)), x)",
            "sinh(sqrt(x)) / (2 * sqrt(x))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(tanh(sqrt(x)), x)",
            "1 / (2 * sqrt(x) * cosh(sqrt(x))^2)",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(ln(cosh(sqrt(x))), x)",
            "tanh(sqrt(x)) / (2 * sqrt(x))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(ln(cosh(sqrt(3*x+1))), x)",
            "3 * tanh(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))",
            vec!["x > -1/3".to_string()],
        ),
        (
            "diff(ln(1/cosh(sqrt(x))), x)",
            "-tanh(sqrt(x)) / (2 * sqrt(x))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(ln(1/cosh(sqrt(3*x+1))), x)",
            "-3 * tanh(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))",
            vec!["x > -1/3".to_string()],
        ),
        (
            "diff(ln(abs(sinh(sqrt(x)))), x)",
            "1 / (2 * tanh(sqrt(x)) * sqrt(x))",
            vec!["sinh(sqrt(x)) ≠ 0".to_string(), "x > 0".to_string()],
        ),
        (
            "diff(ln(abs(sinh(sqrt(3*x+1)))), x)",
            "3 / (2 * tanh(sqrt(3 * x + 1)) * sqrt(3 * x + 1))",
            vec![
                "sinh(sqrt(3 * x + 1)) ≠ 0".to_string(),
                "x > -1/3".to_string(),
            ],
        ),
        (
            "diff(ln(1/sinh(sqrt(x))), x)",
            "-1 / (2 * tanh(sqrt(x)) * sqrt(x))",
            vec!["sinh(sqrt(x)) > 0".to_string(), "x > 0".to_string()],
        ),
        (
            "diff(ln(1/sinh(sqrt(3*x+1))), x)",
            "-3 / (2 * tanh(sqrt(3 * x + 1)) * sqrt(3 * x + 1))",
            vec![
                "sinh(sqrt(3 * x + 1)) > 0".to_string(),
                "x > -1/3".to_string(),
            ],
        ),
        (
            "diff(-1/cosh(sqrt(3*x+1)), x)",
            "3 * sinh(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1) * cosh(sqrt(3 * x + 1))^2)",
            vec!["x > -1/3".to_string()],
        ),
        (
            "diff(-1/sinh(sqrt(3*x+1)), x)",
            "3 * cosh(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1) * sinh(sqrt(3 * x + 1))^2)",
            vec![
                "x > -1/3".to_string(),
                "sinh(sqrt(3 * x + 1)) ≠ 0".to_string(),
            ],
        ),
        (
            "diff(sin(sqrt(2*x)), x)",
            "cos(sqrt(2 * x)) / sqrt(2 * x)",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(tan(sqrt(2*x)), x)",
            "1 / (sqrt(2 * x) * cos(sqrt(2 * x))^2)",
            vec!["cos(sqrt(2 * x)) ≠ 0".to_string(), "x > 0".to_string()],
        ),
        (
            "diff(cot(sqrt(2*x)), x)",
            "-1 / (sqrt(2 * x) * sin(sqrt(2 * x))^2)",
            vec!["sin(sqrt(2 * x)) ≠ 0".to_string(), "x > 0".to_string()],
        ),
        (
            "diff(sec(sqrt(2*x)), x)",
            "sec(sqrt(2 * x)) * tan(sqrt(2 * x)) / sqrt(2 * x)",
            vec!["cos(sqrt(2 * x)) ≠ 0".to_string(), "x > 0".to_string()],
        ),
        (
            "diff(csc(sqrt(2*x)), x)",
            "-csc(sqrt(2 * x)) * cot(sqrt(2 * x)) / sqrt(2 * x)",
            vec!["sin(sqrt(2 * x)) ≠ 0".to_string(), "x > 0".to_string()],
        ),
    ] {
        let mut engine = Engine::new();
        let mut state = SessionState::new();
        state.options_mut().steps_mode = StepsMode::On;
        let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

        let req = EvalRequest {
            raw_input: input.to_string(),
            parsed,
            action: EvalAction::Simplify,
            auto_store: false,
        };

        let output = engine.eval(&mut state, req).expect("eval failed");
        let result_expr = match output.result {
            EvalResult::Expr(expr) => expr,
            other => panic!("expected expression result, got {other:?}"),
        };
        let result = format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result_expr,
            }
        );

        assert_eq!(result, expected_render, "input: {input}");
        assert!(
            !result.contains("^(-1/2)") && !result.contains("^(1/2)"),
            "presentation should keep explicit sqrt forms, got: {result}"
        );

        let expected = parse(expected_render, &mut engine.simplifier.context)
            .unwrap_or_else(|err| panic!("parse expected {expected_render}: {err}"));
        assert!(
            engine.simplifier.are_equivalent(result_expr, expected),
            "input: {input}, expected derivative equivalent to {expected_render}, got: {result}"
        );

        let required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();

        assert_eq!(
            required, expected_required,
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert!(
            output
                .steps
                .iter()
                .any(|step| step.rule_name == "Symbolic Differentiation"),
            "expected the derivative to keep the ordinary symbolic differentiation trace"
        );
    }
}
#[test]
fn square_root_negative_affine_diff_evaluates_with_positive_radicand_condition() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(sqrt(2-x), x)";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    let result = match output.result {
        EvalResult::Expr(expr) => format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: expr,
            }
        ),
        other => panic!("expected expression result, got {other:?}"),
    };

    assert_eq!(result, "-1 / (2 * sqrt(2 - x))");

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert_eq!(
        required,
        vec!["x < 2".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn square_root_bounded_quadratic_diff_evaluates_with_positive_radicand_condition() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(sqrt(1-x^2), x)";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    let result = match output.result {
        EvalResult::Expr(expr) => format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: expr,
            }
        ),
        other => panic!("expected expression result, got {other:?}"),
    };

    assert_eq!(result, "-x / sqrt(1 - x^2)");

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert_eq!(
        required,
        vec!["-1 < x < 1".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn ln_positive_sqrt_shift_diff_keeps_chain_rule_presentation_without_removable_pole() {
    let cases = [
        (
            "diff(ln(sqrt(x)+1), x)",
            "1 / (2 * sqrt(x) * (sqrt(x) + 1))",
        ),
        (
            "diff(ln(1+sqrt(x)), x)",
            "1 / (2 * sqrt(x) * (sqrt(x) + 1))",
        ),
        (
            "diff(ln(sqrt(2*x)+1), x)",
            "1 / (sqrt(2 * x) * (sqrt(2 * x) + 1))",
        ),
        (
            "diff(ln(sqrt(x)+2), x)",
            "1 / (2 * sqrt(x) * (sqrt(x) + 2))",
        ),
        (
            "diff(ln(2+sqrt(x)), x)",
            "1 / (2 * sqrt(x) * (sqrt(x) + 2))",
        ),
        (
            "diff(ln(sqrt(x)+1/2), x)",
            "1 / (2 * sqrt(x) * (sqrt(x) + 1/2))",
        ),
    ];

    for (input, expected_derivative) in cases {
        let mut engine = Engine::new();
        let mut state = SessionState::new();
        state.options_mut().steps_mode = StepsMode::On;
        let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

        let req = EvalRequest {
            raw_input: input.to_string(),
            parsed,
            action: EvalAction::Simplify,
            auto_store: false,
        };

        let output = engine.eval(&mut state, req).expect("eval failed");
        let result_expr = match output.result {
            EvalResult::Expr(expr) => expr,
            other => panic!("expected expression result, got {other:?}"),
        };
        let result = format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result_expr,
            }
        );

        assert_eq!(result, expected_derivative, "unexpected result for {input}");
        assert!(
            !result.contains("x - 1"),
            "presentation should avoid a removable x - 1 pole: {result}"
        );
        assert!(
            !result.contains("x^(-1/2)"),
            "presentation should keep reciprocal-root form: {result}"
        );

        let expected =
            parse(expected_derivative, &mut engine.simplifier.context).expect("parse expected");
        assert!(
            engine.simplifier.are_equivalent(result_expr, expected),
            "post-calculus presentation must stay equivalent for {input}, got: {result}"
        );

        let required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();

        assert_eq!(
            required,
            vec!["x > 0".to_string()],
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert!(
            !required.iter().any(|cond| cond.contains("x - 1")),
            "required conditions should not include removable pole for {input}: {required:?}"
        );
    }
}
#[test]
fn ln_sum_of_equal_derivative_roots_diff_uses_compact_root_product_presentation() {
    let cases = [
        (
            "diff(ln(sqrt(x)+sqrt(x+1)), x)",
            "1 / (2 * sqrt(x) * sqrt(x + 1))",
            "1/(2*sqrt(x)*sqrt(x+1))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(ln(sqrt(x)+sqrt(x-1)), x)",
            "1 / (2 * sqrt(x) * sqrt(x - 1))",
            "1/(2*sqrt(x)*sqrt(x-1))",
            vec!["x > 1".to_string()],
        ),
        (
            "diff(ln(sqrt(2*x+1)+sqrt(2*x+3)), x)",
            "1 / (sqrt(2 * x + 1) * sqrt(2 * x + 3))",
            "1/(sqrt(2*x+1)*sqrt(2*x+3))",
            vec!["x > -1/2".to_string()],
        ),
        (
            "diff(ln(sqrt(x^2+1)+sqrt(x^2+2)), x)",
            "x / (sqrt(x^2 + 1) * sqrt(x^2 + 2))",
            "x/(sqrt(x^2+1)*sqrt(x^2+2))",
            Vec::new(),
        ),
    ];

    for (input, expected_display, expected_expr, expected_conditions) in cases {
        let mut engine = Engine::new();
        let mut state = SessionState::new();
        state.options_mut().steps_mode = StepsMode::On;
        let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

        let output = engine
            .eval(
                &mut state,
                EvalRequest {
                    raw_input: input.to_string(),
                    parsed,
                    action: EvalAction::Simplify,
                    auto_store: false,
                },
            )
            .expect("eval failed");

        let result_expr = match output.result {
            EvalResult::Expr(expr) => expr,
            other => panic!("expected expression result, got {other:?}"),
        };
        let result = format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result_expr,
            }
        );

        assert_eq!(result, expected_display, "input: {input}");
        assert!(
            !result.contains("^(-1/2)") && !result.contains(")^(1/2)"),
            "presentation should keep explicit root-product denominator: {result}"
        );
        let expected =
            parse(expected_expr, &mut engine.simplifier.context).expect("parse expected");
        assert!(
            engine.simplifier.are_equivalent(result_expr, expected),
            "input: {input}, expected equivalent compact root-product derivative, got: {result}"
        );

        let required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();

        assert_eq!(
            required, expected_conditions,
            "input: {input}, unexpected required_conditions: {required:?}"
        );
        assert!(
            output
                .steps
                .iter()
                .any(|step| step.rule_name == "Symbolic Differentiation"),
            "input: {input}, expected the ordinary symbolic differentiation trace"
        );
    }
}
#[test]
fn ln_sum_of_equal_derivative_roots_diff_residual_collapses() {
    let cases = [
        (
            "diff(ln(sqrt(x)+sqrt(x+1)), x) - 1/(2*sqrt(x)*sqrt(x+1))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(ln(sqrt(x)+sqrt(x-1)), x) - 1/(2*sqrt(x)*sqrt(x-1))",
            vec!["x > 1".to_string()],
        ),
        (
            "diff(ln(sqrt(2*x+1)+sqrt(2*x+3)), x) - 1/(sqrt(2*x+1)*sqrt(2*x+3))",
            vec!["x > -1/2".to_string()],
        ),
        (
            "diff(ln(sqrt(x^2+1)+sqrt(x^2+2)), x) - x/(sqrt(x^2+1)*sqrt(x^2+2))",
            Vec::new(),
        ),
    ];

    for (input, expected_conditions) in cases {
        let mut engine = Engine::new();
        let mut state = SessionState::new();
        let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

        let output = engine
            .eval(
                &mut state,
                EvalRequest {
                    raw_input: input.to_string(),
                    parsed,
                    action: EvalAction::Simplify,
                    auto_store: false,
                },
            )
            .expect("eval failed");

        let result = match output.result {
            EvalResult::Expr(expr) => format!(
                "{}",
                DisplayExpr {
                    context: &engine.simplifier.context,
                    id: expr,
                }
            ),
            other => panic!("expected expression result, got {other:?}"),
        };

        assert_eq!(result, "0", "input: {input}");

        let required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();

        assert_eq!(
            required, expected_conditions,
            "input: {input}, unexpected required_conditions: {required:?}"
        );
    }
}
#[test]
fn ln_positive_quadratic_sqrt_shift_diff_does_not_add_redundant_conditions() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let input = "diff(ln(sqrt(x^2+1)+3), x)";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    let result_expr = match output.result {
        EvalResult::Expr(expr) => expr,
        other => panic!("expected expression result, got {other:?}"),
    };
    let result = format!(
        "{}",
        DisplayExpr {
            context: &engine.simplifier.context,
            id: result_expr,
        }
    );

    assert_eq!(result, "x / (sqrt(x^2 + 1) * (sqrt(x^2 + 1) + 3))");
    assert!(
        !result.contains("x^(-1/2)"),
        "presentation should keep reciprocal-root form: {result}"
    );

    let expected = parse(
        "x/(sqrt(x^2+1)*(sqrt(x^2+1)+3))",
        &mut engine.simplifier.context,
    )
    .expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "post-calculus presentation must stay equivalent, got: {result}"
    );

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert!(
        required.is_empty(),
        "strictly positive radicand plus positive shift should not add conditions: {required:?}"
    );
}
#[test]
fn ln_negative_sqrt_shift_diff_keeps_stronger_log_domain_boundary() {
    let cases = [
        (
            "diff(ln(sqrt(x)-1), x)",
            "1 / (2 * sqrt(x) * (sqrt(x) - 1))",
            vec!["x > 1".to_string()],
        ),
        (
            "diff(ln(-1+sqrt(x)), x)",
            "1 / (2 * sqrt(x) * (sqrt(x) - 1))",
            vec!["x > 1".to_string()],
        ),
        (
            "diff(ln(sqrt(2*x)-1), x)",
            "1 / (sqrt(2 * x) * (sqrt(2 * x) - 1))",
            vec!["x > 1/2".to_string()],
        ),
        (
            "diff(ln(sqrt(x)-2), x)",
            "1 / (2 * sqrt(x) * (sqrt(x) - 2))",
            vec!["x > 4".to_string()],
        ),
        (
            "diff(ln(-2+sqrt(x)), x)",
            "1 / (2 * sqrt(x) * (sqrt(x) - 2))",
            vec!["x > 4".to_string()],
        ),
        (
            "diff(ln(sqrt(2*x)-2), x)",
            "1 / (sqrt(2 * x) * (sqrt(2 * x) - 2))",
            vec!["x > 2".to_string()],
        ),
        (
            "diff(ln(sqrt(x^2+4)-2), x)",
            "x / (sqrt(x^2 + 4) * (sqrt(x^2 + 4) - 2))",
            vec!["x ≠ 0".to_string()],
        ),
        (
            "diff(ln(sqrt((2*x+1)^2+4)-2), x)",
            "2 * (2 * x + 1) / (sqrt((2 * x + 1)^2 + 4) * (sqrt((2 * x + 1)^2 + 4) - 2))",
            vec!["x ≠ -1/2".to_string()],
        ),
    ];

    for (input, expected_derivative, expected_required) in cases {
        let mut engine = Engine::new();
        let mut state = SessionState::new();
        state.options_mut().steps_mode = StepsMode::On;
        let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

        let req = EvalRequest {
            raw_input: input.to_string(),
            parsed,
            action: EvalAction::Simplify,
            auto_store: false,
        };

        let output = engine.eval(&mut state, req).expect("eval failed");
        let result_expr = match output.result {
            EvalResult::Expr(expr) => expr,
            other => panic!("expected expression result, got {other:?}"),
        };
        let result = format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result_expr,
            }
        );

        assert_eq!(result, expected_derivative, "unexpected result for {input}");
        assert!(
            !result.contains("x^(-1/2)") && !result.contains("2 * x - 2"),
            "negative shifted sqrt log should use compact chain-rule presentation: {result}"
        );

        let expected =
            parse(expected_derivative, &mut engine.simplifier.context).expect("parse expected");
        assert!(
            engine.simplifier.are_equivalent(result_expr, expected),
            "post-calculus presentation must stay equivalent for {input}, got: {result}"
        );

        let required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();

        assert_eq!(
            required, expected_required,
            "negative shifted sqrt log should keep the stronger log-domain boundary for {input}: {required:?}"
        );
    }
}
#[test]
fn variable_sqrt_square_base_log_abs_diff_normalizes_base_not_one_condition() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(log(sqrt(x^2), abs(x^2-1)), x)";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    let result_expr = match output.result {
        EvalResult::Expr(expr) => expr,
        other => panic!("expected expression result, got {other:?}"),
    };
    let result = format!(
        "{}",
        DisplayExpr {
            context: &engine.simplifier.context,
            id: result_expr,
        }
    );

    assert!(!result.contains("diff("), "got: {result}");

    let expected = parse(
        "((2*x/(x^2-1))*ln(abs(x))-ln(abs(x^2-1))/x)/(ln(abs(x))^2)",
        &mut engine.simplifier.context,
    )
    .expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "expected derivative equivalent to variable abs-base log(abs(u)) rule, got: {result}"
    );

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert_eq!(
        required,
        vec![
            "x ≠ 0".to_string(),
            "x ≠ 1".to_string(),
            "x ≠ -1".to_string()
        ],
        "unexpected required_conditions: {required:?}"
    );
    assert!(
        !required.iter().any(|cond| cond.contains("sqrt(")),
        "sqrt-square base-not-one condition should be normalized: {required:?}"
    );
}
#[test]
fn variable_sqrt_even_power_base_log_abs_diff_normalizes_positive_base_condition() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(log(sqrt((x^2-1)^2), abs(x)), x)";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    let result_expr = match output.result {
        EvalResult::Expr(expr) => expr,
        other => panic!("expected expression result, got {other:?}"),
    };
    let result = format!(
        "{}",
        DisplayExpr {
            context: &engine.simplifier.context,
            id: result_expr,
        }
    );

    assert!(!result.contains("diff("), "got: {result}");

    let expected = parse(
        "(ln(abs(x^2-1))*(x^2-1)-2*ln(abs(x))*x^2)/(x*(x^2*ln(abs(x^2-1))^2-ln(abs(x^2-1))^2))",
        &mut engine.simplifier.context,
    )
    .expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "expected derivative equivalent to variable sqrt-even-power base log(abs(u)) rule, got: {result}"
    );

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert_eq!(
        required,
        vec![
            "x ≠ 1".to_string(),
            "x ≠ -1".to_string(),
            "x ≠ 0".to_string(),
            "x^2 - 2 ≠ 0".to_string()
        ],
        "unexpected required_conditions: {required:?}"
    );
    assert!(
        !required.iter().any(|cond| cond.contains("sqrt(")),
        "sqrt-even-power positivity condition should be normalized: {required:?}"
    );
}
#[test]
fn variable_half_power_even_base_log_abs_diff_normalizes_sqrt_like_conditions() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(log(((x^2-1)^2)^(1/2), abs(x)), x)";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    let result_expr = match output.result {
        EvalResult::Expr(expr) => expr,
        other => panic!("expected expression result, got {other:?}"),
    };
    let result = format!(
        "{}",
        DisplayExpr {
            context: &engine.simplifier.context,
            id: result_expr,
        }
    );

    assert!(!result.contains("diff("), "got: {result}");

    let expected = parse(
        "(ln(abs(x^2-1))*(x^2-1)-2*ln(abs(x))*x^2)/(x*(x^2*ln(abs(x^2-1))^2-ln(abs(x^2-1))^2))",
        &mut engine.simplifier.context,
    )
    .expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "expected derivative equivalent to variable half-power base log(abs(u)) rule, got: {result}"
    );

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert_eq!(
        required,
        vec![
            "x ≠ 1".to_string(),
            "x ≠ -1".to_string(),
            "x ≠ 0".to_string(),
            "x^2 - 2 ≠ 0".to_string()
        ],
        "unexpected required_conditions: {required:?}"
    );
    assert!(
        !required
            .iter()
            .any(|cond| cond.contains("^(1 / 2)") || cond.contains("sqrt(")),
        "sqrt-like half-power conditions should be normalized: {required:?}"
    );
}
#[test]
fn inverse_root_diff_drops_powered_nonzero_guard_under_positive_base() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(sqrt(1/y), y)";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    let result = match output.result {
        EvalResult::Expr(expr) => format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: expr,
            }
        ),
        other => panic!("expected expression result, got {other:?}"),
    };

    assert!(!result.contains("diff("), "got: {result}");
    assert_ne!(result, "0", "inverse-root derivative collapsed to zero");

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert_eq!(
        required,
        vec!["y > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn even_power_quotient_root_diff_expands_positive_quotient_domain_conditions() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(sqrt(x^2/y), y)";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    let result = match output.result {
        EvalResult::Expr(expr) => format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: expr,
            }
        ),
        other => panic!("expected expression result, got {other:?}"),
    };

    assert!(!result.contains("diff("), "got: {result}");
    assert_ne!(
        result, "0",
        "even-power quotient derivative collapsed to zero"
    );

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert_eq!(
        required.len(),
        2,
        "unexpected required_conditions: {required:?}"
    );
    for expected_condition in ["x ≠ 0", "y > 0"] {
        assert!(
            required.iter().any(|cond| cond == expected_condition),
            "missing {expected_condition}; required_conditions: {required:?}"
        );
    }
    assert!(
        !required.iter().any(|cond| cond.contains("x^2 / y")),
        "unexpected composite quotient condition: {required:?}"
    );
}
#[test]
fn even_power_denominator_root_diff_expands_positive_quotient_domain_conditions() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(sqrt(x/y^2), x)";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    let result = match output.result {
        EvalResult::Expr(expr) => format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: expr,
            }
        ),
        other => panic!("expected expression result, got {other:?}"),
    };

    assert!(!result.contains("diff("), "got: {result}");
    assert_ne!(
        result, "0",
        "even-power denominator quotient derivative collapsed to zero"
    );

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert_eq!(
        required.len(),
        2,
        "unexpected required_conditions: {required:?}"
    );
    for expected_condition in ["x > 0", "y ≠ 0"] {
        assert!(
            required.iter().any(|cond| cond == expected_condition),
            "missing {expected_condition}; required_conditions: {required:?}"
        );
    }
    assert!(
        !required.iter().any(|cond| cond.contains("x / y^2")),
        "unexpected composite quotient condition: {required:?}"
    );
}
#[test]
fn multiple_even_power_denominator_root_diff_expands_positive_quotient_domain_conditions() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(sqrt(x/(y^2*z^2)), x)";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    let result = match output.result {
        EvalResult::Expr(expr) => format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: expr,
            }
        ),
        other => panic!("expected expression result, got {other:?}"),
    };

    assert!(!result.contains("diff("), "got: {result}");
    assert_ne!(
        result, "0",
        "multiple even-power denominator quotient derivative collapsed to zero"
    );

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert_eq!(
        required.len(),
        3,
        "unexpected required_conditions: {required:?}"
    );
    for expected_condition in ["x > 0", "y ≠ 0", "z ≠ 0"] {
        assert!(
            required.iter().any(|cond| cond == expected_condition),
            "missing {expected_condition}; required_conditions: {required:?}"
        );
    }
    assert!(
        !required.iter().any(|cond| cond.contains("x / (y^2 * z^2)")),
        "unexpected composite quotient condition: {required:?}"
    );
}
#[test]
fn multiple_even_power_numerator_root_diff_expands_positive_product_domain_conditions() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(sqrt((x^2*z^2)/y), y)";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    let result = match output.result {
        EvalResult::Expr(expr) => format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: expr,
            }
        ),
        other => panic!("expected expression result, got {other:?}"),
    };

    assert!(!result.contains("diff("), "got: {result}");
    assert_ne!(
        result, "0",
        "multiple even-power numerator quotient derivative collapsed to zero"
    );

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert_eq!(
        required.len(),
        3,
        "unexpected required_conditions: {required:?}"
    );
    for expected_condition in ["y > 0", "x ≠ 0", "z ≠ 0"] {
        assert!(
            required.iter().any(|cond| cond == expected_condition),
            "missing {expected_condition}; required_conditions: {required:?}"
        );
    }
    assert!(
        !required.iter().any(|cond| cond.contains("x^2 * z^2")),
        "unexpected composite product condition: {required:?}"
    );
}
#[test]
fn shifted_even_power_denominator_root_diff_drops_expanded_composite_nonzero_condition() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(sqrt(x/(y^2*(z+1)^2)), x)";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    let result = match output.result {
        EvalResult::Expr(expr) => format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: expr,
            }
        ),
        other => panic!("expected expression result, got {other:?}"),
    };

    assert!(!result.contains("diff("), "got: {result}");
    assert_ne!(
        result, "0",
        "shifted even-power denominator quotient derivative collapsed to zero"
    );

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert_eq!(
        required.len(),
        3,
        "unexpected required_conditions: {required:?}"
    );
    for expected_condition in ["x > 0", "y ≠ 0", "z ≠ -1"] {
        assert!(
            required.iter().any(|cond| cond == expected_condition),
            "missing {expected_condition}; required_conditions: {required:?}"
        );
    }
    assert!(
        !required.iter().any(|cond| cond.contains("y^2 * z^2")),
        "unexpected composite product condition: {required:?}"
    );
}
#[test]
fn sqrt_log_diff_residual_collapses_with_function_base_power_merge() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(sqrt(ln(x)), x) - 1/(2*x*sqrt(ln(x)))";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let output = engine
        .eval(
            &mut state,
            EvalRequest {
                raw_input: input.to_string(),
                parsed,
                action: EvalAction::Simplify,
                auto_store: false,
            },
        )
        .expect("eval failed");

    let result = match output.result {
        EvalResult::Expr(expr) => format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: expr,
            }
        ),
        other => panic!("expected expression result, got {other:?}"),
    };

    assert_eq!(result, "0");

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert_eq!(
        required,
        vec!["ln(x) > 0".to_string(), "x > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn surd_quotient_diff_preserves_rational_content_scale() {
    let cases = [
        (
            "diff(arcsin(x/sqrt(1/2)), x)",
            "sqrt(2) / sqrt(1 - 2 * x^2)",
            "1/(sqrt(1/2)*sqrt(1-(x/sqrt(1/2))^2))",
            vec!["-sqrt(2)/2 < x < sqrt(2)/2"],
        ),
        (
            "diff(asinh(x/sqrt(1/2)), x)",
            "sqrt(2) / sqrt(2 * x^2 + 1)",
            "1/(sqrt(1/2)*sqrt(1+(x/sqrt(1/2))^2))",
            vec![],
        ),
        (
            "diff(atanh(x/sqrt(1/2)), x)",
            "sqrt(2) / (1 - 2 * x^2)",
            "1/(sqrt(1/2)*(1-(x/sqrt(1/2))^2))",
            vec!["-sqrt(2)/2 < x < sqrt(2)/2"],
        ),
        (
            "diff(arcsin(x/sqrt(8/9)), x)",
            "3 / sqrt(8 - 9 * x^2)",
            "1/(sqrt(8/9)*sqrt(1-(x/sqrt(8/9))^2))",
            vec!["-2*sqrt(2)/3 < x < 2*sqrt(2)/3"],
        ),
        (
            "diff(atanh(x/sqrt(12/25)), x)",
            "10 * sqrt(3) / (12 - 25 * x^2)",
            "1/(sqrt(12/25)*(1-(x/sqrt(12/25))^2))",
            vec!["-2*sqrt(3)/5 < x < 2*sqrt(3)/5"],
        ),
    ];

    for (input, expected_result, expected_chain_rule, expected_conditions) in cases {
        let mut engine = Engine::new();
        let mut state = SessionState::new();
        state.options_mut().steps_mode = StepsMode::Off;
        let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

        let req = EvalRequest {
            raw_input: input.to_string(),
            parsed,
            action: EvalAction::Simplify,
            auto_store: false,
        };

        let output = engine.eval(&mut state, req).expect("eval failed");
        let result_expr = match output.result {
            EvalResult::Expr(expr) => expr,
            other => panic!("expected expression result, got {other:?}"),
        };
        let result = format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result_expr,
            }
        );

        assert_eq!(result, expected_result, "input: {input}");
        if expected_result.contains("sqrt(2)") {
            assert!(
                result.contains("sqrt(2)") || result.contains("2^(1/2)"),
                "input: {input}, expected retained sqrt(2) scale, got: {result}"
            );
        }

        let expected =
            parse(expected_chain_rule, &mut engine.simplifier.context).expect("parse expected");
        for sample in [-0.25, 0.0, 0.25] {
            let mut vars = HashMap::new();
            vars.insert("x".to_string(), sample);
            let actual_value = eval_f64(&engine.simplifier.context, result_expr, &vars)
                .unwrap_or_else(|| panic!("input: {input}, could not eval result at x={sample}"));
            let expected_value = eval_f64(&engine.simplifier.context, expected, &vars)
                .unwrap_or_else(|| {
                    panic!("input: {input}, could not eval chain-rule form at x={sample}")
                });
            assert!(
                (actual_value - expected_value).abs() < 1e-10,
                "input: {input}, x={sample}, expected numeric chain-rule value {expected_value}, got {actual_value}"
            );
        }

        let required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();
        let expected_required: Vec<String> =
            expected_conditions.into_iter().map(String::from).collect();

        assert_eq!(
            required, expected_required,
            "input: {input}, unexpected required_conditions: {required:?}"
        );
    }
}
#[test]
fn exp_log_sqrt_reciprocal_sqrt_root_diff_stays_compact_and_fast() {
    for (input, expected_result, expected_required) in [
        (
            "diff(sqrt(exp(x)+ln(x)+sqrt(x)+1/sqrt(x)+x), x)",
            "(2 * sqrt(x) + 2 * x * sqrt(x) + 2 * x * sqrt(x) * e^x + x - 1) / (4 * x * sqrt(x) * sqrt(ln(x) + sqrt(x) + e^x + 1 / sqrt(x) + x))",
            vec![
                "x > 0".to_string(),
                "ln(x) + sqrt(x) + e^x + 1 / sqrt(x) + x > 0".to_string(),
            ],
        ),
        (
            "diff(arctan(sqrt(exp(x)+ln(x)+sqrt(x)+1/sqrt(x)+x)), x)",
            "(2 * sqrt(x) + 2 * x * sqrt(x) + 2 * x * sqrt(x) * e^x + x - 1) / (4 * x * sqrt(x) * sqrt(ln(x) + sqrt(x) + e^x + 1 / sqrt(x) + x) * (ln(x) + sqrt(x) + e^x + 1 / sqrt(x) + x + 1))",
            vec![
                "x > 0".to_string(),
                "ln(x) + sqrt(x) + e^x + 1 / sqrt(x) + x > 0".to_string(),
            ],
        ),
    ] {
        let mut engine = Engine::new();
        let mut state = SessionState::new();
        state.options_mut().steps_mode = StepsMode::On;
        let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

        let output = engine
            .eval(
                &mut state,
                EvalRequest {
                    raw_input: input.to_string(),
                    parsed,
                    action: EvalAction::Simplify,
                    auto_store: false,
                },
            )
            .expect("eval failed");

        let result = match output.result {
            EvalResult::Expr(expr) => format!(
                "{}",
                DisplayExpr {
                    context: &engine.simplifier.context,
                    id: expr,
                }
            ),
            other => panic!("expected expression result, got {other:?}"),
        };

        assert_eq!(result, expected_result, "input: {input}");
        assert!(
            output.domain_warnings.is_empty(),
            "warnings: {:?}",
            output.domain_warnings
        );

        let required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();
        assert_eq!(required, expected_required, "input: {input}");
    }
}
#[test]
fn exp_log_sqrt_reciprocal_sqrt_root_diff_residuals_collapse_before_cleanup() {
    for (input, expected_required) in [
        (
            "diff(sqrt(exp(x)+ln(x)+sqrt(x)+1/sqrt(x)+x), x) - (2*sqrt(x)+2*x*sqrt(x)+2*x*sqrt(x)*e^x+x-1)/(4*x*sqrt(x)*sqrt(exp(x)+ln(x)+sqrt(x)+1/sqrt(x)+x))",
            vec![
                "x > 0".to_string(),
                "ln(x) + sqrt(x) + e^x + 1 / sqrt(x) + x > 0".to_string(),
            ],
        ),
        (
            "diff(arctan(sqrt(exp(x)+ln(x)+sqrt(x)+1/sqrt(x)+x)), x) - (2*sqrt(x)+2*x*sqrt(x)+2*x*sqrt(x)*e^x+x-1)/(4*x*sqrt(x)*sqrt(exp(x)+ln(x)+sqrt(x)+1/sqrt(x)+x)*(exp(x)+ln(x)+sqrt(x)+1/sqrt(x)+x+1))",
            vec![
                "x > 0".to_string(),
                "ln(x) + sqrt(x) + e^x + 1 / sqrt(x) + x > 0".to_string(),
            ],
        ),
    ] {
        let mut engine = Engine::new();
        let mut state = SessionState::new();
        state.options_mut().steps_mode = StepsMode::On;
        let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

        let output = engine
            .eval(
                &mut state,
                EvalRequest {
                    raw_input: input.to_string(),
                    parsed,
                    action: EvalAction::Simplify,
                    auto_store: false,
                },
            )
            .expect("eval failed");

        let result = match output.result {
            EvalResult::Expr(expr) => format!(
                "{}",
                DisplayExpr {
                    context: &engine.simplifier.context,
                    id: expr,
                }
            ),
            other => panic!("expected expression result, got {other:?}"),
        };

        assert_eq!(result, "0", "input: {input}");
        assert_eq!(
            output.steps.len(),
            1,
            "expected residual to close before cleanup, got: {:?}",
            output
                .steps
                .iter()
                .map(|step| step.rule_name.as_str())
                .collect::<Vec<_>>()
        );
        assert!(
            output.domain_warnings.is_empty(),
            "warnings: {:?}",
            output.domain_warnings
        );

        let required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();
        assert_eq!(required, expected_required, "input: {input}");
    }
}
#[test]
fn ln_sqrt_affine_gap_diff_uses_compact_public_radicand_and_branch_domain() {
    let cases = [
        (
            "diff(ln(sqrt((2*x+1)^2-4)+(2*x+1)), x)",
            "2 / sqrt((2 * x + 1)^2 - 4)",
            vec!["x > 1/2".to_string()],
        ),
        (
            "diff(ln(sqrt((2*x+1)^2-4)-(2*x+1)), x)",
            "-2 / sqrt((2 * x + 1)^2 - 4)",
            vec!["x < -3/2".to_string()],
        ),
    ];

    for (input, expected_result, expected_conditions) in cases {
        let mut engine = Engine::new();
        let mut state = SessionState::new();
        let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

        let output = engine
            .eval(
                &mut state,
                EvalRequest {
                    raw_input: input.to_string(),
                    parsed,
                    action: EvalAction::Simplify,
                    auto_store: false,
                },
            )
            .expect("eval failed");

        let result = match output.result {
            EvalResult::Expr(expr) => format!(
                "{}",
                DisplayExpr {
                    context: &engine.simplifier.context,
                    id: expr,
                }
            ),
            other => panic!("expected expression result, got {other:?}"),
        };

        assert_eq!(result, expected_result, "input: {input}");

        let required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();

        assert_eq!(
            required, expected_conditions,
            "input: {input}, unexpected required_conditions: {required:?}"
        );
    }
}
#[test]
fn ln_sqrt_quadratic_gap_diff_drops_redundant_sqrt_nonzero_domain() {
    let input = "diff(ln(sqrt((x^2+x+1)^2-4)+(x^2+x+1)), x)";
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let output = engine
        .eval(
            &mut state,
            EvalRequest {
                raw_input: input.to_string(),
                parsed,
                action: EvalAction::Simplify,
                auto_store: false,
            },
        )
        .expect("eval failed");

    let result = match output.result {
        EvalResult::Expr(expr) => format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: expr,
            }
        ),
        other => panic!("expected expression result, got {other:?}"),
    };

    assert_eq!(
        result, "(2 * x + 1) / sqrt((x^2 + x + 1)^2 - 4)",
        "unexpected diff result"
    );

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert_eq!(
        required,
        vec!["x < -1/2 - sqrt(5)/2 or x > -1/2 + sqrt(5)/2"],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn ln_sqrt_affine_gap_diff_residual_collapses_with_branch_domain() {
    let cases = [
        (
            "diff(ln(sqrt((2*x+1)^2-4)+(2*x+1)), x) - 2/sqrt((2*x+1)^2-4)",
            vec!["x > 1/2".to_string()],
        ),
        (
            "diff(ln(sqrt((2*x+1)^2-4)-(2*x+1)), x) + 2/sqrt((2*x+1)^2-4)",
            vec!["x < -3/2".to_string()],
        ),
        (
            "diff(ln(sqrt((x^2+x+1)^2-4)+(x^2+x+1)), x) - (2*x+1)/sqrt((x^2+x+1)^2-4)",
            vec!["x < -1/2 - sqrt(5)/2 or x > -1/2 + sqrt(5)/2".to_string()],
        ),
        (
            "diff(ln(sqrt((2*x+1)^2-4)+(2*x+1)), x) - 2*sqrt((2*x+1)^2-4)/((2*x+1)^2-4)",
            vec!["x > 1/2".to_string()],
        ),
        (
            "diff(ln(sqrt((x^2+x+1)^2-4)+(x^2+x+1)), x) - (2*x+1)*sqrt((x^2+x+1)^2-4)/((x^2+x+1)^2-4)",
            vec!["x < -1/2 - sqrt(5)/2 or x > -1/2 + sqrt(5)/2".to_string()],
        ),
    ];

    for (input, expected_conditions) in cases {
        let mut engine = Engine::new();
        let mut state = SessionState::new();
        let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

        let output = engine
            .eval(
                &mut state,
                EvalRequest {
                    raw_input: input.to_string(),
                    parsed,
                    action: EvalAction::Simplify,
                    auto_store: false,
                },
            )
            .expect("eval failed");

        let result = match output.result {
            EvalResult::Expr(expr) => format!(
                "{}",
                DisplayExpr {
                    context: &engine.simplifier.context,
                    id: expr,
                }
            ),
            other => panic!("expected expression result, got {other:?}"),
        };

        assert_eq!(result, "0", "input: {input}");
        assert!(
            output.steps.iter().any(|step| {
                matches!(
                    step.rule_name.as_str(),
                    "Ln Sqrt Diff Residual" | "Post-calculus residual simplification"
                )
            }),
            "input: {input}, expected direct ln-sqrt residual step, got: {:?}",
            output
                .steps
                .iter()
                .map(|step| step.rule_name.as_str())
                .collect::<Vec<_>>()
        );

        let required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();

        assert_eq!(
            required, expected_conditions,
            "input: {input}, unexpected required_conditions: {required:?}"
        );
    }
}
