use super::*;

#[test]
fn arctan_reciprocal_sqrt_product_diff_keeps_compact_root_fraction_presentation() {
    let cases = [
        (
            "diff(arctan(1/(sqrt(x)*(x+1))), x)",
            "-(3 * x + 1) / (2 * sqrt(x) * (x * (x + 1)^2 + 1))",
        ),
        (
            "diff(arctan(1/(sqrt(x)*(x^2+1))), x)",
            "-(5 * x^2 + 1) / (2 * sqrt(x) * (x * (x^2 + 1)^2 + 1))",
        ),
        (
            "diff(arctan(1/(sqrt(x)*(sqrt(x)+1))), x)",
            "-(2 * sqrt(x) + 1) / (2 * sqrt(x) * (x * (sqrt(x) + 1)^2 + 1))",
        ),
        (
            "diff(arctan(2/(sqrt(x)*(x+1))), x)",
            "-(3 * x + 1) / (sqrt(x) * (x * (x + 1)^2 + 4))",
        ),
        (
            "diff(arctan(1/(sqrt(x)*(2*x+2))), x)",
            "-(3 * x + 1) / (sqrt(x) * (4 * x * (x + 1)^2 + 1))",
        ),
        (
            "diff(arctan(3/(sqrt(x)*(x+1))), x)",
            "-3 * (3 * x + 1) / (2 * sqrt(x) * (x * (x + 1)^2 + 9))",
        ),
        (
            "diff(2*arctan(1/(sqrt(x)*(x+1))), x)",
            "-(3 * x + 1) / ((x * (x + 1)^2 + 1) * sqrt(x))",
        ),
        (
            "diff(1/2*arctan(1/(sqrt(x)*(x+1))), x)",
            "-(3 * x + 1) / (4 * (x * (x + 1)^2 + 1) * sqrt(x))",
        ),
        (
            "diff(1/3*arctan(1/(sqrt(x)*(x+1))), x)",
            "-(3 * x + 1) / (6 * (x * (x + 1)^2 + 1) * sqrt(x))",
        ),
        (
            "diff(arccot(1/(sqrt(x)*(x+1))), x)",
            "(3 * x + 1) / (2 * sqrt(x) * (x * (x + 1)^2 + 1))",
        ),
        (
            "diff(3*arccot(1/(sqrt(x)*(x+1))), x)",
            "3 * (3 * x + 1) / (2 * (x * (x + 1)^2 + 1) * sqrt(x))",
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
            !result.contains("x^(-1/2)")
                && !result.contains("x^(1/2)")
                && !result.contains("x^3 + 2 * x^2 + x"),
            "post-calculus presentation should keep roots and denominator factors compact: {result}"
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
                    "Rationalize Product Denominator" | "Present calculus result in compact form"
                )
            }),
            "arctan reciprocal-root presentation should not expose a rationalize/present round trip; steps: {:?}",
            output
                .steps
                .iter()
                .map(|step| step.rule_name.as_str())
                .collect::<Vec<_>>()
        );
    }

    let residual_input = "diff(arctan(1/(sqrt(x)*(x+1))), x) + \
        (3*x+1)/(2*sqrt(x)*(x*(x+1)^2+1))";
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
    assert_eq!(residual, "0", "residual did not collapse");

    let scaled_residual_input = "diff(2*arctan(1/(sqrt(x)*(x+1))), x) + \
        (3*x+1)/(sqrt(x)*(x*(x+1)^2+1))";
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let parsed =
        parse(scaled_residual_input, &mut engine.simplifier.context).expect("parse residual");
    let output = engine
        .eval(
            &mut state,
            EvalRequest {
                raw_input: scaled_residual_input.to_string(),
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
    assert_eq!(residual, "0", "scaled residual did not collapse");

    let fractional_scaled_residual_input = "diff(1/2*arctan(1/(sqrt(x)*(x+1))), x) + \
        (3*x+1)/(4*sqrt(x)*(x*(x+1)^2+1))";
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let parsed = parse(
        fractional_scaled_residual_input,
        &mut engine.simplifier.context,
    )
    .expect("parse residual");
    let output = engine
        .eval(
            &mut state,
            EvalRequest {
                raw_input: fractional_scaled_residual_input.to_string(),
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
    assert_eq!(residual, "0", "fractional scaled residual did not collapse");

    let third_scaled_residual_input = "diff(1/3*arctan(1/(sqrt(x)*(x+1))), x) + \
        (3*x+1)/(6*sqrt(x)*(x*(x+1)^2+1))";
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let parsed =
        parse(third_scaled_residual_input, &mut engine.simplifier.context).expect("parse residual");
    let output = engine
        .eval(
            &mut state,
            EvalRequest {
                raw_input: third_scaled_residual_input.to_string(),
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
    assert_eq!(residual, "0", "third scaled residual did not collapse");

    let arccot_residual_input = "diff(3*arccot(1/(sqrt(x)*(x+1))), x) - \
        3*(3*x+1)/(2*sqrt(x)*(x*(x+1)^2+1))";
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let parsed =
        parse(arccot_residual_input, &mut engine.simplifier.context).expect("parse residual");
    let output = engine
        .eval(
            &mut state,
            EvalRequest {
                raw_input: arccot_residual_input.to_string(),
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
    assert_eq!(residual, "0", "arccot residual did not collapse");
}
#[test]
fn arctan_reciprocal_sqrt_product_diff_residual_collapses_after_child_expansion() {
    let input = "diff(arctan(1/(sqrt(x)*(x+1))), x) + \
        (6*x+2)/(4*sqrt(x)*(x*(x+1)^2+1))";
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;

    let parsed = parse(input, &mut engine.simplifier.context).expect("parse residual");
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
        .expect("eval residual");
    let result = match output.result {
        EvalResult::Expr(expr) => format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: expr,
            }
        ),
        other => panic!("expected residual expression result, got {other:?}"),
    };

    assert_eq!(result, "0", "residual did not collapse");
    assert!(
        output
            .steps
            .iter()
            .any(|step| step.rule_name.as_str() == "Cancel Opposite Fractions"),
        "residual should close through exact fraction cancellation after child expansion; steps: {:?}",
        output
            .steps
            .iter()
            .map(|step| step.rule_name.as_str())
            .collect::<Vec<_>>()
    );
    assert!(
        !output.steps.iter().any(|step| {
            matches!(
                step.rule_name.as_str(),
                "Add Fractions" | "Zero Property of Division"
            )
        }),
        "residual should not route through generic fraction addition/zero-division cleanup; steps: {:?}",
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
    assert_eq!(required, vec!["x > 0".to_string()]);
}
#[test]
fn arctan_reciprocal_sqrt_quadratic_product_diff_residual_collapses_after_child_expansion() {
    let input = "diff(arctan(1/(sqrt(x)*(x^2+1))), x) + \
        (10*x^2+2)/(4*sqrt(x)*(x*(x^2+1)^2+1))";
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;

    let parsed = parse(input, &mut engine.simplifier.context).expect("parse residual");
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
        .expect("eval residual");
    let result = match output.result {
        EvalResult::Expr(expr) => format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: expr,
            }
        ),
        other => panic!("expected residual expression result, got {other:?}"),
    };

    assert_eq!(result, "0", "quadratic residual did not collapse");
    assert!(
        output
            .steps
            .iter()
            .any(|step| step.rule_name.as_str() == "Cancel Opposite Fractions"),
        "quadratic residual should close through exact fraction cancellation after child expansion; steps: {:?}",
        output
            .steps
            .iter()
            .map(|step| step.rule_name.as_str())
            .collect::<Vec<_>>()
    );
    assert!(
        !output.steps.iter().any(|step| {
            matches!(
                step.rule_name.as_str(),
                "Add Fractions" | "Zero Property of Division"
            )
        }),
        "quadratic residual should not route through generic fraction addition/zero-division cleanup; steps: {:?}",
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
    assert_eq!(required, vec!["x > 0".to_string()]);
}
#[test]
fn inverse_tangent_direct_trig_affine_diff_preserves_chain_rule_presentation() {
    let cases = [
        ("diff(arctan(sin(x)), x)", "cos(x) / (sin(x)^2 + 1)"),
        (
            "diff(arctan(sin(2*x+1)), x)",
            "2 * cos(2 * x + 1) / (sin(2 * x + 1)^2 + 1)",
        ),
        (
            "diff(atan(sin(2*x+1)), x)",
            "2 * cos(2 * x + 1) / (sin(2 * x + 1)^2 + 1)",
        ),
        (
            "diff(atan(sin(1-2*x)), x)",
            "-2 * cos(1 - 2 * x) / (sin(1 - 2 * x)^2 + 1)",
        ),
        (
            "diff(arctan(sin(1-2*x)), x)",
            "-2 * cos(1 - 2 * x) / (sin(1 - 2 * x)^2 + 1)",
        ),
        ("diff(arctan(cos(x)), x)", "-sin(x) / (cos(x)^2 + 1)"),
        (
            "diff(atan(cos(1-2*x)), x)",
            "2 * sin(1 - 2 * x) / (cos(1 - 2 * x)^2 + 1)",
        ),
        (
            "diff(arctan(cos(1-2*x)), x)",
            "2 * sin(1 - 2 * x) / (cos(1 - 2 * x)^2 + 1)",
        ),
        ("diff(arccot(sin(x)), x)", "-cos(x) / (sin(x)^2 + 1)"),
        (
            "diff(arccot(sin(2*x+1)), x)",
            "-2 * cos(2 * x + 1) / (sin(2 * x + 1)^2 + 1)",
        ),
        (
            "diff(arccot(sin(1-2*x)), x)",
            "2 * cos(1 - 2 * x) / (sin(1 - 2 * x)^2 + 1)",
        ),
        (
            "diff(acot(sin(1-2*x)), x)",
            "2 * cos(1 - 2 * x) / (sin(1 - 2 * x)^2 + 1)",
        ),
        ("diff(arccot(cos(x)), x)", "sin(x) / (cos(x)^2 + 1)"),
        (
            "diff(acot(cos(2*x+1)), x)",
            "2 * sin(2 * x + 1) / (cos(2 * x + 1)^2 + 1)",
        ),
        (
            "diff(arccot(cos(2*x+1)), x)",
            "2 * sin(2 * x + 1) / (cos(2 * x + 1)^2 + 1)",
        ),
        (
            "diff(arccot(cos(1-2*x)), x)",
            "-2 * sin(1 - 2 * x) / (cos(1 - 2 * x)^2 + 1)",
        ),
        (
            "diff(acot(cos(1-2*x)), x)",
            "-2 * sin(1 - 2 * x) / (cos(1 - 2 * x)^2 + 1)",
        ),
    ];

    for (input, expected) in cases {
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

        assert_eq!(result, expected, "input: {input}");
        assert!(
            !result.contains("cos(2 * x)^2") && !result.contains("1/2"),
            "post-diff presentation should not degrade through trig half-angle rewrites: {result}"
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
            "inverse-tangent sin/cos affine derivative should not add real-domain conditions: {required:?}"
        );

        assert_eq!(
            output
                .steps
                .iter()
                .filter(|step| step.rule_name.as_str() == "Symbolic Differentiation")
                .count(),
            1,
            "expected one visible differentiation step for {input}: {:?}",
            output
                .steps
                .iter()
                .map(|step| step.rule_name.as_str())
                .collect::<Vec<_>>()
        );

        let residual_input = format!("{input} - ({expected})");
        let parsed_residual =
            parse(&residual_input, &mut engine.simplifier.context).expect("parse residual");
        let residual_output = engine
            .eval(
                &mut state,
                EvalRequest {
                    raw_input: residual_input.clone(),
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
        assert_eq!(
            residual, "0",
            "inverse-tangent sin/cos affine derivative residual did not collapse for {residual_input}"
        );

        let residual_required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &residual_output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();
        assert!(
            residual_required.is_empty(),
            "inverse-tangent sin/cos affine derivative residual should not add real-domain conditions for {residual_input}: {residual_required:?}"
        );
    }
}
#[test]
fn inverse_tangent_over_positive_quadratic_diff_keeps_quotient_compact() {
    let cases = [
        (
            "diff(arctan(2*x)/(4*x^2+1), x)",
            "(2 - 8 * x * arctan(2 * x)) / (4 * x^2 + 1)^2",
        ),
        (
            "diff(arctan(2*x)/(8*x^2+2), x)",
            "(1 - 4 * x * arctan(2 * x)) / (4 * x^2 + 1)^2",
        ),
        (
            "diff(arctan(2*x)/(2*x^2+1/2), x)",
            "(4 - 16 * x * arctan(2 * x)) / (4 * x^2 + 1)^2",
        ),
        (
            "diff(arctan(2*x)/(-4*x^2-1), x)",
            "(8 * x * arctan(2 * x) - 2) / (4 * x^2 + 1)^2",
        ),
        (
            "diff(arctan(2*x+1)/((2*x+1)^2+1), x)",
            "(1/2 - (2 * x + 1) * arctan(2 * x + 1)) / (2 * x^2 + 2 * x + 1)^2",
        ),
        (
            "diff(arctan(2*x+1)/(-((2*x+1)^2+1)), x)",
            "((2 * x + 1) * arctan(2 * x + 1) - 1/2) / (2 * x^2 + 2 * x + 1)^2",
        ),
    ];

    for (input, expected) in cases {
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
        assert_eq!(result, expected, "input: {input}");

        let required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();
        assert!(
            required.is_empty(),
            "positive quadratic quotient derivative should not add synthetic required conditions for {input}: {required:?}"
        );

        let residual_input = format!("{input} - ({expected})");
        let parsed_residual =
            parse(&residual_input, &mut engine.simplifier.context).expect("parse residual");
        let residual_output = engine
            .eval(
                &mut state,
                EvalRequest {
                    raw_input: residual_input.clone(),
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
        assert_eq!(
            residual, "0",
            "positive quadratic quotient derivative residual did not collapse for {residual_input}"
        );
    }
}
#[test]
fn scaled_bounded_inverse_trig_linear_diff_evaluates_to_reciprocal_root() {
    for (input, expected_result, residual_operator, expected_condition, expected_residual_kernel) in [
        (
            "diff(arcsin((x+1)/2), x)",
            "1 / sqrt(3 - x^2 - 2 * x)",
            "-",
            "-3 < x < 1",
            "1/sqrt(3-x^2-2*x)",
        ),
        (
            "diff(arccos((x+1)/2), x)",
            "-1 / sqrt(3 - x^2 - 2 * x)",
            "+",
            "-3 < x < 1",
            "1/sqrt(3-x^2-2*x)",
        ),
        (
            "diff(arcsin((2*x-1)/3), x)",
            "1 / sqrt(x + 2 - x^2)",
            "-",
            "-1 < x < 2",
            "1/sqrt(x+2-x^2)",
        ),
        (
            "diff(arccos((2*x-1)/3), x)",
            "-1 / sqrt(x + 2 - x^2)",
            "+",
            "-1 < x < 2",
            "1/sqrt(x+2-x^2)",
        ),
    ] {
        let mut engine = Engine::new();
        let mut state = SessionState::new();
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

        assert_eq!(result, expected_result, "input: {input}");

        let required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();

        assert_eq!(
            required,
            vec![expected_condition.to_string()],
            "input: {input}, unexpected required_conditions: {required:?}"
        );

        let residual_input = format!("{input} {residual_operator} {expected_residual_kernel}");
        let parsed_residual =
            parse(&residual_input, &mut engine.simplifier.context).expect("parse residual");
        let residual_output = engine
            .eval(
                &mut state,
                EvalRequest {
                    raw_input: residual_input.clone(),
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
        assert_eq!(
            residual, "0",
            "bounded inverse trig linear derivative residual did not collapse for {residual_input}"
        );

        let residual_required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &residual_output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();
        assert_eq!(
            residual_required,
            vec![expected_condition.to_string()],
            "unexpected residual required_conditions for {residual_input}: {residual_required:?}"
        );
    }
}
#[test]
fn scaled_arcsin_unit_interval_residual_collapses_inside_subtraction_context() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(1/2*arcsin(2*x-1), x) - 1/(2*sqrt(x)*sqrt(1-x))";
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
        vec!["x > 0".to_string(), "x < 1".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn scaled_arcsin_unit_interval_diff_reuses_compact_direct_shortcut() {
    for input in ["diff(1/2*arcsin(2*x-1), x)", "diff(arcsin(2*x-1)/2, x)"] {
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

        assert_eq!(result, "1 / (2 * sqrt(x) * sqrt(1 - x))", "input: {input}");
        let rules = output
            .steps
            .iter()
            .map(|step| step.rule_name.as_str())
            .collect::<Vec<_>>();

        assert!(
            rules.contains(&"Symbolic Differentiation"),
            "scaled unit-interval arcsin diff should use the direct diff shortcut for {input}: {rules:?}"
        );
        for noisy_rule in [
            "Rationalize Product Denominator",
            "Merge Sqrt Product",
            "Cancel Power Fraction",
            "Expand Binomial",
        ] {
            assert!(
                !rules.contains(&noisy_rule),
                "scaled unit-interval arcsin diff should avoid generic expansion/cancel noise for {input}: {rules:?}"
            );
        }
    }
}
#[test]
fn arctan_sqrt_variable_over_positive_affine_diff_uses_compact_direct_shortcut() {
    for (input, expected) in [
        (
            "diff(arctan(sqrt(x)/(x+1)), x)",
            "(1 - x) / (2 * sqrt(x) * (x^2 + 3 * x + 1))",
        ),
        (
            "diff(arctan(sqrt(x)/(x+1))/2, x)",
            "(1 - x) / (4 * (x^2 + 3 * x + 1) * sqrt(x))",
        ),
        (
            "diff(2*arctan(sqrt(x)/(x+1)), x)",
            "(1 - x) / ((x^2 + 3 * x + 1) * sqrt(x))",
        ),
        (
            "diff(arctan(sqrt(2*x)/(x+1)), x)",
            "(1 - x) / (sqrt(2 * x) * (x^2 + 4 * x + 1))",
        ),
        (
            "diff(arctan(sqrt(3*x)/(2*x+1)), x)",
            "(3 - 6 * x) / (2 * sqrt(3 * x) * (4 * x^2 + 7 * x + 1))",
        ),
        (
            "diff(arctan(2*sqrt(x)/(x+1)), x)",
            "(1 - x) / (sqrt(x) * (x^2 + 6 * x + 1))",
        ),
        (
            "diff(arctan(3*sqrt(2*x)/(2*x+1)), x)",
            "(3 - 6 * x) / (sqrt(2 * x) * (4 * x^2 + 22 * x + 1))",
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

        let required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();
        assert_eq!(required, vec!["x > 0".to_string()], "input: {input}");

        let rules = output
            .steps
            .iter()
            .map(|step| step.rule_name.as_str())
            .collect::<Vec<_>>();
        assert!(
            rules.contains(&"Symbolic Differentiation"),
            "expected direct symbolic differentiation for {input}: {rules:?}"
        );
        for noisy_rule in [
            "Expand",
            "Distributive Property",
            "Simplify Nested Fraction",
            "Power of a Product",
            "Power of a Quotient",
            "Present calculus result in compact form",
        ] {
            assert!(
                !rules.contains(&noisy_rule),
                "compact arctan sqrt/affine diff should avoid noisy post-calc route for {input}: {rules:?}"
            );
        }
    }
}
#[test]
fn negated_arcsin_unit_interval_diff_uses_post_calculus_root_denominator_presentation() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let input = "diff(-arcsin(2*x-1), x)";
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

    assert_eq!(result, "-1 / (sqrt(x) * sqrt(1 - x))");
    assert!(
        !result.contains("^(-1/2)") && !result.contains("x - x^2"),
        "negated unit-interval arcsin derivative should use educational root-denominator form, got: {result}"
    );

    let expected =
        parse("-1/(sqrt(x)*sqrt(1-x))", &mut engine.simplifier.context).expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "post-calculus presentation must stay equivalent to the expected derivative, got: {result}"
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
        vec!["x > 0".to_string(), "x < 1".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert!(
        output
            .steps
            .iter()
            .any(|step| step.rule_name == "Symbolic Differentiation"),
        "expected the ordinary symbolic differentiation trace"
    );

    let residual_input = "diff(-arcsin(2*x-1), x) + 1/(sqrt(x)*sqrt(1-x))";
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
        .expect("eval residual failed");
    let residual = match residual_output.result {
        EvalResult::Expr(expr) => format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: expr,
            }
        ),
        other => panic!("expected expression residual result, got {other:?}"),
    };
    assert_eq!(residual, "0", "residual did not collapse for {input}");
}
#[test]
fn shifted_arcsin_wrong_orientation_residual_does_not_collapse_across_domains() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(1/2*arcsin(2*x+1), x) - 1/(2*sqrt(x)*sqrt(1-x))";
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
        other => panic!("expected expression residual result, got {other:?}"),
    };

    assert_ne!(
        result, "0",
        "wrong-orientation shifted arcsin residual must not collapse across incompatible domains"
    );

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert!(
        required.contains(&"-1 < x < 0".to_string()),
        "wrong-orientation derivative domain should remain visible: {required:?}"
    );
    assert!(
        required.contains(&"x > 0".to_string()) && required.contains(&"x < 1".to_string()),
        "comparison kernel domain should remain visible: {required:?}"
    );
}
#[test]
fn shifted_arccos_wrong_orientation_residual_does_not_collapse_across_domains() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let valid_input = "diff(1/2*arccos(2*x-1), x) + 1/(2*sqrt(x)*sqrt(1-x))";
    let parsed_valid = parse(valid_input, &mut engine.simplifier.context).expect("parse valid");

    let valid_output = engine
        .eval(
            &mut state,
            EvalRequest {
                raw_input: valid_input.to_string(),
                parsed: parsed_valid,
                action: EvalAction::Simplify,
                auto_store: false,
            },
        )
        .expect("eval valid");
    let valid_result = match valid_output.result {
        EvalResult::Expr(expr) => format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: expr,
            }
        ),
        other => panic!("expected expression result, got {other:?}"),
    };
    assert_eq!(valid_result, "0");

    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let shifted_input = "diff(1/2*arccos(2*x+1), x) + 1/(2*sqrt(x)*sqrt(1-x))";
    let parsed_shifted =
        parse(shifted_input, &mut engine.simplifier.context).expect("parse shifted");

    let shifted_output = engine
        .eval(
            &mut state,
            EvalRequest {
                raw_input: shifted_input.to_string(),
                parsed: parsed_shifted,
                action: EvalAction::Simplify,
                auto_store: false,
            },
        )
        .expect("eval shifted");
    let shifted_result = match shifted_output.result {
        EvalResult::Expr(expr) => format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: expr,
            }
        ),
        other => panic!("expected expression residual result, got {other:?}"),
    };

    assert_ne!(
        shifted_result, "0",
        "wrong-orientation shifted arccos residual must not collapse across incompatible domains"
    );

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &shifted_output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert!(
        required.contains(&"-1 < x < 0".to_string()),
        "wrong-orientation derivative domain should remain visible: {required:?}"
    );
    assert!(
        required.contains(&"x > 0".to_string()) && required.contains(&"x < 1".to_string()),
        "comparison kernel domain should remain visible: {required:?}"
    );
}
#[test]
fn arctan_sqrt_diff_uses_post_calculus_reciprocal_root_presentation() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let input = "diff(arctan(sqrt(x)), x)";
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

    assert_eq!(result, "1 / (2 * sqrt(x) * (x + 1))");
    assert!(
        !result.contains("x^(-1/2)"),
        "presentation regressed: {result}"
    );
    assert!(
        !result.contains("2 * x + 2"),
        "denominator should remain factored in post-calculus presentation: {result}"
    );

    let expected =
        parse("x^(-1/2)/(2*x+2)", &mut engine.simplifier.context).expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "post-calculus presentation must stay equivalent to the canonical derivative, got: {result}"
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
        output.steps.iter().any(|step| matches!(
            step.rule_name.as_str(),
            "Cancel Reciprocal Exponents" | "Square of Square Root"
        )),
        "expected the meaningful root/power cleanup step to remain visible"
    );
    assert!(
        output
            .steps
            .iter()
            .any(|step| step.rule_name == "Present calculus result in compact form"),
        "expected the compact post-calculus presentation step to remain visible"
    );
    assert!(
        output.steps.iter().all(|step| {
            step.rule_name != "Pull Constant From Fraction"
                && step.rule_name != "Simplify Multiplication with Division"
        }),
        "post-calculus presentation should hide mechanical fraction cleanup steps: {:?}",
        output
            .steps
            .iter()
            .map(|step| step.rule_name.as_str())
            .collect::<Vec<_>>()
    );
}
#[test]
fn arctan_sqrt_affine_partition_quotient_diff_uses_compact_real_domain_presentation() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let input = "diff(arctan(sqrt(x/(1-x))), x)";
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

    assert_eq!(result, "1 / (2 * sqrt(x) * sqrt(1 - x))");
    assert!(
        !result.contains("^(-1/2)") && !result.contains("x / (1 - x)"),
        "post-calculus presentation should expose the interval root factors: {result}"
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
        vec!["x > 0".to_string(), "x < 1".to_string()],
        "unexpected required_conditions: {required:?}"
    );

    assert!(
        output
            .steps
            .iter()
            .any(|step| step.rule_name == "Present calculus result in compact form"),
        "expected the compact post-calculus presentation step to remain visible"
    );
}
#[test]
fn arctan_sqrt_affine_partition_quotient_diff_residual_collapses_to_zero() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(arctan(sqrt(x/(1-x))), x) - 1/(2*sqrt(x)*sqrt(1-x))";
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

    assert_eq!(result, "0");

    let mut required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();
    required.sort();

    assert_eq!(
        required,
        vec!["x < 1".to_string(), "x > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn arctan_sqrt_complement_partition_quotient_diff_residual_collapses_to_zero() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(arctan(sqrt((1-x)/x)), x) + 1/(2*sqrt(x)*sqrt(1-x))";
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

    assert_eq!(result, "0");

    let mut required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();
    required.sort();

    assert_eq!(
        required,
        vec!["x < 1".to_string(), "x > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn arctan_sqrt_scaled_affine_partition_quotient_diff_residual_collapses_to_zero() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(arctan(sqrt((2*x+1)/(3-2*x))), x) - 1/(sqrt(2*x+1)*sqrt(3-2*x))";
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

    assert_eq!(result, "0");

    let mut required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();
    required.sort();

    assert_eq!(
        required,
        vec!["x < 3/2".to_string(), "x > -1/2".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn arctan_sqrt_scaled_complement_partition_quotient_diff_residual_collapses_to_zero() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(arctan(sqrt((3-2*x)/(2*x+1))), x) + 1/(sqrt(2*x+1)*sqrt(3-2*x))";
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

    assert_eq!(result, "0");

    let mut required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();
    required.sort();

    assert_eq!(
        required,
        vec!["x < 3/2".to_string(), "x > -1/2".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn arctan_sqrt_affine_quotient_diff_keeps_domain_equivalent_compact_presentation() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let input = "diff(arctan(sqrt((x+1)/(x+3))), x)";
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

    assert_eq!(
        result,
        "1 / (2 * (x + 2) * (x + 3) * sqrt((x + 1) / (x + 3)))"
    );
    assert!(
        !result.contains("^(-1/2)") && !result.contains("sqrt(x + 1) * sqrt(x + 3)"),
        "presentation should keep the quotient root instead of splitting domains: {result}"
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
        vec!["x < -3 or x > -1".to_string()],
        "unexpected required_conditions: {required:?}"
    );

    assert!(
        output
            .steps
            .iter()
            .any(|step| step.rule_name == "Present calculus result in compact form"),
        "expected the compact post-calculus presentation step to remain visible"
    );
}
#[test]
fn arctan_sqrt_scaled_affine_quotient_diff_drops_internal_nonzero_requires() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let input = "diff(arctan(sqrt((2*x+1)/(x+3))), x)";
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

    assert_eq!(
        result,
        "5 / (2 * (3 * x + 4) * (x + 3) * sqrt((2 * x + 1) / (x + 3)))"
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
        vec!["x < -3 or x > -1/2".to_string()],
        "unexpected required_conditions: {required:?}"
    );

    assert!(
        output
            .steps
            .iter()
            .any(|step| step.rule_name == "Present calculus result in compact form"),
        "expected the compact post-calculus presentation step to remain visible"
    );
}
#[test]
fn arctan_sqrt_quadratic_quotient_diff_keeps_compact_quotient_root_presentation() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let input = "diff(arctan(sqrt((x^2+1)/(x^2+3))), x)";
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

    assert_eq!(
        result,
        "x / ((x^2 + 2) * (x^2 + 3) * sqrt((x^2 + 1) / (x^2 + 3)))"
    );
    assert!(
        !result.contains("^(-1/2)") && !result.contains("sqrt(x^2 + 1) * sqrt(x^2 + 3)"),
        "presentation should keep a compact quotient root without strengthening domains: {result}"
    );

    let expected = parse(
        "((x^2+1)/(x^2+3))^(-1/2)*(2*x*(x^2+3)-2*x*(x^2+1))/((x^2+3)*(4*x^2+8))",
        &mut engine.simplifier.context,
    )
    .expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "post-calculus presentation must stay equivalent to the direct derivative, got: {result}"
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
        "unexpected required_conditions: {required:?}"
    );

    assert!(
        output
            .steps
            .iter()
            .any(|step| step.rule_name == "Present calculus result in compact form"),
        "expected the compact post-calculus presentation step to remain visible"
    );
}
#[test]
fn arctan_sqrt_shifted_quadratic_quotient_diff_uses_fast_compact_shortcut_without_steps() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(arctan(sqrt((x^2+x+1)/(x^2+x+3))), x)";
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

    assert_eq!(
        result,
        "(2 * x + 1) / (2 * (x^2 + x + 2) * (x^2 + x + 3) * sqrt((x^2 + x + 1) / (x^2 + x + 3)))"
    );
    assert!(
        !result.contains("^(-1/2)") && !result.contains("sqrt(x^2 + x + 1) * sqrt(x^2 + x + 3)"),
        "shortcut should keep the compact quotient root without strengthening domains: {result}"
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
        "strictly positive quadratic quotient should not require extra conditions: {required:?}"
    );
}
#[test]
fn inverse_reciprocal_trig_negative_affine_sqrt_shifted_quotient_compacts_contextual_diff() {
    for (input, expected_result, expected_required) in [
        (
            "(1 + diff(arcsec(sqrt(5-3*x)), x))/(2 - 3/(2*(5-3*x)*sqrt(4-3*x)))",
            "(1 - 3 / (2 * sqrt(4 - 3 * x) * (5 - 3 * x))) / (2 - 3 / (2 * sqrt(4 - 3 * x) * (5 - 3 * x)))",
            vec![
                "x < 4/3".to_string(),
                "2 - 3 / (2 * sqrt(4 - 3 * x) * (5 - 3 * x)) ≠ 0".to_string(),
            ],
        ),
        (
            "(1 + diff(arccsc(sqrt(5-3*x)), x))/(2 + 3/(2*(5-3*x)*sqrt(4-3*x)))",
            "(3 / (2 * sqrt(4 - 3 * x) * (5 - 3 * x)) + 1) / (3 / (2 * sqrt(4 - 3 * x) * (5 - 3 * x)) + 2)",
            vec!["x < 4/3".to_string()],
        ),
        (
            "(1 + diff(arcsec(sqrt(5-3*x)), x))/(1 - 3/(2*(5-3*x)*sqrt(4-3*x)))",
            "1",
            vec![
                "1 - 3 / (2 * sqrt(4 - 3 * x) * (5 - 3 * x)) ≠ 0".to_string(),
                "x < 4/3".to_string(),
            ],
        ),
    ] {
        let mut engine = Engine::new();
        let mut state = SessionState::new();
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
        assert!(!result.contains("diff"), "diff should be discharged: {result}");
        assert!(
            !result.contains("((4 - 3 * x) / (5 - 3 * x))^(-1/2)")
                && !result.contains("(9 * x^2"),
            "contextual inverse reciprocal trig presentation should stay compact: {result}"
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
    }
}
#[test]
fn unit_interval_bounded_inverse_trig_shifted_quotient_compacts_contextual_diff() {
    for (input, expected_result, expected_required) in [
        (
            "(1 + diff(1/2*arcsin(2*x-1), x))/(1 + 1/(2*sqrt(x)*sqrt(1-x)))",
            "1",
            vec!["x < 1".to_string(), "x > 0".to_string()],
        ),
        (
            "(1 + diff(1/2*arccos(2*x-1), x))/(1 - 1/(2*sqrt(x)*sqrt(1-x)))",
            "1",
            vec![
                "1 - 1 / (2 * sqrt(x) * sqrt(1 - x)) ≠ 0".to_string(),
                "x < 1".to_string(),
                "x > 0".to_string(),
            ],
        ),
        (
            "(1 + diff(1/2*arcsin(2*x-1), x))/(2 + 1/(2*sqrt(x)*sqrt(1-x)))",
            "(1 / (2 * sqrt(x) * sqrt(1 - x)) + 1) / (1 / (2 * sqrt(x) * sqrt(1 - x)) + 2)",
            vec!["x < 1".to_string(), "x > 0".to_string()],
        ),
        (
            "(1 + diff(1/2*arccos(2*x-1), x))/(2 - 1/(2*sqrt(x)*sqrt(1-x)))",
            "(1 - 1 / (2 * sqrt(x) * sqrt(1 - x))) / (2 - 1 / (2 * sqrt(x) * sqrt(1 - x)))",
            vec![
                "2 - 1 / (2 * sqrt(x) * sqrt(1 - x)) ≠ 0".to_string(),
                "x < 1".to_string(),
                "x > 0".to_string(),
            ],
        ),
    ] {
        let mut engine = Engine::new();
        let mut state = SessionState::new();
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
        assert!(
            !result.contains("diff"),
            "diff should be discharged: {result}"
        );
        assert!(
            !result.contains("(x * (1 - x))^(1/2)")
                && !result.contains("(x - x^2)^(-1/2)")
                && !result.contains("(x - x^2)^(1/2)"),
            "contextual bounded inverse trig presentation should keep separate sqrt factors: {result}"
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
    }
}
#[test]
fn shifted_sqrt_inverse_tangent_unit_quotient_compacts_contextual_diff() {
    for (input, expected_required) in [
        (
            "(1 + diff(arctan(sqrt(x+1)), x))/(1 + 1/(2*sqrt(x+1)*(x+2)))",
            vec![
                "1 / (2 * sqrt(x + 1) * (x + 2)) + 1 ≠ 0".to_string(),
                "x > -1".to_string(),
            ],
        ),
        (
            "(1 + diff(arccot(sqrt(x+1)), x))/(1 - 1/(2*sqrt(x+1)*(x+2)))",
            vec![
                "1 - 1 / (2 * sqrt(x + 1) * (x + 2)) ≠ 0".to_string(),
                "x > -1".to_string(),
            ],
        ),
        (
            "(1 + diff(arctan(sqrt(2*x+3)), x))/(1 + 1/(2*sqrt(2*x+3)*(x+2)))",
            vec![
                "1 / (2 * sqrt(2 * x + 3) * (x + 2)) + 1 ≠ 0".to_string(),
                "x > -3/2".to_string(),
            ],
        ),
        (
            "(1 + diff(arccot(sqrt(2*x+3)), x))/(1 - 1/(2*sqrt(2*x+3)*(x+2)))",
            vec![
                "1 - 1 / (2 * sqrt(2 * x + 3) * (x + 2)) ≠ 0".to_string(),
                "x > -3/2".to_string(),
            ],
        ),
    ] {
        let mut engine = Engine::new();
        let mut state = SessionState::new();
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

        assert_eq!(result, "1", "input: {input}");
        assert!(
            !result.contains("diff"),
            "diff should be discharged: {result}"
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
    }
}
#[test]
fn arctan_surd_quotient_diff_uses_compact_sqrt_scale_presentation() {
    let cases = [
        (
            "diff(arctan((2*x+2)/sqrt(6)), x)",
            "2 * sqrt(6) / ((2 * x + 2)^2 + 6)",
            "2/(sqrt(6)*(1+((2*x+2)/sqrt(6))^2))",
        ),
        (
            "diff(arctan((x^2+x+1)/sqrt(7)), x)",
            "(2 * x + 1) * sqrt(7) / ((x^2 + x + 1)^2 + 7)",
            "(2*x+1)/(sqrt(7)*(1+((x^2+x+1)/sqrt(7))^2))",
        ),
        (
            "diff(arctan(-(x^2+x+1)/sqrt(5)), x)",
            "-(2 * x + 1) * sqrt(5) / ((x^2 + x + 1)^2 + 5)",
            "(-2*x-1)/(sqrt(5)*(1+((-(x^2+x+1))/sqrt(5))^2))",
        ),
        (
            "diff(arctan((1-x-x^2)/sqrt(5))/sqrt(5), x)",
            "-(2 * x + 1) / ((x^2 + x - 1)^2 + 5)",
            "(-2*x-1)/(5*(1+((1-x-x^2)/sqrt(5))^2))",
        ),
    ];

    for (input, expected_result, expected_chain_rule) in cases {
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
        assert!(
            !result.contains("1/6") && !result.contains("1/7"),
            "input: {input}, compact arctan quotient derivative should not expose rationalized denominator scale: {result}"
        );

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

        assert!(
            output.required_conditions.is_empty(),
            "arctan quotient over a positive constant sqrt should not add required conditions"
        );
    }
}
#[test]
fn arctan_self_normalized_surd_quotient_diff_uses_compact_gap_presentation() {
    let input = "diff(arctan(x/sqrt(x^2+1)), x)";
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

    assert_eq!(result, "1 / (sqrt(x^2 + 1) * (2 * x^2 + 1))");
    assert!(
        !result.contains("^(-1/2)") && !result.contains("(x^2 + 1)^(1/2) -"),
        "self-normalized arctan quotient should use compact sqrt denominator: {result}"
    );
    assert!(
        output.required_conditions.is_empty(),
        "unexpected required conditions for {input}: {:?}",
        output.required_conditions
    );

    let expected =
        parse("1/(sqrt(x^2+1)*(2*x^2+1))", &mut engine.simplifier.context).expect("parse expected");
    for sample in [-0.5, 0.0, 0.5] {
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), sample);
        let actual_value = eval_f64(&engine.simplifier.context, result_expr, &vars)
            .unwrap_or_else(|| panic!("input: {input}, could not eval result at x={sample}"));
        let expected_value = eval_f64(&engine.simplifier.context, expected, &vars)
            .unwrap_or_else(|| panic!("input: {input}, could not eval expected at x={sample}"));
        assert!(
            (actual_value - expected_value).abs() < 1e-10,
            "input: {input}, x={sample}, expected {expected_value}, got {actual_value}"
        );
    }
}
#[test]
fn arctan_self_normalized_surd_reciprocal_diff_uses_compact_gap_presentation() {
    let cases = [
        (
            "diff(arctan(sqrt(x^2+1)/x), x)",
            "-1 / (sqrt(x^2 + 1) * (2 * x^2 + 1))",
            "-1/(sqrt(x^2+1)*(2*x^2+1))",
            vec!["x ≠ 0"],
        ),
        (
            "diff(arctan(sqrt((2*x+1)^2+1)/(2*x+1)), x)",
            "-2 / (sqrt((2 * x + 1)^2 + 1) * (2 * (2 * x + 1)^2 + 1))",
            "-2/(sqrt((2*x+1)^2+1)*(2*(2*x+1)^2+1))",
            vec!["x ≠ -1/2"],
        ),
    ];

    for (input, expected_result, expected_expr, expected_required) in cases {
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
        assert!(
            !result.contains("^(-1/2)") && !result.contains(" - ("),
            "reciprocal self-normalized arctan quotient should use compact sqrt denominator: {result}"
        );

        let expected =
            parse(expected_expr, &mut engine.simplifier.context).expect("parse expected");
        assert!(
            engine.simplifier.are_equivalent(result_expr, expected),
            "input: {input}, expected compact reciprocal orientation, got: {result}"
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
            "input: {input}, unexpected required conditions: {required:?}"
        );
    }
}
#[test]
fn inverse_reciprocal_trig_affine_scaled_diff_uses_abs_sqrt_presentation() {
    let cases = [
        (
            "diff(arcsec(x)/2, x)",
            "1 / (2 * |x| * sqrt(x^2 - 1))",
            "x < -1 or x > 1",
        ),
        (
            "diff((1/2)*arcsec(x), x)",
            "1 / (2 * |x| * sqrt(x^2 - 1))",
            "x < -1 or x > 1",
        ),
        (
            "diff(arccsc(x)/2, x)",
            "-1 / (2 * |x| * sqrt(x^2 - 1))",
            "x < -1 or x > 1",
        ),
        (
            "diff(arcsec(2*x+1)/3, x)",
            "1 / (3 * |2 * x + 1| * sqrt(x^2 + x))",
            "x < -1 or x > 0",
        ),
    ];

    for (input, expected_result, expected_condition) in cases {
        let mut engine = Engine::new();
        let mut state = SessionState::new();
        state.options_mut().steps_mode = StepsMode::Off;
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

        assert_eq!(result, expected_result, "input: {input}");
        assert!(
            result.contains('|') && result.contains("sqrt("),
            "input: {input}, scaled inverse reciprocal trig derivative should stay abs/sqrt compact: {result}"
        );
        assert!(
            !result.contains("((x^2 - 1) / x^2)"),
            "input: {input}, scaled inverse reciprocal trig derivative should not expose normalized quotient roots: {result}"
        );

        let expected =
            parse(expected_result, &mut engine.simplifier.context).expect("parse expected");
        assert!(
            engine.simplifier.are_equivalent(result_expr, expected),
            "input: {input}, expected derivative equivalent to {expected_result}, got: {result}"
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
            vec![expected_condition.to_string()],
            "input: {input}, scaled inverse reciprocal trig derivative must preserve the real-domain guard"
        );
    }

    let residual_cases = [
        (
            "diff(arcsec(x)/2, x) - 1/(2*abs(x)*sqrt(x^2-1))",
            "x < -1 or x > 1",
        ),
        (
            "1/(2*abs(x)*sqrt(x^2-1)) - diff(arcsec(x)/2, x)",
            "x < -1 or x > 1",
        ),
        (
            "diff(arccsc(x)/2, x) + 1/(2*abs(x)*sqrt(x^2-1))",
            "x < -1 or x > 1",
        ),
        (
            "diff(arcsec(x+1)/2, x) - 1/(2*abs(x+1)*sqrt((x+1)^2-1))",
            "x < -2 or x > 0",
        ),
        (
            "diff(arccsc(2*x+1)/3, x) + 2/(3*abs(2*x+1)*sqrt((2*x+1)^2-1))",
            "x < -1 or x > 0",
        ),
        (
            "diff(arcsec(1-2*x)/3, x) + 2/(3*abs(1-2*x)*sqrt((1-2*x)^2-1))",
            "x < 0 or x > 1",
        ),
    ];

    for (residual_input, expected_condition) in residual_cases {
        let mut engine = Engine::new();
        let mut state = SessionState::new();
        state.options_mut().steps_mode = StepsMode::Off;
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
        let result = match output.result {
            EvalResult::Expr(expr) => format!(
                "{}",
                DisplayExpr {
                    context: &engine.simplifier.context,
                    id: expr,
                }
            ),
            other => panic!("expected expression residual result, got {other:?}"),
        };
        assert_eq!(
            result, "0",
            "scaled inverse reciprocal trig residual did not collapse for {residual_input}"
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
            vec![expected_condition.to_string()],
            "residual {residual_input} must preserve the real-domain guard"
        );
    }
}
#[test]
fn positive_quadratic_inverse_reciprocal_trig_residual_collapses_to_compact_public_form() {
    let cases = [
        (
            "diff(arcsec(x^2+1), x) - 2*x/(abs(x^2+1)*sqrt((x^2+1)^2-1))",
            vec!["x ≠ 0"],
        ),
        (
            "diff(arccsc(x^2+1), x) + 2*x/(abs(x^2+1)*sqrt((x^2+1)^2-1))",
            vec!["x ≠ 0"],
        ),
        (
            "diff(arcsec(x^2+x+3), x) - (2*x+1)/((x^2+x+3)*sqrt(x^4+2*x^3+7*x^2+6*x+8))",
            vec![],
        ),
        (
            "diff(arccsc(x^2+x+3), x) + (2*x+1)/((x^2+x+3)*sqrt(x^4+2*x^3+7*x^2+6*x+8))",
            vec![],
        ),
    ];

    for (input, expected_required) in cases {
        let mut engine = Engine::new();
        let mut state = SessionState::new();
        state.options_mut().steps_mode = StepsMode::Off;
        let parsed = parse(input, &mut engine.simplifier.context).expect("parse residual");

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
            .expect("eval residual");
        let result = match output.result {
            EvalResult::Expr(expr) => format!(
                "{}",
                DisplayExpr {
                    context: &engine.simplifier.context,
                    id: expr,
                }
            ),
            other => panic!("expected expression residual result, got {other:?}"),
        };
        assert_eq!(
            result, "0",
            "positive quadratic inverse reciprocal trig residual did not collapse for {input}"
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
            expected_required
                .iter()
                .map(|cond| cond.to_string())
                .collect::<Vec<_>>(),
            "residual {input} must preserve the real-domain guard"
        );
    }
}
#[test]
fn arctan_scaled_sqrt_diff_uses_post_calculus_reciprocal_root_presentation() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let input = "diff(arctan(sqrt(3*x)), x)";
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

    assert_eq!(result, "3 / (2 * sqrt(3 * x) * (3 * x + 1))");
    assert!(
        !result.contains("(3 * x)^(-1/2)"),
        "presentation regressed: {result}"
    );
    assert!(
        !result.contains("6 * x + 2"),
        "denominator should remain factored in post-calculus presentation: {result}"
    );

    let expected =
        parse("((3*x)^(-1/2)*3)/(6*x+2)", &mut engine.simplifier.context).expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "post-calculus presentation must stay equivalent to the canonical derivative, got: {result}"
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
}
#[test]
fn constant_scaled_arctan_sqrt_diff_uses_post_calculus_reciprocal_root_presentation() {
    for (input, expected_render, canonical_equivalent) in [
        (
            "diff(2*arctan(sqrt(x)), x)",
            "1 / ((x + 1) * sqrt(x))",
            "(x^(-1/2)*2)/(2*x+2)",
        ),
        (
            "diff(-arctan(sqrt(x)), x)",
            "-1 / (2 * (x + 1) * sqrt(x))",
            "-x^(-1/2)/(2*x+2)",
        ),
        (
            "diff(-2*arccot(sqrt(x)), x)",
            "1 / ((x + 1) * sqrt(x))",
            "(x^(-1/2)*2)/(2*x+2)",
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
            !result.contains("x^(-1/2)") && !result.contains("2 * x + 2"),
            "scaled arctan sqrt presentation regressed for {input}: {result}"
        );

        let expected =
            parse(canonical_equivalent, &mut engine.simplifier.context).expect("parse expected");
        assert!(
            engine.simplifier.are_equivalent(result_expr, expected),
            "post-calculus presentation must stay equivalent to the canonical derivative, got: {result}"
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
            "expected compact post-calculus presentation step for {input}"
        );
    }
}
#[test]
fn inverse_tangent_externally_scaled_sqrt_diff_uses_post_calculus_reciprocal_root_presentation() {
    for (input, expected_render, canonical_equivalent) in [
        (
            "diff(arctan(2*sqrt(x)), x)",
            "1 / (sqrt(x) * (4 * x + 1))",
            "x^(-1/2)/(4*x+1)",
        ),
        (
            "diff(arctan(sqrt(x)/2), x)",
            "1 / (sqrt(x) * (x + 4))",
            "x^(-1/2)/(x+4)",
        ),
        (
            "diff(arctan(sqrt(4*x)/3), x)",
            "3 / (sqrt(x) * (4 * x + 9))",
            "6/(sqrt(4*x)*(4*x+9))",
        ),
        (
            "diff(arctan(-2*sqrt(x)), x)",
            "-1 / (sqrt(x) * (4 * x + 1))",
            "-x^(-1/2)/(4*x+1)",
        ),
        (
            "diff(arctan(-sqrt(4*x)/3), x)",
            "-3 / (sqrt(x) * (4 * x + 9))",
            "-6/(sqrt(4*x)*(4*x+9))",
        ),
        (
            "diff(arccot(2*sqrt(x)), x)",
            "-1 / (sqrt(x) * (4 * x + 1))",
            "-x^(-1/2)/(4*x+1)",
        ),
        (
            "diff(arccot(-2*sqrt(x)), x)",
            "1 / (sqrt(x) * (4 * x + 1))",
            "x^(-1/2)/(4*x+1)",
        ),
        (
            "diff(arccot(sqrt(x)/2), x)",
            "-1 / (sqrt(x) * (x + 4))",
            "-x^(-1/2)/(x+4)",
        ),
        (
            "diff(arccot(sqrt(4*x)/3), x)",
            "-3 / (sqrt(x) * (4 * x + 9))",
            "-6/(sqrt(4*x)*(4*x+9))",
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
            !result.contains("^(-1/2)"),
            "externally scaled sqrt presentation regressed for {input}: {result}"
        );

        let expected =
            parse(canonical_equivalent, &mut engine.simplifier.context).expect("parse expected");
        assert!(
            engine.simplifier.are_equivalent(result_expr, expected),
            "post-calculus presentation must stay equivalent to the canonical derivative, got: {result}"
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
            output
                .steps
                .iter()
                .any(|step| step.rule_name == "Symbolic Differentiation"),
            "expected the derivative to keep the ordinary symbolic differentiation trace"
        );
        assert!(
            output.steps.iter().all(|step| {
                step.rule_name != "Power of a Product"
                    && step.rule_name != "Expand"
                    && step.rule_name != "Rationalize Denominator"
                    && step.rule_name != "Rationalize Product Denominator"
            }),
            "externally scaled inverse-trig sqrt diff should avoid generic product/rationalization cleanup: {:?}",
            output
                .steps
                .iter()
                .map(|step| step.rule_name.as_str())
                .collect::<Vec<_>>()
        );
    }
}
#[test]
fn constant_scaled_inverse_tangent_externally_scaled_sqrt_diff_uses_post_calculus_presentation() {
    for (input, expected_render, canonical_equivalent) in [
        (
            "diff(2*arctan(2*sqrt(x)), x)",
            "2 / ((4 * x + 1) * sqrt(x))",
            "(x^(-1/2)*2)/(4*x+1)",
        ),
        (
            "diff(2*arctan(sqrt(x)/2), x)",
            "2 / ((x + 4) * sqrt(x))",
            "(x^(-1/2)*2)/(x+4)",
        ),
        (
            "diff(2*arctan(sqrt(4*x)/3), x)",
            "6 / ((4 * x + 9) * sqrt(x))",
            "12/(sqrt(4*x)*(4*x+9))",
        ),
        (
            "diff(-2*arccot(2*sqrt(x)), x)",
            "2 / ((4 * x + 1) * sqrt(x))",
            "(x^(-1/2)*2)/(4*x+1)",
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
            !result.contains("^(-1/2)"),
            "outer-scaled inverse tangent sqrt presentation regressed for {input}: {result}"
        );

        let expected =
            parse(canonical_equivalent, &mut engine.simplifier.context).expect("parse expected");
        assert!(
            engine.simplifier.are_equivalent(result_expr, expected),
            "post-calculus presentation must stay equivalent to the canonical derivative, got: {result}"
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
            "expected compact post-calculus presentation step for {input}"
        );
    }
}
#[test]
fn bounded_inverse_trig_reciprocal_sqrt_diff_uses_post_calculus_root_denominator_presentation() {
    let cases = [
        (
            "diff(arcsin(1/sqrt(x)), x)",
            "-1 / (2 * x * sqrt(x - 1))",
            vec!["x > 1".to_string()],
            "diff(arcsin(1/sqrt(x)), x) + 1/(2*x*sqrt(x-1))",
        ),
        (
            "diff(arccos(1/sqrt(x+1)), x)",
            "1 / (2 * (x + 1) * sqrt(x))",
            vec!["x > 0".to_string()],
            "diff(arccos(1/sqrt(x+1)), x) - 1/(2*(x+1)*sqrt(x))",
        ),
        (
            "diff(arcsin(2/sqrt(x)), x)",
            "-1 / (x * sqrt(x - 4))",
            vec!["x > 4".to_string()],
            "diff(arcsin(2/sqrt(x)), x) + 1/(x*sqrt(x-4))",
        ),
        (
            "diff(arcsin(-1/sqrt(x)), x)",
            "1 / (2 * x * sqrt(x - 1))",
            vec!["x > 1".to_string()],
            "diff(arcsin(-1/sqrt(x)), x) - 1/(2*x*sqrt(x-1))",
        ),
    ];

    for (input, expected, expected_required, residual_input) in cases {
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
        assert_eq!(result, expected, "input: {input}");
        assert!(
            !result.contains("^(-") && !result.contains(") / x)^"),
            "presentation should hide reciprocal-root internals for {input}: {result}"
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
fn arctan_shifted_sqrt_diff_uses_post_calculus_reciprocal_root_presentation() {
    for (input, expected_render, canonical_equivalent, residual_operator) in [
        (
            "diff(arctan(sqrt(x+1)), x)",
            "1 / (2 * sqrt(x + 1) * (x + 2))",
            "(x+1)^(-1/2)/(2*x+4)",
            "-",
        ),
        (
            "diff(arccot(sqrt(x+1)), x)",
            "-1 / (2 * sqrt(x + 1) * (x + 2))",
            "-(x+1)^(-1/2)/(2*x+4)",
            "+",
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
            !result.contains("(x + 1)^(-1/2)"),
            "presentation regressed for {input}: {result}"
        );
        assert!(
            !result.contains("2 * x + 4"),
            "denominator should remain compact in post-calculus presentation for {input}: {result}"
        );

        let expected =
            parse(canonical_equivalent, &mut engine.simplifier.context).expect("parse expected");
        assert!(
            engine.simplifier.are_equivalent(result_expr, expected),
            "post-calculus presentation must stay equivalent to the canonical derivative, got: {result}"
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
            "unexpected required_conditions for {input}: {required:?}"
        );

        let residual_input = format!("{input} {residual_operator} 1/(2*sqrt(x+1)*(x+2))");
        let parsed_residual =
            parse(&residual_input, &mut engine.simplifier.context).expect("parse residual");
        let residual_output = engine
            .eval(
                &mut state,
                EvalRequest {
                    raw_input: residual_input.clone(),
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
        assert_eq!(
            residual, "0",
            "shifted sqrt inverse-tangent residual did not collapse for {residual_input}"
        );

        let residual_required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &residual_output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();
        assert_eq!(
            residual_required,
            vec!["x > -1".to_string()],
            "unexpected residual required_conditions for {residual_input}: {residual_required:?}"
        );

        assert!(
            output
                .steps
                .iter()
                .any(|step| step.rule_name == "Symbolic Differentiation"),
            "expected the derivative to keep the ordinary symbolic differentiation trace for {input}"
        );
    }
}
#[test]
fn arctan_positive_affine_sqrt_diff_cancels_external_post_calculus_coefficient() {
    for (input, expected_render, canonical_equivalent, residual_operator) in [
        (
            "diff(arctan(sqrt(2*x+3)), x)",
            "1 / (2 * sqrt(2 * x + 3) * (x + 2))",
            "(2*x+3)^(-1/2)/(2*x+4)",
            "-",
        ),
        (
            "diff(arccot(sqrt(2*x+3)), x)",
            "-1 / (2 * sqrt(2 * x + 3) * (x + 2))",
            "-(2*x+3)^(-1/2)/(2*x+4)",
            "+",
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
            !result.contains("(2 * x + 3)^(-1/2)"),
            "presentation regressed for {input}: {result}"
        );
        assert!(
            !result.contains("2 / (2 * sqrt"),
            "external derivative coefficient should be cancelled safely for {input}: {result}"
        );
        assert!(
            !result.contains("2 * x + 4"),
            "post-calculus presentation should keep denominator content factored for {input}: {result}"
        );

        let expected =
            parse(canonical_equivalent, &mut engine.simplifier.context).expect("parse expected");
        assert!(
            engine.simplifier.are_equivalent(result_expr, expected),
            "post-calculus presentation must stay equivalent to the canonical derivative, got: {result}"
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
            vec!["x > -3/2".to_string()],
            "unexpected required_conditions for {input}: {required:?}"
        );

        let residual_input = format!("{input} {residual_operator} 1/(2*sqrt(2*x+3)*(x+2))");
        let parsed_residual =
            parse(&residual_input, &mut engine.simplifier.context).expect("parse residual");
        let residual_output = engine
            .eval(
                &mut state,
                EvalRequest {
                    raw_input: residual_input.clone(),
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
        assert_eq!(
            residual, "0",
            "positive affine sqrt inverse-tangent residual did not collapse for {residual_input}"
        );

        let residual_required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &residual_output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();
        assert_eq!(
            residual_required,
            vec!["x > -3/2".to_string()],
            "unexpected residual required_conditions for {residual_input}: {residual_required:?}"
        );

        assert!(
            output
                .steps
                .iter()
                .any(|step| step.rule_name == "Symbolic Differentiation"),
            "expected the derivative to keep the ordinary symbolic differentiation trace for {input}"
        );
    }
}
#[test]
fn arctan_positive_affine_sqrt_shifted_quotient_compacts_equivalent_denominator_content() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "(1 + diff(arctan(sqrt(2*x+3)), x))/(2 + 1/(sqrt(2*x+3)*(2*x+4)))";
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

    assert_eq!(
        result,
        "(1 / (sqrt(2 * x + 3) * (2 * x + 4)) + 1) / (1 / (sqrt(2 * x + 3) * (2 * x + 4)) + 2)"
    );
    assert!(
        !result.contains("diff"),
        "shifted quotient should compact the derivative side: {result}"
    );
    assert!(
        !result.contains("(2 * x + 3)^(-1/2)"),
        "presentation regressed: {result}"
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
            "1 / (sqrt(2 * x + 3) * (2 * x + 4)) + 2 ≠ 0".to_string(),
            "x > -3/2".to_string(),
        ],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn arctan_negative_affine_sqrt_shifted_quotient_compacts_opposite_orientation() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "(1 + diff(arctan(sqrt(5-3*x)), x))/(2 - 3/(2*sqrt(5-3*x)*(6-3*x)))";
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

    assert_eq!(
        result,
        "(1 - 3 / (2 * sqrt(5 - 3 * x) * (6 - 3 * x))) / (2 - 3 / (2 * sqrt(5 - 3 * x) * (6 - 3 * x)))"
    );
    assert!(
        !result.contains("diff"),
        "shifted quotient should compact the derivative side: {result}"
    );
    assert!(
        !result.contains("(5 - 3 * x)^(-1/2)"),
        "presentation regressed: {result}"
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
            "x < 5/3".to_string(),
            "2 - 3 / (2 * sqrt(5 - 3 * x) * (6 - 3 * x)) ≠ 0".to_string(),
        ],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn arctan_affine_sqrt_diff_cancels_gap_content_in_post_calculus_presentation() {
    for (
        input,
        expected_render,
        canonical_equivalent,
        forbidden_denominator_content,
        expected_condition,
    ) in [
        (
            "diff(arctan(sqrt(4*x+1)), x)",
            "1 / (sqrt(4 * x + 1) * (2 * x + 1))",
            "2*(4*x+1)^(-1/2)/(4*x+2)",
            "4 * x + 2",
            "x > -1/4",
        ),
        (
            "diff(arctan(sqrt(5-3*x)), x)",
            "-1 / (2 * sqrt(5 - 3 * x) * (2 - x))",
            "-3*(5-3*x)^(-1/2)/(2*(6-3*x))",
            "6 - 3 * x",
            "x < 5/3",
        ),
        (
            "diff(arccot(sqrt(4*x+1)), x)",
            "-1 / (sqrt(4 * x + 1) * (2 * x + 1))",
            "-2*(4*x+1)^(-1/2)/(4*x+2)",
            "4 * x + 2",
            "x > -1/4",
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
            !result.contains("^(-1/2)"),
            "presentation regressed for {input}: {result}"
        );
        assert!(
            !result.contains(forbidden_denominator_content),
            "post-calculus presentation should cancel common denominator content for {input}: {result}"
        );

        let expected =
            parse(canonical_equivalent, &mut engine.simplifier.context).expect("parse expected");
        assert!(
            engine.simplifier.are_equivalent(result_expr, expected),
            "post-calculus presentation must stay equivalent to the canonical derivative, got: {result}"
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
            vec![expected_condition.to_string()],
            "unexpected required_conditions for {input}: {required:?}"
        );

        assert!(
            output
                .steps
                .iter()
                .any(|step| step.rule_name == "Symbolic Differentiation"),
            "expected the derivative to keep the ordinary symbolic differentiation trace for {input}"
        );
    }
}
#[test]
fn arctan_negative_affine_sqrt_diff_keeps_sign_and_minimal_domain() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let input = "diff(arctan(sqrt(3-2*x)), x)";
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

    assert_eq!(result, "-1 / (2 * sqrt(3 - 2 * x) * (2 - x))");
    assert!(
        !result.contains("(3 - 2 * x)^(-1/2)"),
        "presentation regressed: {result}"
    );

    let expected =
        parse("-((3-2*x)^(-1/2)/(4-2*x))", &mut engine.simplifier.context).expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "post-calculus presentation must stay equivalent to the canonical derivative, got: {result}"
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
            .any(|step| step.rule_name == "Symbolic Differentiation"),
        "expected the derivative to keep the ordinary symbolic differentiation trace"
    );
}
#[test]
fn arctan_polynomial_sqrt_diff_uses_post_calculus_reciprocal_root_presentation() {
    for (input, expected_render, canonical_equivalent) in [
        (
            "diff(arctan(sqrt(x^2+1)), x)",
            "x / (sqrt(x^2 + 1) * (x^2 + 2))",
            "x*(x^2+1)^(-1/2)/(x^2+2)",
        ),
        (
            "diff(arctan(sqrt(x^2+2*x+2)), x)",
            "(x + 1) / (sqrt(x^2 + 2 * x + 2) * (x^2 + 2 * x + 3))",
            "((2*x+2)*(x^2+2*x+2)^(-1/2))/(2*x^2+4*x+6)",
        ),
        (
            "diff(arctan(sqrt(x^2-2*x+2)), x)",
            "(x - 1) / (sqrt(x^2 - 2 * x + 2) * (x^2 + 3 - 2 * x))",
            "((2*x-2)*(x^2-2*x+2)^(-1/2))/(2*x^2-4*x+6)",
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
            !result.contains("^(-1/2)"),
            "presentation should use a sqrt denominator, got: {result}"
        );
        assert!(
            !result.contains("2 * x^2 + 4 * x + 6"),
            "presentation should keep denominator content factored, got: {result}"
        );

        let expected =
            parse(canonical_equivalent, &mut engine.simplifier.context).expect("parse expected");
        assert!(
            engine.simplifier.are_equivalent(result_expr, expected),
            "post-calculus presentation must stay equivalent to the canonical derivative, got: {result}"
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
fn inverse_tangent_reciprocal_sqrt_polynomial_diff_uses_post_calculus_presentation() {
    for (input, expected_render, canonical_equivalent) in [
        (
            "diff(arctan(1/sqrt(x^2+x+1)), x)",
            "-(2 * x + 1) / (2 * sqrt(x^2 + x + 1) * (x^2 + x + 2))",
            "-(x^2+x+1)^(-1/2)*(2*x+1)/(2*x^2+2*x+4)",
        ),
        (
            "diff(arctan((x^2+x+1)^(-1/2)), x)",
            "-(2 * x + 1) / (2 * sqrt(x^2 + x + 1) * (x^2 + x + 2))",
            "-(x^2+x+1)^(-1/2)*(2*x+1)/(2*x^2+2*x+4)",
        ),
        (
            "diff(arccot(1/sqrt(x^2+x+1)), x)",
            "(2 * x + 1) / (2 * sqrt(x^2 + x + 1) * (x^2 + x + 2))",
            "(x^2+x+1)^(-1/2)*(2*x+1)/(2*x^2+2*x+4)",
        ),
        (
            "diff(arccot((x^2+x+1)^(-1/2)), x)",
            "(2 * x + 1) / (2 * sqrt(x^2 + x + 1) * (x^2 + x + 2))",
            "(x^2+x+1)^(-1/2)*(2*x+1)/(2*x^2+2*x+4)",
        ),
        (
            "diff(arctan(1/sqrt((x^2+x+1)/2)), x)",
            "-(2 * x + 1) / (2 * sqrt((x^2 + x + 1) / 2) * (x^2 + x + 3))",
            "-(2*x+1)/(4*sqrt((x^2+x+1)/2)*(1/2*x^2+1/2*x+3/2))",
        ),
        (
            "diff(arccot(1/sqrt((x^2+x+1)/2)), x)",
            "(2 * x + 1) / (2 * sqrt((x^2 + x + 1) / 2) * (x^2 + x + 3))",
            "(2*x+1)/(4*sqrt((x^2+x+1)/2)*(1/2*x^2+1/2*x+3/2))",
        ),
        (
            "diff(arctan(2/sqrt(x^2+x+1)), x)",
            "-(2 * x + 1) / (sqrt(x^2 + x + 1) * (x^2 + x + 5))",
            "-(x^2+x+1)^(-1/2)*(2*x+1)/(x^2+x+5)",
        ),
        (
            "diff(arccot(2/sqrt(x^2+x+1)), x)",
            "(2 * x + 1) / (sqrt(x^2 + x + 1) * (x^2 + x + 5))",
            "(x^2+x+1)^(-1/2)*(2*x+1)/(x^2+x+5)",
        ),
        (
            "diff(arctan(3/sqrt((x^2+x+1)/2)), x)",
            "-3 * (2 * x + 1) / (2 * sqrt((x^2 + x + 1) / 2) * (x^2 + x + 19))",
            "-3*(2*x+1)*((x^2+x+1)/2)^(-1/2)/(2*(x^2+x+19))",
        ),
        (
            "diff(arccot(3/sqrt((x^2+x+1)/2)), x)",
            "3 * (2 * x + 1) / (2 * sqrt((x^2 + x + 1) / 2) * (x^2 + x + 19))",
            "3*(2*x+1)*((x^2+x+1)/2)^(-1/2)/(2*(x^2+x+19))",
        ),
        (
            "diff(arctan(2*(x^2+x+1)^(-1/2)), x)",
            "-(2 * x + 1) / (sqrt(x^2 + x + 1) * (x^2 + x + 5))",
            "-(x^2+x+1)^(-1/2)*(2*x+1)/(x^2+x+5)",
        ),
        (
            "diff(arccot(2*(x^2+x+1)^(-1/2)), x)",
            "(2 * x + 1) / (sqrt(x^2 + x + 1) * (x^2 + x + 5))",
            "(x^2+x+1)^(-1/2)*(2*x+1)/(x^2+x+5)",
        ),
        (
            "diff(arctan(-2*(x^2+x+1)^(-1/2)), x)",
            "(2 * x + 1) / (sqrt(x^2 + x + 1) * (x^2 + x + 5))",
            "(x^2+x+1)^(-1/2)*(2*x+1)/(x^2+x+5)",
        ),
        (
            "diff(arctan(3*((x^2+x+1)/2)^(-1/2)), x)",
            "-3 * (2 * x + 1) / (2 * sqrt((x^2 + x + 1) / 2) * (x^2 + x + 19))",
            "-3*(2*x+1)*((x^2+x+1)/2)^(-1/2)/(2*(x^2+x+19))",
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
            !result.contains("^(-1/2)"),
            "presentation should use a sqrt denominator, got: {result}"
        );
        assert!(
            !result.contains("2 * x^2 + 2 * x + 4"),
            "presentation should keep denominator content factored, got: {result}"
        );

        let expected =
            parse(canonical_equivalent, &mut engine.simplifier.context).expect("parse expected");
        assert!(
            engine.simplifier.are_equivalent(result_expr, expected),
            "post-calculus presentation must stay equivalent to the canonical derivative, got: {result}"
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
fn inverse_tangent_reciprocal_sqrt_linear_diff_keeps_domain_conditions() {
    for (input, expected_render, canonical_equivalent) in [
        (
            "diff(arctan(2/sqrt(x)), x)",
            "-1 / (sqrt(x) * (x + 4))",
            "-x^(-1/2)/(x+4)",
        ),
        (
            "diff(arctan(2*x^(-1/2)), x)",
            "-1 / (sqrt(x) * (x + 4))",
            "-x^(-1/2)/(x+4)",
        ),
        (
            "diff(arctan(sqrt(1/x)), x)",
            "-1 / (2 * sqrt(x) * (x + 1))",
            "-x^(-1/2)/(2*x+2)",
        ),
        (
            "diff(arctan(sqrt(4/x)), x)",
            "-1 / (sqrt(x) * (x + 4))",
            "-x^(-1/2)/(x+4)",
        ),
        (
            "diff(arccot(2/sqrt(x)), x)",
            "1 / (sqrt(x) * (x + 4))",
            "x^(-1/2)/(x+4)",
        ),
        (
            "diff(arccot(2*x^(-1/2)), x)",
            "1 / (sqrt(x) * (x + 4))",
            "x^(-1/2)/(x+4)",
        ),
        (
            "diff(arccot(sqrt(1/x)), x)",
            "1 / (2 * sqrt(x) * (x + 1))",
            "x^(-1/2)/(2*x+2)",
        ),
        (
            "diff(arccot(sqrt(4/x)), x)",
            "1 / (sqrt(x) * (x + 4))",
            "x^(-1/2)/(x+4)",
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
            !result.contains("^(-1/2)"),
            "presentation should use a sqrt denominator, got: {result}"
        );

        let expected =
            parse(canonical_equivalent, &mut engine.simplifier.context).expect("parse expected");
        assert!(
            engine.simplifier.are_equivalent(result_expr, expected),
            "post-calculus presentation must stay equivalent to the canonical derivative, got: {result}"
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
            output
                .steps
                .iter()
                .any(|step| step.rule_name == "Symbolic Differentiation"),
            "expected the derivative to keep the ordinary symbolic differentiation trace"
        );

        let residual_input = format!("{input} - ({expected_render})");
        let parsed_residual =
            parse(&residual_input, &mut engine.simplifier.context).expect("parse residual");
        let residual_output = engine
            .eval(
                &mut state,
                EvalRequest {
                    raw_input: residual_input.clone(),
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
        assert_eq!(
            residual, "0",
            "linear reciprocal sqrt derivative residual did not collapse for {residual_input}"
        );

        let residual_required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &residual_output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();
        assert_eq!(
            residual_required,
            vec!["x > 0".to_string()],
            "unexpected residual required_conditions for {residual_input}: {residual_required:?}"
        );
    }
}
#[test]
fn inverse_tangent_reciprocal_sqrt_shifted_affine_diff_keeps_domain_conditions() {
    for (input, expected_render, canonical_equivalent) in [
        (
            "diff(arctan(2/sqrt(x+1)), x)",
            "-1 / (sqrt(x + 1) * (x + 5))",
            "-(x+1)^(-1/2)/(x+5)",
        ),
        (
            "diff(arctan(2*(x+1)^(-1/2)), x)",
            "-1 / (sqrt(x + 1) * (x + 5))",
            "-(x+1)^(-1/2)/(x+5)",
        ),
        (
            "diff(arctan(sqrt(1/(x+1))), x)",
            "-1 / (2 * sqrt(x + 1) * (x + 2))",
            "-(x+1)^(-1/2)/(2*x+4)",
        ),
        (
            "diff(arctan(sqrt(4/(x+1))), x)",
            "-1 / (sqrt(x + 1) * (x + 5))",
            "-(x+1)^(-1/2)/(x+5)",
        ),
        (
            "diff(arccot(2/sqrt(x+1)), x)",
            "1 / (sqrt(x + 1) * (x + 5))",
            "(x+1)^(-1/2)/(x+5)",
        ),
        (
            "diff(arccot(2*(x+1)^(-1/2)), x)",
            "1 / (sqrt(x + 1) * (x + 5))",
            "(x+1)^(-1/2)/(x+5)",
        ),
        (
            "diff(arccot(sqrt(1/(x+1))), x)",
            "1 / (2 * sqrt(x + 1) * (x + 2))",
            "(x+1)^(-1/2)/(2*x+4)",
        ),
        (
            "diff(arccot(sqrt(4/(x+1))), x)",
            "1 / (sqrt(x + 1) * (x + 5))",
            "(x+1)^(-1/2)/(x+5)",
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
            !result.contains("^(-1/2)"),
            "presentation should use a sqrt denominator, got: {result}"
        );

        let expected =
            parse(canonical_equivalent, &mut engine.simplifier.context).expect("parse expected");
        assert!(
            engine.simplifier.are_equivalent(result_expr, expected),
            "post-calculus presentation must stay equivalent to the canonical derivative, got: {result}"
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
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert!(
            output
                .steps
                .iter()
                .any(|step| step.rule_name == "Symbolic Differentiation"),
            "expected the derivative to keep the ordinary symbolic differentiation trace"
        );

        let residual_input = format!("{input} - ({expected_render})");
        let parsed_residual =
            parse(&residual_input, &mut engine.simplifier.context).expect("parse residual");
        let residual_output = engine
            .eval(
                &mut state,
                EvalRequest {
                    raw_input: residual_input.clone(),
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
        assert_eq!(
            residual, "0",
            "shifted affine reciprocal sqrt derivative residual did not collapse for {residual_input}"
        );

        let residual_required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &residual_output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();
        assert_eq!(
            residual_required,
            vec!["x > -1".to_string()],
            "unexpected residual required_conditions for {residual_input}: {residual_required:?}"
        );

        if matches!(
            input,
            "diff(arctan(sqrt(1/(x+1))), x)" | "diff(arccot(sqrt(1/(x+1))), x)"
        ) {
            assert_no_redundant_post_calculus_presentation_round_trip(output.steps.as_slice());
        }
    }
}
#[test]
fn inverse_tangent_reciprocal_sqrt_scaled_affine_diff_keeps_domain_conditions() {
    for (input, expected_render, canonical_equivalent) in [
        (
            "diff(arctan(2/sqrt(2*x+3)), x)",
            "-2 / (sqrt(2 * x + 3) * (2 * x + 7))",
            "-2*(2*x+3)^(-1/2)/(2*x+7)",
        ),
        (
            "diff(arctan(2*(2*x+3)^(-1/2)), x)",
            "-2 / (sqrt(2 * x + 3) * (2 * x + 7))",
            "-2*(2*x+3)^(-1/2)/(2*x+7)",
        ),
        (
            "diff(arccot(2/sqrt(2*x+3)), x)",
            "2 / (sqrt(2 * x + 3) * (2 * x + 7))",
            "2*(2*x+3)^(-1/2)/(2*x+7)",
        ),
        (
            "diff(arccot(2*(2*x+3)^(-1/2)), x)",
            "2 / (sqrt(2 * x + 3) * (2 * x + 7))",
            "2*(2*x+3)^(-1/2)/(2*x+7)",
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
            !result.contains("^(-1/2)"),
            "presentation should use a sqrt denominator, got: {result}"
        );

        let expected =
            parse(canonical_equivalent, &mut engine.simplifier.context).expect("parse expected");
        assert!(
            engine.simplifier.are_equivalent(result_expr, expected),
            "post-calculus presentation must stay equivalent to the canonical derivative, got: {result}"
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
            vec!["x > -3/2".to_string()],
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert!(
            output
                .steps
                .iter()
                .any(|step| step.rule_name == "Symbolic Differentiation"),
            "expected the derivative to keep the ordinary symbolic differentiation trace"
        );

        let residual_input = format!("{input} - ({expected_render})");
        let parsed_residual =
            parse(&residual_input, &mut engine.simplifier.context).expect("parse residual");
        let residual_output = engine
            .eval(
                &mut state,
                EvalRequest {
                    raw_input: residual_input.clone(),
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
        assert_eq!(
            residual, "0",
            "scaled affine reciprocal sqrt derivative residual did not collapse for {residual_input}"
        );

        let residual_required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &residual_output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();
        assert_eq!(
            residual_required,
            vec!["x > -3/2".to_string()],
            "unexpected residual required_conditions for {residual_input}: {residual_required:?}"
        );
    }
}
#[test]
fn inverse_tangent_reciprocal_sqrt_negative_affine_diff_keeps_domain_conditions() {
    for (input, expected_render, canonical_equivalent) in [
        (
            "diff(arctan(2/sqrt(3-2*x)), x)",
            "2 / (sqrt(3 - 2 * x) * (7 - 2 * x))",
            "2*(3-2*x)^(-1/2)/(7-2*x)",
        ),
        (
            "diff(arctan(2*(3-2*x)^(-1/2)), x)",
            "2 / (sqrt(3 - 2 * x) * (7 - 2 * x))",
            "2*(3-2*x)^(-1/2)/(7-2*x)",
        ),
        (
            "diff(arccot(2/sqrt(3-2*x)), x)",
            "-2 / (sqrt(3 - 2 * x) * (7 - 2 * x))",
            "-2*(3-2*x)^(-1/2)/(7-2*x)",
        ),
        (
            "diff(arccot(2*(3-2*x)^(-1/2)), x)",
            "-2 / (sqrt(3 - 2 * x) * (7 - 2 * x))",
            "-2*(3-2*x)^(-1/2)/(7-2*x)",
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
            !result.contains("^(-1/2)"),
            "presentation should use a sqrt denominator, got: {result}"
        );

        let expected =
            parse(canonical_equivalent, &mut engine.simplifier.context).expect("parse expected");
        assert!(
            engine.simplifier.are_equivalent(result_expr, expected),
            "post-calculus presentation must stay equivalent to the canonical derivative, got: {result}"
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
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert!(
            output
                .steps
                .iter()
                .any(|step| step.rule_name == "Symbolic Differentiation"),
            "expected the derivative to keep the ordinary symbolic differentiation trace"
        );

        let residual_input = format!("{input} - ({expected_render})");
        let parsed_residual =
            parse(&residual_input, &mut engine.simplifier.context).expect("parse residual");
        let residual_output = engine
            .eval(
                &mut state,
                EvalRequest {
                    raw_input: residual_input.clone(),
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
        assert_eq!(
            residual, "0",
            "negative affine reciprocal sqrt derivative residual did not collapse for {residual_input}"
        );

        let residual_required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &residual_output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();
        assert_eq!(
            residual_required,
            vec!["x < 3/2".to_string()],
            "unexpected residual required_conditions for {residual_input}: {residual_required:?}"
        );
    }
}
#[test]
fn arctan_reciprocal_scaled_polynomial_sqrt_diff_compacts_gap_presentation() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let input = "diff(arctan(sqrt((x^2+x+1)/2)), x)";
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

    assert_eq!(
        result,
        "(2 * x + 1) / (2 * sqrt((x^2 + x + 1) / 2) * (x^2 + x + 3))"
    );
    assert!(
        !result.contains("1/2 * x^2"),
        "post-calculus gap should not expose fractional polynomial content: {result}"
    );
    assert!(
        !result.contains("^(-1/2)"),
        "presentation should use a sqrt denominator, got: {result}"
    );

    let expected = parse(
        "((2*x+1)*((x^2+x+1)/2)^(-1/2))/(4*(1/2*x^2+1/2*x+3/2))",
        &mut engine.simplifier.context,
    )
    .expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "post-calculus presentation must stay equivalent to the canonical derivative, got: {result}"
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
        "unexpected required_conditions: {required:?}"
    );
    assert!(
        output
            .steps
            .iter()
            .any(|step| step.rule_name == "Symbolic Differentiation"),
        "expected the derivative to keep the ordinary symbolic differentiation trace"
    );
}
#[test]
fn arctan_reciprocal_scaled_affine_sqrt_diff_compacts_gap_presentation() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let input = "diff(arctan(sqrt((x+1)/3)), x)";
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

    assert_eq!(result, "1 / (2 * sqrt((x + 1) / 3) * (x + 4))");
    assert!(
        !result.contains("1/3 * x"),
        "post-calculus affine gap should not expose fractional polynomial content: {result}"
    );
    assert!(
        !result.contains("^(-1/2)"),
        "presentation should use a sqrt denominator, got: {result}"
    );

    let expected = parse(
        "1/(6*sqrt((x+1)/3)*(1/3*x+4/3))",
        &mut engine.simplifier.context,
    )
    .expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "post-calculus presentation must stay equivalent to the canonical derivative, got: {result}"
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
    assert!(
        output
            .steps
            .iter()
            .any(|step| step.rule_name == "Symbolic Differentiation"),
        "expected the derivative to keep the ordinary symbolic differentiation trace"
    );
}
#[test]
fn arctan_constant_over_affine_sqrt_diff_preserves_compact_denominator() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let input = "diff(arctan(sqrt(2/(x+1))), x)";
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

    assert_eq!(result, "-1 / (sqrt(2 / (x + 1)) * (x + 1) * (x + 3))");
    assert!(
        !result.contains("^(-1/2)"),
        "presentation should use a sqrt denominator, got: {result}"
    );
    assert!(
        !result.contains("x * (x + 1) + 3 * x + 3"),
        "presentation should preserve the factored denominator, got: {result}"
    );

    let expected = parse(
        "-((2/(x+1))^(-1/2)/(x*(x+1)+3*x+3))",
        &mut engine.simplifier.context,
    )
    .expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "post-calculus presentation must stay equivalent to the canonical derivative, got: {result}"
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
    assert!(
        output.domain_warnings.is_empty(),
        "unexpected domain warnings: {:?}",
        output.domain_warnings
    );
    assert!(
        output
            .steps
            .iter()
            .any(|step| step.rule_name == "Symbolic Differentiation"),
        "expected the derivative to keep the ordinary symbolic differentiation trace"
    );
}
#[test]
fn arctan_constant_over_quadratic_sqrt_diff_avoids_depth_fragile_route() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let input = "diff(arctan(sqrt(2/(x^2+1))), x)";
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

    assert_eq!(
        result,
        "-2 * x / (sqrt(2 / (x^2 + 1)) * (x^2 + 1) * (x^2 + 3))"
    );
    assert!(
        !result.contains("^(-1/2)"),
        "presentation should use a sqrt denominator, got: {result}"
    );

    let expected = parse(
        "((2/(x^2+1))^(-1/2)*x*-2)/((x^2+1)*(x^2+3))",
        &mut engine.simplifier.context,
    )
    .expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "post-calculus presentation must stay equivalent to the canonical derivative, got: {result}"
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
        "unexpected required_conditions: {required:?}"
    );
    assert!(
        output.domain_warnings.is_empty(),
        "unexpected domain warnings: {:?}",
        output.domain_warnings
    );
    assert!(
        output
            .steps
            .iter()
            .any(|step| step.rule_name == "Symbolic Differentiation"),
        "expected the derivative to keep the ordinary symbolic differentiation trace"
    );
}
#[test]
fn shifted_arcsin_diff_displays_surd_interior_domain_interval() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(arcsin((x+1)/sqrt(2)), x)";
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

    assert_eq!(result, "1 / sqrt(2 - (x + 1)^2)");

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert_eq!(
        required,
        vec!["-1 - sqrt(2) < x < -1 + sqrt(2)".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn inverse_function_diff_evaluates_with_required_domain_conditions() {
    let cases = [
        ("diff(arctan(x), x)", "x^2 + 1"),
        ("diff(asinh(x), x)", "x^2 + 1"),
    ];

    for (input, expected_core) in cases {
        let mut engine = Engine::new();
        let mut state = SessionState::new();
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

        assert!(!result.contains("diff("), "input: {input}, got: {result}");
        assert!(
            result.contains(expected_core),
            "input: {input}, got: {result}"
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
            "input: {input}, unexpected required_conditions: {required:?}"
        );
    }
}
#[test]
fn arctan_rational_affine_diff_uses_compact_denominator_presentation() {
    let cases = [
        ("diff(arctan((x+1)/2), x)", "2 / ((x + 1)^2 + 4)"),
        ("diff(arctan(-(x+1)/2), x)", "-2 / ((x + 1)^2 + 4)"),
        ("diff(arctan((x-1)/-2), x)", "-2 / ((x - 1)^2 + 4)"),
    ];

    for (input, expected_display) in cases {
        let mut engine = Engine::new();
        let mut state = SessionState::new();
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

        assert_eq!(result, expected_display, "input: {input}");

        let expected =
            parse(expected_display, &mut engine.simplifier.context).expect("parse expected");
        assert!(
            engine.simplifier.are_equivalent(result_expr, expected),
            "input: {input}, expected compact derivative equivalent to {expected_display}, got: {result}"
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
            "unexpected required_conditions for {input}: {required:?}"
        );
    }
}
#[test]
fn inverse_reciprocal_trig_direct_bounded_trig_diff_returns_undefined_when_real_open_interval_is_empty(
) {
    for input in [
        "diff(arcsec(sin(x)), x)",
        "diff(arccsc(cos(x)), x)",
        "diff(arcsec(cos(x)), x)",
        "diff(arccsc(sin(x)), x)",
        "diff(arccos(csc(x)), x)",
        "diff(arcsin(sec(x)), x)",
        "diff(arccos(sec(x)), x)",
        "diff(arcsin(csc(x)), x)",
        "diff(arccos(1/sin(x)), x)",
        "diff(arcsin(1/cos(x)), x)",
        "diff(arccos(sin(x)^(-1)), x)",
        "diff(arcsin(cos(x)^(-1)), x)",
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
            result, "undefined",
            "input: {input}, empty real-domain inverse reciprocal trig diff should be undefined, got: {result}"
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
            "input: {input}, empty-domain undefined result should not surface redundant Requires: {required:?}"
        );
        assert!(
            output
                .steps
                .iter()
                .any(|step| step.rule_name == "Symbolic Differentiation"),
            "input: {input}, empty real-domain inverse reciprocal trig diff should record the undefined differentiation step"
        );
        assert!(
            output.blocked_hints.len() <= 1,
            "input: {input}, empty real-domain inverse reciprocal trig diff should expose at most one domain hint"
        );
        if let Some(hint) = output.blocked_hints.first() {
            assert_eq!(hint.rule, "Symbolic Differentiation");
            assert!(
                hint.suggestion.contains("real domain is empty"),
                "input: {input}, expected a domain-empty hint, got: {}",
                hint.suggestion
            );
        }
    }
}
#[test]
fn bounded_inverse_trig_diff_evaluates_with_strict_required_domain_conditions() {
    let cases = [
        ("diff(arcsin(x), x)", "1/sqrt(1-x^2)"),
        ("diff(arccos(x), x)", "-1/sqrt(1-x^2)"),
    ];

    for (input, expected_derivative) in cases {
        let mut engine = Engine::new();
        let mut state = SessionState::new();
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

        assert!(!result.contains("diff("), "input: {input}, got: {result}");

        let expected =
            parse(expected_derivative, &mut engine.simplifier.context).expect("parse expected");
        assert!(
            engine.simplifier.are_equivalent(result_expr, expected),
            "input: {input}, expected derivative equivalent to {expected_derivative}, got: {result}"
        );
        if input.contains("arccos") {
            assert!(
                result == "-1 / sqrt(1 - x^2)"
                    && result.starts_with('-')
                    && !result.contains("^(-1/2)")
                    && !result.contains("x^2 - 1"),
                "input: {input}, expected negative reciprocal-root presentation without extra presentation guards, got: {result}"
            );
        } else {
            assert!(
                result.contains("sqrt(1 - x^2)") && !result.contains("^(-1/2)"),
                "input: {input}, expected reciprocal-root presentation, got: {result}"
            );
        }
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
            "input: {input}, required_conditions: {required:?}"
        );
    }
}
#[test]
fn bounded_inverse_trig_self_normalized_projection_diff_compacts_presentation() {
    let cases = [
        (
            "diff(arcsin(x/sqrt(x^2+1)), x)",
            "1 / (x^2 + 1)",
            Some("diff(arcsin(x/sqrt(x^2+1)), x) - 1/(x^2+1)"),
        ),
        (
            "diff(arccos(x/sqrt(x^2+1)), x)",
            "-1 / (x^2 + 1)",
            Some("diff(arccos(x/sqrt(x^2+1)), x) + 1/(x^2+1)"),
        ),
        (
            "diff(arcsin(-x/sqrt(x^2+1)), x)",
            "-1 / (x^2 + 1)",
            Some("diff(arcsin(-x/sqrt(x^2+1)), x) + 1/(x^2+1)"),
        ),
        (
            "diff(arccos(-x/sqrt(x^2+1)), x)",
            "1 / (x^2 + 1)",
            Some("diff(arccos(-x/sqrt(x^2+1)), x) - 1/(x^2+1)"),
        ),
        (
            "diff(arccos((2*x+1)/sqrt((2*x+1)^2+3)), x)",
            "-2 * sqrt(3) / ((2 * x + 1)^2 + 3)",
            None,
        ),
        (
            "diff(arccos((x^2+x+1)/sqrt((x^2+x+1)^2+5)), x)",
            "-sqrt(5) * (2 * x + 1) / ((x^2 + x + 1)^2 + 5)",
            Some(
                "diff(arccos((x^2+x+1)/sqrt((x^2+x+1)^2+5)), x) + sqrt(5)*(2*x+1)/((x^2+x+1)^2+5)",
            ),
        ),
        (
            "diff(arccos(-(x^2+x+1)/sqrt((x^2+x+1)^2+5)), x)",
            "sqrt(5) * (2 * x + 1) / ((x^2 + x + 1)^2 + 5)",
            Some(
                "diff(arccos(-(x^2+x+1)/sqrt((x^2+x+1)^2+5)), x) - sqrt(5)*(2*x+1)/((x^2+x+1)^2+5)",
            ),
        ),
    ];

    for (input, expected_render, residual_input) in cases {
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

        assert_eq!(result, expected_render, "input: {input}");
        assert!(
            !result.contains("^(-1/2)") && !result.contains("sqrt(x^2 + 1)"),
            "projection derivative should use the compact rational form, got: {result}"
        );
        assert!(
            output.required_conditions.is_empty(),
            "self-normalized projection should stay inside the open real inverse-trig domain without extra Requires: {:?}",
            output.required_conditions
        );
        assert!(
            output
                .steps
                .iter()
                .any(|step| step.rule_name == "Symbolic Differentiation"),
            "expected the ordinary symbolic differentiation trace"
        );

        if let Some(residual_input) = residual_input {
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
                other => panic!("expected expression residual result, got {other:?}"),
            };
            assert_eq!(residual, "0", "residual did not collapse for {input}");
        }
    }
}
#[test]
fn bounded_inverse_trig_sqrt_diff_uses_post_calculus_root_denominator_presentation() {
    for (input, expected_render, expected_required, expected_chain_rule) in [
        (
            "diff(arcsin(sqrt(x)), x)",
            "1 / (2 * sqrt(x) * sqrt(1 - x))",
            vec!["x > 0".to_string(), "x < 1".to_string()],
            None,
        ),
        (
            "diff(arccos(sqrt(x)), x)",
            "-1 / (2 * sqrt(x) * sqrt(1 - x))",
            vec!["x > 0".to_string(), "x < 1".to_string()],
            None,
        ),
        (
            "diff(arcsin(sqrt(x)/2), x)",
            "1 / (2 * sqrt(x) * sqrt(4 - x))",
            vec!["x > 0".to_string(), "x < 4".to_string()],
            None,
        ),
        (
            "diff(arccos(sqrt(x)/2), x)",
            "-1 / (2 * sqrt(x) * sqrt(4 - x))",
            vec!["x > 0".to_string(), "x < 4".to_string()],
            None,
        ),
        (
            "diff(arccos(-sqrt(x)/2), x)",
            "1 / (2 * sqrt(x) * sqrt(4 - x))",
            vec!["x > 0".to_string(), "x < 4".to_string()],
            None,
        ),
        (
            "diff(arccos(sqrt(x)/3), x)",
            "-1 / (2 * sqrt(x) * sqrt(9 - x))",
            vec!["x > 0".to_string(), "x < 9".to_string()],
            None,
        ),
        (
            "diff(arcsin(sqrt(4*x)/3), x)",
            "1 / (2 * sqrt(x) * sqrt(9/4 - x))",
            vec!["x > 0".to_string(), "x < 9/4".to_string()],
            None,
        ),
        (
            "diff(arccos(sqrt(4*x)/3), x)",
            "-1 / (2 * sqrt(x) * sqrt(9/4 - x))",
            vec!["x > 0".to_string(), "x < 9/4".to_string()],
            None,
        ),
        (
            "diff(arccos(-sqrt(4*x)/3), x)",
            "1 / (2 * sqrt(x) * sqrt(9/4 - x))",
            vec!["x > 0".to_string(), "x < 9/4".to_string()],
            None,
        ),
        (
            "diff(2*arcsin(sqrt(x)), x)",
            "1 / (sqrt(1 - x) * sqrt(x))",
            vec!["x < 1".to_string(), "x > 0".to_string()],
            None,
        ),
        (
            "diff(2*arccos(sqrt(x)), x)",
            "-1 / (sqrt(1 - x) * sqrt(x))",
            vec!["x < 1".to_string(), "x > 0".to_string()],
            None,
        ),
        (
            "diff(3*arcsin(sqrt(x)), x)",
            "3 / (2 * sqrt(1 - x) * sqrt(x))",
            vec!["x < 1".to_string(), "x > 0".to_string()],
            None,
        ),
        (
            "diff(arcsin(2*sqrt(x)-1), x)",
            "1 / (2 * sqrt(x) * sqrt(sqrt(x) - x))",
            vec!["x > 0".to_string(), "sqrt(x) - x > 0".to_string()],
            None,
        ),
        (
            "diff(2*arcsin(2*sqrt(x)-1), x)",
            "1 / (sqrt(sqrt(x) - x) * sqrt(x))",
            vec!["sqrt(x) - x > 0".to_string(), "x > 0".to_string()],
            None,
        ),
        (
            "diff(2*arccos(2*sqrt(x)-1), x)",
            "-1 / (sqrt(sqrt(x) - x) * sqrt(x))",
            vec!["sqrt(x) - x > 0".to_string(), "x > 0".to_string()],
            None,
        ),
        (
            "diff(3*arcsin(2*sqrt(x)-1), x)",
            "3 / (2 * sqrt(sqrt(x) - x) * sqrt(x))",
            vec!["sqrt(x) - x > 0".to_string(), "x > 0".to_string()],
            None,
        ),
        (
            "diff(arcsin(2*sqrt(2*x)-1), x)",
            "1 / (sqrt(2 * x) * sqrt(sqrt(2 * x) - 2 * x))",
            vec!["x > 0".to_string(), "sqrt(2 * x) - 2 * x > 0".to_string()],
            None,
        ),
        (
            "diff(arccos(2*sqrt(2*x)-1), x)",
            "-1 / (sqrt(2 * x) * sqrt(sqrt(2 * x) - 2 * x))",
            vec!["x > 0".to_string(), "sqrt(2 * x) - 2 * x > 0".to_string()],
            None,
        ),
        (
            "diff(arccos(2*sqrt(x)-1), x)",
            "-1 / (2 * sqrt(x) * sqrt(sqrt(x) - x))",
            vec!["x > 0".to_string(), "sqrt(x) - x > 0".to_string()],
            None,
        ),
        (
            "diff(arcsin(1-2*sqrt(x)), x)",
            "-1 / (2 * sqrt(x) * sqrt(sqrt(x) - x))",
            vec!["x > 0".to_string(), "sqrt(x) - x > 0".to_string()],
            None,
        ),
        (
            "diff(arcsin(1-2*sqrt(2*x)), x)",
            "-1 / (sqrt(2 * x) * sqrt(sqrt(2 * x) - 2 * x))",
            vec!["x > 0".to_string(), "sqrt(2 * x) - 2 * x > 0".to_string()],
            None,
        ),
        (
            "diff(arccos(1-2*sqrt(x)), x)",
            "1 / (2 * sqrt(x) * sqrt(sqrt(x) - x))",
            vec!["x > 0".to_string(), "sqrt(x) - x > 0".to_string()],
            None,
        ),
        (
            "diff(arccos(1-2*sqrt(2*x)), x)",
            "1 / (sqrt(2 * x) * sqrt(sqrt(2 * x) - 2 * x))",
            vec!["x > 0".to_string(), "sqrt(2 * x) - 2 * x > 0".to_string()],
            None,
        ),
        (
            "diff(arcsin(sqrt(2*x)), x)",
            "1 / (sqrt(2 * x) * sqrt(1 - 2 * x))",
            vec!["x < 1/2".to_string(), "x > 0".to_string()],
            None,
        ),
        (
            "diff(arcsin(sqrt(x+1)), x)",
            "1 / (2 * sqrt(x + 1) * sqrt(-x))",
            vec!["x > -1".to_string(), "x < 0".to_string()],
            None,
        ),
        (
            "diff(arcsin(sqrt((x+1)/(x+2))), x)",
            "1 / (2 * (x + 2) * sqrt(x + 1))",
            vec!["x > -1".to_string()],
            None,
        ),
        (
            "diff(arccos(sqrt((x+1)/(x+2))), x)",
            "-1 / (2 * (x + 2) * sqrt(x + 1))",
            vec!["x > -1".to_string()],
            None,
        ),
        (
            "diff(arcsin(sqrt((x+1)/(x+3))), x)",
            "sqrt(2) / (2 * (x + 3) * sqrt(x + 1))",
            vec!["x > -1".to_string()],
            None,
        ),
        (
            "diff(arccos(sqrt((x+1)/(x+3))), x)",
            "-sqrt(2) / (2 * (x + 3) * sqrt(x + 1))",
            vec!["x > -1".to_string()],
            None,
        ),
        (
            "diff(arcsin(sqrt(2*x+3)), x)",
            "1 / (sqrt(2 * x + 3) * sqrt(-2 * x - 2))",
            vec!["x > -3/2".to_string(), "x < -1".to_string()],
            Some("1/(sqrt(2*x+3)*sqrt(-2*x-2))"),
        ),
        (
            "diff(arccos(sqrt(2*x+3)), x)",
            "-1 / (sqrt(2 * x + 3) * sqrt(-2 * x - 2))",
            vec!["x > -3/2".to_string(), "x < -1".to_string()],
            Some("-1/(sqrt(2*x+3)*sqrt(-2*x-2))"),
        ),
        (
            "diff(arcsin(sqrt(5-3*x)), x)",
            "-3 / (2 * sqrt(5 - 3 * x) * sqrt(3 * x - 4))",
            vec!["x < 5/3".to_string(), "x > 4/3".to_string()],
            None,
        ),
        (
            "diff(arccos(sqrt(5-3*x)), x)",
            "3 / (2 * sqrt(5 - 3 * x) * sqrt(3 * x - 4))",
            vec!["x < 5/3".to_string(), "x > 4/3".to_string()],
            None,
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
            !result.contains("^(-1/2)"),
            "presentation should use explicit sqrt denominators, got: {result}"
        );
        if input.contains("arccos") && expected_render.starts_with('-') {
            assert!(
                result.starts_with('-'),
                "arccos sqrt derivative should keep negative orientation, got: {result}"
            );
        }
        if let Some(expected_chain_rule) = expected_chain_rule {
            let expected =
                parse(expected_chain_rule, &mut engine.simplifier.context).expect("parse expected");
            let mut vars = HashMap::new();
            vars.insert("x".to_string(), -1.25);
            let actual_value = eval_f64(&engine.simplifier.context, result_expr, &vars)
                .unwrap_or_else(|| panic!("input: {input}, could not eval result"));
            let expected_value = eval_f64(&engine.simplifier.context, expected, &vars)
                .unwrap_or_else(|| panic!("input: {input}, could not eval chain-rule form"));
            assert!(
                (actual_value - expected_value).abs() < 1e-10,
                "input: {input}, expected numeric chain-rule value {expected_value}, got {actual_value}"
            );
        }

        let mut required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();
        required.sort();

        let mut expected_required = expected_required;
        expected_required.sort();
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

    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let residual_input = "diff(arcsin(2*sqrt(x)-1), x) - 1/(2*sqrt(x)*sqrt(sqrt(x)-x))";
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
    let residual_result = match output.result {
        EvalResult::Expr(expr) => format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: expr,
            }
        ),
        other => panic!("expected expression residual result, got {other:?}"),
    };
    assert_eq!(residual_result, "0");

    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let residual_input = "diff(arcsin(sqrt((x+1)/(x+2))), x) - 1/(2*(x+2)*sqrt(x+1))";
    let parsed =
        parse(residual_input, &mut engine.simplifier.context).expect("parse quotient residual");
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
        .expect("eval quotient residual");
    let residual_result = match output.result {
        EvalResult::Expr(expr) => format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: expr,
            }
        ),
        other => panic!("expected expression residual result, got {other:?}"),
    };
    assert_eq!(residual_result, "0");

    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let residual_input = "diff(arcsin(sqrt((x+1)/(x+3))), x) - sqrt(2)/(2*(x+3)*sqrt(x+1))";
    let parsed = parse(residual_input, &mut engine.simplifier.context)
        .expect("parse non-square quotient residual");
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
        .expect("eval non-square quotient residual");
    let residual_result = match output.result {
        EvalResult::Expr(expr) => format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: expr,
            }
        ),
        other => panic!("expected expression residual result, got {other:?}"),
    };
    assert_eq!(residual_result, "0");

    let parsed = parse(
        "diff(arccos(sqrt((x+1)/(x+3))), x) + sqrt(2)/(2*(x+3)*sqrt(x+1))",
        &mut engine.simplifier.context,
    )
    .expect("parse non-square arccos quotient residual");
    let output = engine
        .eval(
            &mut state,
            EvalRequest {
                raw_input: "diff(arccos(sqrt((x+1)/(x+3))), x) + sqrt(2)/(2*(x+3)*sqrt(x+1))"
                    .to_string(),
                parsed,
                action: EvalAction::Simplify,
                auto_store: false,
            },
        )
        .expect("eval non-square arccos quotient residual");
    let residual_result = match output.result {
        EvalResult::Expr(expr) => format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: expr,
            }
        ),
        other => panic!("expected expression residual result, got {other:?}"),
    };
    assert_eq!(residual_result, "0");
    let residual_required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|condition| condition.display(&engine.simplifier.context))
    .collect();
    assert_eq!(
        residual_required,
        vec!["x > -1".to_string()],
        "non-square arccos quotient residual should not retain redundant open-interval guard"
    );
}
#[test]
fn bounded_inverse_trig_sqrt_rationalized_residual_uses_direct_fast_path() {
    for input in [
        "diff(arcsin(sqrt(x)), x) - sqrt(x)*sqrt(1-x)/(2*x*(1-x))",
        "diff(arccos(sqrt(x)), x) + sqrt(x)*sqrt(1-x)/(2*x*(1-x))",
    ] {
        let mut engine = Engine::new();
        let mut state = SessionState::new();
        state.options_mut().steps_mode = StepsMode::On;
        let parsed = parse(input, &mut engine.simplifier.context).expect("parse residual");
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
            .expect("eval residual");
        let result = match output.result {
            EvalResult::Expr(expr) => format!(
                "{}",
                DisplayExpr {
                    context: &engine.simplifier.context,
                    id: expr,
                }
            ),
            other => panic!("expected expression residual result, got {other:?}"),
        };
        assert_eq!(result, "0", "residual did not collapse for {input}");
        assert_eq!(
            output.steps.len(),
            1,
            "expected direct residual simplification for {input}"
        );
        assert_eq!(
            output.steps[0].rule_name,
            "Resolve inverse-trig derivative residual"
        );

        let required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();
        assert_eq!(required, vec!["x < 1".to_string(), "x > 0".to_string()]);
    }
}
#[test]
fn bounded_inverse_trig_sqrt_rationalized_wrong_orientation_does_not_collapse() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(arcsin(sqrt(x)), x) + sqrt(x)*sqrt(1-x)/(2*x*(1-x))";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse residual");
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
        .expect("eval residual");
    let result = match output.result {
        EvalResult::Expr(expr) => format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: expr,
            }
        ),
        other => panic!("expected expression residual result, got {other:?}"),
    };
    assert_ne!(result, "0");
}
#[test]
fn bounded_inverse_trig_surd_quotient_diff_compacts_open_interval_gap() {
    let cases = [
        (
            "diff(arcsin((x^2+x+1)/sqrt(7)), x)",
            "(2 * x + 1) / sqrt(7 - (x^2 + x + 1)^2)",
            "(2*x+1)/(sqrt(7)*sqrt(1-((x^2+x+1)/sqrt(7))^2))",
            "7 - (x^2 + x + 1)^2 > 0",
        ),
        (
            "diff(arccos((x^2+x+1)/sqrt(7)), x)",
            "-(2 * x + 1) / sqrt(7 - (x^2 + x + 1)^2)",
            "-(2*x+1)/(sqrt(7)*sqrt(1-((x^2+x+1)/sqrt(7))^2))",
            "7 - (x^2 + x + 1)^2 > 0",
        ),
        (
            "diff(arcsin((x^2+2*x+1)/sqrt(7)), x)",
            "2 * (x + 1) / sqrt(7 - (x + 1)^4)",
            "(2*x+2)/(sqrt(7)*sqrt(1-((x^2+2*x+1)/sqrt(7))^2))",
            "7 - (x + 1)^4 > 0",
        ),
        (
            "diff(arccos((x^2+2*x+1)/sqrt(7)), x)",
            "-2 * (x + 1) / sqrt(7 - (x + 1)^4)",
            "-(2*x+2)/(sqrt(7)*sqrt(1-((x^2+2*x+1)/sqrt(7))^2))",
            "7 - (x + 1)^4 > 0",
        ),
        (
            "diff(arcsin((x^2+x+1)/sqrt(2/3)), x)",
            "(2 * x + 1) * sqrt(3) / sqrt(2 - 3 * (x^2 + x + 1)^2)",
            "(2*x+1)/(sqrt(2/3)*sqrt(1-((x^2+x+1)/sqrt(2/3))^2))",
            "2 - 3 * (x^2 + x + 1)^2 > 0",
        ),
        (
            "diff(arccos(sqrt(3/2)*(x^2+x+1)), x)",
            "-(2 * x + 1) * sqrt(3) / sqrt(2 - 3 * (x^2 + x + 1)^2)",
            "-(sqrt(3/2)*(2*x+1))/sqrt(1-(sqrt(3/2)*(x^2+x+1))^2)",
            "2 - 3 * (x^2 + x + 1)^2 > 0",
        ),
    ];

    for (input, expected_result, expected_chain_rule, expected_required) in cases {
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
        assert!(
            result.contains("sqrt(")
                && !result.contains("^(-1/2)")
                && !result.contains("1 -")
                && !result.contains("x^4"),
            "input: {input}, expected compact reciprocal-root normalized gap, got: {result}"
        );

        let expected =
            parse(expected_chain_rule, &mut engine.simplifier.context).expect("parse expected");
        let samples: &[f64] = if input.contains("2/3") || input.contains("3/2") {
            &[-0.6, -0.5, -0.4]
        } else {
            &[-0.25, 0.0, 0.25]
        };
        for &sample in samples {
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

        assert_eq!(
            required,
            vec![expected_required.to_string()],
            "input: {input}, unexpected required_conditions: {required:?}"
        );
        assert!(
            !required
                .iter()
                .any(|cond| cond.contains("sqrt(") || cond.contains("1/7") || cond.contains("≠")),
            "input: {input}, required condition should be the strict normalized gap: {required:?}"
        );
    }
}
#[test]
fn bounded_inverse_trig_surd_quotient_diff_compacts_scaled_fourth_gap_condition() {
    let cases = [
        (
            "diff(arcsin(((x^2+x+1)^2)/sqrt(2/3)), x)",
            "2 * (2 * x^3 + 3 * x^2 + 3 * x + 1) * sqrt(3) / sqrt(2 - 3 * (x^2 + x + 1)^4)",
            "(4*x^3+6*x^2+6*x+2)/(sqrt(2/3)*sqrt(1-(((x^2+x+1)^2)/sqrt(2/3))^2))",
        ),
        (
            "diff(arcsin(((x^2+x+1)^2)/sqrt(2/3))/sqrt(3), x)",
            "2 * (2 * x^3 + 3 * x^2 + 3 * x + 1) / sqrt(2 - 3 * (x^2 + x + 1)^4)",
            "((4*x^3+6*x^2+6*x+2)/(sqrt(2/3)*sqrt(1-(((x^2+x+1)^2)/sqrt(2/3))^2)))/sqrt(3)",
        ),
        (
            "diff((1/sqrt(3))*arcsin(((x^2+x+1)^2)/sqrt(2/3)), x)",
            "2 * (2 * x^3 + 3 * x^2 + 3 * x + 1) / sqrt(2 - 3 * (x^2 + x + 1)^4)",
            "(1/sqrt(3))*((4*x^3+6*x^2+6*x+2)/(sqrt(2/3)*sqrt(1-(((x^2+x+1)^2)/sqrt(2/3))^2)))",
        ),
        (
            "diff(sqrt(3)^(-1)*arcsin(((x^2+x+1)^2)/sqrt(2/3)), x)",
            "2 * (2 * x^3 + 3 * x^2 + 3 * x + 1) / sqrt(2 - 3 * (x^2 + x + 1)^4)",
            "sqrt(3)^(-1)*((4*x^3+6*x^2+6*x+2)/(sqrt(2/3)*sqrt(1-(((x^2+x+1)^2)/sqrt(2/3))^2)))",
        ),
        (
            "diff(arcsin(((x^2+x+1)^2)/sqrt(2/3))*1/sqrt(3), x)",
            "2 * (2 * x^3 + 3 * x^2 + 3 * x + 1) / sqrt(2 - 3 * (x^2 + x + 1)^4)",
            "((4*x^3+6*x^2+6*x+2)/(sqrt(2/3)*sqrt(1-(((x^2+x+1)^2)/sqrt(2/3))^2)))*(1/sqrt(3))",
        ),
        (
            "diff(arcsin(sqrt(3/2)*(x^2+x+1)^2), x)",
            "2 * (2 * x^3 + 3 * x^2 + 3 * x + 1) * sqrt(3) / sqrt(2 - 3 * (x^2 + x + 1)^4)",
            "(sqrt(3/2)*(4*x^3+6*x^2+6*x+2))/sqrt(1-(sqrt(3/2)*(x^2+x+1)^2)^2)",
        ),
        (
            "diff(arccos(sqrt(3/2)*(x^2+x+1)^2), x)",
            "-2 * (2 * x^3 + 3 * x^2 + 3 * x + 1) * sqrt(3) / sqrt(2 - 3 * (x^2 + x + 1)^4)",
            "-(sqrt(3/2)*(4*x^3+6*x^2+6*x+2))/sqrt(1-(sqrt(3/2)*(x^2+x+1)^2)^2)",
        ),
    ];

    for (input, expected_result, expected_chain_rule) in cases {
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
        assert!(!result.contains("diff("), "input: {input}, got: {result}");

        let expected =
            parse(expected_chain_rule, &mut engine.simplifier.context).expect("parse expected");
        for sample in [-0.6, -0.5, -0.4] {
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

        assert_eq!(
            required,
            vec!["2 - 3 * (x^2 + x + 1)^4 > 0".to_string()],
            "input: {input}, unexpected required_conditions: {required:?}"
        );
    }
}
#[test]
fn bounded_inverse_trig_surd_quotient_scaled_diff_starts_with_direct_derivative_trace() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let input = "diff(arcsin((x^2+x+1)^2/sqrt(2/3))*1/sqrt(3), x)";
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

    assert_eq!(
        result,
        "2 * (2 * x^3 + 3 * x^2 + 3 * x + 1) / sqrt(2 - 3 * (x^2 + x + 1)^4)"
    );
    assert_eq!(
        output
            .steps
            .first()
            .map(|step| step.rule_name.as_str()),
        Some("Symbolic Differentiation"),
        "scaled bounded inverse-trig surd quotient diff should not expand the target before deriving; steps: {:?}",
        output
            .steps
            .iter()
            .map(|step| step.rule_name.as_str())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        output.steps.len(),
        1,
        "compact direct diff presentation should suppress redundant post-diff expansion/repair steps; steps: {:?}",
        output
            .steps
            .iter()
            .map(|step| step.rule_name.as_str())
            .collect::<Vec<_>>()
    );
    assert!(
        output.steps.iter().all(|step| {
            !matches!(
                step.rule_name.as_str(),
                "Expand Expression" | "Expand Binomial" | "Present calculus result in compact form"
            )
        }),
        "compact direct diff presentation should not expose expansion/repair noise; steps: {:?}",
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

    assert_eq!(required, vec!["2 - 3 * (x^2 + x + 1)^4 > 0".to_string()]);
}
#[test]
fn bounded_inverse_trig_surd_quotient_negative_orientation_compacts_derivative_content() {
    let cases = [
        (
            "diff(asin((1-2*x)^2/sqrt(5)), x)",
            "4 * (2 * x - 1) / sqrt(5 - (1 - 2 * x)^4)",
            "(8*x-4)/(sqrt(5)*sqrt(1-((1-2*x)^2/sqrt(5))^2))",
        ),
        (
            "diff(acos((1-2*x)^2/sqrt(5)), x)",
            "-4 * (2 * x - 1) / sqrt(5 - (1 - 2 * x)^4)",
            "-(8*x-4)/(sqrt(5)*sqrt(1-((1-2*x)^2/sqrt(5))^2))",
        ),
    ];

    for (input, expected_result, expected_chain_rule) in cases {
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
        assert!(
            !result.contains("1/2") && !result.contains("8 * x - 4"),
            "input: {input}, expected compact derivative content presentation, got: {result}"
        );

        let expected =
            parse(expected_chain_rule, &mut engine.simplifier.context).expect("parse expected");
        for sample in [0.0, 0.25, 0.5] {
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

        assert_eq!(
            required,
            vec!["5 - (1 - 2 * x)^4 > 0".to_string()],
            "input: {input}, unexpected required_conditions: {required:?}"
        );
    }
}
#[test]
fn inverse_reciprocal_trig_diff_evaluates_with_explicit_domain_conditions() {
    let representative_inputs = [
        "diff(arcsec(x), x)",
        "diff(arccsc(2*x), x)",
        "diff(arcsec(sqrt(x+1)), x)",
        "diff(arccsc(sqrt(3-2*x)), x)",
        "diff(arcsec(sqrt(1-2*x)), x)",
        "diff(arcsec(sqrt(x^2+1)), x)",
        "diff(arccsc(sqrt(x^2+2)), x)",
        "diff(arcsec(x^2+1), x)",
        "diff(arcsec(((1/3)*(x^2+x+3))^2), x)",
        "diff(arccsc(sqrt(3/2)*(x^2+x+3)), x)",
        "diff(arccot(x), x)",
    ];
    let cases: Vec<_> = inverse_reciprocal_trig_diff_exhaustive_cases()
        .into_iter()
        .filter(|(input, _, _)| representative_inputs.contains(input))
        .collect();
    assert_eq!(cases.len(), representative_inputs.len());
    assert_inverse_reciprocal_trig_diff_cases(cases);
}
#[test]
#[ignore = "exhaustive inverse reciprocal trig diff contract is debug-slow; CI keeps representative structural smoke"]
fn inverse_reciprocal_trig_diff_evaluates_with_explicit_domain_conditions_exhaustive() {
    assert_inverse_reciprocal_trig_diff_cases(inverse_reciprocal_trig_diff_exhaustive_cases());
}
#[test]
fn affine_arcsin_diff_drops_scaled_nonnegative_domain_shadow() {
    let cases = [
        ("diff(arcsin(2*x+1), x)", false),
        ("diff(arccos(2*x+1), x)", true),
    ];

    for (input, expect_negative) in cases {
        let mut engine = Engine::new();
        let mut state = SessionState::new();
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

        if expect_negative {
            assert_eq!(result, "-1 / sqrt(-x^2 - x)", "input: {input}");
        } else {
            assert_eq!(result, "1 / sqrt(-x^2 - x)", "input: {input}");
        }

        let required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();

        assert_eq!(
            required,
            vec!["-1 < x < 0".to_string()],
            "input: {input}, unexpected required_conditions: {required:?}"
        );
    }
}
#[test]
fn bounded_inverse_trig_polynomial_rationalized_residual_uses_direct_fast_path() {
    let cases = [
        (
            "diff(arcsin(2*x+1), x) - 2*sqrt(1-(2*x+1)^2)/(1-(2*x+1)^2)",
            vec!["-1 < x < 0".to_string()],
        ),
        (
            "diff(arccos(2*x+1), x) + 2*sqrt(1-(2*x+1)^2)/(1-(2*x+1)^2)",
            vec!["-1 < x < 0".to_string()],
        ),
        (
            "diff(arcsin(3*x-1), x) - 3*sqrt(1-(3*x-1)^2)/(1-(3*x-1)^2)",
            vec!["0 < x < 2/3".to_string()],
        ),
        (
            "diff(arcsin(x^2), x) - 2*x*sqrt(1-x^4)/(1-x^4)",
            vec!["1 - x^4 > 0".to_string()],
        ),
        (
            "diff(arccos(x^2), x) + 2*x*sqrt(1-x^4)/(1-x^4)",
            vec!["1 - x^4 > 0".to_string()],
        ),
        (
            "diff(arcsin(x^2+x), x) - (2*x+1)*sqrt(1-(x^2+x)^2)/(1-(x^2+x)^2)",
            vec!["1 - (x^2 + x)^2 > 0".to_string()],
        ),
        (
            "diff(arcsin(x^3), x) - 3*x^2*sqrt(1-x^6)/(1-x^6)",
            vec!["1 - x^6 > 0".to_string()],
        ),
        (
            "diff(arccos(x^3), x) + 3*x^2*sqrt(1-x^6)/(1-x^6)",
            vec!["1 - x^6 > 0".to_string()],
        ),
        (
            "diff(arcsin(x^3+x), x) - (3*x^2+1)*sqrt(1-(x^3+x)^2)/(1-(x^3+x)^2)",
            vec!["1 - x^6 - 2 * x^4 - x^2 > 0".to_string()],
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

        let step_rules = output
            .steps
            .iter()
            .map(|step| step.rule_name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            step_rules,
            vec!["Resolve inverse-trig derivative residual"],
            "input: {input}, expected direct residual fast path"
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
#[test]
fn arctan_additive_trig_root_diff_keeps_radicand_plus_one_factored() {
    for (input, expected_result) in [
        (
            "diff(arctan(sqrt(sin(x^2)+cos(x)+4)), x)",
            "(2 * x * cos(x^2) - sin(x)) / (2 * sqrt(sin(x^2) + cos(x) + 4) * (sin(x^2) + cos(x) + 5))",
        ),
        (
            "diff(arctan(sqrt(sin(2*x)+cos(x)+4)), x)",
            "(cos(2 * x) - 1/2 * sin(x)) / (sqrt(sin(2 * x) + cos(x) + 4) * (sin(2 * x) + cos(x) + 5))",
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
            !result.contains("2 * sin(x^2)")
                && !result.contains("2 * cos(x)")
                && !result.contains("+ 1 + 4"),
            "arctan/sqrt presentation should preserve the compact radicand+1 factor: {result}"
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
            "bounded positive trig radicand should not surface new requirements: {required:?}"
        );
    }
}
#[test]
fn arctan_additive_trig_root_diff_residual_collapses_with_factored_gap() {
    for input in [
        "diff(arctan(sqrt(sin(x^2)+cos(x)+4)), x) - \
        (2*x*cos(x^2)-sin(x))/(2*sqrt(sin(x^2)+cos(x)+4)*(sin(x^2)+cos(x)+5))",
        "diff(arctan(sqrt(sin(2*x)+cos(x)+4)), x) - \
        (cos(2*x)-sin(x)/2)/(sqrt(sin(2*x)+cos(x)+4)*(sin(2*x)+cos(x)+5))",
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
        assert!(
            output.steps.len() <= 2,
            "expected arctan additive-trig root residual to close before cleanup, got: {:?}",
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
        assert!(
            required.is_empty(),
            "bounded positive trig radicand should not surface new requirements: {required:?}"
        );
    }
}
#[test]
fn arctan_sec_csc_exp_sqrt_root_diff_uses_inline_reciprocal_trig_presentation_and_residual() {
    for (direct_input, expected, expected_required, residual_input) in [
        (
            "diff(arctan(sqrt(sec(x)+exp(x)+sqrt(x)+x)), x)",
            "(e^x + tan(x) * sec(x) + 1 / (2 * sqrt(x)) + 1) / (2 * sqrt(sec(x) + sqrt(x) + e^x + x) * (sec(x) + sqrt(x) + e^x + x + 1))",
            vec![
                "sec(x) + sqrt(x) + e^x + x > 0".to_string(),
                "cos(x) ≠ 0".to_string(),
                "x > 0".to_string(),
            ],
            "diff(arctan(sqrt(sec(x)+exp(x)+sqrt(x)+x)), x) - (e^x+sec(x)*tan(x)+1/(2*sqrt(x))+1)/(2*sqrt(sec(x)+sqrt(x)+e^x+x)*(sec(x)+sqrt(x)+e^x+x+1))",
        ),
        (
            "diff(arctan(sqrt(csc(x)+exp(x)+sqrt(x)+x)), x)",
            "(e^x + 1 / (2 * sqrt(x)) + 1 - csc(x) * cot(x)) / (2 * sqrt(csc(x) + sqrt(x) + e^x + x) * (csc(x) + sqrt(x) + e^x + x + 1))",
            vec![
                "csc(x) + sqrt(x) + e^x + x > 0".to_string(),
                "sin(x) ≠ 0".to_string(),
                "x > 0".to_string(),
            ],
            "diff(arctan(sqrt(csc(x)+exp(x)+sqrt(x)+x)), x) - (e^x+1/(2*sqrt(x))+1-csc(x)*cot(x))/(2*sqrt(csc(x)+sqrt(x)+e^x+x)*(csc(x)+sqrt(x)+e^x+x+1))",
        ),
    ] {
        let mut engine = Engine::new();
        let mut state = SessionState::new();
        state.options_mut().steps_mode = StepsMode::On;

        let parsed = parse(direct_input, &mut engine.simplifier.context).expect("parse direct");
        let output = engine
            .eval(
                &mut state,
                EvalRequest {
                    raw_input: direct_input.to_string(),
                    parsed,
                    action: EvalAction::Simplify,
                    auto_store: false,
                },
            )
            .expect("eval direct");

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

        assert_eq!(result, expected, "input: {direct_input}");
        assert_eq!(
            output.steps.len(),
            1,
            "arctan reciprocal trig root diff should stay on direct presentation: {:?}",
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
        assert_eq!(required, expected_required, "input: {direct_input}");

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
        assert_eq!(result, "0", "input: {residual_input}");
        assert_eq!(
            output.steps.len(),
            1,
            "arctan reciprocal trig residual should close before cleanup, got: {:?}",
            output
                .steps
                .iter()
                .map(|step| step.rule_name.as_str())
                .collect::<Vec<_>>()
        );
    }
}
#[test]
fn sec_log_reciprocal_sqrt_arctan_residual_subtractive_wrappers_stay_bounded() {
    let residual = "diff(arctan(sqrt(sec(x)+ln(x)+1/sqrt(x)+x)),x)-(2*sqrt(x)+2*x*sqrt(x)+2*x*tan(x)*sec(x)*sqrt(x)-1)/(4*x*sqrt(x)*sqrt(sec(x)+ln(x)+1/sqrt(x)+x)*(sec(x)+ln(x)+1/sqrt(x)+x+1))";

    for (input, expected) in [
        (format!("(1-({residual}))/(x+2)"), "1 / (x + 2)"),
        (format!("(({residual})-1)/(x+2)"), "-1 / (x + 2)"),
    ] {
        let mut engine = Engine::new();
        let mut state = SessionState::new();
        let parsed = parse(&input, &mut engine.simplifier.context).expect("parse");

        let output = engine
            .eval(
                &mut state,
                EvalRequest {
                    raw_input: input.clone(),
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
            vec![
                "x > 0".to_string(),
                "sec(x) + ln(x) + 1 / sqrt(x) + x > 0".to_string(),
                "cos(x) ≠ 0".to_string(),
            ],
            "input: {input}"
        );
    }
}
#[test]
fn arctan_sqrt_exp_trig_log_root_diff_stays_compact_and_fast() {
    for (input, expected_result, expected_required) in [
        (
            "diff(sqrt(exp(sin(x))+ln(x)+sqrt(x)), x)",
            "(2 * sqrt(x) + 2 * x * cos(x) * sqrt(x) * e^sin(x) + x) / (4 * x * sqrt(x) * sqrt(ln(x) + sqrt(x) + e^sin(x)))",
            vec![
                "x > 0".to_string(),
                "ln(x) + sqrt(x) + e^sin(x) > 0".to_string(),
            ],
        ),
        (
            "diff(arctan(sqrt(exp(sin(x))+ln(x)+sqrt(x))), x)",
            "(2 * sqrt(x) + 2 * x * cos(x) * sqrt(x) * e^sin(x) + x) / (4 * x * sqrt(x) * sqrt(ln(x) + sqrt(x) + e^sin(x)) * (ln(x) + sqrt(x) + e^sin(x) + 1))",
            vec![
                "x > 0".to_string(),
                "ln(x) + sqrt(x) + e^sin(x) > 0".to_string(),
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
        let step_names = output
            .steps
            .iter()
            .map(|step| step.rule_name.as_str())
            .collect::<Vec<_>>();
        assert!(
            step_names.contains(&"Symbolic Differentiation"),
            "expected symbolic differentiation step, got: {step_names:?}"
        );
        assert!(
            !step_names
                .iter()
                .any(|name| *name == "Rationalize Product Denominator"
                    || *name == "N-ary Mul Combine Powers"),
            "unexpected noisy cleanup in compact elementary diff path: {step_names:?}"
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
        for condition in expected_required {
            assert!(
                required.contains(&condition),
                "missing required condition {condition:?}; got {required:?}; input: {input}"
            );
        }
    }
}
#[test]
fn arctan_sqrt_exp_trig_log_root_diff_residual_collapses_before_cleanup() {
    let input = "diff(arctan(sqrt(exp(sin(x))+ln(x)+sqrt(x))), x) - (2*x*sqrt(x)*cos(x)*e^sin(x)+2*sqrt(x)+x)/(4*x*sqrt(x)*sqrt(exp(sin(x))+ln(x)+sqrt(x))*(exp(sin(x))+ln(x)+sqrt(x)+1))";
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

    assert_eq!(result, "0");
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
            "x > 0".to_string(),
            "ln(x) + sqrt(x) + e^sin(x) > 0".to_string()
        ]
    );
}
#[test]
fn arctan_sqrt_tan_mixed_sqrt_reciprocal_sqrt_diff_uses_inner_common_denominator_presentation() {
    for (input, expected, expected_required) in [
        (
            "diff(arctan(sqrt(tan(x)+sqrt(x)+1/sqrt(x)+x)), x)",
            "(2 * x * sqrt(x) + 2 * x * sqrt(x) * sec(x)^2 + x - 1) / (4 * x * sqrt(x) * sqrt(tan(x) + sqrt(x) + 1 / sqrt(x) + x) * (tan(x) + sqrt(x) + 1 / sqrt(x) + x + 1))",
            vec![
                "x > 0".to_string(),
                "tan(x) + sqrt(x) + 1 / sqrt(x) + x > 0".to_string(),
                "cos(x) ≠ 0".to_string(),
            ],
        ),
        (
            "diff(arctan(sqrt(tan(x)+2*sqrt(x)-3/sqrt(x)+x)), x)",
            "(2 * x * sqrt(x) + 2 * x * sqrt(x) * sec(x)^2 + 2 * x + 3) / (4 * x * sqrt(x) * sqrt(tan(x) + 2 * sqrt(x) - 3 / sqrt(x) + x) * (tan(x) + 2 * sqrt(x) - 3 / sqrt(x) + x + 1))",
            vec![
                "x > 0".to_string(),
                "tan(x) + 2 * sqrt(x) - 3 / sqrt(x) + x > 0".to_string(),
                "cos(x) ≠ 0".to_string(),
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

        assert_eq!(result, expected, "input: {input}");
        assert!(
            !result.contains("x^(-3/2)") && !result.contains("sqrt(x) * x^("),
            "post-calculus presentation should avoid mixed half-power noise: {result}"
        );
        assert_eq!(
            output.steps.len(),
            1,
            "expected direct arctan/sqrt/tan/root presentation, got: {:?}",
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
        assert_eq!(required, expected_required, "input: {input}");
    }
}
#[test]
fn arctan_sqrt_tan_mixed_sqrt_reciprocal_sqrt_diff_residual_collapses_before_cleanup() {
    for (input, expected_required) in [
        (
            "diff(arctan(sqrt(tan(x)+sqrt(x)+1/sqrt(x)+x)), x) - (2*x*sqrt(x)*sec(x)^2+2*x*sqrt(x)+x-1)/(4*x*sqrt(x)*sqrt(tan(x)+sqrt(x)+1/sqrt(x)+x)*(tan(x)+sqrt(x)+1/sqrt(x)+x+1))",
            vec![
                "x > 0".to_string(),
                "tan(x) + sqrt(x) + 1 / sqrt(x) + x > 0".to_string(),
                "cos(x) ≠ 0".to_string(),
            ],
        ),
        (
            "diff(arctan(sqrt(tan(x)+2*sqrt(x)-3/sqrt(x)+x)), x) - (2*x*sqrt(x)*sec(x)^2+2*x*sqrt(x)+2*x+3)/(4*x*sqrt(x)*sqrt(tan(x)+2*sqrt(x)-3/sqrt(x)+x)*(tan(x)+2*sqrt(x)-3/sqrt(x)+x+1))",
            vec![
                "x > 0".to_string(),
                "tan(x) + 2 * sqrt(x) - 3 / sqrt(x) + x > 0".to_string(),
                "cos(x) ≠ 0".to_string(),
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
            "expected arctan/sqrt/tan/root residual to close before cleanup, got: {:?}",
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
        assert_eq!(required, expected_required, "input: {input}");
    }
}
