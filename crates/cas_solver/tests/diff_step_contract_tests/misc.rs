use super::*;

#[test]
fn polynomial_diff_first_step_omits_zero_products_and_unit_factors() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(x^3 + 2*x^2 - 5*x + 1, x)";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    let first = output
        .steps
        .iter()
        .find(|step| step.rule_name.as_str() == "Symbolic Differentiation")
        .expect("symbolic differentiation step");

    assert_eq!(first.rule_name.as_str(), "Symbolic Differentiation");

    let local_after = format!(
        "{}",
        DisplayExpr {
            context: &engine.simplifier.context,
            id: first.after,
        }
    );
    let global_after = format!(
        "{}",
        DisplayExpr {
            context: &engine.simplifier.context,
            id: first.global_after.expect("global_after"),
        }
    );

    assert!(
        !local_after.contains("0 ·"),
        "first local diff step still contains zero-product noise: {local_after}"
    );
    assert!(
        !local_after.contains("· 1"),
        "first local diff step still contains unit-factor noise: {local_after}"
    );
    assert!(
        !global_after.contains("0 ·"),
        "first global diff step still contains zero-product noise: {global_after}"
    );
    assert!(
        !global_after.contains("· 1"),
        "first global diff step still contains unit-factor noise: {global_after}"
    );

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
    assert_eq!(result, "3 * x^2 + 4 * x - 5");
}
#[test]
fn absolute_value_diff_composite_arguments_preserve_nonsmooth_conditions() {
    let cases = [
        (
            "diff(abs(sin(x)), x)",
            "sin(x) * cos(x) / |sin(x)|",
            vec!["sin(x) ≠ 0".to_string()],
        ),
        (
            "diff(abs(x^2-1), x)",
            "((x^2 - 1) * x * 2)/|x^2 - 1|",
            vec!["x ≠ 1".to_string(), "x ≠ -1".to_string()],
        ),
    ];

    for (input, expected_derivative, expected_conditions) in cases {
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

        assert_eq!(result, expected_derivative, "input: {input}");

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
fn reciprocal_diff_evaluates_with_nonzero_domain_condition() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(1/x, x)";
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

    assert_eq!(result, "-1 / x^2");

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert_eq!(
        required,
        vec!["x ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn absolute_value_diff_quotient_argument_uses_compact_domain_safe_form() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(abs((x-1)/(x+1)), x)";
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

    assert_eq!(result, "(2 * x - 2) / (|(x - 1) / (x + 1)| * (x + 1)^3)");
    assert!(
        output.steps.len() <= 3,
        "unexpected noisy abs quotient derivative route: {} steps",
        output.steps.len()
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
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn absolute_value_diff_evaluates_with_nonsmooth_point_condition() {
    let cases = [
        // d/dx |h| presents as the textbook sign(h) (affine h), with the
        // non-differentiable point excluded via the h != 0 condition.
        ("diff(abs(x), x)", "sign(x)", vec!["x ≠ 0".to_string()]),
        (
            "diff(abs(2*x+1), x)",
            "2 * sign(2 * x + 1)",
            vec!["x ≠ -1/2".to_string()],
        ),
    ];

    for (input, expected_derivative, expected_conditions) in cases {
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

        assert_eq!(result, expected_derivative, "input: {input}");

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
fn sign_polynomial_diff_evaluates_zero_with_nonsmooth_conditions() {
    let cases = [
        ("diff(sign(x), x)", vec!["x ≠ 0".to_string()]),
        ("diff(sign(2*x+1), x)", vec!["x ≠ -1/2".to_string()]),
        (
            "diff(sign(x^2-1), x)",
            vec!["x ≠ 1".to_string(), "x ≠ -1".to_string()],
        ),
    ];

    for (input, expected_conditions) in cases {
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
fn sign_nonpolynomial_diff_remains_residual_with_inner_domain_condition() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(sign(sqrt(x)), x)";
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

    assert_eq!(result, "diff(sign(x^(1/2)), x)");

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert_eq!(
        required,
        vec!["x ≥ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn affine_total_domain_inverse_diff_drops_redundant_quadratic_conditions() {
    let cases = [
        ("diff(arctan(2*x+1), x)", "2 / ((2*x + 1)^2 + 1)"),
        ("diff(asinh(2*x+1), x)", "2 * ((2*x + 1)^2 + 1)^(-1/2)"),
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

        let expected = parse(expected_derivative, &mut engine.simplifier.context)
            .expect("parse expected derivative");
        assert!(
            engine.simplifier.are_equivalent(result_expr, expected),
            "input: {input}, expected derivative equivalent to {expected_derivative}, got: {result}"
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
