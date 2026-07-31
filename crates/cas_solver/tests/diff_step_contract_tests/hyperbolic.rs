use super::*;

#[test]
fn affine_hyperbolic_cubic_primitive_diff_residual_collapses() {
    for input in [
        "diff(1/2*(1/3*cosh(2*x+1)^3-cosh(2*x+1)), x) - sinh(2*x+1)^3",
        "diff(1/2*(sinh(2*x+1)+1/3*sinh(2*x+1)^3), x) - cosh(2*x+1)^3",
        "diff(-1/2*(sinh(1-2*x)+1/3*sinh(1-2*x)^3), x) - cosh(1-2*x)^3",
    ] {
        let mut engine = Engine::new();
        let mut state = SessionState::new();
        state.options_mut().steps_mode = StepsMode::Off;

        let output =
            evaluate_eval_command_output(&mut engine, &mut state, input, false).expect("eval");
        let result = output
            .result_line
            .as_ref()
            .expect("result line")
            .line
            .as_str();

        assert_eq!(result, "Result: 0", "input: {input}");
        assert!(
            output.metadata.requires_lines.is_empty(),
            "input: {input}, unexpected required conditions: {:?}",
            output.metadata.requires_lines
        );
    }
}
#[test]
fn hyperbolic_diff_evaluates_to_symbolic_derivatives() {
    let cases = [
        ("diff(sinh(x), x)", "cosh(x)"),
        ("diff(cosh(x), x)", "sinh(x)"),
        ("diff(tanh(x), x)", "1 / cosh(x)^2"),
    ];

    for (input, expected) in cases {
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

        assert_eq!(result, expected, "input: {input}");
    }
}
#[test]
fn affine_tanh_diff_uses_compact_chain_quotient_with_pole_conditions() {
    let cases = [
        (
            "diff(tanh(2*x+1), x)",
            "2/cosh(2*x+1)^2",
            Vec::<String>::new(),
        ),
        (
            "diff(sinh(1-2*x)/cosh(1-2*x), x)",
            "-2/cosh(1-2*x)^2",
            Vec::<String>::new(),
        ),
    ];

    for (input, expected_derivative, expected_conditions) in cases {
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

        assert!(!result.contains("diff("), "input: {input}, got: {result}");
        assert!(
            !result.contains("1 *") && !result.contains("1·"),
            "affine tanh derivative should not expose a unit multiplier: {result}"
        );

        let expected =
            parse(expected_derivative, &mut engine.simplifier.context).expect("parse expected");
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

        assert_eq!(
            required, expected_conditions,
            "input: {input}, unexpected required_conditions: {required:?}"
        );
        assert!(
            output
                .steps
                .iter()
                .any(|step| step.rule_name.as_str() == "Symbolic Differentiation"),
            "expected a visible differentiation step for {input}"
        );
    }
}
#[test]
fn affine_linear_times_tanh_diff_keeps_product_rule_shape() {
    let cases = [
        (
            "diff((x+1)*tanh(2*x+1), x)",
            "tanh(2 * x + 1) + (2 * x + 2) / cosh(2 * x + 1)^2",
        ),
        (
            "diff((3*x+2)*tanh(2*x+1), x)",
            "3 * tanh(2 * x + 1) + (6 * x + 4) / cosh(2 * x + 1)^2",
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

        assert_eq!(result, expected, "unexpected result for {input}");
        assert!(
            !result.contains("diff("),
            "unexpected residual diff for {input}: {result}"
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
            Vec::<String>::new(),
            "input {input}: unexpected required_conditions: {required:?}"
        );
        assert!(
            output
                .steps
                .iter()
                .any(|step| step.rule_name.as_str() == "Symbolic Differentiation"),
            "expected a visible differentiation step for {input}"
        );
    }
}
#[test]
fn affine_hyperbolic_coth_quotient_diff_uses_direct_derivative() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let input = "diff(cosh(2*x+1)/sinh(2*x+1), x)";
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

    let expected =
        parse("-2/sinh(2*x+1)^2", &mut engine.simplifier.context).expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "expected affine hyperbolic coth derivative, got: {result}"
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
        vec!["sinh(2 * x + 1) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );

    assert!(
        output
            .steps
            .iter()
            .any(|step| step.rule_name.as_str() == "Symbolic Differentiation"),
        "expected a visible differentiation step"
    );
    assert!(
        !output
            .steps
            .iter()
            .any(|step| step.rule_name.as_str() == "Pythagorean Identity"),
        "hyperbolic coth quotient derivative should not require a post-quotient identity collapse"
    );
}
#[test]
fn affine_linear_times_hyperbolic_coth_diff_keeps_product_rule_shape() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let input = "diff((x+1)*cosh(2*x+1)/sinh(2*x+1), x)";
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
        "1 / tanh(2 * x + 1) - (2 * x + 2) / sinh(2 * x + 1)^2"
    );
    assert!(
        !result.contains("diff("),
        "unexpected residual diff: {result}"
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
        vec!["sinh(2 * x + 1) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn log_abs_trig_and_hyperbolic_diff_compacts_with_domain_conditions() {
    let cases = [
        (
            "diff(ln(abs(sin(2*x + 1)))/2, x)",
            "cot(2 * x + 1)",
            "sin(2 * x + 1) ≠ 0",
        ),
        (
            "diff(-ln(abs(cos(2*x + 1)))/2, x)",
            "tan(2 * x + 1)",
            "cos(2 * x + 1) ≠ 0",
        ),
        (
            "diff(ln(abs(sinh(2*x + 1)))/2, x)",
            "1 / tanh(2 * x + 1)",
            "sinh(2 * x + 1) ≠ 0",
        ),
        ("diff(ln(abs(cosh(2*x + 1)))/2, x)", "tanh(2 * x + 1)", ""),
    ];

    for (input, expected_derivative, expected_condition) in cases {
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

        assert_eq!(result, expected_derivative, "input: {input}");

        let required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();

        let expected_conditions = if expected_condition.is_empty() {
            Vec::<String>::new()
        } else {
            vec![expected_condition.to_string()]
        };

        assert_eq!(
            required, expected_conditions,
            "input: {input}, unexpected required_conditions: {required:?}"
        );
    }
}
#[test]
fn hyperbolic_log_abs_diff_residuals_cancel_as_public_diff_contract() {
    let cases = [
        "diff(ln(abs(sinh(2*x+1))), x)/2 - 1/tanh(2*x+1)",
        "diff(ln(abs(sinh(2*x+1))), x)/2 - cosh(2*x+1)/sinh(2*x+1)",
        "diff(ln(abs(cosh(2*x+1))), x)/2 - tanh(2*x+1)",
        "diff(ln(abs(cosh(2*x+1))), x)/2 - sinh(2*x+1)/cosh(2*x+1)",
    ];

    for input in cases {
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

        assert_eq!(result, "0", "input: {input}");
        assert!(
            output
                .steps
                .iter()
                .any(|step| step.rule_name.as_str() == "Hyperbolic Diff Residual"),
            "expected visible hyperbolic residual cancellation step for {input}"
        );
    }
}
