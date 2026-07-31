use super::*;

#[test]
fn chain_rule_power_composition_diff_evaluates_to_simplified_product() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff((x^2+1)^3, x)";
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

    assert_eq!(result, "6 * x * (x^2 + 1)^2");

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
fn trinomial_power_diff_preserves_raw_target_until_derivative() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let input = "diff((x^2+2*x+1)^3, x)";
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
        parse("3*(2*x+2)*(x^2+2*x+1)^2", &mut engine.simplifier.context).expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "input: {input}, expected compact chain-rule derivative, got: {result}"
    );
    assert!(
        result.contains("(x^2 + 2 * x + 1)^2"),
        "input: {input}, expected compact polynomial-power factor, got: {result}"
    );
    assert!(
        result.contains("(x + 1)"),
        "input: {input}, expected common linear factor in compact derivative, got: {result}"
    );
    assert!(
        !result.contains("x^4") && !result.contains("x * x^2"),
        "input: {input}, derivative should not expand the polynomial-power factor, got: {result}"
    );
    assert!(
        !result.contains("+ 6 * x *"),
        "input: {input}, derivative should factor the shared compact power and scale, got: {result}"
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
        "polynomial power derivative should not add domain conditions: {required:?}"
    );

    let diff_step_index = output
        .steps
        .iter()
        .position(|step| step.rule_name.as_str() == "Symbolic Differentiation")
        .expect("expected a visible differentiation step");
    assert!(
        !output.steps[..diff_step_index].iter().any(|step| {
            step.rule_name.contains("Expand") || step.rule_name.contains("Expansion")
        }),
        "target should not expand before differentiation; steps: {:?}",
        output
            .steps
            .iter()
            .map(|step| step.rule_name.as_str())
            .collect::<Vec<_>>()
    );
    assert!(
        !output
            .steps
            .iter()
            .any(|step| step.rule_name.as_str() == "Expandir la expresión"),
        "compact power derivative should not distribute and refactor the chain-rule product; steps: {:?}",
        output
            .steps
            .iter()
            .map(|step| step.rule_name.as_str())
            .collect::<Vec<_>>()
    );
}
#[test]
fn reciprocal_polynomial_power_diff_preserves_compact_target_until_derivative() {
    let cases = [
        ("diff(-1/(2*(x^2+x-1)^2), x)", "(2*x+1)/(x^2+x-1)^3"),
        ("diff(-1/(2*(x^2+x-1)^3), x)", "(3*x+3/2)/(x^2+x-1)^4"),
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

        assert!(!result.contains("diff("), "input: {input}, got: {result}");

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
            required,
            vec!["x^2 + x - 1 ≠ 0".to_string()],
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
fn log_abs_quotient_diff_uses_direct_domain_safe_log_rule() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(ln(abs((x-1)/(x+1))), x)";
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

    assert_eq!(result, "2 / (x^2 - 1)");
    assert!(
        output.steps.len() <= 3,
        "unexpected noisy log abs quotient derivative route: {} steps",
        output.steps.len()
    );

    let first = output
        .steps
        .iter()
        .find(|step| step.rule_name.as_str() == "Symbolic Differentiation")
        .expect("symbolic differentiation step");
    let first_after = format!(
        "{}",
        DisplayExpr {
            context: &engine.simplifier.context,
            id: first.after,
        }
    );
    assert!(
        !first_after.contains('|'),
        "direct ln(abs(f/g)) derivative should not carry abs noise: {first_after}"
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
        vec!["x ≠ 1".to_string(), "x ≠ -1".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn log_power_diff_uses_compact_numeric_post_calculus_presentation() {
    for (input, expected_result, expected_required) in [
        ("diff(ln(x^2), x)", "2 / x", vec!["x ≠ 0".to_string()]),
        (
            "diff(ln((x+1)^2), x)",
            "2 / (x + 1)",
            vec!["x ≠ -1".to_string()],
        ),
        ("diff(ln(x^3), x)", "3 / x", vec!["x > 0".to_string()]),
        ("diff(ln(x^-2), x)", "-2 / x", vec!["x ≠ 0".to_string()]),
        (
            "diff(ln((x+1)^-2), x)",
            "-2 / (x + 1)",
            vec!["x ≠ -1".to_string()],
        ),
        ("diff(ln(x^-3), x)", "-3 / x", vec!["x > 0".to_string()]),
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
        assert!(
            !result.contains("1 *"),
            "post-calculus log-power presentation should not keep unit products: {result}"
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
            "input: {input}, unexpected required_conditions: {required:?}"
        );
    }
}
#[test]
fn log_abs_product_diff_uses_direct_domain_safe_log_rule() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(ln(abs((x-1)*(x+1))), x)";
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

    assert_eq!(result, "(x * 2)/(x^2 - 1)");
    assert!(
        output.steps.len() <= 4,
        "unexpected noisy log abs product derivative route: {} steps",
        output.steps.len()
    );

    let first = output
        .steps
        .iter()
        .find(|step| step.rule_name.as_str() == "Symbolic Differentiation")
        .expect("symbolic differentiation step");
    let first_after = format!(
        "{}",
        DisplayExpr {
            context: &engine.simplifier.context,
            id: first.after,
        }
    );
    assert!(
        !first_after.contains('|'),
        "direct ln(abs(f*g)) derivative should not carry abs noise: {first_after}"
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
        vec!["x ≠ 1".to_string(), "x ≠ -1".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn generic_log_abs_composite_diff_uses_direct_domain_safe_log_rule() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(ln(abs(x^2-1)), x)";
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

    assert_eq!(result, "(x * 2)/(x^2 - 1)");
    assert!(
        output.steps.len() <= 3,
        "unexpected noisy generic log abs derivative route: {} steps",
        output.steps.len()
    );

    let first = output
        .steps
        .iter()
        .find(|step| step.rule_name.as_str() == "Symbolic Differentiation")
        .expect("symbolic differentiation step");
    let first_after = format!(
        "{}",
        DisplayExpr {
            context: &engine.simplifier.context,
            id: first.after,
        }
    );
    assert!(
        !first_after.contains('|'),
        "direct generic ln(abs(u)) derivative should not carry abs noise: {first_after}"
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
        vec!["x ≠ 1".to_string(), "x ≠ -1".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn product_rule_log_diff_evaluates_with_positive_domain_condition() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(x*ln(x), x)";
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

    assert_eq!(result, "ln(x) + 1");

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
fn unary_log2_diff_evaluates_with_positive_argument_condition() {
    assert_unary_constant_base_log_diff("diff(log2(x), x)", "1/(x*ln(2))");
}
#[test]
fn unary_log10_diff_evaluates_with_positive_argument_condition() {
    assert_unary_constant_base_log_diff("diff(log10(x), x)", "1/(x*ln(10))");
}
#[test]
fn symbolic_base_log_diff_evaluates_without_redundant_ln_base_condition() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(log(y, x), x)";
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

    assert_eq!(result, "1 / (x * ln(y))");

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
    assert!(required.contains(&"x > 0".to_string()), "{required:?}");
    assert!(required.contains(&"y ≠ 1".to_string()), "{required:?}");
    assert!(required.contains(&"y > 0".to_string()), "{required:?}");
    assert!(
        !required.iter().any(|cond| cond.contains("ln(y)")),
        "unexpected redundant ln condition: {required:?}"
    );
}
#[test]
fn variable_base_variable_argument_log_diff_evaluates_with_domain_conditions() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(log(x, x + 1), x)";
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
    assert_ne!(
        result, "0",
        "variable-base log derivative collapsed to zero"
    );

    let expected = parse(
        "(ln(x)/(x+1)-ln(x+1)/x)/ln(x)^2",
        &mut engine.simplifier.context,
    )
    .expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "expected derivative equivalent to change-of-base quotient rule, got: {result}"
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
    assert!(
        required.contains(&"x > 0".to_string()),
        "base positivity condition missing: {required:?}"
    );
    assert!(
        required.contains(&"x ≠ 1".to_string()),
        "argument/base boundary condition missing: {required:?}"
    );
    assert!(
        !required.iter().any(|cond| cond.contains("ln(x)")),
        "unexpected redundant ln(base) condition: {required:?}"
    );
}
#[test]
fn variable_base_log_abs_diff_uses_direct_domain_safe_arg_rule() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(log(x, abs(x^2-1)), x)";
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
        "((2*x/(x^2-1))*ln(x)-ln(abs(x^2-1))/x)/(ln(x)^2)",
        &mut engine.simplifier.context,
    )
    .expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "expected derivative equivalent to direct log(abs(u)) change-of-base rule, got: {result}"
    );

    let first = output
        .steps
        .iter()
        .find(|step| step.rule_name.as_str() == "Symbolic Differentiation")
        .expect("symbolic differentiation step");
    let first_after = format!(
        "{}",
        DisplayExpr {
            context: &engine.simplifier.context,
            id: first.after,
        }
    );
    assert!(
        !first_after.contains("(x^2 - 1)/(|x^2 - 1|)")
            && !first_after.contains("/(|x^2 - 1|)")
            && !first_after.contains("|x^2 - 1|)^2"),
        "direct variable-base log(abs(u)) derivative should not carry abs noise: {first_after}"
    );

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    let mut sorted_required = required.clone();
    sorted_required.sort();
    assert_eq!(
        sorted_required,
        vec!["x > 0".to_string(), "x ≠ 1".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert!(
        !required.iter().any(|cond| cond.contains('|')),
        "absolute-value domain conditions should be normalized: {required:?}"
    );
}
#[test]
fn variable_abs_base_log_abs_diff_uses_direct_domain_safe_base_and_arg_rule() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(log(abs(x), abs(x^2-1)), x)";
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
        "expected derivative equivalent to direct variable abs-base log(abs(u)) rule, got: {result}"
    );

    let first = output
        .steps
        .iter()
        .find(|step| step.rule_name.as_str() == "Symbolic Differentiation")
        .expect("symbolic differentiation step");
    let first_after = format!(
        "{}",
        DisplayExpr {
            context: &engine.simplifier.context,
            id: first.after,
        }
    );
    assert!(
        !first_after.contains("x/|x|")
            && !first_after.contains("x / |x|")
            && !first_after.contains("/|x|")
            && !first_after.contains("/ |x|"),
        "direct variable abs-base log(abs(u)) derivative should not carry abs-base noise: {first_after}"
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
        !required.iter().any(|cond| cond.contains('|')),
        "absolute-value domain conditions should be normalized: {required:?}"
    );
    assert!(
        !required.iter().any(|cond| cond.contains("ln(")),
        "unexpected redundant ln(base) condition: {required:?}"
    );
}
#[test]
fn variable_abs_even_power_base_log_abs_diff_drops_impossible_base_not_one_condition() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(log(abs((x^2-1)^2), abs(x)), x)";
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
        "(2*ln(abs(x^2-1))*(x^2-1)^2+4*ln(abs(x))*x^2-4*ln(abs(x))*x^4)/(4*x*ln(abs(x^2-1))^2*(x^2-1)^2)",
        &mut engine.simplifier.context,
    )
    .expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "expected derivative equivalent to variable abs-even-power base log(abs(u)) rule, got: {result}"
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
        !required.iter().any(|cond| cond == "x^4 + 2 - 2 * x^2 ≠ 0"),
        "impossible positive boundary should not leak into public diff conditions: {required:?}"
    );
}
#[test]
fn variable_abs_higher_even_power_base_log_abs_diff_drops_positive_factor_boundary() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(log(abs((x^2-1)^4), abs(x)), x)";
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
        "(4*ln(abs(x^2-1))*x^2-8*ln(abs(x))*x^2-4*ln(abs(x^2-1)))/(16*x*(x^2*ln(abs(x^2-1))^2-ln(abs(x^2-1))^2))",
        &mut engine.simplifier.context,
    )
    .expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "expected derivative equivalent to variable abs-higher-even-power base log(abs(u)) rule, got: {result}"
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
            .any(|cond| cond == "x^6 + 6 * x^2 - 4 * x^4 - 4 ≠ 0"),
        "positive factor boundary should not leak into public diff conditions: {required:?}"
    );
}
#[test]
fn variable_base_product_log_diff_normalizes_reciprocal_domain_conditions() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(log(x, x*y), x)";
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
    assert_ne!(result, "0", "product log derivative collapsed to zero");

    let expected =
        parse("ln(1/y)/(x*ln(x)^2)", &mut engine.simplifier.context).expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "expected derivative equivalent to reciprocal-log quotient rule, got: {result}"
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
    for expected_condition in ["x ≠ 1", "x > 0", "y > 0"] {
        assert!(
            required.iter().any(|cond| cond == expected_condition),
            "missing {expected_condition}; required_conditions: {required:?}"
        );
    }
    assert!(
        !required
            .iter()
            .any(|cond| cond.contains("1 / y") || cond.contains("x * y")),
        "unexpected redundant reciprocal/product condition: {required:?}"
    );
}
#[test]
fn variable_base_quotient_log_diff_reduces_positive_numerator_domain_conditions() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(log(x, x/y), y)";
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

    let expected = parse("-1/(y*ln(x))", &mut engine.simplifier.context).expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "expected derivative equivalent to quotient log derivative, got: {result}"
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
    for expected_condition in ["x ≠ 1", "x > 0", "y > 0"] {
        assert!(
            required.iter().any(|cond| cond == expected_condition),
            "missing {expected_condition}; required_conditions: {required:?}"
        );
    }
    assert!(
        !required
            .iter()
            .any(|cond| cond.contains("x / y") || cond == "y ≠ 0"),
        "unexpected redundant quotient/nonzero condition: {required:?}"
    );
}
#[test]
fn chain_rule_exp_composition_diff_evaluates_to_simplified_product() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(exp(x^2), x)";
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

    assert_eq!(result, "2 * x * e^(x^2)");

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
fn eval_steps_collapse_additive_zero_tail_for_log_fraction_gap_regression() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;

    let input = "log(x*sqrt(x)) + log(sqrt(x)/x^2)";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    let last = output.steps.last().expect("last step");

    let last_after = format!(
        "{}",
        DisplayExpr {
            context: &engine.simplifier.context,
            id: last.global_after.expect("global_after"),
        }
    );
    assert_eq!(last_after, "0");
}
#[test]
fn generic_log_abs_diff_normalizes_nonzero_domain_conditions() {
    let cases = [
        (
            "diff(ln(abs(2*x+1)), x)",
            "2 / (2 * x + 1)",
            vec!["x ≠ -1/2".to_string()],
        ),
        (
            "diff(ln(abs(x*y)), x)",
            "1 / x",
            vec!["x ≠ 0".to_string(), "y ≠ 0".to_string()],
        ),
        (
            "diff(ln(abs(x^2-1)), x)",
            "(x * 2)/(x^2 - 1)",
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
fn log_cube_by_parts_antiderivative_diff_contract() {
    let cases = [
        (
            "diff((x^2+1)*(ln(x^2+1)^3 - 3*ln(x^2+1)^2 + 6*ln(x^2+1) - 6), x)",
            "2 * x * ln(x^2 + 1)^3",
        ),
        (
            "diff((x^2+1)*(ln(x^2+1)^3 - 3*ln(x^2+1)^2 + 6*ln(x^2+1) - 6), x) - 2*x*ln(x^2+1)^3",
            "0",
        ),
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
