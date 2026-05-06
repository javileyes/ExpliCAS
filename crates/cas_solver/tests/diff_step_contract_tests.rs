use cas_formatter::DisplayExpr;
use cas_math::eval_f64;
use cas_parser::parse;
use cas_session::SessionState;
use cas_solver::runtime::{Engine, EvalAction, EvalRequest, EvalResult, StepsMode};
use cas_solver_core::domain_normalization::normalize_and_dedupe_conditions;
use std::collections::HashMap;

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
fn constant_diff_preserves_independent_input_domain_conditions() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(ln(y)*(z+1)^(-2), x)";
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
        vec!["y > 0".to_string(), "z + 1 ≠ 0".to_string()],
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
fn constant_scaled_trinomial_power_diff_preserves_raw_target_until_derivative() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let input = "diff(1/9 * (x^2+x-1)^3, x)";
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
        parse("1/3*(x^2+x-1)^2*(2*x+1)", &mut engine.simplifier.context).expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "input: {input}, expected scaled compact chain-rule derivative, got: {result}"
    );
    assert!(
        result.contains("(x^2 + x - 1)^2"),
        "input: {input}, expected compact polynomial-power factor, got: {result}"
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
        "scaled polynomial power derivative should not add domain conditions: {required:?}"
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
fn scaled_arcsin_linear_diff_evaluates_to_reciprocal_root() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(arcsin((x+1)/2), x)";
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

    assert_eq!(result, "1 / sqrt(3 - x^2 - 2 * x)");

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert_eq!(
        required,
        vec!["3 - x^2 - 2 * x > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn scaled_asinh_linear_diff_evaluates_to_reciprocal_root() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(asinh((x+1)/2), x)";
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

    assert_eq!(result, "1 / sqrt((x + 1)^2 + 4)");

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
fn inverse_hyperbolic_asinh_surd_quotient_diff_compacts_positive_gap() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::Off;
    let input = "diff(asinh((x^2+x+1)/sqrt(7)), x)";
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
        result, "(2 * x + 1) / sqrt(x^4 + 2 * x^3 + 3 * x^2 + 2 * x + 8)",
        "input: {input}, unexpected derivative result"
    );
    assert!(
        result.contains("sqrt(x^4 + 2 * x^3 + 3 * x^2 + 2 * x + 8)")
            && !result.contains("^(-1/2)")
            && !result.contains("1/7"),
        "input: {input}, expected normalized positive gap in a sqrt denominator, got: {result}"
    );

    let expected = parse(
        "(2*x+1)/(sqrt(7)*sqrt(1+((x^2+x+1)/sqrt(7))^2))",
        &mut engine.simplifier.context,
    )
    .expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "input: {input}, expected derivative equivalent to chain rule, got: {result}"
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
        "asinh is defined on all real inputs and the positive denominator gap should be proved: {required:?}"
    );
}

#[test]
fn product_rule_trig_polynomial_diff_evaluates_to_simplified_sum() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(x^2 * sin(x), x)";
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

    assert!(
        result == "2 * x * sin(x) + cos(x) * x^2" || result == "cos(x) * x^2 + 2 * x * sin(x)",
        "unexpected product-rule derivative presentation: {result}"
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
fn absolute_value_diff_evaluates_with_nonsmooth_point_condition() {
    let cases = [
        ("diff(abs(x), x)", "x / |x|", vec!["x ≠ 0".to_string()]),
        (
            "diff(abs(2*x+1), x)",
            "((2 * x + 1) * 2)/|2 * x + 1|",
            vec!["2 * x + 1 ≠ 0".to_string()],
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
            vec!["x - 1 ≠ 0".to_string(), "x + 1 ≠ 0".to_string()],
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
        vec!["x + 1 ≠ 0".to_string(), "x - 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
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
        vec!["x - 1 ≠ 0".to_string(), "x + 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
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
        vec!["x - 1 ≠ 0".to_string(), "x + 1 ≠ 0".to_string()],
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
        vec!["x - 1 ≠ 0".to_string(), "x + 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
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
        vec!["3 - 2 * x > 0".to_string()],
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
        vec!["3 - 2 * x > 0".to_string()],
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
        vec!["2 * x + 5 > 0".to_string()],
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
fn arctan_shifted_sqrt_diff_uses_post_calculus_reciprocal_root_presentation() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let input = "diff(arctan(sqrt(x+1)), x)";
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

    assert_eq!(result, "1 / (2 * sqrt(x + 1) * (x + 2))");
    assert!(
        !result.contains("(x + 1)^(-1/2)"),
        "presentation regressed: {result}"
    );
    assert!(
        !result.contains("2 * x + 4"),
        "denominator should remain compact in post-calculus presentation: {result}"
    );

    let expected =
        parse("(x+1)^(-1/2)/(2*x+4)", &mut engine.simplifier.context).expect("parse expected");
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
        vec!["x + 1 > 0".to_string()],
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
fn arctan_positive_affine_sqrt_diff_cancels_external_post_calculus_coefficient() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let input = "diff(arctan(sqrt(2*x+3)), x)";
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

    assert_eq!(result, "1 / (sqrt(2 * x + 3) * (2 * x + 4))");
    assert!(
        !result.contains("(2 * x + 3)^(-1/2)"),
        "presentation regressed: {result}"
    );
    assert!(
        !result.contains("2 / (2 * sqrt"),
        "external derivative coefficient should be cancelled safely: {result}"
    );

    let expected =
        parse("(2*x+3)^(-1/2)/(2*x+4)", &mut engine.simplifier.context).expect("parse expected");
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
        vec!["2 * x + 3 > 0".to_string()],
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

    assert_eq!(result, "-1 / (sqrt(3 - 2 * x) * (4 - 2 * x))");
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
        vec!["3 - 2 * x > 0".to_string()],
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
fn arccot_polynomial_sqrt_diff_uses_negative_post_calculus_reciprocal_root_presentation() {
    for (input, expected_render, canonical_equivalent, expected_required) in [
        (
            "diff(arccot(sqrt(x)), x)",
            "-1 / (2 * sqrt(x) * (x + 1))",
            "-x^(-1/2)/(2*x+2)",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(arccot(sqrt(3*x)), x)",
            "-3 / (2 * sqrt(3 * x) * (3 * x + 1))",
            "((3*x)^(-3/2)*x*-9)/(6*x+2)",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(arccot(sqrt(x^2+1)), x)",
            "-x / (sqrt(x^2 + 1) * (x^2 + 2))",
            "-x*(x^2+1)^(-1/2)/(x^2+2)",
            Vec::new(),
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
            result.starts_with('-'),
            "arccot sqrt derivative should keep the negative orientation, got: {result}"
        );
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
        vec!["x + 1 > 0".to_string()],
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
        vec!["x + 1 > 0".to_string()],
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
fn arccot_constant_over_polynomial_sqrt_diff_preserves_compact_denominator() {
    for (input, expected_render, canonical_equivalent, expected_required) in [
        (
            "diff(arccot(sqrt(2/(x+1))), x)",
            "1 / (sqrt(2 / (x + 1)) * (x + 1) * (x + 3))",
            "(2/(x+1))^(-3/2)/(1/2*(x^3+5*x^2+7*x+3))",
            vec!["x + 1 > 0".to_string()],
        ),
        (
            "diff(arccot(sqrt(2/(x^2+1))), x)",
            "2 * x / (sqrt(2 / (x^2 + 1)) * (x^2 + 1) * (x^2 + 3))",
            "((2/(x^2+1))^(-1/2)*x*2)/(x^4+4*x^2+3)",
            Vec::new(),
        ),
        (
            "diff(arccot(sqrt(2/(1-x))), x)",
            "-1 / (sqrt(2 / (1 - x)) * (1 - x) * (3 - x))",
            "-1/(sqrt(2/(1-x))*(1-x)*(3-x))",
            vec!["1 - x > 0".to_string()],
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
            !result.contains("^(-1/2)") && !result.contains("^(-3/2)"),
            "presentation should use a sqrt denominator, got: {result}"
        );
        assert!(
            !result.contains("x^3 + 5 * x^2 + 7 * x + 3") && !result.contains("x^4 + 4 * x^2 + 3"),
            "presentation should preserve the compact factored denominator, got: {result}"
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
            required, expected_required,
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert!(
            output.domain_warnings.is_empty(),
            "unexpected domain warnings for {input}: {:?}",
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
        vec!["x + 1 > 0".to_string()],
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
            vec!["x + 1 > 0".to_string()],
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
            vec!["x + 1 > 0".to_string(), "cos(sqrt(x + 1)) ≠ 0".to_string()],
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
            "diff(-1/cosh(sqrt(3*x+1)), x)",
            "3 * sinh(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1) * cosh(sqrt(3 * x + 1))^2)",
            vec!["3 * x + 1 > 0".to_string()],
        ),
        (
            "diff(-1/sinh(sqrt(3*x+1)), x)",
            "3 * cosh(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1) * sinh(sqrt(3 * x + 1))^2)",
            vec![
                "3 * x + 1 > 0".to_string(),
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
            vec!["x > 0".to_string(), "cos(sqrt(2 * x)) ≠ 0".to_string()],
        ),
        (
            "diff(cot(sqrt(2*x)), x)",
            "-1 / (sqrt(2 * x) * sin(sqrt(2 * x))^2)",
            vec!["x > 0".to_string(), "sin(sqrt(2 * x)) ≠ 0".to_string()],
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
        vec!["2 - x > 0".to_string()],
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
        vec!["1 - x^2 > 0".to_string()],
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
fn constant_base_log_diff_evaluates_with_positive_argument_condition() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(log(2, x), x)";
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

    assert_eq!(result, "1 / (x * ln(2))");

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

fn assert_unary_constant_base_log_diff(input: &str, expected_derivative: &str) {
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
fn constant_base_log_chain_rule_diff_evaluates_without_argument_condition() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(log(2, x^2+1), x)";
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

    assert_eq!(result, "(x * 2)/(ln(2) * (x^2 + 1))");

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
fn constant_base_log_abs_diff_uses_direct_domain_safe_log_rule() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(log(2, abs(x^2-1)), x)";
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

    assert_eq!(result, "(x * 2)/(ln(2) * (x^2 - 1))");
    assert!(
        output.steps.len() <= 3,
        "unexpected noisy fixed-base log abs derivative route: {} steps",
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
        "direct log(base, abs(u)) derivative should not carry abs noise: {first_after}"
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
        vec!["x - 1 ≠ 0".to_string(), "x + 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
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
    assert!(required.contains(&"y - 1 ≠ 0".to_string()), "{required:?}");
    assert!(required.contains(&"y > 0".to_string()), "{required:?}");
    assert!(
        !required.iter().any(|cond| cond.contains("ln(y)")),
        "unexpected redundant ln condition: {required:?}"
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
            vec!["x - 1 > 0".to_string()],
        ),
        (
            "diff(ln(-1+sqrt(x)), x)",
            "1 / (2 * sqrt(x) * (sqrt(x) - 1))",
            vec!["x - 1 > 0".to_string()],
        ),
        (
            "diff(ln(sqrt(2*x)-1), x)",
            "1 / (sqrt(2 * x) * (sqrt(2 * x) - 1))",
            vec!["2 * x - 1 > 0".to_string()],
        ),
        (
            "diff(ln(sqrt(x)-2), x)",
            "1 / (2 * sqrt(x) * (sqrt(x) - 2))",
            vec!["x - 4 > 0".to_string()],
        ),
        (
            "diff(ln(-2+sqrt(x)), x)",
            "1 / (2 * sqrt(x) * (sqrt(x) - 2))",
            vec!["x - 4 > 0".to_string()],
        ),
        (
            "diff(ln(sqrt(2*x)-2), x)",
            "1 / (sqrt(2 * x) * (sqrt(2 * x) - 2))",
            vec!["x - 2 > 0".to_string()],
        ),
        (
            "diff(ln(sqrt(x^2+4)-2), x)",
            "x / (sqrt(x^2 + 4) * (sqrt(x^2 + 4) - 2))",
            vec!["x ≠ 0".to_string()],
        ),
        (
            "diff(ln(sqrt((2*x+1)^2+4)-2), x)",
            "2 * (2 * x + 1) / (sqrt((2 * x + 1)^2 + 4) * (sqrt((2 * x + 1)^2 + 4) - 2))",
            vec!["2 * x + 1 ≠ 0".to_string()],
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
fn variable_base_constant_argument_log_diff_evaluates_with_base_domain_conditions() {
    let cases = [
        (
            "diff(log(x, 2), x)",
            "-ln(2)/(x*ln(x)^2)",
            vec!["x - 1 ≠ 0", "x > 0"],
        ),
        (
            "diff(log(x, y), x)",
            "-ln(y)/(x*ln(x)^2)",
            vec!["x - 1 ≠ 0", "x > 0", "y > 0"],
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
        let expected: Vec<String> = expected_conditions.into_iter().map(String::from).collect();

        assert_eq!(
            required, expected,
            "input: {input}, unexpected required_conditions: {required:?}"
        );
        assert!(
            !required.iter().any(|cond| cond.contains("ln(x)")),
            "unexpected redundant ln(base) condition: {required:?}"
        );
    }
}

#[test]
fn variable_base_polynomial_constant_argument_log_diff_keeps_factored_presentation() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let input = "diff(log(x^2+1, 2), x)";
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

    assert_eq!(result, "-2 * ln(2) * x / ((x^2 + 1) * ln(x^2 + 1)^2)");
    assert!(
        !result.contains("ln(x^2 + 1)^2 + x^2 * ln(x^2 + 1)^2"),
        "post-calculus presentation should keep the denominator factored: {result}"
    );
    assert!(!result.contains("diff("), "got: {result}");

    let expected = parse(
        "-2*x*ln(2)/((x^2+1)*ln(x^2+1)^2)",
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

    assert_eq!(
        required,
        vec!["x ≠ 0".to_string()],
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
fn variable_base_polynomial_constant_argument_log_diff_avoids_negative_unit_factor_noise() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let input = "diff(log(x^2+x+1, 2), x)";
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
        "-ln(2) * (2 * x + 1) / ((x^2 + x + 1) * ln(x^2 + x + 1)^2)"
    );
    assert!(
        !result.contains("-1 *") && !result.contains("-1·"),
        "post-calculus presentation should not expose a negative unit factor: {result}"
    );
    assert!(!result.contains("diff("), "got: {result}");

    let expected = parse(
        "-ln(2)*(2*x+1)/((x^2+x+1)*ln(x^2+x+1)^2)",
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

    assert_eq!(
        required,
        vec!["x ≠ 0".to_string(), "x + 1 ≠ 0".to_string()],
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
        required.contains(&"x - 1 ≠ 0".to_string()),
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

    assert_eq!(
        required,
        vec!["x > 0".to_string(), "x - 1 ≠ 0".to_string()],
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
            "x - 1 ≠ 0".to_string(),
            "x + 1 ≠ 0".to_string()
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
            "x - 1 ≠ 0".to_string(),
            "x + 1 ≠ 0".to_string()
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
            "x - 1 ≠ 0".to_string(),
            "x + 1 ≠ 0".to_string(),
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
            "x - 1 ≠ 0".to_string(),
            "x + 1 ≠ 0".to_string(),
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
            "x - 1 ≠ 0".to_string(),
            "x + 1 ≠ 0".to_string(),
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
            "x - 1 ≠ 0".to_string(),
            "x + 1 ≠ 0".to_string(),
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
fn variable_base_power_log_diff_simplifies_constant_with_minimal_domain_conditions() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(log(x^2, x^3), x)";
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

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert_eq!(
        required,
        vec!["x - 1 ≠ 0".to_string(), "x > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert!(
        !required
            .iter()
            .any(|cond| cond == "x ≠ 0" || cond == "x + 1 ≠ 0" || cond.contains("x^3")),
        "unexpected redundant power-domain condition: {required:?}"
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
    for expected_condition in ["x - 1 ≠ 0", "x > 0", "y > 0"] {
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
    for expected_condition in ["x - 1 ≠ 0", "x > 0", "y > 0"] {
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
    for expected_condition in ["x > 0", "y ≠ 0", "z + 1 ≠ 0"] {
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
fn chain_rule_trig_composition_diff_evaluates_to_simplified_product() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(sin(x^2), x)";
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

    assert_eq!(result, "2 * x * cos(x^2)");

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
fn reciprocal_trig_diff_evaluates_with_pole_conditions() {
    let cases = [
        ("diff(sec(x), x)", "sin(x)/cos(x)^2", "cos(x) ≠ 0"),
        ("diff(csc(x), x)", "-cos(x)/sin(x)^2", "sin(x) ≠ 0"),
        ("diff(cot(x), x)", "-1/sin(x)^2", "sin(x) ≠ 0"),
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
            vec![expected_condition.to_string()],
            "input: {input}, unexpected required_conditions: {required:?}"
        );
    }
}

#[test]
fn affine_sec_csc_diff_uses_chain_rule_with_pole_conditions() {
    let cases = [
        (
            "diff(sec(2*x+1), x)",
            "2*sec(2*x+1)*tan(2*x+1)",
            "2 * sec(2 * x + 1) * tan(2 * x + 1)",
            "cos(2 * x + 1) ≠ 0",
        ),
        (
            "diff(1/cos(2*x+1), x)",
            "2*sec(2*x+1)*tan(2*x+1)",
            "2 * sec(2 * x + 1) * tan(2 * x + 1)",
            "cos(2 * x + 1) ≠ 0",
        ),
        (
            "diff(csc(1-2*x), x)",
            "2*csc(1-2*x)*cot(1-2*x)",
            "2 * csc(1 - 2 * x) * cot(1 - 2 * x)",
            "sin(1 - 2 * x) ≠ 0",
        ),
        (
            "diff(1/sin(1-2*x), x)",
            "2*csc(1-2*x)*cot(1-2*x)",
            "2 * csc(1 - 2 * x) * cot(1 - 2 * x)",
            "sin(1 - 2 * x) ≠ 0",
        ),
    ];

    for (input, expected_derivative, expected_display, expected_condition) in cases {
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
        assert_eq!(result, expected_display, "input: {input}");

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
            vec![expected_condition.to_string()],
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
fn affine_linear_times_tan_diff_keeps_product_rule_shape() {
    let cases = [
        (
            "diff((x+1)*tan(2*x+1), x)",
            "tan(2 * x + 1) + (2 * x + 2) / cos(2 * x + 1)^2",
        ),
        (
            "diff((3*x+2)*tan(2*x+1), x)",
            "3 * tan(2 * x + 1) + (6 * x + 4) / cos(2 * x + 1)^2",
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
            vec!["cos(2 * x + 1) ≠ 0".to_string()],
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
fn affine_linear_times_cot_diff_keeps_product_rule_shape() {
    let cases = [
        (
            "diff((x+1)*cot(2*x+1), x)",
            "cot(2 * x + 1) - (2 * x + 2) / sin(2 * x + 1)^2",
        ),
        (
            "diff((3*x+2)*cot(2*x+1), x)",
            "3 * cot(2 * x + 1) - (6 * x + 4) / sin(2 * x + 1)^2",
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
            vec!["sin(2 * x + 1) ≠ 0".to_string()],
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
fn affine_linear_times_sec_csc_diff_avoids_reciprocal_expansion_timeout_shape() {
    let cases = [
        (
            "diff((x+1)*sec(2*x+1), x)",
            "(cos(2 * x + 1) + 2 * sin(2 * x + 1) + 2 * x * sin(2 * x + 1)) / cos(2 * x + 1)^2",
            "cos(2 * x + 1) ≠ 0",
        ),
        (
            "diff((3*x+2)*sec(2*x+1), x)",
            "(3 * cos(2 * x + 1) + 4 * sin(2 * x + 1) + 6 * x * sin(2 * x + 1)) / cos(2 * x + 1)^2",
            "cos(2 * x + 1) ≠ 0",
        ),
        (
            "diff((x+1)*csc(2*x+1), x)",
            "csc(2 * x + 1) - cos(2 * x + 1) * (2 * x + 2) / sin(2 * x + 1)^2",
            "sin(2 * x + 1) ≠ 0",
        ),
        (
            "diff((3*x+2)*csc(2*x+1), x)",
            "3 * csc(2 * x + 1) - cos(2 * x + 1) * (6 * x + 4) / sin(2 * x + 1)^2",
            "sin(2 * x + 1) ≠ 0",
        ),
    ];

    for (input, expected_display, expected_condition) in cases {
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
        assert_eq!(result, expected_display, "unexpected result for {input}");

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
fn affine_cot_diff_uses_direct_reciprocal_trig_derivative_after_canonicalization() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let input = "diff(cot(2*x+1), x)";
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
        parse("-2/sin(2*x+1)^2", &mut engine.simplifier.context).expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "expected affine cot derivative, got: {result}"
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
        vec!["sin(2 * x + 1) ≠ 0".to_string()],
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
            .any(|step| step.rule_name.as_str() == "Pythagorean with Generic Coefficient"),
        "affine cot derivative should not require a post-quotient pythagorean collapse"
    );
}

#[test]
fn rational_affine_tan_cot_diff_avoids_half_angle_cleanup_warnings() {
    let cases = [
        (
            "diff(tan((3*x+2)/2), x)",
            "3 / (cos(3 * x + 2) + 1)",
            "3/(2*cos((3*x+2)/2)^2)",
            vec!["cos(3 * x + 2) + 1 ≠ 0".to_string()],
        ),
        (
            "diff(cot((2-3*x)/2), x)",
            "3 / (1 - cos(2 - 3 * x))",
            "3/(2*sin((2-3*x)/2)^2)",
            vec!["1 - cos(2 - 3 * x) ≠ 0".to_string()],
        ),
    ];

    for (input, expected_display, expected_derivative, expected_conditions) in cases {
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
        assert!(
            output.domain_warnings.is_empty(),
            "input: {input}, unexpected warnings: {:?}",
            output.domain_warnings
        );

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
fn rational_affine_sec_csc_diff_avoids_half_angle_cleanup_warnings() {
    let cases = [
        (
            "diff(sec((3*x+2)/2), x)",
            "3/2*sec((3*x+2)/2)*tan((3*x+2)/2)",
            "3/2 * sec((3 * x + 2) / 2) * tan((3 * x + 2) / 2)",
            "cos(",
        ),
        (
            "diff(csc((2-3*x)/2), x)",
            "3/2*csc((2-3*x)/2)*cot((2-3*x)/2)",
            "3/2 * csc((2 - 3 * x) / 2) * cot((2 - 3 * x) / 2)",
            "sin(",
        ),
    ];

    for (input, expected_derivative, expected_display, expected_condition_fn) in cases {
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
        assert!(
            output.domain_warnings.is_empty(),
            "input: {input}, unexpected warnings: {:?}",
            output.domain_warnings
        );

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
            !result.contains("+ 1 - 1") && !result.contains("1 - (2 *"),
            "input: {input}, noisy half-angle cleanup survived: {result}"
        );
        assert_eq!(result, expected_display, "input: {input}");

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
            required.len(),
            1,
            "input: {input}, unexpected required_conditions: {required:?}"
        );
        assert!(
            required[0].contains(expected_condition_fn),
            "input: {input}, unexpected required condition: {:?}",
            required
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
fn reciprocal_trig_half_angle_fraction_passthrough_keeps_compact_form() {
    let cases = [
        "3*sin((3*x+2)/2)/(1+cos(3*x+2))",
        "3*cos((2-3*x)/2)/(1-cos(2-3*x))",
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
        assert!(
            output.domain_warnings.is_empty(),
            "input: {input}, unexpected warnings: {:?}",
            output.domain_warnings
        );

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
            !result.contains("+ 1 - 1") && !result.contains("1 - (2 *"),
            "input: {input}, noisy half-angle cleanup survived: {result}"
        );
        assert!(
            !result.contains("^2"),
            "input: {input}, expected compact half-angle fraction to be preserved, got: {result}"
        );

        let expected = parse(input, &mut engine.simplifier.context).expect("parse expected");
        assert!(
            engine.simplifier.are_equivalent(result_expr, expected),
            "input: {input}, expected passthrough equivalent to source, got: {result}"
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
            1,
            "input: {input}, unexpected required_conditions: {required:?}"
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
fn generic_log_abs_diff_normalizes_nonzero_domain_conditions() {
    let cases = [
        (
            "diff(ln(abs(2*x+1)), x)",
            "2 / (2 * x + 1)",
            vec!["2 * x + 1 ≠ 0".to_string()],
        ),
        (
            "diff(ln(abs(x*y)), x)",
            "1 / x",
            vec!["x ≠ 0".to_string(), "y ≠ 0".to_string()],
        ),
        (
            "diff(ln(abs(x^2-1)), x)",
            "(x * 2)/(x^2 - 1)",
            vec!["x - 1 ≠ 0".to_string(), "x + 1 ≠ 0".to_string()],
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

#[test]
fn shifted_linear_asinh_diff_preserves_compact_radicand() {
    let cases = [
        ("diff(asinh(2*x+1), x)", "2 / sqrt((2 * x + 1)^2 + 1)"),
        ("diff(asinh(3-2*x), x)", "-2 / sqrt((3 - 2 * x)^2 + 1)"),
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

#[test]
fn asinh_sqrt_diff_uses_post_calculus_root_denominator_presentation() {
    for (input, expected_render, expected_required) in [
        (
            "diff(asinh(sqrt(x)), x)",
            "1 / (2 * sqrt(x) * sqrt(x + 1))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(asinh(sqrt(2*x)), x)",
            "1 / (sqrt(2 * x) * sqrt(2 * x + 1))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(asinh(sqrt(x+1)), x)",
            "1 / (2 * sqrt(x + 1) * sqrt(x + 2))",
            vec!["x + 1 > 0".to_string()],
        ),
        (
            "diff(asinh(sqrt(1-2*x)), x)",
            "-1 / (sqrt(1 - 2 * x) * sqrt(2 - 2 * x))",
            vec!["1 - 2 * x > 0".to_string()],
        ),
        (
            "diff(asinh(sqrt(x^2+1)), x)",
            "x / (sqrt(x^2 + 1) * sqrt(x^2 + 2))",
            Vec::new(),
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
fn asinh_affine_by_parts_primitive_diff_contract() {
    let cases = [
        (
            "diff(1/2*((2*x+1)*asinh(2*x+1)-sqrt((2*x+1)^2+1)), x)",
            "asinh(2 * x + 1)",
        ),
        (
            "diff(1/2*(sqrt((1-2*x)^2+1)-asinh(1-2*x)*(1-2*x)), x)",
            "asinh(1 - 2 * x)",
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

#[test]
fn inverse_hyperbolic_acosh_diff_evaluates_with_domain_safe_conditions() {
    let cases = [
        (
            "diff(acosh(x), x)",
            "(x - 1)^(-1/2)/(x + 1)^(1/2)",
            vec!["x - 1 > 0"],
        ),
        (
            "diff(acosh(2*x+1), x)",
            "2*(2*x)^(-1/2)/(2*x+2)^(1/2)",
            vec!["x > 0"],
        ),
        (
            "diff(acosh(x^2+1), x)",
            "2*x/(sqrt(x^2)*sqrt(x^2+2))",
            vec!["x ≠ 0"],
        ),
        (
            "diff(acosh((x+1)^2+1), x)",
            "(2*x+2)/(sqrt(x^2+2*x+1)*sqrt(x^2+2*x+3))",
            vec!["x + 1 ≠ 0"],
        ),
        (
            "diff(acosh((2*x+1)^2+1), x)",
            "(8*x+4)/(sqrt(4*x^2+4*x+1)*sqrt(4*x^2+4*x+3))",
            vec!["2 * x + 1 ≠ 0"],
        ),
        (
            "diff(acosh((1-2*x)^2+1), x)",
            "(8*x-4)/(sqrt(4*x^2+1-4*x)*sqrt(4*x^2+3-4*x))",
            vec!["2 * x - 1 ≠ 0"],
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
        if input == "diff(acosh(x^2+1), x)" {
            assert!(
                result.contains("|x|"),
                "even-power acosh derivative must preserve branch sign with abs: {result}"
            );
        }
        if input == "diff(acosh((x+1)^2+1), x)" {
            assert!(
                result.contains("|x + 1|"),
                "expanded shifted-square acosh derivative must preserve branch sign with abs: {result}"
            );
        }
        if input == "diff(acosh((2*x+1)^2+1), x)" {
            assert!(
                result.contains("|2 * x + 1|"),
                "scaled shifted-square acosh derivative must preserve branch sign with abs: {result}"
            );
        }
        if input == "diff(acosh((1-2*x)^2+1), x)" {
            assert!(
                result.contains("|1 - 2 * x|"),
                "negatively oriented shifted-square acosh derivative must preserve branch sign with abs: {result}"
            );
        }
        let required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();
        let expected: Vec<String> = expected_conditions.into_iter().map(String::from).collect();

        assert_eq!(
            required, expected,
            "input: {input}, unexpected required_conditions: {required:?}"
        );
    }
}

#[test]
fn acosh_sqrt_diff_uses_post_calculus_root_denominator_presentation() {
    for (input, expected_render, expected_required) in [
        (
            "diff(acosh(sqrt(x)), x)",
            "1 / (2 * sqrt(x) * sqrt(sqrt(x) - 1) * sqrt(sqrt(x) + 1))",
            vec!["x - 1 > 0".to_string()],
        ),
        (
            "diff(acosh(sqrt(x+1)), x)",
            "1 / (2 * sqrt(x + 1) * sqrt(sqrt(x + 1) - 1) * sqrt(sqrt(x + 1) + 1))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(acosh(sqrt(x^2+1)), x)",
            "x / (sqrt(x^2 + 1) * sqrt(sqrt(x^2 + 1) - 1) * sqrt(sqrt(x^2 + 1) + 1))",
            vec!["x ≠ 0".to_string()],
        ),
        (
            "diff(acosh(sqrt((x+1)^2+1)), x)",
            "(x + 1) / (sqrt((x + 1)^2 + 1) * sqrt(sqrt((x + 1)^2 + 1) - 1) * sqrt(sqrt((x + 1)^2 + 1) + 1))",
            vec!["x + 1 ≠ 0".to_string()],
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
}

#[test]
fn inverse_hyperbolic_atanh_diff_evaluates_with_open_unit_interval_condition() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(atanh(x), x)";
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

    let expected = parse("1/(1-x^2)", &mut engine.simplifier.context).expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "input: {input}, expected derivative equivalent to 1/(1-x^2), got: {result}"
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
        vec!["1 - x^2 > 0".to_string()],
        "input: {input}, unexpected required_conditions: {required:?}"
    );
}

#[test]
fn negative_atanh_polynomial_diff_keeps_fraction_sign_out_of_numerator() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(-atanh(x^2), x)";
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
        result.starts_with('-'),
        "expected outer negative sign: {result}"
    );
    assert!(
        !result.starts_with("-("),
        "post-calculus fraction presentation should not wrap the whole quotient: {result}"
    );
    assert!(
        !result.contains("* -2"),
        "post-calculus fraction presentation should not bury the sign in the numerator: {result}"
    );
    assert!(
        !result.contains("/(("),
        "post-calculus fraction presentation should not double-wrap the denominator: {result}"
    );
    assert!(
        !result.contains("-x * 2"),
        "post-calculus fraction presentation should put the numeric coefficient first: {result}"
    );

    let expected = parse("-2*x/(1-x^4)", &mut engine.simplifier.context).expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "input: {input}, expected derivative equivalent to -2*x/(1-x^4), got: {result}"
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
        vec!["1 - x^4 > 0".to_string()],
        "input: {input}, unexpected required_conditions: {required:?}"
    );
}

#[test]
fn atanh_sqrt_diff_uses_post_calculus_root_denominator_presentation() {
    for (input, expected_render, expected_required) in [
        (
            "diff(atanh(sqrt(x)), x)",
            "-1 / (2 * sqrt(x) * (x - 1))",
            vec!["1 - x > 0".to_string(), "x > 0".to_string()],
        ),
        (
            "diff(atanh(sqrt(2*x)), x)",
            "-1 / (sqrt(2 * x) * (2 * x - 1))",
            vec!["1 - 2 * x > 0".to_string(), "x > 0".to_string()],
        ),
        (
            "diff(atanh(sqrt(x+1)), x)",
            "-1 / (2 * sqrt(x + 1) * x)",
            vec!["-x > 0".to_string(), "x + 1 > 0".to_string()],
        ),
        (
            "diff(atanh(sqrt(1-2*x)), x)",
            "-1 / (sqrt(1 - 2 * x) * 2 * x)",
            vec!["1 - 2 * x > 0".to_string(), "x > 0".to_string()],
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
            !required
                .iter()
                .any(|condition| condition.contains("^(1/2)^2")),
            "atanh sqrt condition should be compact, got: {required:?}"
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
fn inverse_hyperbolic_atanh_affine_diff_preserves_open_interval_condition_without_steps() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::Off;
    let input = "diff(atanh(x+1), x)";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    assert!(
        output.steps.is_empty(),
        "steps-off contract should not rely on recorded steps"
    );
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

    let expected = parse("1/(1-(x+1)^2)", &mut engine.simplifier.context).expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "input: {input}, expected derivative equivalent to 1/(1-(x+1)^2), got: {result}"
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
        vec!["-x^2 - 2 * x > 0".to_string()],
        "input: {input}, unexpected required_conditions: {required:?}"
    );
}

#[test]
fn inverse_hyperbolic_atanh_scaled_affine_diff_dedupes_boundary_conditions_without_steps() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::Off;
    let input = "diff(atanh(2*x+1), x)";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    assert!(
        output.steps.is_empty(),
        "steps-off contract should not rely on recorded steps"
    );
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
        parse("2/(1-(2*x+1)^2)", &mut engine.simplifier.context).expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "input: {input}, expected derivative equivalent to 2/(1-(2*x+1)^2), got: {result}"
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
        vec!["-x^2 - x > 0".to_string()],
        "input: {input}, unexpected required_conditions: {required:?}"
    );
}

#[test]
fn inverse_hyperbolic_atanh_surd_polynomial_diff_uses_compact_open_interval_condition() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::Off;
    let input = "diff(atanh(x^2/sqrt(3)), x)";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    assert!(
        output.steps.is_empty(),
        "steps-off contract should not rely on recorded steps"
    );
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

    let expected = parse(
        "2*x/(sqrt(3)*(1-(x^2/sqrt(3))^2))",
        &mut engine.simplifier.context,
    )
    .expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "input: {input}, expected derivative equivalent to chain rule over atanh, got: {result}"
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
        vec!["3 - x^4 > 0".to_string()],
        "input: {input}, unexpected required_conditions: {required:?}"
    );
    assert!(
        !required.iter().any(|cond| cond.contains("sqrt(")),
        "atanh open-interval condition should not leak sqrt denominator form: {required:?}"
    );
}

#[test]
fn inverse_hyperbolic_atanh_shifted_surd_polynomial_diff_uses_compact_open_interval_condition() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::Off;
    let input = "diff(atanh((x+1)^2/sqrt(3)), x)";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    assert!(
        output.steps.is_empty(),
        "steps-off contract should not rely on recorded steps"
    );
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

    assert_eq!(
        result, "3^(1/2) * (2 * x + 2) / (3 - (x + 1)^4)",
        "input: {input}, unexpected derivative result"
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
        vec!["3 - (x + 1)^4 > 0".to_string()],
        "input: {input}, unexpected required_conditions: {required:?}"
    );
    assert!(
        !required.iter().any(|cond| cond.contains("sqrt(")),
        "atanh open-interval condition should not leak sqrt denominator form: {required:?}"
    );
}

#[test]
fn inverse_hyperbolic_atanh_negatively_oriented_shifted_surd_polynomial_diff_compacts_result() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::Off;
    let input = "diff(atanh((1-2*x)^2/sqrt(3)), x)";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    assert!(
        output.steps.is_empty(),
        "steps-off contract should not rely on recorded steps"
    );
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
        result, "3^(1/2) * (8 * x - 4) / (3 - (1 - 2 * x)^4)",
        "input: {input}, unexpected derivative result"
    );
    assert!(!result.contains("diff("), "input: {input}, got: {result}");

    let expected = parse(
        "(8*x-4)/(sqrt(3)*(1-((1-2*x)^2/sqrt(3))^2))",
        &mut engine.simplifier.context,
    )
    .expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "input: {input}, expected derivative equivalent to chain rule over atanh, got: {result}"
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
        vec!["3 - (1 - 2 * x)^4 > 0".to_string()],
        "input: {input}, unexpected required_conditions: {required:?}"
    );
    assert!(
        !required.iter().any(|cond| cond.contains("sqrt(")),
        "atanh open-interval condition should not leak sqrt denominator form: {required:?}"
    );
}

#[test]
fn inverse_hyperbolic_atanh_quadratic_surd_diff_normalizes_result_denominator_domain() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::Off;
    let input = "diff(atanh((x^2+x+1)/sqrt(7)), x)";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    assert!(
        output.steps.is_empty(),
        "steps-off contract should not rely on recorded steps"
    );
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
        result, "7^(1/2) * (2 * x + 1) / (6 - x^4 - 2 * x^3 - 3 * x^2 - 2 * x)",
        "input: {input}, unexpected derivative result"
    );
    assert!(!result.contains("diff("), "input: {input}, got: {result}");

    let expected = parse(
        "(2*x+1)/(sqrt(7)*(1-((x^2+x+1)/sqrt(7))^2))",
        &mut engine.simplifier.context,
    )
    .expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "input: {input}, expected derivative equivalent to chain rule over atanh, got: {result}"
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
        vec!["6 - x^4 - 2 * x^3 - 3 * x^2 - 2 * x > 0".to_string()],
        "input: {input}, unexpected required_conditions: {required:?}"
    );
    assert!(
        !required.iter().any(|cond| cond.contains("≠")),
        "positive atanh interval should dominate derivative denominator nonzero guard: {required:?}"
    );
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
            vec!["1 - x^2 > 0".to_string()],
            "input: {input}, required_conditions: {required:?}"
        );
    }
}

#[test]
fn bounded_inverse_trig_sqrt_diff_uses_post_calculus_root_denominator_presentation() {
    for (input, expected_render, expected_required) in [
        (
            "diff(arcsin(sqrt(x)), x)",
            "1 / (2 * sqrt(x) * sqrt(1 - x))",
            vec!["x > 0".to_string(), "1 - x > 0".to_string()],
        ),
        (
            "diff(arccos(sqrt(x)), x)",
            "-1 / (2 * sqrt(x) * sqrt(1 - x))",
            vec!["x > 0".to_string(), "1 - x > 0".to_string()],
        ),
        (
            "diff(arcsin(sqrt(2*x)), x)",
            "1 / (sqrt(2 * x) * sqrt(1 - 2 * x))",
            vec!["1 - 2 * x > 0".to_string(), "x > 0".to_string()],
        ),
        (
            "diff(arcsin(sqrt(x+1)), x)",
            "1 / (2 * sqrt(x + 1) * sqrt(-x))",
            vec!["x + 1 > 0".to_string(), "-x > 0".to_string()],
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
        if input.contains("arccos") {
            assert!(
                result.starts_with('-'),
                "arccos sqrt derivative should keep negative orientation, got: {result}"
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
fn bounded_inverse_trig_surd_quotient_diff_compacts_open_interval_gap() {
    let cases = [
        (
            "diff(arcsin((x^2+x+1)/sqrt(7)), x)",
            "(2 * x + 1) / sqrt(6 - x^4 - 2 * x^3 - 3 * x^2 - 2 * x)",
            "(2*x+1)/(sqrt(7)*sqrt(1-((x^2+x+1)/sqrt(7))^2))",
        ),
        (
            "diff(arccos((x^2+x+1)/sqrt(7)), x)",
            "-(2 * x + 1) / sqrt(6 - x^4 - 2 * x^3 - 3 * x^2 - 2 * x)",
            "-(2*x+1)/(sqrt(7)*sqrt(1-((x^2+x+1)/sqrt(7))^2))",
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
            result.contains("sqrt(6 - x^4 - 2 * x^3 - 3 * x^2 - 2 * x)")
                && !result.contains("^(-1/2)")
                && !result.contains("1/7"),
            "input: {input}, expected compact reciprocal-root normalized gap, got: {result}"
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

        let required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();

        assert_eq!(
            required,
            vec!["6 - x^4 - 2 * x^3 - 3 * x^2 - 2 * x > 0".to_string()],
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
fn surd_quotient_diff_preserves_rational_content_scale() {
    let cases = [
        (
            "diff(arcsin(x/sqrt(1/2)), x)",
            "sqrt(2) / sqrt(1 - 2 * x^2)",
            "1/(sqrt(1/2)*sqrt(1-(x/sqrt(1/2))^2))",
            vec!["1 - 2 * x^2 > 0"],
        ),
        (
            "diff(asinh(x/sqrt(1/2)), x)",
            "sqrt(2) / sqrt(2 * x^2 + 1)",
            "1/(sqrt(1/2)*sqrt(1+(x/sqrt(1/2))^2))",
            vec![],
        ),
        (
            "diff(atanh(x/sqrt(1/2)), x)",
            "2^(1/2) / (1 - 2 * x^2)",
            "1/(sqrt(1/2)*(1-(x/sqrt(1/2))^2))",
            vec!["1 - 2 * x^2 > 0"],
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
        assert!(
            result.contains("sqrt(2)") || result.contains("2^(1/2)"),
            "input: {input}, expected retained sqrt(2) scale, got: {result}"
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
fn inverse_reciprocal_trig_diff_evaluates_with_explicit_domain_conditions() {
    let cases = [
        (
            "diff(arcsec(x), x)",
            "1 / (|x| * sqrt(x^2 - 1))",
            vec!["x^2 - 1 > 0"],
        ),
        (
            "diff(arccsc(x), x)",
            "-1 / (|x| * sqrt(x^2 - 1))",
            vec!["x^2 - 1 > 0"],
        ),
        (
            "diff(arcsec(2*x), x)",
            "2 / (|2 * x| * sqrt(4 * x^2 - 1))",
            vec!["4 * x^2 - 1 > 0"],
        ),
        (
            "diff(arccsc(2*x), x)",
            "-2 / (|2 * x| * sqrt(4 * x^2 - 1))",
            vec!["4 * x^2 - 1 > 0"],
        ),
        (
            "diff(arcsec(x+1), x)",
            "1 / (|x + 1| * sqrt(x^2 + 2 * x))",
            vec!["x^2 + 2 * x > 0"],
        ),
        (
            "diff(arccsc(x+1), x)",
            "-1 / (|x + 1| * sqrt(x^2 + 2 * x))",
            vec!["x^2 + 2 * x > 0"],
        ),
        (
            "diff(arcsec(sqrt(x)), x)",
            "1 / (2*x*sqrt(x-1))",
            vec!["x - 1 > 0"],
        ),
        (
            "diff(arccsc(sqrt(x)), x)",
            "-1 / (2*x*sqrt(x-1))",
            vec!["x - 1 > 0"],
        ),
        (
            "diff(arcsec(sqrt(x+1)), x)",
            "1 / (2*(x+1)*sqrt(x))",
            vec!["x > 0"],
        ),
        (
            "diff(arccsc(sqrt(x+1)), x)",
            "-1 / (2*(x+1)*sqrt(x))",
            vec!["x > 0"],
        ),
        (
            "diff(arcsec(sqrt(2*x)), x)",
            "1 / (2*x*sqrt(2*x-1))",
            vec!["2 * x - 1 > 0"],
        ),
        (
            "diff(arccsc(sqrt(2*x)), x)",
            "-1 / (2*x*sqrt(2*x-1))",
            vec!["2 * x - 1 > 0"],
        ),
        (
            "diff(arcsec(sqrt(3-2*x)), x)",
            "-1 / ((3-2*x)*sqrt(2-2*x))",
            vec!["1 - x > 0"],
        ),
        (
            "diff(arccsc(sqrt(3-2*x)), x)",
            "1 / ((3-2*x)*sqrt(2-2*x))",
            vec!["1 - x > 0"],
        ),
        (
            "diff(arcsec(sqrt(1-2*x)), x)",
            "-1 / ((1-2*x)*sqrt(-2*x))",
            vec!["-x > 0"],
        ),
        (
            "diff(arccsc(sqrt(1-2*x)), x)",
            "1 / ((1-2*x)*sqrt(-2*x))",
            vec!["-x > 0"],
        ),
        (
            "diff(arcsec(x^2+1), x)",
            "2*x/((x^2+1)*sqrt(x^4+2*x^2))",
            vec!["x ≠ 0"],
        ),
        (
            "diff(arcsec(x^2+1/2), x)",
            "2*x/((x^2+1/2)*sqrt(x^4+x^2-3/4))",
            vec!["4 * x^4 + 4 * x^2 - 3 > 0"],
        ),
        (
            "diff(arcsec(x^2+x+3), x)",
            "(2*x+1)/((x^2+x+3)*sqrt(x^4+2*x^3+7*x^2+6*x+8))",
            vec![],
        ),
        (
            "diff(arcsec((x^2+1)^2), x)",
            "4*x/((x^2+1)*sqrt((x^2+1)^4-1))",
            vec!["x ≠ 0"],
        ),
        (
            "diff(arcsec((x^2+x+3)^2), x)",
            "2*(2*x+1)/((x^2+x+3)*sqrt((x^2+x+3)^4-1))",
            vec![],
        ),
        (
            "diff(arcsec((2*x^2+2*x+6)^2), x)",
            "4*(2*x+1)/((2*x^2+2*x+6)*sqrt((2*x^2+2*x+6)^4-1))",
            vec![],
        ),
        (
            "diff(arcsec(((x^2+x+3)/2)^2), x)",
            "2*(2*x+1)/((x^2+x+3)*sqrt(((x^2+x+3)/2)^4-1))",
            vec![],
        ),
        (
            "diff(arcsec(((1/2)*(x^2+x+3))^2), x)",
            "2*(2*x+1)/((x^2+x+3)*sqrt(((1/2)*(x^2+x+3))^4-1))",
            vec![],
        ),
        (
            "diff(arcsec(((1/3)*(x^2+x+3))^2), x)",
            "2*(2*x+1)/((x^2+x+3)*sqrt(((1/3)*(x^2+x+3))^4-1))",
            vec!["(x^2 + x + 3)^4 - 81 > 0"],
        ),
        (
            "diff(arccsc(x^2+1), x)",
            "-2*x/((x^2+1)*sqrt(x^4+2*x^2))",
            vec!["x ≠ 0"],
        ),
        (
            "diff(arccsc(x^2+1/2), x)",
            "-2*x/((x^2+1/2)*sqrt(x^4+x^2-3/4))",
            vec!["4 * x^4 + 4 * x^2 - 3 > 0"],
        ),
        (
            "diff(arccsc(x^2+x+3), x)",
            "-(2*x+1)/((x^2+x+3)*sqrt(x^4+2*x^3+7*x^2+6*x+8))",
            vec![],
        ),
        (
            "diff(arccsc((x^2+1)^2), x)",
            "-4*x/((x^2+1)*sqrt((x^2+1)^4-1))",
            vec!["x ≠ 0"],
        ),
        (
            "diff(arccsc((x^2+x+3)^2), x)",
            "-2*(2*x+1)/((x^2+x+3)*sqrt((x^2+x+3)^4-1))",
            vec![],
        ),
        (
            "diff(arccsc((2*x^2+2*x+6)^2), x)",
            "-4*(2*x+1)/((2*x^2+2*x+6)*sqrt((2*x^2+2*x+6)^4-1))",
            vec![],
        ),
        (
            "diff(arccsc(((x^2+x+3)/2)^2), x)",
            "-2*(2*x+1)/((x^2+x+3)*sqrt(((x^2+x+3)/2)^4-1))",
            vec![],
        ),
        (
            "diff(arccsc(((1/2)*(x^2+x+3))^2), x)",
            "-2*(2*x+1)/((x^2+x+3)*sqrt(((1/2)*(x^2+x+3))^4-1))",
            vec![],
        ),
        (
            "diff(arccsc(((1/3)*(x^2+x+3))^2), x)",
            "-2*(2*x+1)/((x^2+x+3)*sqrt(((1/3)*(x^2+x+3))^4-1))",
            vec!["(x^2 + x + 3)^4 - 81 > 0"],
        ),
        (
            "diff(arcsec((x^2+x+3)/sqrt(2)), x)",
            "sqrt(2)*(2*x+1)*(x^4+2*x^3+7*x^2+6*x+7)^(-1/2)/(x^2+x+3)",
            vec![],
        ),
        (
            "diff(arcsec((x^2+x+3)/sqrt(1/2)), x)",
            "(2*x+1)*(2*x^4+4*x^3+14*x^2+12*x+17)^(-1/2)/(x^2+x+3)",
            vec![],
        ),
        (
            "diff(arcsec(sqrt(2)*(x^2+x+3)), x)",
            "(2*x+1)*(2*x^4+4*x^3+14*x^2+12*x+17)^(-1/2)/(x^2+x+3)",
            vec![],
        ),
        (
            "diff(arccsc(sqrt(2)*(x^2+x+3)), x)",
            "-(2*x+1)*(2*x^4+4*x^3+14*x^2+12*x+17)^(-1/2)/(x^2+x+3)",
            vec![],
        ),
        (
            "diff(arccsc((x^2+x+3)/sqrt(2)), x)",
            "-sqrt(2)*(2*x+1)*(x^4+2*x^3+7*x^2+6*x+7)^(-1/2)/(x^2+x+3)",
            vec![],
        ),
        ("diff(arccot(x), x)", "-1/(x^2 + 1)", vec![]),
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
        if matches!(input, "diff(arcsec(x^2+1), x)" | "diff(arccsc(x^2+1), x)") {
            assert!(
                !result.contains("1 - 1 /"),
                "positive argument inverse reciprocal trig derivative should not expose reciprocal-square gap: {result}"
            );
            assert!(
                result.contains("x^4 + 2 * x^2"),
                "positive argument inverse reciprocal trig derivative should expose the direct gap: {result}"
            );
        }
        if matches!(
            input,
            "diff(arcsec(x^2+1), x)"
                | "diff(arccsc(x^2+1), x)"
                | "diff(arcsec(x^2+1/2), x)"
                | "diff(arccsc(x^2+1/2), x)"
                | "diff(arcsec(x^2+x+3), x)"
                | "diff(arccsc(x^2+x+3), x)"
                | "diff(arcsec((x^2+1)^2), x)"
                | "diff(arccsc((x^2+1)^2), x)"
                | "diff(arcsec((x^2+x+3)^2), x)"
                | "diff(arccsc((x^2+x+3)^2), x)"
                | "diff(arcsec((2*x^2+2*x+6)^2), x)"
                | "diff(arccsc((2*x^2+2*x+6)^2), x)"
                | "diff(arcsec(((x^2+x+3)/2)^2), x)"
                | "diff(arccsc(((x^2+x+3)/2)^2), x)"
                | "diff(arcsec(((1/2)*(x^2+x+3))^2), x)"
                | "diff(arccsc(((1/2)*(x^2+x+3))^2), x)"
                | "diff(arcsec(((1/3)*(x^2+x+3))^2), x)"
                | "diff(arccsc(((1/3)*(x^2+x+3))^2), x)"
        ) {
            assert!(
                result.contains("sqrt(") && !result.contains("^(-1/2)"),
                "positive quadratic inverse reciprocal trig derivative should use a compact sqrt denominator: {result}"
            );
            assert!(
                !result.contains(") / ((") || !result.contains(") * (x^4"),
                "positive quadratic inverse reciprocal trig derivative should not expose sqrt(gap)/gap shape: {result}"
            );
        }
        if matches!(
            input,
            "diff(arcsec(((1/2)*(x^2+x+3))^2), x)" | "diff(arccsc(((1/2)*(x^2+x+3))^2), x)"
        ) {
            assert!(
                !result.contains("* 1"),
                "multiplicative rational scale presentation should not leak unit factors: {result}"
            );
            assert!(
                result.contains("(x^2 + x + 3) / 2"),
                "multiplicative rational scale presentation should reuse compact quotient base in the gap: {result}"
            );
        }
        if matches!(
            input,
            "diff(arcsec(((1/3)*(x^2+x+3))^2), x)" | "diff(arccsc(((1/3)*(x^2+x+3))^2), x)"
        ) {
            assert!(
                !result.contains("x^8"),
                "post-calculus presentation should keep the result gap compact: {result}"
            );
            assert!(
                result.contains("(x^2 + x + 3) / 3"),
                "multiplicative rational scale presentation should reuse compact quotient base in the gap: {result}"
            );
        }
        if matches!(
            input,
            "diff(arcsec(x), x)"
                | "diff(arccsc(x), x)"
                | "diff(arcsec(2*x), x)"
                | "diff(arccsc(2*x), x)"
                | "diff(arcsec(x+1), x)"
                | "diff(arccsc(x+1), x)"
        ) {
            assert!(
                result.contains('|') && result.contains("sqrt(") && !result.contains("^("),
                "affine inverse reciprocal trig derivative should use an abs-safe sqrt denominator: {result}"
            );
        }
        if matches!(
            input,
            "diff(arcsec(sqrt(x)), x)"
                | "diff(arccsc(sqrt(x)), x)"
                | "diff(arcsec(sqrt(x+1)), x)"
                | "diff(arccsc(sqrt(x+1)), x)"
                | "diff(arcsec(sqrt(2*x)), x)"
                | "diff(arccsc(sqrt(2*x)), x)"
                | "diff(arcsec(sqrt(3-2*x)), x)"
                | "diff(arccsc(sqrt(3-2*x)), x)"
                | "diff(arcsec(sqrt(1-2*x)), x)"
                | "diff(arccsc(sqrt(1-2*x)), x)"
        ) {
            let expected_display = match input {
                "diff(arcsec(sqrt(x)), x)" => "1 / (2 * x * sqrt(x - 1))",
                "diff(arccsc(sqrt(x)), x)" => "-1 / (2 * x * sqrt(x - 1))",
                "diff(arcsec(sqrt(x+1)), x)" => "1 / (2 * (x + 1) * sqrt(x))",
                "diff(arccsc(sqrt(x+1)), x)" => "-1 / (2 * (x + 1) * sqrt(x))",
                "diff(arcsec(sqrt(2*x)), x)" => "1 / (2 * x * sqrt(2 * x - 1))",
                "diff(arccsc(sqrt(2*x)), x)" => "-1 / (2 * x * sqrt(2 * x - 1))",
                "diff(arcsec(sqrt(3-2*x)), x)" => "-1 / ((3 - 2 * x) * sqrt(2 - 2 * x))",
                "diff(arccsc(sqrt(3-2*x)), x)" => "1 / ((3 - 2 * x) * sqrt(2 - 2 * x))",
                "diff(arcsec(sqrt(1-2*x)), x)" => "-1 / ((1 - 2 * x) * sqrt(-2 * x))",
                "diff(arccsc(sqrt(1-2*x)), x)" => "1 / ((1 - 2 * x) * sqrt(-2 * x))",
                _ => unreachable!(),
            };
            assert_eq!(
                result, expected_display,
                "sqrt-affine inverse reciprocal trig derivative should use compact post-calculus presentation"
            );
            assert!(
                !result.contains("^(-1/2)") && !result.contains(" / x)^"),
                "sqrt-affine inverse reciprocal trig derivative should not expose reciprocal-root internals: {result}"
            );
        }
        if matches!(
            input,
            "diff(arcsec((x^2+x+3)/sqrt(2)), x)" | "diff(arccsc((x^2+x+3)/sqrt(2)), x)"
        ) {
            assert!(
                !result.contains("1 - 1 /") && !result.contains("1 - 2 /"),
                "positive surd quotient inverse reciprocal trig derivative should expose direct gap: {result}"
            );
            assert!(
                result.contains("x^4 + 2 * x^3 + 7 * x^2 + 6 * x + 7"),
                "positive surd quotient inverse reciprocal trig derivative should expose q^2-k: {result}"
            );
        }
        if matches!(
            input,
            "diff(arcsec((x^2+x+3)/sqrt(1/2)), x)"
                | "diff(arcsec(sqrt(2)*(x^2+x+3)), x)"
                | "diff(arccsc(sqrt(2)*(x^2+x+3)), x)"
        ) {
            assert!(
                !result.contains("x^8") && !result.contains("1 - 1 /"),
                "scaled surd inverse reciprocal trig derivative should keep compact direct gap: {result}"
            );
            assert!(
                result.contains("2 * x^4 + 4 * x^3 + 14 * x^2 + 12 * x + 17"),
                "scaled surd inverse reciprocal trig derivative should expose value-preserving gap: {result}"
            );
        }

        let required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();
        let expected: Vec<String> = expected_conditions.into_iter().map(String::from).collect();

        assert_eq!(
            required, expected,
            "input: {input}, unexpected required_conditions: {required:?}"
        );
    }
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
            vec!["-x^2 - x > 0".to_string()],
            "input: {input}, unexpected required_conditions: {required:?}"
        );
    }
}
