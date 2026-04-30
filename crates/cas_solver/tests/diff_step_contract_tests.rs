use cas_formatter::DisplayExpr;
use cas_parser::parse;
use cas_session::SessionState;
use cas_solver::runtime::{Engine, EvalAction, EvalRequest, EvalResult, StepsMode};
use cas_solver_core::domain_normalization::normalize_and_dedupe_conditions;

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

    assert_eq!(result, "2 * x * sin(x) + cos(x) * x^2");

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

    assert_eq!(result, "1/2 * x^(-1/2)");

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

    assert_eq!(result, "1/2 * (x + 1)^(-1/2)");

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

    assert_eq!(result, "x * (x^2 + 1)^(-1/2)");

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

    assert_eq!(result, "-1/2 * (2 - x)^(-1/2)");

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

    assert_eq!(result, "-(x * (1 - x^2)^(-1/2))");

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
        required,
        vec!["x - 1 ≠ 0".to_string(), "x > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert!(
        !required.iter().any(|cond| cond.contains("ln(x)")),
        "unexpected redundant ln(base) condition: {required:?}"
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
fn inverse_hyperbolic_acosh_diff_evaluates_with_domain_safe_conditions() {
    let cases = [
        (
            "diff(acosh(x), x)",
            "(x - 1)^(-1/2)/(x + 1)^(1/2)",
            vec!["x - 1 > 0", "x + 1 > 0"],
        ),
        (
            "diff(acosh(2*x+1), x)",
            "2*(2*x)^(-1/2)/(2*x+2)^(1/2)",
            vec!["x > 0"],
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
        assert!(
            result.contains("(1 - x^2)^(-1/2)"),
            "input: {input}, expected reciprocal radical power form, got: {result}"
        );
        if input.contains("arccos") {
            assert!(
                result.starts_with('-') && !result.contains("x^2 - 1"),
                "input: {input}, expected negative reciprocal radical form, got: {result}"
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
fn inverse_reciprocal_trig_diff_evaluates_with_explicit_domain_conditions() {
    let cases = [
        (
            "diff(arcsec(x), x)",
            "((x^2 - 1)/x^2)^(1/2)/(x^2 - 1)",
            vec!["x^2 - 1 > 0"],
        ),
        (
            "diff(arccsc(x), x)",
            "-(((x^2 - 1)/x^2)^(1/2)/(x^2 - 1))",
            vec!["x^2 - 1 > 0"],
        ),
        (
            "diff(arcsec(2*x), x)",
            "(4*((4*x^2 - 1)/(4*x^2))^(1/2))/(8*x^2 - 2)",
            vec!["4 * x^2 - 1 > 0"],
        ),
        (
            "diff(arccsc(2*x), x)",
            "(-4*((4*x^2 - 1)/(4*x^2))^(1/2))/(8*x^2 - 2)",
            vec!["4 * x^2 - 1 > 0"],
        ),
        (
            "diff(arcsec(x+1), x)",
            "((x^2 + 2*x)/(x^2 + 2*x + 1))^(1/2)/(x^2 + 2*x)",
            vec!["x^2 + 2 * x > 0"],
        ),
        (
            "diff(arccsc(x+1), x)",
            "-(((x^2 + 2*x)/(x^2 + 2*x + 1))^(1/2)/(x^2 + 2*x))",
            vec!["x^2 + 2 * x > 0"],
        ),
        (
            "diff(arcsec(x^2+1), x)",
            "(2*x*(1 - 1/(x^2 + 1)^2)^(1/2))/((x^2 + 1)^2 - 1)",
            vec!["x ≠ 0"],
        ),
        (
            "diff(arcsec((x^2+1)^2), x)",
            "(2*x*(2*x^2 + 2)*sqrt(((x^2 + 1)^4 - 1)/(x^2 + 1)^4))/((x^2 + 1)^4 - 1)",
            vec!["x ≠ 0"],
        ),
        (
            "diff(arccsc(x^2+1), x)",
            "(-2*x*(1 - 1/(x^2 + 1)^2)^(1/2))/((x^2 + 1)^2 - 1)",
            vec!["x ≠ 0"],
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
            assert_eq!(result, "-((-x^2 - x)^(-1/2))", "input: {input}");
        } else {
            assert_eq!(result, "(-x^2 - x)^(-1/2)", "input: {input}");
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
