use super::*;

#[test]
fn tan_exp_root_diff_residual_collapses_before_generic_cleanup() {
    let input = "diff(sqrt(tan(x)+exp(x)+x), x) - \
        (cos(x)^2+e^x*cos(x)^2+1)/(2*cos(x)^2*sqrt(tan(x)+e^x+x))";
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
    let rules = output
        .steps
        .iter()
        .map(|step| step.rule_name.as_str())
        .collect::<Vec<_>>();
    assert!(
        rules.iter().any(|rule| matches!(
            *rule,
            "Resolve calculus calls and simplify matching residual before general simplification"
                | "Resolve a matching calculus residual inside its wrapper before general simplification"
                | "Post-calculus residual simplification"
                | "Cancel Equal Fractions Difference"
                | "Cancelar términos opuestos"
        )),
        "held post-calculus fraction residual should close before generic cleanup; steps: {rules:?}"
    );
    assert!(
        rules.iter().all(|rule| !matches!(
            *rule,
            "Add Fractions"
                | "Zero Property of Division"
                | "Rationalize Denominator"
                | "Rationalize Product Denominator"
        )),
        "residual should not enter generic fraction/rationalization cleanup: {rules:?}"
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
        vec!["cos(x) ≠ 0".to_string(), "tan(x) + e^x + x > 0".to_string(),]
    );
}
#[test]
fn exp_trig_by_parts_primitive_diff_residual_collapses() {
    for input in [
        "diff(1/4*exp(2*x+1)*(sin(2*x+1)-cos(2*x+1)), x) - exp(2*x+1)*sin(2*x+1)",
        "diff(1/4*exp(2*x+1)*(sin(2*x+1)+cos(2*x+1)), x) - exp(2*x+1)*cos(2*x+1)",
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
fn reciprocal_trig_affine_diff_omits_non_actionable_cycle_hints() {
    for (input, expected_condition, expected_terms) in [
        (
            "diff(sec((3*x+2)/2), x)",
            "cos((3 * x + 2) / 2) ≠ 0",
            ["sec((3 * x + 2) / 2)", "tan((3 * x + 2) / 2)"],
        ),
        (
            "diff(csc((2-3*x)/2), x)",
            "sin((2 - 3 * x) / 2) ≠ 0",
            ["csc((2 - 3 * x) / 2)", "cot((2 - 3 * x) / 2)"],
        ),
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

        for expected_term in expected_terms {
            assert!(
                result.contains(expected_term),
                "input: {input}, result should keep compact reciprocal trig factors; got: {result}"
            );
        }

        assert!(
            output.metadata.hint_lines.is_empty(),
            "input: {input}, successful derivative should not surface stale cycle hints: {:?}",
            output.metadata.hint_lines
        );
        assert!(
            output
                .metadata
                .requires_lines
                .iter()
                .any(|line| line.contains(expected_condition)),
            "input: {input}, required domain should remain visible: {:?}",
            output.metadata.requires_lines
        );
    }
}
#[test]
fn reciprocal_trig_power_diff_keeps_compact_post_calculus_presentation() {
    // Each reciprocal-trig power is undefined where its base function blows up, so the derivative
    // carries that domain condition (tan/sec → cos ≠ 0, cot/csc → sin ≠ 0).
    let cases = [
        ("diff(tan(x)^2/2, x)", "tan(x) * sec(x)^2", "cos(x) ≠ 0"),
        (
            "diff(tan(2*x+1)^2/2, x)",
            "2 * tan(2 * x + 1) * sec(2 * x + 1)^2",
            "cos(2 * x + 1) ≠ 0",
        ),
        ("diff(-cot(x)^2/2, x)", "cot(x) * csc(x)^2", "sin(x) ≠ 0"),
    ];

    for (input, expected, expected_condition) in cases {
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
            .as_str()
            .strip_prefix("Result: ")
            .unwrap_or_else(|| {
                output
                    .result_line
                    .as_ref()
                    .expect("result line")
                    .line
                    .as_str()
            });

        assert_eq!(result, expected, "input: {input}");
        assert!(
            !result.contains("sin(") && !result.contains("cos("),
            "input: {input}, post-calculus presentation should keep reciprocal trig factors: {result}"
        );
        assert!(
            output
                .metadata
                .requires_lines
                .iter()
                .any(|line| line.contains(expected_condition)),
            "input: {input}, expected required condition `{expected_condition}`, got: {:?}",
            output.metadata.requires_lines
        );
    }
}
#[test]
fn log_tan_cot_sqrt_diff_conditions_are_compact_domain_guards() {
    for (input, expected_equivalent, expected_required) in [
        (
            "diff(ln(tan(sqrt(x))), x)",
            "1/(sqrt(x)*sin(2*sqrt(x)))",
            vec![
                // tan(sqrt(x)) requires cos(sqrt(x)) ≠ 0; the derivative is valid only on that domain.
                "cos(sqrt(x)) ≠ 0".to_string(),
                "sin(sqrt(x)) / cos(sqrt(x)) > 0".to_string(),
                "x > 0".to_string(),
            ],
        ),
        (
            "diff(ln(cot(sqrt(x))), x)",
            "-1/(sqrt(x)*sin(2*sqrt(x)))",
            vec![
                "cos(sqrt(x)) / sin(sqrt(x)) > 0".to_string(),
                "x > 0".to_string(),
            ],
        ),
    ] {
        let mut engine = Engine::new();
        let mut state = SessionState::new();
        let parsed = parse(input, &mut engine.simplifier.context).expect("parse input");
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
        let expected = parse(expected_equivalent, &mut engine.simplifier.context)
            .unwrap_or_else(|err| panic!("parse expected {expected_equivalent}: {err}"));

        assert!(
            engine.simplifier.are_equivalent(result_expr, expected),
            "input: {input}, derivative is not equivalent to {expected_equivalent}"
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
            "input: {input}, unexpected required_conditions: {required:?}"
        );
    }
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
        vec!["y > 0".to_string(), "z ≠ -1".to_string()],
        "unexpected required_conditions: {required:?}"
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
fn arccot_negative_affine_sqrt_shifted_quotient_compacts_contextual_diff() {
    for (input, expected_result, expected_required) in [
        (
            "(1 + diff(arccot(sqrt(5-3*x)), x))/(2 + 3/(2*sqrt(5-3*x)*(6-3*x)))",
            "(3 / (2 * sqrt(5 - 3 * x) * (6 - 3 * x)) + 1) / (3 / (2 * sqrt(5 - 3 * x) * (6 - 3 * x)) + 2)",
            vec![
                "x < 5/3".to_string(),
                "3 / (2 * sqrt(5 - 3 * x) * (6 - 3 * x)) + 2 ≠ 0".to_string(),
            ],
        ),
        (
            "(1 - diff(arccot(sqrt(5-3*x)), x))/(2 - 3/(2*sqrt(5-3*x)*(6-3*x)))",
            "(1 - 3 / (2 * sqrt(5 - 3 * x) * (6 - 3 * x))) / (2 - 3 / (2 * sqrt(5 - 3 * x) * (6 - 3 * x)))",
            vec![
                "x < 5/3".to_string(),
                "2 - 3 / (2 * sqrt(5 - 3 * x) * (6 - 3 * x)) ≠ 0".to_string(),
            ],
        ),
        (
            "(1 + diff(arccot(sqrt(5-3*x)), x))/(1 + 3/(2*sqrt(5-3*x)*(6-3*x)))",
            "1",
            vec![
                "x < 5/3".to_string(),
                "3 / (2 * sqrt(5 - 3 * x) * (6 - 3 * x)) + 1 ≠ 0".to_string(),
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
            !result.contains("(120 - 60 * x)") && !result.contains("depth_overflow"),
            "contextual post-calculus presentation should stay compact: {result}"
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
fn scaled_arccot_sqrt_diff_residual_collapses_without_fraction_add_roundtrip() {
    for (input, expected_required) in [
        (
            "diff(arccot(2*sqrt(x)), x) + 1/(sqrt(x)*(4*x+1))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(arccot(-2*sqrt(x)), x) - 1/(sqrt(x)*(4*x+1))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(arccot(2*sqrt(2*x+3)), x) + 2/(sqrt(2*x+3)*(8*x+13))",
            vec!["x > -3/2".to_string()],
        ),
        (
            "diff(arccot(-2*sqrt(2*x+3)), x) - 2/(sqrt(2*x+3)*(8*x+13))",
            vec!["x > -3/2".to_string()],
        ),
        (
            "diff(arccot(3*sqrt(2*x+1)), x) + 3/(sqrt(2*x+1)*(18*x+10))",
            vec!["x > -1/2".to_string()],
        ),
        (
            "diff(arctan(3*sqrt(2*x+1)), x) - 3/(sqrt(2*x+1)*(18*x+10))",
            vec!["x > -1/2".to_string()],
        ),
        (
            "diff(arccot(3*sqrt(2*x+1)), x) + 6/(4*sqrt(2*x+1)*(9*x+5))",
            vec!["x > -1/2".to_string()],
        ),
        (
            "diff(arctan(3*sqrt(2*x+1)), x) - 6/(4*sqrt(2*x+1)*(9*x+5))",
            vec!["x > -1/2".to_string()],
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

        assert_eq!(result, "0", "input: {input}");

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
            output.steps.len() <= 5,
            "scaled arccot sqrt residual should collapse without a long cleanup route: {:?}",
            output
                .steps
                .iter()
                .map(|step| step.rule_name.as_str())
                .collect::<Vec<_>>()
        );
        assert!(
            output.steps.iter().any(|step| {
                step.rule_name == "Cancelar la subexpresión idénticamente nula"
                    || step.rule_name == "Cancel Exact Additive Pairs"
                    || step.rule_name == "Cancel Opposite Fractions"
                    || step.rule_name == "Cancel Equal Fractions Difference"
            }),
            "scaled arccot sqrt residual should use exact additive cancellation: {:?}",
            output
                .steps
                .iter()
                .map(|step| step.rule_name.as_str())
                .collect::<Vec<_>>()
        );
        assert!(
            output.steps.iter().all(|step| {
                step.rule_name != "Add Fractions"
                    && step.rule_name != "Zero Property of Division"
                    && step.rule_name != "Rationalize Denominator"
                    && step.rule_name != "Rationalize Product Denominator"
            }),
            "scaled arccot sqrt residual should avoid fraction-add/rationalization cleanup: {:?}",
            output
                .steps
                .iter()
                .map(|step| step.rule_name.as_str())
                .collect::<Vec<_>>()
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
        (
            "diff(arccot(sqrt(x^2+2*x+2)), x)",
            "-(x + 1) / (sqrt(x^2 + 2 * x + 2) * (x^2 + 2 * x + 3))",
            "-((2*x+2)*(x^2+2*x+2)^(-1/2))/(2*x^2+4*x+6)",
            Vec::new(),
        ),
        (
            "diff(arccot(sqrt(x^2-2*x+2)), x)",
            "-(x - 1) / (sqrt(x^2 - 2 * x + 2) * (x^2 + 3 - 2 * x))",
            "-((2*x-2)*(x^2-2*x+2)^(-1/2))/(2*x^2-4*x+6)",
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
        if input == "diff(arccot(sqrt(x)), x)" {
            assert!(
                output
                    .steps
                    .iter()
                    .all(|step| step.rule_name != "arccot(x) → arctan(1/x)"),
                "arccot sqrt diff should use the direct arccot route instead of converting through arctan"
            );
            assert!(
                output
                    .steps
                    .iter()
                    .any(|step| step.rule_name == "Present calculus result in compact form"),
                "expected the compact post-calculus presentation step to remain visible"
            );
            assert!(
                output
                    .steps
                    .iter()
                    .all(|step| step.rule_name != "Expand"
                        && step.rule_name != "Distributive Property"),
                "post-calculus presentation should hide expansion round trips: {:?}",
                output
                    .steps
                    .iter()
                    .map(|step| step.rule_name.as_str())
                    .collect::<Vec<_>>()
            );
        }
        if input == "diff(arccot(sqrt(x^2+1)), x)" {
            assert!(
                output
                    .steps
                    .iter()
                    .all(|step| step.rule_name != "arccot(x) → arctan(1/x)"),
                "arccot sqrt diff should use the direct arccot route instead of converting through arctan"
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
                        && step.rule_name != "Rationalize Product Denominator"
                }),
                "post-calculus presentation should hide rationalization round trips: {:?}",
                output
                    .steps
                    .iter()
                    .map(|step| step.rule_name.as_str())
                    .collect::<Vec<_>>()
            );
        }
        if input == "diff(arctan(sqrt(1/(x+1))), x)" {
            assert!(
                output.steps.len() <= 5,
                "shifted reciprocal sqrt arctan diff should avoid the generic abs cleanup route; got {} steps",
                output.steps.len()
            );
            assert!(
                output.steps.iter().all(|step| {
                    step.rule_name != "Abs Squared Identity"
                        && step.rule_name != "Heuristic Poly Normalize"
                }),
                "shifted reciprocal sqrt arctan diff should not route through abs-square/poly-normalize cleanup"
            );
        }
        if input == "diff(arccot(sqrt(1/(x+1))), x)" {
            assert!(
                output.steps.len() <= 5,
                "shifted reciprocal sqrt arccot diff should avoid the generic abs cleanup route; got {} steps",
                output.steps.len()
            );
            assert!(
                output.steps.iter().all(|step| {
                    step.rule_name != "Abs Squared Identity"
                        && step.rule_name != "Heuristic Poly Normalize"
                }),
                "shifted reciprocal sqrt arccot diff should not route through abs-square/poly-normalize cleanup"
            );
        }
    }
}
#[test]
fn negative_arccot_sqrt_diff_uses_direct_positive_root_fraction_presentation() {
    for (input, expected_render, canonical_equivalent, expected_required) in [
        (
            "diff(arccot(-sqrt(x)), x)",
            "1 / (2 * sqrt(x) * (x + 1))",
            "x^(-1/2)/(2*x+2)",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(arccot(-sqrt(x^2+1)), x)",
            "x / (sqrt(x^2 + 1) * (x^2 + 2))",
            "x*(x^2+1)^(-1/2)/(x^2+2)",
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
            !result.contains("^(-1/2)") && !result.contains("^(-3/2)"),
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
            output.steps.len() <= 6,
            "negative arccot sqrt diff should stay on a bounded direct route, got steps: {:?}",
            output
                .steps
                .iter()
                .map(|step| step.rule_name.as_str())
                .collect::<Vec<_>>()
        );
        assert!(
            output.steps.iter().all(|step| {
                step.rule_name != "arccot(x) → arctan(1/x)"
                    && step.rule_name != "Inverse Trig Negative Argument"
                    && step.rule_name != "Simplify Complex Fraction"
            }),
            "negative arccot sqrt diff should avoid inverse-trig rewrite and nested-fraction cleanup: {:?}",
            output
                .steps
                .iter()
                .map(|step| step.rule_name.as_str())
                .collect::<Vec<_>>()
        );
    }
}
#[test]
fn arccot_constant_over_polynomial_sqrt_diff_preserves_compact_denominator() {
    for (input, expected_render, canonical_equivalent, expected_required) in [
        (
            "diff(arccot(sqrt(2/(x+1))), x)",
            "1 / (sqrt(2 / (x + 1)) * (x + 1) * (x + 3))",
            "(2/(x+1))^(-3/2)/(1/2*(x^3+5*x^2+7*x+3))",
            vec!["x > -1".to_string()],
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
            vec!["x < 1".to_string()],
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
        vec!["x ≠ 1".to_string(), "x ≠ -1".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}
#[test]
fn variable_base_constant_argument_log_diff_evaluates_with_base_domain_conditions() {
    let cases = [
        (
            "diff(log(x, 2), x)",
            "-ln(2)/(x*ln(x)^2)",
            vec!["x ≠ 1", "x > 0"],
        ),
        (
            "diff(log(x, y), x)",
            "-ln(y)/(x*ln(x)^2)",
            vec!["x ≠ 1", "x > 0", "y > 0"],
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
        vec!["x ≠ 0".to_string(), "x ≠ -1".to_string()],
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
        vec!["x ≠ 1".to_string(), "x > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert!(
        !required
            .iter()
            .any(|cond| cond == "x ≠ 0" || cond == "x ≠ -1" || cond.contains("x^3")),
        "unexpected redundant power-domain condition: {required:?}"
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
fn chain_rule_nested_trig_exp_composition_diff_evaluates_to_simplified_product() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(sin(e^(x^2)), x)";
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

    assert_eq!(result, "2 * x * cos(e^(x^2)) * e^(x^2)");

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

    let residual_input = "diff(sin(e^(x^2)), x) - 2*x*cos(e^(x^2))*e^(x^2)";
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
        other => panic!("expected expression result, got {other:?}"),
    };

    assert_eq!(residual, "0");

    let residual_required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &residual_output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert!(
        residual_required.is_empty(),
        "unexpected residual required_conditions: {residual_required:?}"
    );
}
#[test]
fn chain_rule_log_trig_composition_diff_keeps_real_domain_condition() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(ln(sin(x)), x)";
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

    assert_eq!(result, "cot(x)");

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert_eq!(
        required,
        vec!["sin(x) > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );

    let residual_input = "diff(ln(sin(x)), x) - cot(x)";
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
        other => panic!("expected expression result, got {other:?}"),
    };

    assert_eq!(residual, "0");

    let residual_required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &residual_output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();

    assert_eq!(
        residual_required,
        vec!["sin(x) > 0".to_string()],
        "unexpected residual required_conditions: {residual_required:?}"
    );
}
#[test]
fn sqrt_log_plus_constant_diff_residual_collapses_with_required_conditions() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "diff(sqrt(ln(x)+1), x) - 1/(2*x*sqrt(ln(x)+1))";
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
        vec!["ln(x) + 1 > 0".to_string(), "x > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
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
            vec![
                // tan((3x+2)/2) requires cos((3x+2)/2) ≠ 0 on the differand's domain.
                "cos((3 * x + 2) / 2) ≠ 0".to_string(),
                "cos(3 * x + 2) + 1 ≠ 0".to_string(),
            ],
        ),
        (
            "diff(cot((2-3*x)/2), x)",
            "3 / (1 - cos(2 - 3 * x))",
            "3/(2*sin((2-3*x)/2)^2)",
            vec![
                "1 - cos(2 - 3 * x) ≠ 0".to_string(),
                // cot((2-3x)/2) requires sin((2-3x)/2) ≠ 0 on the differand's domain.
                "sin((2 - 3 * x) / 2) ≠ 0".to_string(),
            ],
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
fn additive_trig_root_diff_scaled_fraction_residual_collapses_before_trig_expansion() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let input =
        "diff(sqrt(sin(2*x)+cos(x)+4), x) - (2*cos(2*x)-sin(x))/(2*sqrt(sin(2*x)+cos(x)+4))";
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
        "expected the residual to close before trig expansion, got: {:?}",
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

    assert!(
        required.is_empty(),
        "bounded positive trig radicand should not surface new requirements: {required:?}"
    );
}
#[test]
fn additive_trig_root_diff_reciprocal_term_uses_direct_presentation_and_residual() {
    for (input, expected_result, expected_required) in [
        (
            "diff(sqrt(sin(2*x)+cos(x)+1/x), x)",
            "(2 * cos(2 * x) * x^2 - sin(x) * x^2 - 1) / (2 * x^2 * sqrt(sin(2 * x) + cos(x) + 1 / x))",
            vec![
                "sin(2 * x) + cos(x) + 1 / x > 0".to_string(),
                "x ≠ 0".to_string(),
            ],
        ),
        (
            "diff(sqrt(sin(2*x)+cos(x)+2/x), x)",
            "(2 * cos(2 * x) * x^2 - sin(x) * x^2 - 2) / (2 * x^2 * sqrt(sin(2 * x) + cos(x) + 2 / x))",
            vec![
                "sin(2 * x) + cos(x) + 2 / x > 0".to_string(),
                "x ≠ 0".to_string(),
            ],
        ),
        (
            "diff(sqrt(sin(2*x)+cos(x)-2/x), x)",
            "(2 * cos(2 * x) * x^2 + 2 - sin(x) * x^2) / (2 * x^2 * sqrt(sin(2 * x) + cos(x) - 2 / x))",
            vec![
                "x ≠ 0".to_string(),
                "sin(2 * x) + cos(x) - 2 / x > 0".to_string(),
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
        assert_eq!(
            output.steps.len(),
            1,
            "expected reciprocal term route to stay direct, got: {:?}",
            output
                .steps
                .iter()
                .map(|step| step.rule_name.as_str())
                .collect::<Vec<_>>()
        );
        assert_eq!(output.steps[0].rule_name, "Symbolic Differentiation");
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

    for input in [
        "diff(sqrt(sin(2*x)+cos(x)+1/x), x) - \
            (2*cos(2*x)*x^2 - sin(x)*x^2 - 1)/(2*x^2*sqrt(sin(2*x)+cos(x)+1/x))",
        "diff(sqrt(sin(2*x)+cos(x)+2/x), x) - \
            (2*cos(2*x)*x^2 - sin(x)*x^2 - 2)/(2*x^2*sqrt(sin(2*x)+cos(x)+2/x))",
        "diff(sqrt(sin(2*x)+cos(x)-2/x), x) - \
            (2*cos(2*x)*x^2 + 2 - sin(x)*x^2)/(2*x^2*sqrt(sin(2*x)+cos(x)-2/x))",
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
            other => panic!("expected expression result, got {other:?}"),
        };

        assert_eq!(result, "0", "input: {input}");
        assert_eq!(
            output.steps.len(),
            1,
            "expected reciprocal term residual to close before cleanup, got: {:?}",
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
    }

    let input = "diff(sqrt(sin(2*x)+cos(x)+1/x), x) - \
        (cos(2*x)-sin(x)/2-1/(2*x^2))/sqrt(sin(2*x)+cos(x)+1/x)";
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse inline residual");

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
        .expect("eval inline residual");

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

    assert_eq!(result, "0", "inline residual did not collapse");
    assert_eq!(
        output.steps.len(),
        1,
        "expected inline reciprocal residual to close before cleanup, got: {:?}",
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
    assert!(
        required.contains(&"sin(2 * x) + cos(x) + 1 / x > 0".to_string()),
        "missing radicand positivity: {required:?}"
    );
    assert!(
        required.contains(&"x ≠ 0".to_string()),
        "missing reciprocal domain guard: {required:?}"
    );
}
#[test]
fn raw_tangent_trig_root_diff_preserves_direct_presentation_and_residual() {
    for (input, expected_result, expected_step_rule) in [
        (
            "diff(sqrt(tan(x)+sin(x)+x), x)",
            "(cos(x) + sec(x)^2 + 1) / (2 * sqrt(sin(x) + tan(x) + x))",
            "Calcular la derivada",
        ),
        (
            "diff(sqrt(tan(x)+sin(x)+x), x) - (cos(x)^2+cos(x)^3+1)/(2*cos(x)^2*sqrt(sin(x)+tan(x)+x))",
            "0",
            "Resolve a matching calculus residual inside its wrapper before general simplification",
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
        assert_eq!(
            output.steps.len(),
            1,
            "raw tangent root diff should stay on a one-step presentation route: {:?}",
            output
                .steps
                .iter()
                .map(|step| step.rule_name.as_str())
                .collect::<Vec<_>>()
        );
        assert_eq!(output.steps[0].rule_name, expected_step_rule, "input: {input}");

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
                "cos(x) ≠ 0".to_string(),
                "sin(x) + tan(x) + x > 0".to_string(),
            ],
            "input: {input}"
        );
    }
}
#[test]
fn tan_exp_root_diff_scaled_fraction_residual_collapses_before_cleanup() {
    for input in [
        "diff(sqrt(tan(x)+exp(x)+x), x) - (2*cos(x)^2+2*e^x*cos(x)^2+2)/(4*cos(x)^2*sqrt(tan(x)+e^x+x))",
        "diff(sqrt(tan(x)+exp(x)+x), x) - (sec(x)^2+e^x+1)/(2*sqrt(tan(x)+exp(x)+x))",
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
            "expected the residual to close before cleanup, got: {:?}",
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
            vec!["cos(x) ≠ 0".to_string(), "tan(x) + e^x + x > 0".to_string()],
            "input: {input}"
        );
    }
}
#[test]
fn cot_exp_root_diff_uses_direct_presentation_and_residual() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;

    let direct_input = "diff(sqrt(cot(x)+exp(x)+x), x)";
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

    assert_eq!(
        result,
        "(e^x + 1 - csc(x)^2) / (2 * sqrt(cot(x) + e^x + x))"
    );
    assert_eq!(
        output.steps.len(),
        1,
        "cot root diff should stay on the direct presentation route: {:?}",
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
        vec!["cot(x) + e^x + x > 0".to_string(), "sin(x) ≠ 0".to_string()]
    );

    let residual_input =
        "diff(sqrt(cot(x)+exp(x)+x), x) - (1-csc(x)^2+e^x)/(2*sqrt(cot(x)+exp(x)+x))";
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
    assert_eq!(result, "0");
    assert_eq!(
        output.steps.len(),
        1,
        "cot/csc residual should close before cleanup, got: {:?}",
        output
            .steps
            .iter()
            .map(|step| step.rule_name.as_str())
            .collect::<Vec<_>>()
    );
}
#[test]
fn cot_sqrt_root_diff_uses_direct_csc_presentation_and_residual() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;

    let direct_input = "diff(sqrt(cot(x)+sqrt(x)+x), x)";
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

    assert_eq!(
        result,
        "(1 / (2 * sqrt(x)) + 1 - csc(x)^2) / (2 * sqrt(cot(x) + sqrt(x) + x))"
    );
    assert_eq!(
        output.steps.len(),
        1,
        "cot/sqrt root diff should stay on a direct presentation route: {:?}",
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
            "cot(x) + sqrt(x) + x > 0".to_string(),
            "sin(x) ≠ 0".to_string(),
            "x > 0".to_string(),
        ]
    );

    let residual_input = "diff(sqrt(cot(x)+sqrt(x)+x), x) - (2*sqrt(x)+1-2*sqrt(x)*csc(x)^2)/(4*sqrt(x)*sqrt(cot(x)+sqrt(x)+x))";
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
    assert_eq!(result, "0");
    assert_eq!(
        output.steps.len(),
        1,
        "cot/sqrt residual should close before cleanup, got: {:?}",
        output
            .steps
            .iter()
            .map(|step| step.rule_name.as_str())
            .collect::<Vec<_>>()
    );
}
#[test]
fn sec_csc_sqrt_root_diff_uses_direct_reciprocal_trig_presentation_and_residual() {
    for (direct_input, expected, expected_required, residual_input) in [
        (
            "diff(sqrt(sec(x)+sqrt(x)+x), x)",
            "(tan(x) * sec(x) + 1 / (2 * sqrt(x)) + 1) / (2 * sqrt(sec(x) + sqrt(x) + x))",
            vec![
                "cos(x) ≠ 0".to_string(),
                "sec(x) + sqrt(x) + x > 0".to_string(),
                "x > 0".to_string(),
            ],
            "diff(sqrt(sec(x)+sqrt(x)+x), x) - (sec(x)*tan(x)+1/(2*sqrt(x))+1)/(2*sqrt(sec(x)+sqrt(x)+x))",
        ),
        (
            "diff(sqrt(csc(x)+sqrt(x)+x), x)",
            "(1 / (2 * sqrt(x)) + 1 - csc(x) * cot(x)) / (2 * sqrt(csc(x) + sqrt(x) + x))",
            vec![
                "csc(x) + sqrt(x) + x > 0".to_string(),
                "sin(x) ≠ 0".to_string(),
                "x > 0".to_string(),
            ],
            "diff(sqrt(csc(x)+sqrt(x)+x), x) - (1/(2*sqrt(x))+1-csc(x)*cot(x))/(2*sqrt(csc(x)+sqrt(x)+x))",
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
            "reciprocal trig root diff should stay on a direct presentation route: {:?}",
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
            "reciprocal trig residual should close before cleanup, got: {:?}",
            output
                .steps
                .iter()
                .map(|step| step.rule_name.as_str())
                .collect::<Vec<_>>()
        );
    }
}
#[test]
fn sec_csc_exp_root_diff_uses_direct_reciprocal_trig_presentation_and_residual() {
    for (direct_input, expected, expected_required, residual_input) in [
        (
            "diff(sqrt(sec(x)+exp(x)+x), x)",
            "(e^x + tan(x) * sec(x) + 1) / (2 * sqrt(sec(x) + e^x + x))",
            vec!["cos(x) ≠ 0".to_string(), "sec(x) + e^x + x > 0".to_string()],
            "diff(sqrt(sec(x)+exp(x)+x), x) - (sec(x)*tan(x)+e^x+1)/(2*sqrt(sec(x)+e^x+x))",
        ),
        (
            "diff(sqrt(csc(x)+exp(x)+x), x)",
            "(e^x + 1 - csc(x) * cot(x)) / (2 * sqrt(csc(x) + e^x + x))",
            vec!["csc(x) + e^x + x > 0".to_string(), "sin(x) ≠ 0".to_string()],
            "diff(sqrt(csc(x)+exp(x)+x), x) - (e^x+1-csc(x)*cot(x))/(2*sqrt(csc(x)+e^x+x))",
        ),
        (
            "diff(sqrt(sec(x)+exp(x)+sqrt(x)+x), x)",
            "(e^x + tan(x) * sec(x) + 1 / (2 * sqrt(x)) + 1) / (2 * sqrt(sec(x) + sqrt(x) + e^x + x))",
            vec![
                "cos(x) ≠ 0".to_string(),
                "sec(x) + sqrt(x) + e^x + x > 0".to_string(),
                "x > 0".to_string(),
            ],
            "diff(sqrt(sec(x)+exp(x)+sqrt(x)+x), x) - (e^x+sec(x)*tan(x)+1/(2*sqrt(x))+1)/(2*sqrt(sec(x)+sqrt(x)+e^x+x))",
        ),
        (
            "diff(sqrt(csc(x)+exp(x)+sqrt(x)+x), x)",
            "(e^x + 1 / (2 * sqrt(x)) + 1 - csc(x) * cot(x)) / (2 * sqrt(csc(x) + sqrt(x) + e^x + x))",
            vec![
                "csc(x) + sqrt(x) + e^x + x > 0".to_string(),
                "sin(x) ≠ 0".to_string(),
                "x > 0".to_string(),
            ],
            "diff(sqrt(csc(x)+exp(x)+sqrt(x)+x), x) - (e^x+1/(2*sqrt(x))+1-csc(x)*cot(x))/(2*sqrt(csc(x)+sqrt(x)+e^x+x))",
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
            "reciprocal trig exp root diff should stay on a direct presentation route: {:?}",
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
            "reciprocal trig exp residual should close before cleanup, got: {:?}",
            output
                .steps
                .iter()
                .map(|step| step.rule_name.as_str())
                .collect::<Vec<_>>()
        );
    }
}
#[test]
fn tan_ln_root_diff_uses_inline_presentation_and_residual() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;

    let direct_input = "diff(sqrt(tan(x)+ln(x)+x), x)";
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

    assert_eq!(
        result,
        "(sec(x)^2 + 1 / x + 1) / (2 * sqrt(tan(x) + ln(x) + x))"
    );
    assert_eq!(
        output.steps.len(),
        1,
        "tan ln root diff should stay on a direct inline presentation route: {:?}",
        output
            .steps
            .iter()
            .map(|step| step.rule_name.as_str())
            .collect::<Vec<_>>()
    );

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
        vec![
            "cos(x) ≠ 0".to_string(),
            "tan(x) + ln(x) + x > 0".to_string(),
            "x > 0".to_string(),
        ],
        "input: {direct_input}"
    );

    let residual_input =
        "diff(sqrt(tan(x)+ln(x)+x), x) - (sec(x)^2+1/x+1)/(2*sqrt(tan(x)+ln(x)+x))";
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
        "tan ln root residual should close before cleanup, got: {:?}",
        output
            .steps
            .iter()
            .map(|step| step.rule_name.as_str())
            .collect::<Vec<_>>()
    );
}
#[test]
fn scaled_tan_ln_root_diff_uses_inline_presentation_and_residual() {
    for (direct_input, expected, expected_positive_radicand, residual_input) in [
        (
            "diff(sqrt(tan(x)+2*ln(x)+x), x)",
            "(sec(x)^2 + 2 / x + 1) / (2 * sqrt(tan(x) + 2 * ln(x) + x))",
            "tan(x) + 2 * ln(x) + x > 0",
            "diff(sqrt(tan(x)+2*ln(x)+x), x) - (sec(x)^2+2/x+1)/(2*sqrt(tan(x)+2*ln(x)+x))",
        ),
        (
            "diff(sqrt(tan(x)-ln(x)+x), x)",
            "(sec(x)^2 + 1 - 1 / x) / (2 * sqrt(tan(x) - ln(x) + x))",
            "tan(x) - ln(x) + x > 0",
            "diff(sqrt(tan(x)-ln(x)+x), x) - (sec(x)^2-1/x+1)/(2*sqrt(tan(x)-ln(x)+x))",
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
            "scaled tan ln root diff should stay on a direct inline presentation route: {:?}",
            output
                .steps
                .iter()
                .map(|step| step.rule_name.as_str())
                .collect::<Vec<_>>()
        );

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
            vec![
                "cos(x) ≠ 0".to_string(),
                expected_positive_radicand.to_string(),
                "x > 0".to_string(),
            ],
            "input: {direct_input}"
        );

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
            "scaled tan ln root residual should close before cleanup, got: {:?}",
            output
                .steps
                .iter()
                .map(|step| step.rule_name.as_str())
                .collect::<Vec<_>>()
        );
    }
}
#[test]
fn tan_ln_sqrt_root_diff_uses_inline_presentation_and_residual() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;

    let direct_input = "diff(sqrt(tan(x)+ln(x)+sqrt(x)+x), x)";
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

    assert_eq!(
        result,
        "(sec(x)^2 + 1 / x + 1 / (2 * sqrt(x)) + 1) / (2 * sqrt(tan(x) + ln(x) + sqrt(x) + x))"
    );
    assert_eq!(
        output.steps.len(),
        1,
        "tan ln sqrt root diff should stay on a direct inline presentation route: {:?}",
        output
            .steps
            .iter()
            .map(|step| step.rule_name.as_str())
            .collect::<Vec<_>>()
    );

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
        vec![
            "cos(x) ≠ 0".to_string(),
            "tan(x) + ln(x) + sqrt(x) + x > 0".to_string(),
            "x > 0".to_string(),
        ],
        "input: {direct_input}"
    );

    let residual_input = "diff(sqrt(tan(x)+ln(x)+sqrt(x)+x), x) - (sec(x)^2+1/x+1/(2*sqrt(x))+1)/(2*sqrt(tan(x)+ln(x)+sqrt(x)+x))";
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
        "tan ln sqrt root residual should close before cleanup, got: {:?}",
        output
            .steps
            .iter()
            .map(|step| step.rule_name.as_str())
            .collect::<Vec<_>>()
    );
}
#[test]
fn tan_ln_reciprocal_sqrt_root_diff_uses_inline_presentation_and_residual() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;

    let direct_input = "diff(sqrt(tan(x)+ln(x)+1/sqrt(x)+x), x)";
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

    assert_eq!(
        result,
        "(sec(x)^2 + 1 / x + 1 - 1/2 * x^(-3/2)) / (2 * sqrt(tan(x) + ln(x) + 1 / sqrt(x) + x))"
    );
    assert_eq!(
        output.steps.len(),
        1,
        "tan ln reciprocal-sqrt root diff should stay on a direct inline presentation route: {:?}",
        output
            .steps
            .iter()
            .map(|step| step.rule_name.as_str())
            .collect::<Vec<_>>()
    );

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
        vec![
            "cos(x) ≠ 0".to_string(),
            "tan(x) + ln(x) + 1 / sqrt(x) + x > 0".to_string(),
            "x > 0".to_string(),
        ],
        "input: {direct_input}"
    );

    let residual_input = "diff(sqrt(tan(x)+ln(x)+1/sqrt(x)+x), x) - (sec(x)^2+1/x-1/(2*x^(3/2))+1)/(2*sqrt(tan(x)+ln(x)+1/sqrt(x)+x))";
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
        "tan ln reciprocal-sqrt root residual should close before cleanup, got: {:?}",
        output
            .steps
            .iter()
            .map(|step| step.rule_name.as_str())
            .collect::<Vec<_>>()
    );
}
#[test]
fn sec_csc_ln_root_diff_uses_direct_common_denominator_presentation_and_residual() {
    for (direct_input, expected, expected_required, residual_input) in [
        (
            "diff(sqrt(sec(x)+ln(x)+x), x)",
            "(x * tan(x) * sec(x) + x + 1) / (2 * x * sqrt(sec(x) + ln(x) + x))",
            vec![
                "sec(x) + ln(x) + x > 0".to_string(),
                "cos(x) ≠ 0".to_string(),
                "x > 0".to_string(),
            ],
            "diff(sqrt(sec(x)+ln(x)+x), x) - (x*sec(x)*tan(x)+x+1)/(2*x*sqrt(sec(x)+ln(x)+x))",
        ),
        (
            "diff(sqrt(csc(x)+ln(x)+x), x)",
            "(x + 1 - x * csc(x) * cot(x)) / (2 * x * sqrt(csc(x) + ln(x) + x))",
            vec![
                "csc(x) + ln(x) + x > 0".to_string(),
                "sin(x) ≠ 0".to_string(),
                "x > 0".to_string(),
            ],
            "diff(sqrt(csc(x)+ln(x)+x), x) - (x+1-x*csc(x)*cot(x))/(2*x*sqrt(csc(x)+ln(x)+x))",
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
            "reciprocal trig ln root diff should stay on a direct presentation route: {:?}",
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
            "reciprocal trig ln residual should close before cleanup, got: {:?}",
            output
                .steps
                .iter()
                .map(|step| step.rule_name.as_str())
                .collect::<Vec<_>>()
        );
    }
}
#[test]
fn sec_csc_ln_sqrt_root_diff_uses_direct_common_denominator_presentation_and_residual() {
    for (direct_input, expected, expected_required, residual_input) in [
        (
            "diff(sqrt(sec(x)+ln(x)+sqrt(x)+x), x)",
            "(2 * sqrt(x) + 2 * x * sqrt(x) + 2 * x * tan(x) * sec(x) * sqrt(x) + x) / (4 * x * sqrt(x) * sqrt(sec(x) + ln(x) + sqrt(x) + x))",
            vec![
                "x > 0".to_string(),
                "sec(x) + ln(x) + sqrt(x) + x > 0".to_string(),
                "cos(x) ≠ 0".to_string(),
            ],
            "diff(sqrt(sec(x)+ln(x)+sqrt(x)+x), x) - (2*sqrt(x)+2*x*sqrt(x)+2*x*tan(x)*sec(x)*sqrt(x)+x)/(4*x*sqrt(x)*sqrt(sec(x)+ln(x)+sqrt(x)+x))",
        ),
        (
            "diff(sqrt(csc(x)+ln(x)+sqrt(x)+x), x)",
            "(2 * sqrt(x) + 2 * x * sqrt(x) + x - 2 * x * csc(x) * cot(x) * sqrt(x)) / (4 * x * sqrt(x) * sqrt(csc(x) + ln(x) + sqrt(x) + x))",
            vec![
                "x > 0".to_string(),
                "csc(x) + ln(x) + sqrt(x) + x > 0".to_string(),
                "sin(x) ≠ 0".to_string(),
            ],
            "diff(sqrt(csc(x)+ln(x)+sqrt(x)+x), x) - (2*sqrt(x)+2*x*sqrt(x)+x-2*x*csc(x)*cot(x)*sqrt(x))/(4*x*sqrt(x)*sqrt(csc(x)+ln(x)+sqrt(x)+x))",
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
            "reciprocal trig ln/sqrt root diff should stay on a direct presentation route: {:?}",
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
            "reciprocal trig ln/sqrt residual should close before cleanup, got: {:?}",
            output
                .steps
                .iter()
                .map(|step| step.rule_name.as_str())
                .collect::<Vec<_>>()
        );
    }
}
#[test]
fn sec_csc_ln_reciprocal_sqrt_root_diff_uses_compact_common_denominator_presentation() {
    for (direct_input, expected, expected_required, residual_input) in [
        (
            "diff(sqrt(sec(x)+ln(x)+1/sqrt(x)+x), x)",
            "(2 * sqrt(x) + 2 * x * sqrt(x) + 2 * x * tan(x) * sec(x) * sqrt(x) - 1) / (4 * x * sqrt(x) * sqrt(sec(x) + ln(x) + 1 / sqrt(x) + x))",
            vec![
                "x > 0".to_string(),
                "sec(x) + ln(x) + 1 / sqrt(x) + x > 0".to_string(),
                "cos(x) ≠ 0".to_string(),
            ],
            "diff(sqrt(sec(x)+ln(x)+1/sqrt(x)+x), x) - (2*sqrt(x)+2*x*sqrt(x)+2*x*tan(x)*sec(x)*sqrt(x)-1)/(4*x*sqrt(x)*sqrt(sec(x)+ln(x)+1/sqrt(x)+x))",
        ),
        (
            "diff(sqrt(csc(x)+ln(x)+1/sqrt(x)+x), x)",
            "(2 * sqrt(x) + 2 * x * sqrt(x) - 2 * x * csc(x) * cot(x) * sqrt(x) - 1) / (4 * x * sqrt(x) * sqrt(csc(x) + ln(x) + 1 / sqrt(x) + x))",
            vec![
                "x > 0".to_string(),
                "csc(x) + ln(x) + 1 / sqrt(x) + x > 0".to_string(),
                "sin(x) ≠ 0".to_string(),
            ],
            "diff(sqrt(csc(x)+ln(x)+1/sqrt(x)+x), x) - (2*sqrt(x)+2*x*sqrt(x)-2*x*csc(x)*cot(x)*sqrt(x)-1)/(4*x*sqrt(x)*sqrt(csc(x)+ln(x)+1/sqrt(x)+x))",
        ),
        (
            "diff(arctan(sqrt(sec(x)+ln(x)+1/sqrt(x)+x)), x)",
            "(2 * sqrt(x) + 2 * x * sqrt(x) + 2 * x * tan(x) * sec(x) * sqrt(x) - 1) / (4 * x * sqrt(x) * sqrt(sec(x) + ln(x) + 1 / sqrt(x) + x) * (sec(x) + ln(x) + 1 / sqrt(x) + x + 1))",
            vec![
                "x > 0".to_string(),
                "sec(x) + ln(x) + 1 / sqrt(x) + x > 0".to_string(),
                "cos(x) ≠ 0".to_string(),
            ],
            "diff(arctan(sqrt(sec(x)+ln(x)+1/sqrt(x)+x)), x) - (2*sqrt(x)+2*x*sqrt(x)+2*x*tan(x)*sec(x)*sqrt(x)-1)/(4*x*sqrt(x)*sqrt(sec(x)+ln(x)+1/sqrt(x)+x)*(sec(x)+ln(x)+1/sqrt(x)+x+1))",
        ),
        (
            "diff(arctan(sqrt(csc(x)+ln(x)+1/sqrt(x)+x)), x)",
            "(2 * sqrt(x) + 2 * x * sqrt(x) - 2 * x * csc(x) * cot(x) * sqrt(x) - 1) / (4 * x * sqrt(x) * sqrt(csc(x) + ln(x) + 1 / sqrt(x) + x) * (csc(x) + ln(x) + 1 / sqrt(x) + x + 1))",
            vec![
                "x > 0".to_string(),
                "csc(x) + ln(x) + 1 / sqrt(x) + x > 0".to_string(),
                "sin(x) ≠ 0".to_string(),
            ],
            "diff(arctan(sqrt(csc(x)+ln(x)+1/sqrt(x)+x)), x) - (2*sqrt(x)+2*x*sqrt(x)-2*x*csc(x)*cot(x)*sqrt(x)-1)/(4*x*sqrt(x)*sqrt(csc(x)+ln(x)+1/sqrt(x)+x)*(csc(x)+ln(x)+1/sqrt(x)+x+1))",
        ),
        (
            "diff(sqrt(csc(x)-2*ln(x)+3/sqrt(x)+x), x)",
            "(2 * x * sqrt(x) - 4 * sqrt(x) - 2 * x * csc(x) * cot(x) * sqrt(x) - 3) / (4 * x * sqrt(x) * sqrt(csc(x) - 2 * ln(x) + 3 / sqrt(x) + x))",
            vec![
                "x > 0".to_string(),
                "csc(x) - 2 * ln(x) + 3 / sqrt(x) + x > 0".to_string(),
                "sin(x) ≠ 0".to_string(),
            ],
            "diff(sqrt(csc(x)-2*ln(x)+3/sqrt(x)+x), x) - (2*x*sqrt(x)-2*x*csc(x)*cot(x)*sqrt(x)-4*sqrt(x)-3)/(4*x*sqrt(x)*sqrt(csc(x)-2*ln(x)+3/sqrt(x)+x))",
        ),
        (
            "diff(arctan(sqrt(csc(x)-2*ln(x)+3/sqrt(x)+x)), x)",
            "(2 * x * sqrt(x) - 4 * sqrt(x) - 2 * x * csc(x) * cot(x) * sqrt(x) - 3) / (4 * x * sqrt(x) * sqrt(csc(x) - 2 * ln(x) + 3 / sqrt(x) + x) * (csc(x) - 2 * ln(x) + 3 / sqrt(x) + x + 1))",
            vec![
                "x > 0".to_string(),
                "csc(x) - 2 * ln(x) + 3 / sqrt(x) + x > 0".to_string(),
                "sin(x) ≠ 0".to_string(),
            ],
            "diff(arctan(sqrt(csc(x)-2*ln(x)+3/sqrt(x)+x)), x) - (2*x*sqrt(x)-2*x*csc(x)*cot(x)*sqrt(x)-4*sqrt(x)-3)/(4*x*sqrt(x)*sqrt(csc(x)-2*ln(x)+3/sqrt(x)+x)*(csc(x)-2*ln(x)+3/sqrt(x)+x+1))",
        ),
        (
            "diff(sqrt(sec(x)+ln(x)+sqrt(x)+1/sqrt(x)+x), x)",
            "(2 * sqrt(x) + 2 * x * sqrt(x) + 2 * x * tan(x) * sec(x) * sqrt(x) + x - 1) / (4 * x * sqrt(x) * sqrt(sec(x) + ln(x) + sqrt(x) + 1 / sqrt(x) + x))",
            vec![
                "x > 0".to_string(),
                "sec(x) + ln(x) + sqrt(x) + 1 / sqrt(x) + x > 0".to_string(),
                "cos(x) ≠ 0".to_string(),
            ],
            "diff(sqrt(sec(x)+ln(x)+sqrt(x)+1/sqrt(x)+x), x) - (2*sqrt(x)+2*x*sqrt(x)+2*x*tan(x)*sec(x)*sqrt(x)+x-1)/(4*x*sqrt(x)*sqrt(sec(x)+ln(x)+sqrt(x)+1/sqrt(x)+x))",
        ),
        (
            "diff(sqrt(csc(x)+ln(x)+sqrt(x)+1/sqrt(x)+x), x)",
            "(2 * sqrt(x) + 2 * x * sqrt(x) + x - 2 * x * csc(x) * cot(x) * sqrt(x) - 1) / (4 * x * sqrt(x) * sqrt(csc(x) + ln(x) + sqrt(x) + 1 / sqrt(x) + x))",
            vec![
                "x > 0".to_string(),
                "csc(x) + ln(x) + sqrt(x) + 1 / sqrt(x) + x > 0".to_string(),
                "sin(x) ≠ 0".to_string(),
            ],
            "diff(sqrt(csc(x)+ln(x)+sqrt(x)+1/sqrt(x)+x), x) - (2*sqrt(x)+2*x*sqrt(x)+x-2*x*csc(x)*cot(x)*sqrt(x)-1)/(4*x*sqrt(x)*sqrt(csc(x)+ln(x)+sqrt(x)+1/sqrt(x)+x))",
        ),
        (
            "diff(arctan(sqrt(sec(x)+ln(x)+sqrt(x)+1/sqrt(x)+x)), x)",
            "(2 * sqrt(x) + 2 * x * sqrt(x) + 2 * x * tan(x) * sec(x) * sqrt(x) + x - 1) / (4 * x * sqrt(x) * sqrt(sec(x) + ln(x) + sqrt(x) + 1 / sqrt(x) + x) * (sec(x) + ln(x) + sqrt(x) + 1 / sqrt(x) + x + 1))",
            vec![
                "x > 0".to_string(),
                "sec(x) + ln(x) + sqrt(x) + 1 / sqrt(x) + x > 0".to_string(),
                "cos(x) ≠ 0".to_string(),
            ],
            "diff(arctan(sqrt(sec(x)+ln(x)+sqrt(x)+1/sqrt(x)+x)), x) - (2*sqrt(x)+2*x*sqrt(x)+2*x*tan(x)*sec(x)*sqrt(x)+x-1)/(4*x*sqrt(x)*sqrt(sec(x)+ln(x)+sqrt(x)+1/sqrt(x)+x)*(sec(x)+ln(x)+sqrt(x)+1/sqrt(x)+x+1))",
        ),
        (
            "diff(arctan(sqrt(csc(x)+ln(x)+sqrt(x)+1/sqrt(x)+x)), x)",
            "(2 * sqrt(x) + 2 * x * sqrt(x) + x - 2 * x * csc(x) * cot(x) * sqrt(x) - 1) / (4 * x * sqrt(x) * sqrt(csc(x) + ln(x) + sqrt(x) + 1 / sqrt(x) + x) * (csc(x) + ln(x) + sqrt(x) + 1 / sqrt(x) + x + 1))",
            vec![
                "x > 0".to_string(),
                "csc(x) + ln(x) + sqrt(x) + 1 / sqrt(x) + x > 0".to_string(),
                "sin(x) ≠ 0".to_string(),
            ],
            "diff(arctan(sqrt(csc(x)+ln(x)+sqrt(x)+1/sqrt(x)+x)), x) - (2*sqrt(x)+2*x*sqrt(x)+x-2*x*csc(x)*cot(x)*sqrt(x)-1)/(4*x*sqrt(x)*sqrt(csc(x)+ln(x)+sqrt(x)+1/sqrt(x)+x)*(csc(x)+ln(x)+sqrt(x)+1/sqrt(x)+x+1))",
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
        assert!(
            !result.contains("x^(-3/2)") && !result.contains("x * x^("),
            "post-calculus presentation should avoid raw reciprocal half-powers: {result}"
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
            "reciprocal-sqrt residual should close before cleanup, got: {:?}",
            output
                .steps
                .iter()
                .map(|step| step.rule_name.as_str())
                .collect::<Vec<_>>()
        );
    }
}
#[test]
fn tan_exp_linear_root_diff_scaled_fraction_residual_collapses_before_cleanup() {
    for (input, expected_required) in [
        (
            "diff(sqrt(tan(x)+exp(2*x)+x), x) - (2*cos(x)^2+4*e^(2*x)*cos(x)^2+2)/(4*cos(x)^2*sqrt(tan(x)+e^(2*x)+x))",
            vec![
                "cos(x) ≠ 0".to_string(),
                "tan(x) + e^(2 * x) + x > 0".to_string(),
            ],
        ),
        (
            "diff(sqrt(tan(x)+exp(2*x+1)+x), x) - (2*cos(x)^2+4*e^(2*x+1)*cos(x)^2+2)/(4*cos(x)^2*sqrt(tan(x)+e^(2*x+1)+x))",
            vec![
                "cos(x) ≠ 0".to_string(),
                "tan(x) + e^(2 * x + 1) + x > 0".to_string(),
            ],
        ),
        (
            "diff(sqrt(tan(x)+exp(-2*x)+x), x) - (2*cos(x)^2+2-4*e^(-2*x)*cos(x)^2)/(4*cos(x)^2*sqrt(tan(x)+e^(-2*x)+x))",
            vec![
                "cos(x) ≠ 0".to_string(),
                "tan(x) + e^(-2 * x) + x > 0".to_string(),
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
            "expected the exp-linear residual to close before cleanup, got: {:?}",
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

        for condition in expected_required {
            assert!(
                required.contains(&condition),
                "missing required condition {condition:?}; got {required:?}; input: {input}"
            );
        }
    }
}
#[test]
fn tan_exp_linear_root_diff_uses_direct_presentation_before_cleanup() {
    for (input, expected_result, expected_required) in [
        (
            "diff(sqrt(tan(x)+exp(2*x)+x), x)",
            "(sec(x)^2 + 2 * e^(2 * x) + 1) / (2 * sqrt(tan(x) + e^(2 * x) + x))",
            vec![
                "cos(x) ≠ 0".to_string(),
                "tan(x) + e^(2 * x) + x > 0".to_string(),
            ],
        ),
        (
            "diff(sqrt(tan(x)+exp(2*x+1)+x), x)",
            "(sec(x)^2 + 2 * e^(2 * x + 1) + 1) / (2 * sqrt(tan(x) + e^(2 * x + 1) + x))",
            vec![
                "cos(x) ≠ 0".to_string(),
                "tan(x) + e^(2 * x + 1) + x > 0".to_string(),
            ],
        ),
        (
            "diff(sqrt(tan(x)+exp(-2*x)+x), x)",
            "(sec(x)^2 + 1 - 2 * e^(-2 * x)) / (2 * sqrt(tan(x) + e^(-2 * x) + x))",
            vec![
                "cos(x) ≠ 0".to_string(),
                "tan(x) + e^(-2 * x) + x > 0".to_string(),
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
        assert_eq!(
            output.steps.len(),
            1,
            "expected direct presentation to avoid general cleanup, got: {:?}",
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

        for condition in expected_required {
            assert!(
                required.contains(&condition),
                "missing required condition {condition:?}; got {required:?}; input: {input}"
            );
        }
    }
}
#[test]
fn tan_exp_sqrt_root_diff_cross_family_contract_stays_compact() {
    for (input, expected_result, expected_required) in [
        (
            "diff(sqrt(tan(x)+exp(x)+sqrt(x)+x), x)",
            "(e^x + sec(x)^2 + 1 / (2 * sqrt(x)) + 1) / (2 * sqrt(tan(x) + sqrt(x) + e^x + x))",
            vec![
                "cos(x) ≠ 0".to_string(),
                "tan(x) + sqrt(x) + e^x + x > 0".to_string(),
                "x > 0".to_string(),
            ],
        ),
        (
            "diff(arctan(sqrt(tan(x)+exp(x)+sqrt(x)+x)), x)",
            "(e^x + sec(x)^2 + 1 / (2 * sqrt(x)) + 1) / (2 * sqrt(tan(x) + sqrt(x) + e^x + x) * (tan(x) + sqrt(x) + e^x + x + 1))",
            vec![
                "x > 0".to_string(),
                "tan(x) + sqrt(x) + e^x + x > 0".to_string(),
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

        assert_eq!(result, expected_result, "input: {input}");
        assert_eq!(
            output.steps.len(),
            1,
            "expected cross-family diff presentation to stay direct, got: {:?}",
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

        let mut required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();
        let mut expected_required = expected_required;
        required.sort();
        expected_required.sort();
        assert_eq!(required, expected_required, "input: {input}");
    }
}
#[test]
fn tan_exp_sqrt_root_diff_cross_family_residuals_collapse_before_cleanup() {
    for (input, expected_required) in [
        (
            "diff(sqrt(tan(x)+exp(x)+sqrt(x)+x), x) - (2*sqrt(x)+2*sqrt(x)*e^x+2*sqrt(x)*sec(x)^2+1)/(4*sqrt(x)*sqrt(tan(x)+sqrt(x)+e^x+x))",
            vec![
                "cos(x) ≠ 0".to_string(),
                "tan(x) + sqrt(x) + e^x + x > 0".to_string(),
                "x > 0".to_string(),
            ],
        ),
        (
            "diff(sqrt(tan(x)+exp(x)+sqrt(x)+x), x) - (sec(x)^2+e^x+1+1/(2*sqrt(x)))/(2*sqrt(tan(x)+exp(x)+sqrt(x)+x))",
            vec![
                "cos(x) ≠ 0".to_string(),
                "tan(x) + sqrt(x) + e^x + x > 0".to_string(),
                "x > 0".to_string(),
            ],
        ),
        (
            "diff(arctan(sqrt(tan(x)+exp(x)+sqrt(x)+x)), x) - (2*sqrt(x)+2*sqrt(x)*e^x+2*sqrt(x)*sec(x)^2+1)/(4*sqrt(x)*sqrt(tan(x)+sqrt(x)+e^x+x)*(tan(x)+sqrt(x)+e^x+x+1))",
            vec![
                "x > 0".to_string(),
                "tan(x) + sqrt(x) + e^x + x > 0".to_string(),
                "cos(x) ≠ 0".to_string(),
            ],
        ),
        (
            "(e^x + sec(x)^2 + 1/(2*sqrt(x)) + 1)/(2*sqrt(tan(x)+sqrt(x)+e^x+x)) - (2*sqrt(x)+2*sqrt(x)*e^x+2*sqrt(x)*sec(x)^2+1)/(4*sqrt(x)*sqrt(tan(x)+sqrt(x)+e^x+x))",
            vec![
                "tan(x) + sqrt(x) + e^x + x > 0".to_string(),
                "x > 0".to_string(),
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
            "expected cross-family residual to close before cleanup, got: {:?}",
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

        let mut required: Vec<String> = normalize_and_dedupe_conditions(
            &mut engine.simplifier.context,
            &output.required_conditions,
        )
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();
        let mut expected_required = expected_required;
        required.sort();
        expected_required.sort();
        assert_eq!(required, expected_required, "input: {input}");
    }
}
#[test]
fn exp_trig_log_sqrt_reciprocal_sqrt_root_diff_stays_compact_and_fast() {
    for (input, expected_result, expected_required) in [
        (
            "diff(sqrt(exp(sin(x))+ln(x)+sqrt(x)+1/sqrt(x)+x), x)",
            "(2 * sqrt(x) + 2 * x * sqrt(x) + 2 * x * cos(x) * sqrt(x) * e^sin(x) + x - 1) / (4 * x * sqrt(x) * sqrt(ln(x) + sqrt(x) + e^sin(x) + 1 / sqrt(x) + x))",
            vec![
                "x > 0".to_string(),
                "ln(x) + sqrt(x) + e^sin(x) + 1 / sqrt(x) + x > 0".to_string(),
            ],
        ),
        (
            "diff(arctan(sqrt(exp(sin(x))+ln(x)+sqrt(x)+1/sqrt(x)+x)), x)",
            "(2 * sqrt(x) + 2 * x * sqrt(x) + 2 * x * cos(x) * sqrt(x) * e^sin(x) + x - 1) / (4 * x * sqrt(x) * sqrt(ln(x) + sqrt(x) + e^sin(x) + 1 / sqrt(x) + x) * (ln(x) + sqrt(x) + e^sin(x) + 1 / sqrt(x) + x + 1))",
            vec![
                "x > 0".to_string(),
                "ln(x) + sqrt(x) + e^sin(x) + 1 / sqrt(x) + x > 0".to_string(),
            ],
        ),
        (
            "diff(sqrt(exp(sin(2*x))+ln(x)+sqrt(x)+1/sqrt(x)+x), x)",
            "(2 * sqrt(x) + 2 * x * sqrt(x) + 4 * x * cos(2 * x) * sqrt(x) * e^sin(2 * x) + x - 1) / (4 * x * sqrt(x) * sqrt(ln(x) + sqrt(x) + e^sin(2 * x) + 1 / sqrt(x) + x))",
            vec![
                "x > 0".to_string(),
                "ln(x) + sqrt(x) + e^sin(2 * x) + 1 / sqrt(x) + x > 0".to_string(),
            ],
        ),
        (
            "diff(arctan(sqrt(exp(sin(2*x))+ln(x)+sqrt(x)+1/sqrt(x)+x)), x)",
            "(2 * sqrt(x) + 2 * x * sqrt(x) + 4 * x * cos(2 * x) * sqrt(x) * e^sin(2 * x) + x - 1) / (4 * x * sqrt(x) * sqrt(ln(x) + sqrt(x) + e^sin(2 * x) + 1 / sqrt(x) + x) * (ln(x) + sqrt(x) + e^sin(2 * x) + 1 / sqrt(x) + x + 1))",
            vec![
                "x > 0".to_string(),
                "ln(x) + sqrt(x) + e^sin(2 * x) + 1 / sqrt(x) + x > 0".to_string(),
            ],
        ),
        (
            "diff(sqrt(exp(cos(x))+ln(x)+sqrt(x)+1/sqrt(x)+x), x)",
            "(2 * sqrt(x) + 2 * x * sqrt(x) + x - 2 * x * sin(x) * sqrt(x) * e^cos(x) - 1) / (4 * x * sqrt(x) * sqrt(ln(x) + sqrt(x) + e^cos(x) + 1 / sqrt(x) + x))",
            vec![
                "x > 0".to_string(),
                "ln(x) + sqrt(x) + e^cos(x) + 1 / sqrt(x) + x > 0".to_string(),
            ],
        ),
        (
            "diff(arctan(sqrt(exp(cos(x))+ln(x)+sqrt(x)+1/sqrt(x)+x)), x)",
            "(2 * sqrt(x) + 2 * x * sqrt(x) + x - 2 * x * sin(x) * sqrt(x) * e^cos(x) - 1) / (4 * x * sqrt(x) * sqrt(ln(x) + sqrt(x) + e^cos(x) + 1 / sqrt(x) + x) * (ln(x) + sqrt(x) + e^cos(x) + 1 / sqrt(x) + x + 1))",
            vec![
                "x > 0".to_string(),
                "ln(x) + sqrt(x) + e^cos(x) + 1 / sqrt(x) + x > 0".to_string(),
            ],
        ),
        (
            "diff(sqrt(-exp(sin(x))+ln(x)+sqrt(x)+1/sqrt(x)+x), x)",
            "(2 * sqrt(x) + 2 * x * sqrt(x) + x - 2 * x * cos(x) * sqrt(x) * e^sin(x) - 1) / (4 * x * sqrt(x) * sqrt(ln(x) + sqrt(x) + 1 / sqrt(x) + x - e^sin(x)))",
            vec![
                "x > 0".to_string(),
                "ln(x) + sqrt(x) + 1 / sqrt(x) + x - e^sin(x) > 0".to_string(),
            ],
        ),
        (
            "diff(arctan(sqrt(-exp(sin(x))+ln(x)+sqrt(x)+1/sqrt(x)+x)), x)",
            "(2 * sqrt(x) + 2 * x * sqrt(x) + x - 2 * x * cos(x) * sqrt(x) * e^sin(x) - 1) / (4 * x * sqrt(x) * sqrt(ln(x) + sqrt(x) + 1 / sqrt(x) + x - e^sin(x)) * (ln(x) + sqrt(x) + 1 / sqrt(x) + x + 1 - e^sin(x)))",
            vec![
                "x > 0".to_string(),
                "ln(x) + sqrt(x) + 1 / sqrt(x) + x - e^sin(x) > 0".to_string(),
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
        for condition in expected_required {
            assert!(
                required.contains(&condition),
                "missing required condition {condition:?}; got {required:?}; input: {input}"
            );
        }
    }
}
#[test]
fn exp_trig_log_sqrt_reciprocal_sqrt_root_diff_residuals_collapse_before_cleanup() {
    for (input, expected_required) in [
        (
            "diff(sqrt(exp(sin(x))+ln(x)+sqrt(x)+1/sqrt(x)+x), x) - (2*sqrt(x)+2*x*sqrt(x)+2*x*cos(x)*sqrt(x)*e^sin(x)+x-1)/(4*x*sqrt(x)*sqrt(exp(sin(x))+ln(x)+sqrt(x)+1/sqrt(x)+x))",
            vec![
                "x > 0".to_string(),
                "ln(x) + sqrt(x) + e^sin(x) + 1 / sqrt(x) + x > 0".to_string(),
            ],
        ),
        (
            "diff(arctan(sqrt(exp(sin(x))+ln(x)+sqrt(x)+1/sqrt(x)+x)), x) - (2*sqrt(x)+2*x*sqrt(x)+2*x*cos(x)*sqrt(x)*e^sin(x)+x-1)/(4*x*sqrt(x)*sqrt(exp(sin(x))+ln(x)+sqrt(x)+1/sqrt(x)+x)*(exp(sin(x))+ln(x)+sqrt(x)+1/sqrt(x)+x+1))",
            vec![
                "x > 0".to_string(),
                "ln(x) + sqrt(x) + e^sin(x) + 1 / sqrt(x) + x > 0".to_string(),
            ],
        ),
        (
            "diff(sqrt(exp(sin(2*x))+ln(x)+sqrt(x)+1/sqrt(x)+x), x) - (2*sqrt(x)+2*x*sqrt(x)+4*x*cos(2*x)*sqrt(x)*e^sin(2*x)+x-1)/(4*x*sqrt(x)*sqrt(exp(sin(2*x))+ln(x)+sqrt(x)+1/sqrt(x)+x))",
            vec![
                "x > 0".to_string(),
                "ln(x) + sqrt(x) + e^sin(2 * x) + 1 / sqrt(x) + x > 0".to_string(),
            ],
        ),
        (
            "diff(arctan(sqrt(exp(sin(2*x))+ln(x)+sqrt(x)+1/sqrt(x)+x)), x) - (2*sqrt(x)+2*x*sqrt(x)+4*x*cos(2*x)*sqrt(x)*e^sin(2*x)+x-1)/(4*x*sqrt(x)*sqrt(exp(sin(2*x))+ln(x)+sqrt(x)+1/sqrt(x)+x)*(exp(sin(2*x))+ln(x)+sqrt(x)+1/sqrt(x)+x+1))",
            vec![
                "x > 0".to_string(),
                "ln(x) + sqrt(x) + e^sin(2 * x) + 1 / sqrt(x) + x > 0".to_string(),
            ],
        ),
        (
            "diff(sqrt(exp(cos(x))+ln(x)+sqrt(x)+1/sqrt(x)+x), x) - (2*sqrt(x)+2*x*sqrt(x)+x-2*x*sin(x)*sqrt(x)*e^cos(x)-1)/(4*x*sqrt(x)*sqrt(exp(cos(x))+ln(x)+sqrt(x)+1/sqrt(x)+x))",
            vec![
                "x > 0".to_string(),
                "ln(x) + sqrt(x) + e^cos(x) + 1 / sqrt(x) + x > 0".to_string(),
            ],
        ),
        (
            "diff(arctan(sqrt(exp(cos(x))+ln(x)+sqrt(x)+1/sqrt(x)+x)), x) - (2*sqrt(x)+2*x*sqrt(x)+x-2*x*sin(x)*sqrt(x)*e^cos(x)-1)/(4*x*sqrt(x)*sqrt(exp(cos(x))+ln(x)+sqrt(x)+1/sqrt(x)+x)*(exp(cos(x))+ln(x)+sqrt(x)+1/sqrt(x)+x+1))",
            vec![
                "x > 0".to_string(),
                "ln(x) + sqrt(x) + e^cos(x) + 1 / sqrt(x) + x > 0".to_string(),
            ],
        ),
        (
            "diff(sqrt(-exp(sin(x))+ln(x)+sqrt(x)+1/sqrt(x)+x), x) - (2*sqrt(x)+2*x*sqrt(x)+x-2*x*cos(x)*sqrt(x)*e^sin(x)-1)/(4*x*sqrt(x)*sqrt(-exp(sin(x))+ln(x)+sqrt(x)+1/sqrt(x)+x))",
            vec![
                "x > 0".to_string(),
                "ln(x) + sqrt(x) + 1 / sqrt(x) + x - e^sin(x) > 0".to_string(),
            ],
        ),
        (
            "diff(arctan(sqrt(-exp(sin(x))+ln(x)+sqrt(x)+1/sqrt(x)+x)), x) - (2*sqrt(x)+2*x*sqrt(x)+x-2*x*cos(x)*sqrt(x)*e^sin(x)-1)/(4*x*sqrt(x)*sqrt(-exp(sin(x))+ln(x)+sqrt(x)+1/sqrt(x)+x)*(-exp(sin(x))+ln(x)+sqrt(x)+1/sqrt(x)+x+1))",
            vec![
                "x > 0".to_string(),
                "ln(x) + sqrt(x) + 1 / sqrt(x) + x - e^sin(x) > 0".to_string(),
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
        for condition in expected_required {
            assert!(
                required.contains(&condition),
                "missing required condition {condition:?}; got {required:?}; input: {input}"
            );
        }
    }
}
#[test]
fn sqrt_exp_trig_log_root_diff_common_denominator_residual_collapses_before_cleanup() {
    let input = "diff(sqrt(exp(sin(x))+ln(x)+sqrt(x)), x) - (2*sqrt(x)+2*x*sqrt(x)*cos(x)*e^sin(x)+x)/(4*x*sqrt(x)*sqrt(ln(x)+sqrt(x)+e^sin(x)))";
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
fn sqrt_exp_trig_log_polynomial_root_diff_power_term_residual_collapses_before_cleanup() {
    let input = "diff(sqrt(exp(sin(x))+ln(x)+sqrt(x)+x^2), x) - (2*sqrt(x)+2*x*sqrt(x)*cos(x)*e^sin(x)+x+4*x^2*sqrt(x))/(4*x*sqrt(x)*sqrt(ln(x)+sqrt(x)+e^sin(x)+x^2))";
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
            "ln(x) + sqrt(x) + e^sin(x) + x^2 > 0".to_string()
        ]
    );
}
#[test]
fn sqrt_exp_trig_log_signed_sqrt_term_residual_collapses_before_cleanup() {
    let input = "diff(sqrt(exp(sin(x))+ln(x)-sqrt(x)), x) - (2*sqrt(x)+2*x*sqrt(x)*cos(x)*e^sin(x)-x)/(4*x*sqrt(x)*sqrt(ln(x)-sqrt(x)+e^sin(x)))";
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
            "ln(x) - sqrt(x) + e^sin(x) > 0".to_string()
        ]
    );
}
#[test]
fn additive_trig_root_diff_reciprocal_sqrt_scaled_residual_collapses_before_cleanup() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let input = "diff(sqrt(sin(2*x)+cos(x)+sqrt(x)), x) - (4*cos(2*x)+x^(-1/2)-2*sin(x))/(4*sqrt(sin(2*x)+cos(x)+sqrt(x)))";
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
        "expected the reciprocal-sqrt residual to close before cleanup, got: {:?}",
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
        vec![
            "sin(2 * x) + cos(x) + sqrt(x) > 0".to_string(),
            "x > 0".to_string()
        ]
    );

    let input = "diff(sqrt(sin(2*x)+cos(x)+sqrt(x)), x) - \
        (4*sqrt(x)*cos(2*x)+1-2*sqrt(x)*sin(x))/(4*sqrt(x)*sqrt(sin(2*x)+cos(x)+sqrt(x)))";
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let parsed =
        parse(input, &mut engine.simplifier.context).expect("parse sqrt-denominator residual");

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
        .expect("eval sqrt-denominator residual");

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
        "expected the common-denominator sqrt residual to close before cleanup, got: {:?}",
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
        vec![
            "sin(2 * x) + cos(x) + sqrt(x) > 0".to_string(),
            "x > 0".to_string()
        ]
    );

    let input = "diff(sqrt(sin(2*x)+cos(x)-sqrt(x)), x) - \
        (4*sqrt(x)*cos(2*x)-1-2*sqrt(x)*sin(x))/(4*sqrt(x)*sqrt(sin(2*x)+cos(x)-sqrt(x)))";
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let parsed = parse(input, &mut engine.simplifier.context)
        .expect("parse negative sqrt-denominator residual");

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
        .expect("eval negative sqrt-denominator residual");

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
        "expected the negative common-denominator sqrt residual to close before cleanup, got: {:?}",
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
        vec![
            "sin(2 * x) + cos(x) - sqrt(x) > 0".to_string(),
            "x > 0".to_string()
        ]
    );
}
#[test]
fn tan_sqrt_root_diff_uses_inline_post_calculus_presentation() {
    for (input, expected, expected_required) in [
        (
            "diff(sqrt(tan(x)+sqrt(x)+x), x)",
            "(sec(x)^2 + 1 / (2 * sqrt(x)) + 1) / (2 * sqrt(tan(x) + sqrt(x) + x))",
            vec![
                "cos(x) ≠ 0".to_string(),
                "tan(x) + sqrt(x) + x > 0".to_string(),
                "x > 0".to_string(),
            ],
        ),
        (
            "diff(sqrt(tan(x)+sqrt(x)+2*x+1), x)",
            "(sec(x)^2 + 1 / (2 * sqrt(x)) + 2) / (2 * sqrt(tan(x) + sqrt(x) + 2 * x + 1))",
            vec![
                "cos(x) ≠ 0".to_string(),
                "tan(x) + sqrt(x) + 2 * x + 1 > 0".to_string(),
                "x > 0".to_string(),
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
            !result.contains("x^(-1/2)") && !result.contains("x^(1/2)"),
            "post-calculus presentation should use sqrt notation: {result}"
        );
        assert_eq!(
            output.steps.len(),
            1,
            "expected direct tan/sqrt presentation, got: {:?}",
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
fn tan_reciprocal_sqrt_root_diff_uses_sqrt_denominator_presentation() {
    for (input, expected, expected_required) in [
        (
            "diff(sqrt(tan(x)+1/sqrt(x)+x), x)",
            "(2 * x * sqrt(x) + 2 * x * sqrt(x) * sec(x)^2 - 1) / (4 * x * sqrt(x) * sqrt(tan(x) + 1 / sqrt(x) + x))",
            vec![
                "x > 0".to_string(),
                "tan(x) + 1 / sqrt(x) + x > 0".to_string(),
                "cos(x) ≠ 0".to_string(),
            ],
        ),
        (
            "diff(sqrt(tan(x)-1/sqrt(x)+x), x)",
            "(2 * x * sqrt(x) + 2 * x * sqrt(x) * sec(x)^2 + 1) / (4 * x * sqrt(x) * sqrt(tan(x) - 1 / sqrt(x) + x))",
            vec![
                "x > 0".to_string(),
                "tan(x) - 1 / sqrt(x) + x > 0".to_string(),
                "cos(x) ≠ 0".to_string(),
            ],
        ),
        (
            "diff(sqrt(tan(x)+2/sqrt(x)+x), x)",
            "(2 * x * sqrt(x) + 2 * x * sqrt(x) * sec(x)^2 - 2) / (4 * x * sqrt(x) * sqrt(tan(x) + 2 / sqrt(x) + x))",
            vec![
                "x > 0".to_string(),
                "tan(x) + 2 / sqrt(x) + x > 0".to_string(),
                "cos(x) ≠ 0".to_string(),
            ],
        ),
        (
            "diff(sqrt(tan(x)-3/sqrt(x)+x), x)",
            "(2 * x * sqrt(x) + 2 * x * sqrt(x) * sec(x)^2 + 3) / (4 * x * sqrt(x) * sqrt(tan(x) - 3 / sqrt(x) + x))",
            vec![
                "x > 0".to_string(),
                "tan(x) - 3 / sqrt(x) + x > 0".to_string(),
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
            !result.contains("x^(-3/2)") && !result.contains("cos(x)^2"),
            "post-calculus presentation should avoid half-power/cos-square denominator noise: {result}"
        );
        assert_eq!(
            output.steps.len(),
            1,
            "expected direct tan/reciprocal-sqrt presentation, got: {:?}",
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
fn tan_reciprocal_sqrt_root_diff_residual_collapses_before_cleanup() {
    for (input, expected_required) in [
        (
            "diff(sqrt(tan(x)+1/sqrt(x)+x), x) - (2*x*sqrt(x)*sec(x)^2+2*x*sqrt(x)-1)/(4*x*sqrt(x)*sqrt(tan(x)+1/sqrt(x)+x))",
            vec![
                "x > 0".to_string(),
                "tan(x) + 1 / sqrt(x) + x > 0".to_string(),
                "cos(x) ≠ 0".to_string(),
            ],
        ),
        (
            "diff(sqrt(tan(x)-1/sqrt(x)+x), x) - (2*x*sqrt(x)*sec(x)^2+2*x*sqrt(x)+1)/(4*x*sqrt(x)*sqrt(tan(x)-1/sqrt(x)+x))",
            vec![
                "x > 0".to_string(),
                "tan(x) - 1 / sqrt(x) + x > 0".to_string(),
                "cos(x) ≠ 0".to_string(),
            ],
        ),
        (
            "diff(sqrt(tan(x)+2/sqrt(x)+x), x) - (2*x*sqrt(x)*sec(x)^2+2*x*sqrt(x)-2)/(4*x*sqrt(x)*sqrt(tan(x)+2/sqrt(x)+x))",
            vec![
                "x > 0".to_string(),
                "tan(x) + 2 / sqrt(x) + x > 0".to_string(),
                "cos(x) ≠ 0".to_string(),
            ],
        ),
        (
            "diff(sqrt(tan(x)-3/sqrt(x)+x), x) - (2*x*sqrt(x)*sec(x)^2+2*x*sqrt(x)+3)/(4*x*sqrt(x)*sqrt(tan(x)-3/sqrt(x)+x))",
            vec![
                "x > 0".to_string(),
                "tan(x) - 3 / sqrt(x) + x > 0".to_string(),
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
            "expected tan/reciprocal-sqrt residual to close before cleanup, got: {:?}",
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
fn tan_mixed_sqrt_reciprocal_sqrt_root_diff_uses_common_root_denominator_presentation() {
    for (input, expected, expected_required) in [
        (
            "diff(sqrt(tan(x)+sqrt(x)+1/sqrt(x)+x), x)",
            "(2 * x * sqrt(x) + 2 * x * sqrt(x) * sec(x)^2 + x - 1) / (4 * x * sqrt(x) * sqrt(tan(x) + sqrt(x) + 1 / sqrt(x) + x))",
            vec![
                "x > 0".to_string(),
                "tan(x) + sqrt(x) + 1 / sqrt(x) + x > 0".to_string(),
                "cos(x) ≠ 0".to_string(),
            ],
        ),
        (
            "diff(sqrt(tan(x)+2*sqrt(x)-3/sqrt(x)+x), x)",
            "(2 * x * sqrt(x) + 2 * x * sqrt(x) * sec(x)^2 + 2 * x + 3) / (4 * x * sqrt(x) * sqrt(tan(x) + 2 * sqrt(x) - 3 / sqrt(x) + x))",
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
            "expected direct mixed tan/root presentation, got: {:?}",
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
fn tan_mixed_sqrt_reciprocal_sqrt_root_diff_residual_collapses_before_cleanup() {
    for (input, expected_required) in [
        (
            "diff(sqrt(tan(x)+sqrt(x)+1/sqrt(x)+x), x) - (2*x*sqrt(x)*sec(x)^2+2*x*sqrt(x)+x-1)/(4*x*sqrt(x)*sqrt(tan(x)+sqrt(x)+1/sqrt(x)+x))",
            vec![
                "x > 0".to_string(),
                "tan(x) + sqrt(x) + 1 / sqrt(x) + x > 0".to_string(),
                "cos(x) ≠ 0".to_string(),
            ],
        ),
        (
            "diff(sqrt(tan(x)+2*sqrt(x)-3/sqrt(x)+x), x) - (2*x*sqrt(x)*sec(x)^2+2*x*sqrt(x)+2*x+3)/(4*x*sqrt(x)*sqrt(tan(x)+2*sqrt(x)-3/sqrt(x)+x))",
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
            "expected mixed tan/root residual to close before cleanup, got: {:?}",
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
fn tan_sqrt_root_diff_sec_sqrt_residual_collapses_before_cleanup() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let input = "diff(sqrt(tan(x)+sqrt(x)+x), x) - (2*x^(1/2)*sec(x)^2+1+2*x^(1/2))/(4*x^(1/2)*sqrt(tan(x)+sqrt(x)+x))";
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
        "expected the tan/sqrt residual to close before cleanup, got: {:?}",
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
        vec![
            "cos(x) ≠ 0".to_string(),
            "tan(x) + sqrt(x) + x > 0".to_string(),
            "x > 0".to_string()
        ]
    );
}
#[test]
fn tan_sqrt_affine_root_diff_sec_sqrt_residual_collapses_before_cleanup() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let input = "diff(sqrt(tan(x)+sqrt(x)+2*x+1), x) - (2*x^(1/2)*sec(x)^2+1+4*x^(1/2))/(4*x^(1/2)*sqrt(tan(x)+sqrt(x)+2*x+1))";
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
        "expected the affine tan/sqrt residual to close before cleanup, got: {:?}",
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
        vec![
            "cos(x) ≠ 0".to_string(),
            "tan(x) + sqrt(x) + 2 * x + 1 > 0".to_string(),
            "x > 0".to_string()
        ]
    );
}
