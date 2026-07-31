use super::*;

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
    let cases = [
        (
            "diff(asinh((x^2+x+1)/sqrt(7)), x)",
            "(2 * x + 1) / sqrt((x^2 + x + 1)^2 + 7)",
            "(2*x+1)/(sqrt(7)*sqrt(1+((x^2+x+1)/sqrt(7))^2))",
        ),
        (
            "diff(asinh((1-2*x)^2/sqrt(5)), x)",
            "4 * (2 * x - 1) / sqrt((1 - 2 * x)^4 + 5)",
            "(8*x-4)/(sqrt(5)*sqrt(1+((1-2*x)^2/sqrt(5))^2))",
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
            !result.contains("^(-1/2)") && !result.contains("1/7") && !result.contains("1/5"),
            "input: {input}, expected compact positive gap in a sqrt denominator, got: {result}"
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

        assert!(
            required.is_empty(),
            "asinh is defined on all real inputs and the positive denominator gap should be proved: {required:?}"
        );
    }
}
#[test]
fn ln_positive_sqrt_polynomial_gap_diff_uses_asinh_presentation() {
    for (input, expected_display, expected_expr) in [
        (
            "diff(ln(sqrt(x^2+1)+x), x)",
            "1 / sqrt(x^2 + 1)",
            "1/sqrt(x^2+1)",
        ),
        (
            "diff(ln(sqrt((2*x+1)^2+4)+(2*x+1)), x)",
            "2 / sqrt((2 * x + 1)^2 + 4)",
            "2/sqrt((2*x+1)^2+4)",
        ),
        (
            "diff(ln(sqrt(x^2+1)-x), x)",
            "-1 / sqrt(x^2 + 1)",
            "-1/sqrt(x^2+1)",
        ),
        (
            "diff(ln(sqrt((2*x+1)^2+4)-(2*x+1)), x)",
            "-2 / sqrt((2 * x + 1)^2 + 4)",
            "-2/sqrt((2*x+1)^2+4)",
        ),
        (
            "diff(ln(sqrt(x^4+1)+x^2), x)",
            "2 * x / sqrt(x^4 + 1)",
            "2*x/sqrt(x^4+1)",
        ),
        (
            "diff(ln(sqrt(x^4+1)-x^2), x)",
            "-2 * x / sqrt(x^4 + 1)",
            "-2*x/sqrt(x^4+1)",
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

        assert_eq!(result, expected_display, "input: {input}");
        assert!(
            !result.contains(" - x") && !result.contains("^(-1/2)"),
            "asinh-form derivative should hide the conjugate/fractional-power residual: {result}"
        );
        let expected =
            parse(expected_expr, &mut engine.simplifier.context).expect("parse expected");
        assert!(
            engine.simplifier.are_equivalent(result_expr, expected),
            "input: {input}, expected compact logarithmic asinh-form derivative, got: {result}"
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
            "input: {input}, positive polynomial gap should not invent required conditions: {required:?}"
        );
        assert!(
            output
                .steps
                .iter()
                .any(|step| step.rule_name == "Symbolic Differentiation"),
            "input: {input}, expected the derivative to keep the ordinary symbolic differentiation trace"
        );
    }
}
#[test]
fn ln_negative_sqrt_polynomial_gap_diff_uses_acosh_branch_presentation() {
    let (input, expected_display, expected_expr, expected_required) = (
        "diff(ln(sqrt(x^2-1)+x), x)",
        "1 / sqrt(x^2 - 1)",
        "1/sqrt(x^2-1)",
        "x > 1",
    );

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

    assert_eq!(result, expected_display, "input: {input}");
    assert!(
        !result.contains("^(-1/2)") && !result.contains(" - x)"),
        "acosh-form derivative should hide the conjugate/fractional-power residual: {result}"
    );
    let expected = parse(expected_expr, &mut engine.simplifier.context).expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "input: {input}, expected compact logarithmic acosh-form derivative, got: {result}"
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
        vec![expected_required.to_string()],
        "input: {input}, expected only the compact acosh branch guard: {required:?}"
    );
    assert!(
        output
            .steps
            .iter()
            .any(|step| step.rule_name == "Symbolic Differentiation"),
        "input: {input}, expected the derivative to keep the ordinary symbolic differentiation trace"
    );
}
#[test]
fn asinh_surd_quotient_diff_uses_compact_sqrt_scale_presentation() {
    let cases = [
        (
            "diff(asinh((1-x-x^2)/sqrt(5))/sqrt(5), x)",
            "(-2 * x - 1) / (sqrt(5) * sqrt((1 - x - x^2)^2 + 5))",
            "(-2*x-1)/(5*sqrt(1+((1-x-x^2)/sqrt(5))^2))",
        ),
        (
            "diff((1/sqrt(5))*asinh((1-x-x^2)/sqrt(5)), x)",
            "(-2 * x - 1) / (sqrt(5) * sqrt((1 - x - x^2)^2 + 5))",
            "(-2*x-1)/(5*sqrt(1+((1-x-x^2)/sqrt(5))^2))",
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
            !result.contains("^(-1/2)") && !result.contains("1/5 *"),
            "input: {input}, compact asinh quotient derivative should not expose presimplified reciprocal-root scale: {result}"
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
            "asinh quotient over a positive constant sqrt should not add required conditions"
        );
    }
}
#[test]
fn asinh_sqrt_constant_over_polynomial_diff_places_surd_scale_in_numerator() {
    let cases = [
        (
            "diff(asinh(sqrt(3/x)), x)",
            "-sqrt(3) / (2 * x * sqrt(x + 3))",
        ),
        (
            "diff(asinh(sqrt(12/x)), x)",
            "-sqrt(3) / (x * sqrt(x + 12))",
        ),
    ];

    for (input, expected_result) in cases {
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
            !result.contains("/ (sqrt(3)")
                && !result.contains("/ (sqrt(12)")
                && !result.contains("sqrt(12)"),
            "asinh reciprocal-root presentation should not leave c/sqrt(c) scale in the denominator: {result}"
        );

        let expected =
            parse(expected_result, &mut engine.simplifier.context).expect("parse expected");
        assert!(
            engine.simplifier.are_equivalent(result_expr, expected),
            "post-calculus presentation must stay equivalent to the compact expected form, got: {result}"
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
}
#[test]
fn atanh_sqrt_constant_over_polynomial_diff_combines_sqrt_scale_presentation() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::Off;
    let input = "diff(atanh(sqrt(3/x)), x)";
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

    assert_eq!(result, "3 / (2 * sqrt(3 * x) * (3 - x))");
    assert!(
        !result.contains("sqrt(3) * sqrt(x)"),
        "atanh reciprocal-root presentation should combine constant and polynomial roots: {result}"
    );

    let expected =
        parse("3/(2*sqrt(3*x)*(3-x))", &mut engine.simplifier.context).expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "post-calculus presentation must stay equivalent to the integrand-shaped derivative, got: {result}"
    );

    let required: Vec<String> = normalize_and_dedupe_conditions(
        &mut engine.simplifier.context,
        &output.required_conditions,
    )
    .iter()
    .map(|cond| cond.display(&engine.simplifier.context))
    .collect();
    assert_eq!(required, vec!["x > 3".to_string()]);

    let residual_input = "diff(atanh(sqrt(3/x)), x) - 3/(2*sqrt(3*x)*(3-x))";
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
    assert_eq!(residual, "0");
}
#[test]
fn inverse_hyperbolic_sqrt_polynomial_constant_divisor_diff_uses_shared_compact_root_presentation()
{
    let cases = [
        (
            "diff(acosh(sqrt(x+1))/2, x)",
            "1 / (4 * sqrt(x) * sqrt(x + 1))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(-acosh(sqrt(x+1))/2, x)",
            "-1 / (4 * sqrt(x) * sqrt(x + 1))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(acosh(sqrt(x+1))/sqrt(5), x)",
            "1 / (2 * sqrt(5) * sqrt(x + 1) * sqrt(x))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(asinh(sqrt(x+1))/sqrt(5), x)",
            "1 / (2 * sqrt(5) * sqrt(x + 1) * sqrt(x + 2))",
            vec!["x > -1".to_string()],
        ),
        (
            "diff(atanh(sqrt(x+1)/3)/sqrt(5), x)",
            "3 / (2 * sqrt(5) * sqrt(x + 1) * (8 - x))",
            vec!["x < 8".to_string(), "x > -1".to_string()],
        ),
    ];

    for (input, expected_result, expected_required) in cases {
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
            !result.contains("^(-1/2)") && !result.contains("^(1/2)"),
            "constant-scaled inverse hyperbolic sqrt-polynomial derivative should keep compact root notation: {result}"
        );
        assert!(
            output.domain_warnings.is_empty(),
            "input: {input}, unexpected domain warnings: {:?}",
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
            required, expected_required,
            "input: {input}, unexpected required conditions: {required:?}"
        );
    }
}
#[test]
fn atanh_scaled_sqrt_diff_compacts_open_interval_domain_conditions() {
    for (input, expected_render, expected_required, expected_residual) in [
        (
            "diff(atanh(sqrt(x)/2), x)",
            "1 / (sqrt(x) * (4 - x))",
            vec!["x > 0".to_string(), "x < 4".to_string()],
            "diff(atanh(sqrt(x)/2), x) - 1/((4-x)*sqrt(x))",
        ),
        (
            "diff(atanh(-sqrt(x)/2), x)",
            "-1 / (sqrt(x) * (4 - x))",
            vec!["x > 0".to_string(), "x < 4".to_string()],
            "diff(atanh(-sqrt(x)/2), x) + 1/((4-x)*sqrt(x))",
        ),
        (
            "diff(atanh(sqrt(x)/3), x)",
            "3 / (2 * sqrt(x) * (9 - x))",
            vec!["x > 0".to_string(), "x < 9".to_string()],
            "diff(atanh(sqrt(x)/3), x) - 3/(2*(9-x)*sqrt(x))",
        ),
        (
            "diff(2*atanh(sqrt(x)/3), x)",
            "3 / ((9 - x) * sqrt(x))",
            vec!["x > 0".to_string(), "x < 9".to_string()],
            "diff(2*atanh(sqrt(x)/3), x) - 3/((9-x)*sqrt(x))",
        ),
        (
            "diff(2*atanh(sqrt(4*x)/3), x)",
            "6 / ((9 - 4 * x) * sqrt(x))",
            vec!["x > 0".to_string(), "x < 9/4".to_string()],
            "diff(2*atanh(sqrt(4*x)/3), x) - 6/((9-4*x)*sqrt(x))",
        ),
        (
            "diff(-2*atanh(-sqrt(x)/2), x)",
            "2 / ((4 - x) * sqrt(x))",
            vec!["x > 0".to_string(), "x < 4".to_string()],
            "diff(-2*atanh(-sqrt(x)/2), x) - 2/((4-x)*sqrt(x))",
        ),
        (
            "diff(atanh(sqrt(2*x+3)/3), x)",
            "3 / (2 * sqrt(2 * x + 3) * (3 - x))",
            vec!["x < 3".to_string(), "x > -3/2".to_string()],
            "diff(atanh(sqrt(2*x+3)/3), x) - 3/(2*(3-x)*sqrt(2*x+3))",
        ),
    ] {
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

        assert_eq!(result, expected_render, "input: {input}");

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
            "scaled atanh sqrt derivative should render the real open interval compactly"
        );

        let parsed_residual =
            parse(expected_residual, &mut engine.simplifier.context).expect("parse residual");
        let residual_output = engine
            .eval(
                &mut state,
                EvalRequest {
                    raw_input: expected_residual.to_string(),
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
fn atanh_surd_quotient_diff_uses_compact_sqrt_scale_presentation() {
    let cases = [
        (
            "diff(atanh((x^2+x+1)/sqrt(7))/sqrt(7), x)",
            "(2 * x + 1) / (7 - (x^2 + x + 1)^2)",
            "(2*x+1)/(7*(1-((x^2+x+1)/sqrt(7))^2))",
        ),
        (
            "diff((1/sqrt(7))*atanh((x^2+x+1)/sqrt(7)), x)",
            "(2 * x + 1) / (7 - (x^2 + x + 1)^2)",
            "(2*x+1)/(7*(1-((x^2+x+1)/sqrt(7))^2))",
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
            !result.contains("x^4") && !result.contains("1/7 *"),
            "input: {input}, compact atanh quotient derivative should preserve the squared denominator gap: {result}"
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
            vec!["7 - (x^2 + x + 1)^2 > 0".to_string()],
            "input: {input}, compact atanh quotient derivative must preserve the open-interval domain"
        );
    }
}
#[test]
fn atanh_self_normalized_surd_quotient_diff_uses_compact_intrinsic_domain_presentation() {
    let cases = [
        (
            "diff(atanh(x/sqrt(x^2+3)), x)",
            "1 / sqrt(x^2 + 3)",
            "1/sqrt(x^2+3)",
        ),
        (
            "diff(atanh((2*x+1)/sqrt((2*x+1)^2+3)), x)",
            "2 / sqrt((2 * x + 1)^2 + 3)",
            "2/sqrt((2*x+1)^2+3)",
        ),
        (
            "diff(atanh(-x/sqrt(x^2+3)), x)",
            "-1 / sqrt(x^2 + 3)",
            "-1/sqrt(x^2+3)",
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
            !result.contains("^(-1/2)") && !result.contains("1 -"),
            "input: {input}, compact atanh self-normalized derivative should not leak chain-rule scaffolding: {result}"
        );

        let expected =
            parse(expected_chain_rule, &mut engine.simplifier.context).expect("parse expected");
        for sample in [-0.5, 0.0, 0.5] {
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
        assert!(
            required.is_empty(),
            "input: {input}, n/sqrt(n^2+c) with c>0 has intrinsic atanh open-interval domain, got {required:?}"
        );
    }
}
#[test]
fn acosh_affine_scaled_diff_uses_compact_root_product_presentation() {
    let cases = [
        (
            "diff(acosh(x+1)/2, x)",
            "1 / (2 * sqrt(x) * sqrt(x + 2))",
            "1/(2*sqrt(x)*sqrt(x+2))",
        ),
        (
            "diff((1/2)*acosh(x+1), x)",
            "1 / (2 * sqrt(x) * sqrt(x + 2))",
            "1/(2*sqrt(x)*sqrt(x+2))",
        ),
    ];

    for (input, expected_result, residual_target) in cases {
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
            !result.contains("^(-1/2)"),
            "input: {input}, scaled acosh affine derivative should not expose reciprocal powers: {result}"
        );

        let residual_input = format!("{input} - {residual_target}");
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
        assert_eq!(residual, "0", "residual did not collapse for {input}");

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
            "input: {input}, scaled acosh affine derivative must preserve the real-domain guard"
        );
    }
}
#[test]
fn acosh_fractional_affine_diff_absorbs_common_root_denominator() {
    let cases = [
        (
            "diff(acosh((x+1)/2), x)",
            "1 / (sqrt(x - 1) * sqrt(x + 3))",
            vec!["x > 1".to_string()],
        ),
        (
            "diff(acosh(-(x+1)/2), x)",
            "-1 / (sqrt(-x - 3) * sqrt(1 - x))",
            vec!["x < -3".to_string()],
        ),
        (
            "diff(acosh((2*x+1)/3), x)",
            "1 / (sqrt(x - 1) * sqrt(x + 2))",
            vec!["x > 1".to_string()],
        ),
        (
            "diff(acosh(-(2*x+1)/3), x)",
            "-1 / (sqrt(-x - 2) * sqrt(1 - x))",
            vec!["x < -2".to_string()],
        ),
        (
            "diff(acosh(2/5*x+1), x)",
            "1 / (sqrt(x) * sqrt(x + 5))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(acosh(2/3*x+1), x)",
            "1 / (sqrt(x) * sqrt(x + 3))",
            vec!["x > 0".to_string()],
        ),
    ];

    for (input, expected_display, expected_required) in cases {
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
        assert!(
            !result.contains("sqrt(2 *"),
            "input: {input}, acosh fractional-affine presentation should absorb common positive root content: {result}"
        );

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

        assert_eq!(
            required, expected_required,
            "input: {input}, unexpected required_conditions: {required:?}"
        );
    }
}
#[test]
fn shifted_linear_asinh_diff_preserves_compact_radicand() {
    let cases = [
        ("diff(asinh(2*x+1), x)", "2 / sqrt((2 * x + 1)^2 + 1)"),
        ("diff(asinh(3-2*x), x)", "-2 / sqrt((3 - 2 * x)^2 + 1)"),
        ("diff(asinh(-(x+1)/2), x)", "-1 / sqrt((x + 1)^2 + 4)"),
        ("diff(asinh(x^2), x)", "2 * x / sqrt(x^4 + 1)"),
        ("diff(asinh(x^3), x)", "3 * x^2 / sqrt(x^6 + 1)"),
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
            "diff(-asinh(sqrt(x)), x)",
            "-1 / (2 * sqrt(x + 1) * sqrt(x))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(2*asinh(sqrt(x)), x)",
            "1 / (sqrt(x + 1) * sqrt(x))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(asinh(sqrt(2*x)), x)",
            "1 / (sqrt(2 * x) * sqrt(2 * x + 1))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(asinh(sqrt(x/2)), x)",
            "1 / (2 * sqrt(x) * sqrt(x + 2))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(asinh(sqrt(4*x)/3), x)",
            "1 / (2 * sqrt(x) * sqrt(x + 9/4))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(2*asinh(sqrt(4*x)/3), x)",
            "1 / (sqrt(x + 9/4) * sqrt(x))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(asinh(sqrt(x+1)), x)",
            "1 / (2 * sqrt(x + 1) * sqrt(x + 2))",
            vec!["x > -1".to_string()],
        ),
        (
            "diff(asinh(sqrt(1-2*x)), x)",
            "-1 / (sqrt(1 - 2 * x) * sqrt(2 - 2 * x))",
            vec!["x < 1/2".to_string()],
        ),
        (
            "diff(asinh(sqrt(x^2+1)), x)",
            "x / (sqrt(x^2 + 1) * sqrt(x^2 + 2))",
            Vec::new(),
        ),
        (
            "diff(asinh(sqrt(x^2+2*x+2)), x)",
            "(x + 1) / (sqrt(x^2 + 2 * x + 2) * sqrt(x^2 + 2 * x + 3))",
            Vec::new(),
        ),
        (
            "diff(asinh(sqrt(x^2-2*x+2)), x)",
            "(x - 1) / (sqrt(x^2 - 2 * x + 2) * sqrt(x^2 + 3 - 2 * x))",
            Vec::new(),
        ),
        (
            "diff(asinh(sqrt(1/x)), x)",
            "-1 / (2 * x * sqrt(x + 1))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(asinh(sqrt(4/x)), x)",
            "-1 / (x * sqrt(x + 4))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(asinh(sqrt(1/(x+1))), x)",
            "-1 / (2 * (x + 1) * sqrt(x + 2))",
            vec!["x > -1".to_string()],
        ),
        (
            "diff(asinh(sqrt(4/(2*x+4))), x)",
            "-1 / ((x + 2) * sqrt(2 * x + 8))",
            vec!["x > -2".to_string()],
        ),
        (
            "diff(asinh(sqrt(2/(x+3))), x)",
            "-sqrt(2) / (2 * (x + 3) * sqrt(x + 5))",
            vec!["x > -3".to_string()],
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
        assert!(
            !result.contains("1 / (sqrt(2)"),
            "presentation should place irrational constant scale in the numerator, got: {result}"
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
        if matches!(
            input,
            "diff(asinh(sqrt(1/(x+1))), x)" | "diff(asinh(sqrt(1/x)), x)"
        ) {
            assert_no_redundant_post_calculus_presentation_round_trip(output.steps.as_slice());
        }
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
            "1 / (sqrt(x - 1) * sqrt(x + 1))",
            vec!["x > 1"],
        ),
        (
            "diff(acosh(2*x+1), x)",
            "2 / (sqrt(2 * x) * sqrt(2 * x + 2))",
            vec!["x > 0"],
        ),
        (
            "diff(acosh(1-2*x), x)",
            "-2 / (sqrt(-2 * x) * sqrt(2 - 2 * x))",
            vec!["x < 0"],
        ),
        (
            "diff(acosh(x^2+1), x)",
            "2*x/(sqrt(x^2)*sqrt(x^2+2))",
            vec!["x ≠ 0"],
        ),
        (
            "diff(acosh((x+1)^2+1), x)",
            "(2*x+2)/(sqrt(x^2+2*x+1)*sqrt(x^2+2*x+3))",
            vec!["x ≠ -1"],
        ),
        (
            "diff(acosh((2*x+1)^2+1), x)",
            "(8*x+4)/(sqrt(4*x^2+4*x+1)*sqrt(4*x^2+4*x+3))",
            vec!["x ≠ -1/2"],
        ),
        (
            "diff(acosh((1-2*x)^2+1), x)",
            "(8*x-4)/(sqrt(4*x^2+1-4*x)*sqrt(4*x^2+3-4*x))",
            vec!["x ≠ 1/2"],
        ),
        (
            "diff(acosh(x^2+x+3), x)",
            "(2*x+1)/(sqrt(x^2+x+2)*sqrt(x^2+x+4))",
            vec![],
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
        if input == "diff(acosh(x^2+x+3), x)" {
            assert_eq!(
                result, "(2 * x + 1) / (sqrt(x^2 + x + 2) * sqrt(x^2 + x + 4))",
                "strictly-positive quadratic acosh derivative should use sqrt denominator presentation"
            );
            assert!(
                !result.contains("^(-1/2)") && !result.contains("|"),
                "strictly-positive quadratic acosh derivative should avoid inverse powers and unnecessary abs: {result}"
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
fn inverse_hyperbolic_acosh_surd_polynomial_diff_uses_compact_real_domain() {
    for (input, expected_result, expected_chain_rule) in [
        (
            "diff(acosh((x^2+x)/sqrt(5)), x)",
            "(2 * x + 1) / sqrt((x^2 + x)^2 - 5)",
            "(2*x+1)/(sqrt(5)*sqrt(((x^2+x)/sqrt(5))^2-1))",
        ),
        (
            "diff(-acosh((x^2+x)/sqrt(5)), x)",
            "-(2 * x + 1) / sqrt((x^2 + x)^2 - 5)",
            "-(2*x+1)/(sqrt(5)*sqrt(((x^2+x)/sqrt(5))^2-1))",
        ),
        (
            "diff(2*acosh((x^2+x)/sqrt(5)), x)",
            "2 * (2 * x + 1) / sqrt((x^2 + x)^2 - 5)",
            "2*(2*x+1)/(sqrt(5)*sqrt(((x^2+x)/sqrt(5))^2-1))",
        ),
    ] {
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

        assert_eq!(result, expected_result, "input: {input}");
        assert!(
            !result.contains("diff(") && !result.contains("^(-1/2)"),
            "input: {input}, got: {result}"
        );

        let expected =
            parse(expected_chain_rule, &mut engine.simplifier.context).expect("parse expected");
        assert!(
            engine.simplifier.are_equivalent(result_expr, expected),
            "input: {input}, expected derivative equivalent to chain rule over acosh, got: {result}"
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
            vec!["x^2 + x - sqrt(5) > 0".to_string()],
            "input: {input}, unexpected required_conditions: {required:?}"
        );
    }
}
#[test]
fn inverse_hyperbolic_acosh_negative_oriented_surd_polynomial_diff_preserves_sign() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::Off;
    let input = "diff(acosh((1-2*x)^2/sqrt(5)), x)";
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
        result, "4 * (2 * x - 1) / sqrt((1 - 2 * x)^4 - 5)",
        "input: {input}, unexpected derivative result"
    );
    assert!(!result.contains("diff("), "input: {input}, got: {result}");

    let expected = parse(
        "(8*x-4)/(sqrt(5)*sqrt(((1-2*x)^2/sqrt(5))^2-1))",
        &mut engine.simplifier.context,
    )
    .expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "input: {input}, expected derivative equivalent to chain rule over acosh, got: {result}"
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
        vec!["(1 - 2 * x)^2 - sqrt(5) > 0".to_string()],
        "input: {input}, unexpected required_conditions: {required:?}"
    );
}
#[test]
fn acosh_sqrt_diff_uses_post_calculus_root_denominator_presentation() {
    for (input, expected_render, expected_required) in [
        (
            "diff(acosh(sqrt(x)), x)",
            "1 / (2 * sqrt(x) * sqrt(x - 1))",
            vec!["x > 1".to_string()],
        ),
        (
            "diff(-acosh(sqrt(x)), x)",
            "-1 / (2 * sqrt(x - 1) * sqrt(x))",
            vec!["x > 1".to_string()],
        ),
        (
            "diff(2*acosh(sqrt(x)), x)",
            "1 / (sqrt(x - 1) * sqrt(x))",
            vec!["x > 1".to_string()],
        ),
        (
            "diff(acosh(sqrt(x+1)), x)",
            "1 / (2 * sqrt(x + 1) * sqrt(x))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(acosh(sqrt(x/2)), x)",
            "1 / (2 * sqrt(x) * sqrt(x - 2))",
            vec!["x > 2".to_string()],
        ),
        (
            "diff(acosh(sqrt(x)/3), x)",
            "1 / (2 * sqrt(x) * sqrt(x - 9))",
            vec!["x > 9".to_string()],
        ),
        (
            "diff(2*acosh(sqrt(x)/3), x)",
            "1 / (sqrt(x - 9) * sqrt(x))",
            vec!["x > 9".to_string()],
        ),
        (
            "diff(acosh(sqrt(2*x+3)/3), x)",
            "1 / (sqrt(2 * x + 3) * sqrt(2 * x - 6))",
            vec!["x > 3".to_string()],
        ),
        (
            "diff(-acosh(sqrt(x+1)), x)",
            "-1 / (2 * sqrt(x) * sqrt(x + 1))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(acosh(sqrt(x^2+1)), x)",
            "x / (sqrt(x^2 + 1) * |x|)",
            vec!["x ≠ 0".to_string()],
        ),
        (
            "diff(acosh(sqrt((x+1)^2+1)), x)",
            "(x + 1) / (sqrt((x + 1)^2 + 1) * |x + 1|)",
            vec!["x ≠ -1".to_string()],
        ),
        (
            "diff(acosh(sqrt((2*x+1)/3)), x)",
            "1 / (sqrt(2 * x + 1) * sqrt(2 * x - 2))",
            vec!["x > 1".to_string()],
        ),
        (
            "diff(acosh(sqrt(x^2+2*x+2)), x)",
            "(x + 1) / (sqrt(x^2 + 2 * x + 2) * |x + 1|)",
            vec!["x ≠ -1".to_string()],
        ),
        (
            "diff(acosh(sqrt(x^2-2*x+2)), x)",
            "(x - 1) / (sqrt(x^2 - 2 * x + 2) * |x - 1|)",
            vec!["x ≠ 1".to_string()],
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
fn acosh_sqrt_abs_safe_diff_residual_collapses_with_required_nonzero() {
    for (input, expected_required) in [
        (
            "diff(acosh(sqrt(x^2+1)), x)-x/(abs(x)*sqrt(x^2+1))",
            vec!["x ≠ 0".to_string()],
        ),
        (
            "diff(acosh(sqrt(x^2+2*x+2)), x)-(x+1)/(abs(x+1)*sqrt(x^2+2*x+2))",
            vec!["x ≠ -1".to_string()],
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

        assert_eq!(result, "0", "input: {input}");

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
        vec!["-1 < x < 1".to_string()],
        "input: {input}, unexpected required_conditions: {required:?}"
    );
}
#[test]
fn atanh_fractional_affine_diff_preserves_open_interval_gap_presentation() {
    let cases = [
        (
            "diff(atanh((x+1)/2), x)",
            "2 / (4 - (x + 1)^2)",
            vec!["-3 < x < 1".to_string()],
        ),
        (
            "diff(atanh(-(x+1)/2), x)",
            "-2 / (4 - (x + 1)^2)",
            vec!["-3 < x < 1".to_string()],
        ),
        (
            "diff(atanh((2*x+1)/3), x)",
            "6 / (9 - (2 * x + 1)^2)",
            vec!["-2 < x < 1".to_string()],
        ),
        (
            "diff(atanh(-(2*x+1)/3), x)",
            "-6 / (9 - (2 * x + 1)^2)",
            vec!["-2 < x < 1".to_string()],
        ),
    ];

    for (input, expected_display, expected_required) in cases {
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

        assert_eq!(
            required, expected_required,
            "input: {input}, unexpected required_conditions: {required:?}"
        );
    }
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
            "1 / (2 * sqrt(x) * (1 - x))",
            vec!["x < 1".to_string(), "x > 0".to_string()],
        ),
        (
            "diff(-atanh(sqrt(x)), x)",
            "-1 / (2 * (1 - x) * sqrt(x))",
            vec!["x < 1".to_string(), "x > 0".to_string()],
        ),
        (
            "diff(2*atanh(sqrt(x)), x)",
            "1 / ((1 - x) * sqrt(x))",
            vec!["x < 1".to_string(), "x > 0".to_string()],
        ),
        (
            "diff(atanh(sqrt(2*x)), x)",
            "1 / (sqrt(2 * x) * (1 - 2 * x))",
            vec!["x < 1/2".to_string(), "x > 0".to_string()],
        ),
        (
            "diff(atanh(sqrt(x/2)), x)",
            "1 / (sqrt(2 * x) * (2 - x))",
            vec!["x < 2".to_string(), "x > 0".to_string()],
        ),
        (
            "diff(atanh(sqrt(x/3)), x)",
            "3 / (2 * sqrt(3 * x) * (3 - x))",
            vec!["x < 3".to_string(), "x > 0".to_string()],
        ),
        (
            "diff(atanh(sqrt(x+1)), x)",
            "-1 / (2 * sqrt(x + 1) * x)",
            vec!["x < 0".to_string(), "x > -1".to_string()],
        ),
        (
            "diff(atanh(sqrt(1-2*x)), x)",
            "-1 / (2 * sqrt(1 - 2 * x) * x)",
            vec!["x < 1/2".to_string(), "x > 0".to_string()],
        ),
        (
            "diff(atanh(sqrt(3-2*x)), x)",
            "-1 / (2 * sqrt(3 - 2 * x) * (x - 1))",
            vec!["x < 3/2".to_string(), "x > 1".to_string()],
        ),
        (
            "diff(atanh(sqrt(1/x)), x)",
            "1 / (2 * sqrt(x) * (1 - x))",
            vec!["x > 1".to_string()],
        ),
        (
            "diff(atanh(sqrt(4/x)), x)",
            "1 / (sqrt(x) * (4 - x))",
            vec!["x > 4".to_string()],
        ),
        (
            "diff(atanh(sqrt(1/(x+1))), x)",
            "-1 / (2 * sqrt(x + 1) * x)",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(atanh(sqrt(1/(3*x+1))), x)",
            "-1 / (2 * sqrt(3 * x + 1) * x)",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(atanh(sqrt(4/(2*x+4))), x)",
            "-1 / (sqrt(2 * x + 4) * x)",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(atanh(sqrt(9/(3*x+6))), x)",
            "3 / (2 * sqrt(3 * x + 6) * (1 - x))",
            vec!["x > 1".to_string()],
        ),
        (
            "diff(atanh(sqrt(2/(x+3))), x)",
            "-1 / (sqrt(2 * x + 6) * (x + 1))",
            vec!["x > -1".to_string()],
        ),
        (
            "diff(atanh(sqrt(x/(x+1))), x)",
            "1 / (2 * sqrt(x) * sqrt(x + 1))",
            vec!["x > 0".to_string()],
        ),
        (
            "diff(atanh(sqrt((x+1)/(x+3))), x)",
            "1 / (2 * sqrt(x + 1) * sqrt(x + 3))",
            vec!["x > -1".to_string()],
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
        if matches!(
            input,
            "diff(atanh(sqrt(1/(x+1))), x)" | "diff(atanh(sqrt(1/x)), x)"
        ) {
            assert_no_redundant_post_calculus_presentation_round_trip(output.steps.as_slice());
        }
    }
}
#[test]
fn atanh_sqrt_diff_returns_undefined_when_real_open_interval_is_empty() {
    for input in [
        "diff(atanh(sqrt(x^2+1)), x)",
        "diff(atanh(sqrt(x^2+2)), x)",
        "diff(atanh(sqrt(x^2+2*x+2)), x)",
        "diff(atanh(sqrt((x+1)^2+1)), x)",
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
            "input: {input}, empty real-domain atanh sqrt should be undefined, got: {result}"
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
            "input: {input}, empty-domain undefined result should not surface public Requires: {required:?}"
        );
        assert!(
            output
                .steps
                .iter()
                .any(|step| step.rule_name == "Symbolic Differentiation"),
            "input: {input}, empty real-domain atanh sqrt should record the undefined differentiation step"
        );
        assert!(
            output.blocked_hints.len() <= 1,
            "input: {input}, resolved undefined output should expose at most one raw domain hint"
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
        vec!["-2 < x < 0".to_string()],
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
        vec!["-1 < x < 0".to_string()],
        "input: {input}, unexpected required_conditions: {required:?}"
    );
}
#[test]
fn inverse_hyperbolic_atanh_negative_affine_diff_preserves_open_interval_condition_without_steps() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::Off;
    let input = "diff(atanh(3-2*x), x)";
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
        parse("-2/(1-(3-2*x)^2)", &mut engine.simplifier.context).expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(result_expr, expected),
        "input: {input}, expected derivative equivalent to -2/(1-(3-2*x)^2), got: {result}"
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
        vec!["1 < x < 2".to_string()],
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
        result, "2 * (x + 1) * sqrt(3) / (3 - (x + 1)^4)",
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
        result, "4 * (2 * x - 1) * sqrt(3) / (3 - (1 - 2 * x)^4)",
        "input: {input}, unexpected derivative result"
    );
    assert!(!result.contains("diff("), "input: {input}, got: {result}");

    let expected = parse(
        "(8*x-4)/(sqrt(3)*(1-((1-2*x)^2/sqrt(3))^2))",
        &mut engine.simplifier.context,
    )
    .expect("parse expected");
    for sample in [0.0, 0.25, 0.5] {
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), sample);
        let actual_value = eval_f64(&engine.simplifier.context, result_expr, &vars)
            .unwrap_or_else(|| panic!("input: {input}, could not eval result at x={sample}"));
        let expected_value =
            eval_f64(&engine.simplifier.context, expected, &vars).unwrap_or_else(|| {
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
        result, "(2 * x + 1) * sqrt(7) / (7 - (x^2 + x + 1)^2)",
        "input: {input}, unexpected derivative result"
    );
    assert!(!result.contains("diff("), "input: {input}, got: {result}");

    let expected = parse(
        "(2*x+1)/(sqrt(7)*(1-((x^2+x+1)/sqrt(7))^2))",
        &mut engine.simplifier.context,
    )
    .expect("parse expected");
    for sample in [-0.5, 0.0, 0.5] {
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), sample);
        let actual_value = eval_f64(&engine.simplifier.context, result_expr, &vars)
            .unwrap_or_else(|| panic!("input: {input}, could not eval result at x={sample}"));
        let expected_value =
            eval_f64(&engine.simplifier.context, expected, &vars).unwrap_or_else(|| {
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
        vec!["7 - (x^2 + x + 1)^2 > 0".to_string()],
        "input: {input}, unexpected required_conditions: {required:?}"
    );
    assert!(
        !required.iter().any(|cond| cond.contains("≠")),
        "positive atanh interval should dominate derivative denominator nonzero guard: {required:?}"
    );
}
#[test]
fn acosh_affine_fused_reciprocal_sqrt_residual_collapses_publicly() {
    let cases = [
        (
            "diff(acosh(2*x-1), x) - 2/sqrt((2*x-1)^2-1)",
            vec!["x > 1".to_string()],
        ),
        (
            "diff(acosh(1-2*x), x) - (-2/sqrt((2*x-1)^2-1))",
            vec!["x < 0".to_string()],
        ),
        (
            "diff(acosh(2*x-1), x) - 2*sqrt((2*x-1)^2-1)/((2*x-1)^2-1)",
            vec!["x > 1".to_string()],
        ),
        (
            "diff(acosh(1-2*x), x) + 2*sqrt((2*x-1)^2-1)/((2*x-1)^2-1)",
            vec!["x < 0".to_string()],
        ),
        (
            "diff(acosh(x^2+x+3), x) - (2*x+1)*sqrt((x^2+x+3)^2-1)/((x^2+x+3)^2-1)",
            vec![],
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
            output
                .steps
                .iter()
                .any(|step| step.rule_name.as_str() == "Acosh Reciprocal-Root Diff Residual"),
            "input: {input}, expected direct acosh residual step, got: {:?}",
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
