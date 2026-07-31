use cas_formatter::DisplayExpr;
use cas_math::eval_f64;
use cas_parser::parse;
use cas_session::SessionState;
use cas_solver::command_api::eval::evaluate_eval_command_output;
use cas_solver::runtime::{Engine, EvalAction, EvalRequest, EvalResult, StepsMode};
use cas_solver_core::domain_normalization::normalize_and_dedupe_conditions;
use std::collections::HashMap;

fn assert_no_redundant_post_calculus_presentation_round_trip(
    steps: &[cas_solver_core::step_model::Step],
) {
    assert!(
        !steps.iter().any(|step| {
            step.rule_name.as_str() == "Rationalize Product Denominator"
                || step.rule_name.as_str() == "Combine Constants"
                || step.rule_name.as_str() == "N-ary Mul Combine Powers"
                || step.rule_name.as_str() == "Present calculus result in compact form"
                || step.rule_name.as_str() == "Pull Constant From Fraction"
                || step.rule_name.as_str() == "Normalize Negation in Product"
        }),
        "post-calculus presentation should not expose a rationalize/present round trip; steps: {:?}",
        steps
            .iter()
            .map(|step| step.rule_name.as_str())
            .collect::<Vec<_>>()
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

type InverseReciprocalTrigDiffCase = (&'static str, &'static str, Vec<&'static str>);

fn inverse_reciprocal_trig_diff_exhaustive_cases() -> Vec<InverseReciprocalTrigDiffCase> {
    vec![
        (
            "diff(arcsec(x), x)",
            "1 / (|x| * sqrt(x^2 - 1))",
            vec!["x < -1 or x > 1"],
        ),
        (
            "diff(arccsc(x), x)",
            "-1 / (|x| * sqrt(x^2 - 1))",
            vec!["x < -1 or x > 1"],
        ),
        (
            "diff(arcsec(2*x), x)",
            "2 / (|2 * x| * sqrt(4 * x^2 - 1))",
            vec!["x < -1/2 or x > 1/2"],
        ),
        (
            "diff(arccsc(2*x), x)",
            "-2 / (|2 * x| * sqrt(4 * x^2 - 1))",
            vec!["x < -1/2 or x > 1/2"],
        ),
        (
            "diff(arcsec(x+1), x)",
            "1 / (|x + 1| * sqrt(x^2 + 2 * x))",
            vec!["x < -2 or x > 0"],
        ),
        (
            "diff(arccsc(x+1), x)",
            "-1 / (|x + 1| * sqrt(x^2 + 2 * x))",
            vec!["x < -2 or x > 0"],
        ),
        (
            "diff(arcsec(sqrt(x)), x)",
            "1 / (2*x*sqrt(x-1))",
            vec!["x > 1"],
        ),
        (
            "diff(arccsc(sqrt(x)), x)",
            "-1 / (2*x*sqrt(x-1))",
            vec!["x > 1"],
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
            vec!["x > 1/2"],
        ),
        (
            "diff(arccsc(sqrt(2*x)), x)",
            "-1 / (2*x*sqrt(2*x-1))",
            vec!["x > 1/2"],
        ),
        (
            "diff(arcsec(sqrt(3-2*x)), x)",
            "-1 / ((3-2*x)*sqrt(2-2*x))",
            vec!["x < 1"],
        ),
        (
            "diff(arccsc(sqrt(3-2*x)), x)",
            "1 / ((3-2*x)*sqrt(2-2*x))",
            vec!["x < 1"],
        ),
        (
            "diff(arcsec(sqrt(1-2*x)), x)",
            "-1 / ((1-2*x)*sqrt(-2*x))",
            vec!["x < 0"],
        ),
        (
            "diff(arccsc(sqrt(1-2*x)), x)",
            "1 / ((1-2*x)*sqrt(-2*x))",
            vec!["x < 0"],
        ),
        (
            "diff(arcsec(sqrt(x^2+1)), x)",
            "x/((x^2+1)*abs(x))",
            vec!["x ≠ 0"],
        ),
        (
            "diff(arccsc(sqrt(x^2+1)), x)",
            "-x/((x^2+1)*abs(x))",
            vec!["x ≠ 0"],
        ),
        (
            "diff(arcsec(sqrt(x^2+2)), x)",
            "x/((x^2+2)*sqrt(x^2+1))",
            vec![],
        ),
        (
            "diff(arccsc(sqrt(x^2+2)), x)",
            "-x/((x^2+2)*sqrt(x^2+1))",
            vec![],
        ),
        (
            "diff(arcsec(sqrt(x^2+2*x+2)), x)",
            "(x+1)/((x^2+2*x+2)*abs(x+1))",
            vec!["x ≠ -1"],
        ),
        (
            "diff(arccsc(sqrt(x^2+2*x+2)), x)",
            "-(x+1)/((x^2+2*x+2)*abs(x+1))",
            vec!["x ≠ -1"],
        ),
        (
            "diff(arcsec(sqrt(x^2-2*x+2)), x)",
            "(x-1)/((x^2-2*x+2)*abs(x-1))",
            vec!["x ≠ 1"],
        ),
        (
            "diff(arccsc(sqrt(x^2-2*x+2)), x)",
            "-(x-1)/((x^2-2*x+2)*abs(x-1))",
            vec!["x ≠ 1"],
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
            "(2*x+1)*sqrt(2)/((x^2+x+3)*sqrt((x^2+x+3)^2-2))",
            vec![],
        ),
        (
            "diff(arcsec((x^2+x+3)/sqrt(1/2)), x)",
            "(2*x+1)/((x^2+x+3)*sqrt(2*(x^2+x+3)^2-1))",
            vec![],
        ),
        (
            "diff(arcsec((x^2+x+3)/sqrt(2/3)), x)",
            "(2*x+1)*sqrt(2)/((x^2+x+3)*sqrt(3*(x^2+x+3)^2-2))",
            vec![],
        ),
        (
            "diff(arcsec(sqrt(2)*(x^2+x+3)), x)",
            "(2*x+1)/((x^2+x+3)*sqrt(2*(x^2+x+3)^2-1))",
            vec![],
        ),
        (
            "diff(arccsc(sqrt(3/2)*(x^2+x+3)), x)",
            "-(2*x+1)*sqrt(2)/((x^2+x+3)*sqrt(3*(x^2+x+3)^2-2))",
            vec![],
        ),
        (
            "diff(arccsc(sqrt(2)*(x^2+x+3)), x)",
            "-(2*x+1)/((x^2+x+3)*sqrt(2*(x^2+x+3)^2-1))",
            vec![],
        ),
        (
            "diff(arccsc((x^2+x+3)/sqrt(2)), x)",
            "-(2*x+1)*sqrt(2)/((x^2+x+3)*sqrt((x^2+x+3)^2-2))",
            vec![],
        ),
        ("diff(arccot(x), x)", "-1/(x^2 + 1)", vec![]),
    ]
}

fn assert_inverse_reciprocal_trig_diff_cases(
    cases: impl IntoIterator<Item = InverseReciprocalTrigDiffCase>,
) {
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
            "diff(arcsec(sqrt(x^2+1)), x)"
                | "diff(arccsc(sqrt(x^2+1)), x)"
                | "diff(arcsec(sqrt(x^2+2)), x)"
                | "diff(arccsc(sqrt(x^2+2)), x)"
                | "diff(arcsec(sqrt(x^2+2*x+2)), x)"
                | "diff(arccsc(sqrt(x^2+2*x+2)), x)"
                | "diff(arcsec(sqrt(x^2-2*x+2)), x)"
                | "diff(arccsc(sqrt(x^2-2*x+2)), x)"
        ) {
            assert!(
                output
                    .steps
                    .iter()
                    .any(|step| step.rule_name == "Present calculus result in compact form"),
                "sqrt-quadratic inverse reciprocal trig derivative should use compact post-calculus presentation"
            );
            assert!(
                !result.contains("^(-1/2)") && !result.contains("^(-3/2)"),
                "sqrt-quadratic inverse reciprocal trig derivative should not expose fractional-power internals: {result}"
            );
        }
        if matches!(
            input,
            "diff(arcsec(sqrt(x^2+1)), x)"
                | "diff(arccsc(sqrt(x^2+1)), x)"
                | "diff(arcsec(sqrt(x^2+2*x+2)), x)"
                | "diff(arccsc(sqrt(x^2+2*x+2)), x)"
                | "diff(arcsec(sqrt(x^2-2*x+2)), x)"
                | "diff(arccsc(sqrt(x^2-2*x+2)), x)"
        ) {
            assert!(
                result.contains('|'),
                "perfect-square gap should display an abs-safe denominator: {result}"
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
                result.contains("sqrt((x^2 + x + 3)^2 - 2)")
                    && !result.contains("^(-1/2)")
                    && !result.contains("x^4 + 2 * x^3 + 7 * x^2 + 6 * x + 7"),
                "positive surd quotient inverse reciprocal trig derivative should expose compact q^2-k under sqrt: {result}"
            );
        }
        if matches!(
            input,
            "diff(arcsec((x^2+x+3)/sqrt(1/2)), x)"
                | "diff(arcsec((x^2+x+3)/sqrt(2/3)), x)"
                | "diff(arcsec(sqrt(2)*(x^2+x+3)), x)"
                | "diff(arccsc(sqrt(3/2)*(x^2+x+3)), x)"
                | "diff(arccsc(sqrt(2)*(x^2+x+3)), x)"
        ) {
            assert!(
                !result.contains("x^8")
                    && !result.contains("1 - 1 /")
                    && !result.contains("^(-1/2)"),
                "scaled surd inverse reciprocal trig derivative should keep compact direct gap: {result}"
            );
            let expected_gap = match input {
                "diff(arcsec((x^2+x+3)/sqrt(2/3)), x)" | "diff(arccsc(sqrt(3/2)*(x^2+x+3)), x)" => {
                    "sqrt(3 * (x^2 + x + 3)^2 - 2)"
                }
                _ => "sqrt(2 * (x^2 + x + 3)^2 - 1)",
            };
            assert!(
                result.contains(expected_gap),
                "scaled surd inverse reciprocal trig derivative should expose compact value-preserving gap: {result}"
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

mod exponential_log;
mod hyperbolic;
mod inverse_hyperbolic;
mod inverse_trig;
mod misc;
mod roots_radicals;
mod trigonometric;
