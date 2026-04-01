use cas_ast::ordering::compare_expr;
use cas_formatter::DisplayExpr;
use cas_parser::parse;
use cas_session::SessionState;
use cas_solver::api::ImplicitCondition;
use cas_solver::runtime::{Engine, EvalAction, EvalRequest, EvalResult};

#[test]
fn cos_diff_over_sin_diff_contracts_to_tan_avg_with_nonzero_guards() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "(cos(x) - cos(3*x)) / (sin(3*x) - sin(x))";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    let result_id = match output.result {
        EvalResult::Expr(expr) => expr,
        other => panic!("expected expression result, got {other:?}"),
    };
    let result = DisplayExpr {
        context: &engine.simplifier.context,
        id: result_id,
    }
    .to_string();
    assert_eq!(result, "tan(2 * x)");

    let expected_shared = parse("sin(x)", &mut engine.simplifier.context).expect("shared");
    let expected_result_den = parse("cos(2*x)", &mut engine.simplifier.context).expect("cos");
    assert!(
        output.required_conditions.iter().any(|cond| {
            matches!(
                cond,
                ImplicitCondition::NonZero(expr)
                    if compare_expr(&engine.simplifier.context, *expr, expected_shared)
                        == std::cmp::Ordering::Equal
            )
        }),
        "expected introduced nonzero guard sin(x) ≠ 0, got {:?}",
        output.required_conditions
    );
    assert!(
        output.required_conditions.iter().any(|cond| {
            matches!(
                cond,
                ImplicitCondition::NonZero(expr)
                    if compare_expr(&engine.simplifier.context, *expr, expected_result_den)
                        == std::cmp::Ordering::Equal
            )
        }),
        "expected tan denominator guard cos(2*x) ≠ 0, got {:?}",
        output.required_conditions
    );
}
