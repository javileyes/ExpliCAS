use cas_ast::ordering::compare_expr;
use cas_formatter::DisplayExpr;
use cas_parser::parse;
use cas_session::SessionState;
use cas_solver::api::ImplicitCondition;
use cas_solver::runtime::{Engine, EvalAction, EvalRequest, EvalResult};

#[test]
fn consecutive_factorial_ratio_simplifies_and_preserves_denominator_nonzero_require() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "(n + 1)! / n!";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    let result = match output.result {
        EvalResult::Expr(expr) => DisplayExpr {
            context: &engine.simplifier.context,
            id: expr,
        }
        .to_string(),
        other => panic!("expected expression result, got {other:?}"),
    };

    assert_eq!(result, "n + 1");

    let expected_den = parse("n!", &mut engine.simplifier.context).expect("den");
    assert_eq!(
        output.required_conditions.len(),
        1,
        "expected exactly one inherited denominator requirement"
    );
    assert!(output.required_conditions.iter().any(|cond| {
        matches!(
            cond,
            ImplicitCondition::NonZero(den)
                if compare_expr(&engine.simplifier.context, *den, expected_den)
                    == std::cmp::Ordering::Equal
        )
    }));
}
