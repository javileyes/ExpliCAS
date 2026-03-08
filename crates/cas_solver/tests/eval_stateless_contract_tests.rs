use cas_parser::parse;
use cas_solver::{Engine, EvalAction, EvalOptions, EvalRequest, EvalResult};

#[test]
fn eval_stateless_simplify_ignores_auto_store() {
    let mut engine = Engine::new();
    let parsed = parse("x + x", &mut engine.simplifier.context).expect("parse x + x");

    let req = EvalRequest {
        raw_input: "x + x".to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: true, // Should be ignored in stateless path.
    };

    let output = engine
        .eval_stateless(EvalOptions::default(), req)
        .expect("stateless eval");

    assert!(
        output.stored_id.is_none(),
        "stateless eval must not persist entries"
    );

    let got = match output.result {
        EvalResult::Expr(id) => id,
        other => panic!("expected EvalResult::Expr, got {other:?}"),
    };
    let expected = parse("2*x", &mut engine.simplifier.context).expect("parse expected");
    assert!(
        engine.simplifier.are_equivalent(got, expected),
        "stateless simplify should preserve algebraic semantics"
    );
}

#[test]
fn eval_stateless_supports_equiv_action() {
    let mut engine = Engine::new();
    let parsed = parse("x + x", &mut engine.simplifier.context).expect("parse lhs");
    let other = parse("2*x", &mut engine.simplifier.context).expect("parse rhs");

    let req = EvalRequest {
        raw_input: "x + x".to_string(),
        parsed,
        action: EvalAction::Equiv { other },
        auto_store: false,
    };

    let output = engine
        .eval_stateless(EvalOptions::default(), req)
        .expect("stateless equiv");
    match output.result {
        EvalResult::Bool(value) => assert!(value, "x + x must be equivalent to 2*x"),
        other => panic!("expected EvalResult::Bool, got {other:?}"),
    }
}

#[test]
fn eval_stateless_rejects_session_references() {
    let mut engine = Engine::new();
    let parsed = parse("#1 + x", &mut engine.simplifier.context).expect("parse session ref");

    let req = EvalRequest {
        raw_input: "#1 + x".to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let err = engine
        .eval_stateless(EvalOptions::default(), req)
        .expect_err("stateless eval must reject session refs");
    let msg = err.to_string();
    assert!(
        msg.contains("requires stateful eval"),
        "unexpected error message: {msg}"
    );
}
