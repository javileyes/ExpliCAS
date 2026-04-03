use cas_formatter::DisplayExpr;
use cas_parser::parse;
use cas_session::SessionState;
use cas_solver::runtime::{Engine, EvalAction, EvalRequest, EvalResult};

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
