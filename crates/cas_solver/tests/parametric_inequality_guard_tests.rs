//! Contract tests for the parametric monotone-inequality guard (2026-07-11).
//! `solve(a*x > b)` returned the unconditional ray `(b/a, infinity)` — wrong for
//! the whole `a < 0` half of the parameter space — and the factored linear
//! collect dropped the operator entirely for `(a^2+1)*x > b` (a DISCRETE
//! boundary). Family F1 of docs/AUDITORIA_FRONTERA_2026-07-09.md. An order
//! relation whose next monotone step runs through a var-free NON-NUMERIC
//! constant now transforms EXACTLY when the sign is proven (positive keeps the
//! direction, negative flips) and declines honestly otherwise, with the same
//! message as the quadratic strategy's guard.

use cas_ast::{Equation, RelOp};
use cas_parser::parse;
use cas_solver::api::solve;
use cas_solver::command_api::solve::display_solution_set;
use cas_solver::runtime::Simplifier;

fn solve_result(lhs: &str, op: RelOp, rhs: &str) -> Result<String, String> {
    let mut simplifier = Simplifier::with_default_rules();
    let lhs = parse(lhs, &mut simplifier.context).expect("parse lhs");
    let rhs = parse(rhs, &mut simplifier.context).expect("parse rhs");
    let eq = Equation { lhs, rhs, op };
    match solve(&eq, "x", &mut simplifier) {
        Ok((set, _)) => Ok(display_solution_set(&simplifier.context, &set)),
        Err(e) => Err(format!("{e:?}")),
    }
}

#[test]
fn undecidable_parametric_coefficients_decline_honestly() {
    for (lhs, op, rhs) in [
        ("a*x", RelOp::Gt, "b"),
        ("a*x", RelOp::Lt, "b"),
        ("-a*x", RelOp::Gt, "b"),
        ("2*a*x", RelOp::Gt, "b"),
        ("a*x", RelOp::Gt, "0"),
        ("x/a", RelOp::Gt, "1"),
        ("a/x", RelOp::Gt, "1"),
        ("sqrt(x)", RelOp::Lt, "a"),
        ("a*sqrt(x)", RelOp::Gt, "1"),
    ] {
        let out = solve_result(lhs, op, rhs);
        let err = out.expect_err(&format!("{lhs} must decline"));
        assert!(
            err.contains("symbolic coefficients"),
            "{lhs}: wrong decline message: {err}"
        );
    }
}

#[test]
fn provably_signed_coefficients_transform_exactly() {
    // Structurally provable positive: the previously operator-dropping collect.
    assert_eq!(
        solve_result("(a^2 + 1)*x", RelOp::Gt, "b").unwrap(),
        "(b / (a^2 + 1), infinity)"
    );
    // Constant irrational coefficients keep their exact rays.
    assert_eq!(
        solve_result("sqrt(2)*x", RelOp::Gt, "1").unwrap(),
        "(2^(-1/2), infinity)"
    );
    assert_eq!(
        solve_result("e*x", RelOp::Gt, "1").unwrap(),
        "(1 / e, infinity)"
    );
    assert_eq!(
        solve_result("pi*x", RelOp::Lt, "pi").unwrap(),
        "(-infinity, 1)"
    );
}

#[test]
fn sibling_owners_are_unchanged() {
    // Rational coefficients: historical routes untouched.
    assert_eq!(
        solve_result("-2*x", RelOp::Gt, "6").unwrap(),
        "(-infinity, -3)"
    );
    // Parametric EQUATIONS keep their solutions.
    assert_eq!(solve_result("a*x", RelOp::Eq, "b").unwrap(), "{ b / a }");
    // Even-root thresholds with DECIDABLE signs keep their exact sets.
    assert_eq!(
        solve_result("sqrt(x)", RelOp::Lt, "-2").unwrap(),
        "No solution"
    );
    assert_eq!(solve_result("sqrt(x)", RelOp::Lt, "2").unwrap(), "[0, 4)");
    assert_eq!(
        solve_result("sqrt(x)", RelOp::Lt, "sqrt(2)").unwrap(),
        "[0, 2)"
    );
    // The existing GOOD parametric conditional survives (exp owner).
    let exp_form = solve_result("e^x", RelOp::Gt, "a").unwrap();
    assert!(exp_form.contains("if a > 0"), "got {exp_form}");
}
