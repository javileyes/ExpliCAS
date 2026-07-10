//! Contract tests for `U(x)/√f = k` and the moved/surd single-radical forms
//! (2026-07-11). Family F23 of docs/AUDITORIA_FRONTERA_2026-07-09.md: the Mul
//! isolation moved the VAR-CARRYING reciprocal power and echoed the un-refolded
//! `solve(x − 1/(1−x²)^(−1/2) = 0)`. Three coupled extensions close the chain
//! without touching the isolation core: the quotient normalizer `√f = U/k`, the
//! MOVED single-radical form `c·√f + R = 0`, and the SURD extension of the
//! `g(r) ≥ 0` extraneous-root filter via the exact const-sign cascade.

use cas_ast::{Equation, RelOp};
use cas_parser::parse;
use cas_solver::api::solve;
use cas_solver::command_api::solve::display_solution_set;
use cas_solver::runtime::Simplifier;

fn solve_display(lhs: &str, rhs: &str) -> String {
    let mut simplifier = Simplifier::with_default_rules();
    let lhs = parse(lhs, &mut simplifier.context).expect("parse lhs");
    let rhs = parse(rhs, &mut simplifier.context).expect("parse rhs");
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let (set, _) = solve(&eq, "x", &mut simplifier).expect("solve");
    display_solution_set(&simplifier.context, &set)
}

#[test]
fn quotient_over_sqrt_normalizes_to_the_bare_radical() {
    assert_eq!(solve_display("x/sqrt(1 - x^2)", "1"), "{ 2^(-1/2) }");
    // Negative threshold keeps only the negative root through g(r) >= 0.
    assert_eq!(solve_display("x/sqrt(1 - x^2)", "-1"), "{ -(2^(-1/2)) }");
    // tan(arcsin(x)) folds to the same quotient and inherits the fix.
    assert_eq!(solve_display("tan(arcsin(x))", "1"), "{ 2^(-1/2) }");
    // Coefficients, shifted radicands, rational thresholds.
    assert_eq!(solve_display("2*x/sqrt(1 - x^2)", "1"), "{ 5^(-1/2) }");
    assert_eq!(solve_display("x/sqrt(4 - x^2)", "1"), "{ sqrt(2) }");
    assert_eq!(solve_display("x/sqrt(x^2 + 1)", "1/2"), "{ 3^(-1/2) }");
}

#[test]
fn surd_roots_pass_the_extraneous_filter_exactly() {
    // Direct bare-radical forms with SURD roots (previously declined by the
    // rational-only scope of the g(r) >= 0 filter).
    assert_eq!(solve_display("sqrt(1 - x^2)", "x"), "{ 2^(-1/2) }");
    // The classic extraneous-root case: (1+sqrt(5))/2 is dropped (g(r) < 0),
    // (1-sqrt(5))/2 kept.
    assert_eq!(
        solve_display("sqrt(x + 1)", "-x"),
        "{ 1/2 * (1 - sqrt(5)) }"
    );
}

#[test]
fn sibling_owners_are_unchanged() {
    // Constant numerator: the reciprocal-root owner (U2).
    assert_eq!(solve_display("1/sqrt(x)", "2"), "{ 1/4 }");
    // Rational (non-radical) quotient: the rational owner.
    assert_eq!(
        solve_display("x/(1 - x^2)", "1"),
        "{ -1/2 - 1/2 * sqrt(5), 1/2 * sqrt(5) - 1/2 }"
    );
    // Rational-rooted radical forms keep their exact sets.
    assert_eq!(solve_display("sqrt(5*x^2 + 9*x - 2)", "3*x"), "{ 1/4, 2 }");
    assert_eq!(solve_display("sqrt(x^2 - 4)", "x - 1"), "{ 5/2 }");
}
