//! Contract tests for linear inequalities whose collected coefficient is a
//! SYMBOLIC CONSTANT (2026-07-13b, family F3 of
//! docs/AUDITORIA_FRONTERA_2026-07-13b.md). The log-linearization of
//! `e^x {op} 2^x` recurses with `x {op} x·ln2`, and the equation-only
//! linear-collect returned the boundary root `{0}` with the operator DROPPED.
//! `try_symbolic_linear_coeff_inequality` collects the difference to
//! `c1·x + c0`, decides `sign(c1)` with the exact const-sign oracles, and
//! recurses on the ray (operator flipped for a provably-negative coefficient);
//! an undecidable sign declines honestly. Answers verified against sympy.

use cas_ast::{Equation, RelOp};
use cas_parser::parse;
use cas_solver::api::solve;
use cas_solver::command_api::solve::display_solution_set;
use cas_solver::runtime::Simplifier;

fn solve_display(lhs: &str, op: RelOp, rhs: &str) -> String {
    let mut simplifier = Simplifier::with_default_rules();
    let lhs = parse(lhs, &mut simplifier.context).expect("parse lhs");
    let rhs = parse(rhs, &mut simplifier.context).expect("parse rhs");
    let eq = Equation { lhs, rhs, op };
    match solve(&eq, "x", &mut simplifier) {
        Ok((set, _)) => display_solution_set(&simplifier.context, &set)
            .replace(" * ", "·")
            .replace('*', "·"),
        Err(e) => format!("ERR: {e}"),
    }
}

#[test]
fn transcendental_base_pairs_solve_to_the_monotone_ray() {
    // e vs 2: ln-linearizes to x {op} x·ln2, coeff 1-ln2 > 0.
    assert_eq!(solve_display("e^x", RelOp::Lt, "2^x"), "(-infinity, 0)");
    assert_eq!(solve_display("e^x", RelOp::Gt, "2^x"), "(0, infinity)");
    assert_eq!(solve_display("e^x", RelOp::Leq, "2^x"), "(-infinity, 0]");
    // Orientation swap: 2^x > e^x has coeff ln2-1 < 0 (flip).
    assert_eq!(solve_display("2^x", RelOp::Gt, "e^x"), "(-infinity, 0)");
    // pi vs 2, and the doubled-exponent member.
    assert_eq!(solve_display("pi^x", RelOp::Lt, "2^x"), "(-infinity, 0)");
    assert_eq!(solve_display("e^(2*x)", RelOp::Lt, "4^x"), "(-infinity, 0)");
    // Shifted members carry the boundary into the ray endpoint.
    assert_eq!(
        solve_display("e^x", RelOp::Lt, "2*2^x"),
        "(-infinity, ln(2) / (-ln(2) + 1))"
    );
    assert_eq!(
        solve_display("e^(x + 1)", RelOp::Lt, "2^x"),
        "(-infinity, 1 / (ln(2) - 1))"
    );
}

#[test]
fn direct_symbolic_coefficient_forms_solve_or_decline_honestly() {
    // Var on both sides with ln coefficients.
    assert_eq!(solve_display("x", RelOp::Lt, "x*ln(2)"), "(-infinity, 0)");
    assert_eq!(
        solve_display("x*ln(2)", RelOp::Lt, "x*ln(3)"),
        "(0, infinity)"
    );
    // One-sided variant of the same dropped-operator family.
    assert_eq!(
        solve_display("x + x*ln(2)", RelOp::Lt, "3"),
        "(-infinity, 3 / (ln(2) + 1))"
    );
    // Compound surd coefficient (2·sqrt(2) − 1 > 0 via the exact surd oracle).
    assert_eq!(
        solve_display("2*sqrt(2)*x", RelOp::Lt, "x"),
        "(-infinity, 0)"
    );
    // Named-constant coefficients: e < pi is decidable by bounds.
    assert_eq!(solve_display("x*e", RelOp::Lt, "x*pi"), "(0, infinity)");
    // A provably-zero coefficient difference is a constant relation (AllReals here).
    assert_eq!(
        solve_display("x*ln(2)", RelOp::Lt, "x*ln(2) + 1"),
        "All real numbers"
    );
    // A genuinely-symbolic (parameter) coefficient keeps the honest decline.
    let a = solve_display("a*x", RelOp::Lt, "0");
    assert!(a.contains("ERR"), "{a}");
}

#[test]
fn rational_coefficient_owners_are_untouched() {
    assert_eq!(
        solve_display("2*x + 1", RelOp::Lt, "x + 3"),
        "(-infinity, 2)"
    );
    assert_eq!(solve_display("2*x", RelOp::Lt, "x"), "(-infinity, 0)");
    assert_eq!(solve_display("-2*x", RelOp::Gt, "6"), "(-infinity, -3)");
    assert_eq!(solve_display("x^2", RelOp::Lt, "4"), "(-2, 2)");
    // Integer-base exponential pairs keep the numeric-base terminal.
    assert_eq!(solve_display("3^x", RelOp::Lt, "2^x"), "(-infinity, 0)");
    // Equations are untouched (boundary root is the correct answer there).
    let mut simplifier = Simplifier::with_default_rules();
    let lhs = parse("e^x", &mut simplifier.context).expect("lhs");
    let rhs = parse("2^x", &mut simplifier.context).expect("rhs");
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let (set, _) = solve(&eq, "x", &mut simplifier).expect("solve");
    assert_eq!(display_solution_set(&simplifier.context, &set), "{ 0 }");
}
