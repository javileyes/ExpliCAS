//! Contract tests for `c/g(x) {op} k`, `k ≠ 0`, with a sign-indefinite NON-POLYNOMIAL
//! denominator (2026-07-10). The legacy isolation multiplied by `g` without casing on
//! its sign and returned the single naive interval between the boundary roots —
//! including the pole and the whole wrong-sign region: `1/(abs(x)-1) > 1` → `(-2, 2)`,
//! `1/ln(x) > 2` → `(0, e^(1/2))`. The denominator sign-split closes families F7 AND
//! F5 of docs/AUDITORIA_FRONTERA_2026-07-09.md with one handler.

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
    let (set, _) = solve(&eq, "x", &mut simplifier).expect("solve");
    display_solution_set(&simplifier.context, &set)
}

#[test]
fn reciprocal_of_abs_affine_vs_nonzero_const_splits_on_denominator_sign() {
    assert_eq!(
        solve_display("1/(abs(x) - 1)", RelOp::Gt, "1"),
        "(-2, -1) U (1, 2)"
    );
    // Non-strict: the boundary |x| = 2 (where the value IS 1) closes; poles stay open.
    assert_eq!(
        solve_display("1/(abs(x) - 1)", RelOp::Geq, "1"),
        "[-2, -1) U (1, 2]"
    );
    // The flipped direction recovers the negative-denominator interior (-1, 1).
    assert_eq!(
        solve_display("1/(abs(x) - 1)", RelOp::Lt, "1"),
        "(-infinity, -2) U (-1, 1) U (2, infinity)"
    );
    assert_eq!(
        solve_display("2/(abs(x) - 1)", RelOp::Gt, "1"),
        "(-3, -1) U (1, 3)"
    );
    assert_eq!(
        solve_display("1/(abs(x - 3) - 1)", RelOp::Gt, "2"),
        "(3/2, 2) U (4, 9/2)"
    );
    // Negative threshold: the positive-denominator side holds everywhere.
    assert_eq!(
        solve_display("1/(abs(x) - 1)", RelOp::Gt, "-1"),
        "(-infinity, -1) U (1, infinity)"
    );
    // Leading negation folds into the numerator.
    assert_eq!(
        solve_display("-1/(abs(x) - 1)", RelOp::Gt, "1"),
        "(-1, 0) U (0, 1)"
    );
}

#[test]
fn reciprocal_of_ln_vs_nonzero_const_splits_on_denominator_sign() {
    assert_eq!(solve_display("1/ln(x)", RelOp::Gt, "2"), "(1, e^(1/2))");
    assert_eq!(
        solve_display("1/ln(x)", RelOp::Lt, "2"),
        "(0, 1) U (e^(1/2), infinity)"
    );
    assert_eq!(solve_display("1/ln(x)", RelOp::Geq, "2"), "(1, e^(1/2)]");
    assert_eq!(solve_display("2/ln(x)", RelOp::Gt, "4"), "(1, e^(1/2))");
    assert_eq!(
        solve_display("1/ln(x)", RelOp::Gt, "-1"),
        "(0, 1 / e) U (1, infinity)"
    );
    assert_eq!(
        solve_display("1/(ln(x) - 1)", RelOp::Gt, "2"),
        "(e, e^(3/2))"
    );
    assert_eq!(solve_display("1/log(2, x)", RelOp::Gt, "1"), "(1, 2)");
}

#[test]
fn sibling_owners_are_unchanged() {
    // Polynomial denominators: owned by the rational-inequality path.
    assert_eq!(solve_display("1/(x - 1)", RelOp::Gt, "2"), "(1, 3/2)");
    // Bare-abs denominator: owned by the A/|g| handler.
    assert_eq!(
        solve_display("1/abs(x)", RelOp::Gt, "2"),
        "(-1/2, 0) U (0, 1/2)"
    );
    // Reciprocal of a root: owned by the U2 handler.
    assert_eq!(solve_display("1/sqrt(x)", RelOp::Gt, "2"), "(0, 1/4)");
    // Exponential interior: owned by the exponential substitution.
    assert_eq!(solve_display("1/(e^x - 1)", RelOp::Gt, "1"), "(0, ln(2))");
    // vs-zero forms: owned by the strict reduction.
    assert_eq!(solve_display("1/(abs(x) - 1)", RelOp::Lt, "0"), "(-1, 1)");
}
