//! Contract tests for `|f(x)| {op} |g(x)|` with polynomial arguments (2026-07-10).
//! No handler owned the non-affine shape (single-abs wants ONE distinct abs, the
//! threshold handler a CONSTANT side, multi-abs affine arguments), so it fell to
//! the generic path: `|x²−1| < |x+1|` → "No solution" (truth `(0, 2)`), `≤` kept
//! only degenerate boundary points, `>` leaked a mangled conditional. Family F15
//! of docs/AUDITORIA_FRONTERA_2026-07-09.md. Both sides are non-negative, so the
//! relation reduces EXACTLY to the polynomial inequality `f² − g² {op} 0`.

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
fn quadratic_vs_linear_abs_inequality_reduces_to_squares() {
    assert_eq!(
        solve_display("abs(x^2 - 1)", RelOp::Lt, "abs(x + 1)"),
        "(0, 2)"
    );
    // x = −1 zeroes BOTH sides: included for ≤, excluded for <, excluded for >.
    assert_eq!(
        solve_display("abs(x^2 - 1)", RelOp::Leq, "abs(x + 1)"),
        "[-1, -1] U [0, 2]"
    );
    assert_eq!(
        solve_display("abs(x^2 - 1)", RelOp::Gt, "abs(x + 1)"),
        "(-infinity, -1) U (-1, 0) U (2, infinity)"
    );
    assert_eq!(
        solve_display("abs(x^2 - 1)", RelOp::Geq, "abs(x + 1)"),
        "(-infinity, 0] U [2, infinity)"
    );
}

#[test]
fn shifted_and_swapped_argument_shapes() {
    assert_eq!(
        solve_display("abs(x^2 - 4)", RelOp::Lt, "abs(x + 2)"),
        "(1, 3)"
    );
    assert_eq!(
        solve_display("abs(x^2 - 1)", RelOp::Lt, "abs(x - 1)"),
        "(-2, 0)"
    );
    assert_eq!(
        solve_display("abs(x^2 - 9)", RelOp::Lt, "abs(x - 3)"),
        "(-4, -2)"
    );
}

#[test]
fn sibling_owners_are_unchanged() {
    // Affine-vs-affine: existing correct owner (degree gate declines here).
    assert_eq!(
        solve_display("abs(2*x + 1)", RelOp::Gt, "abs(x - 1)"),
        "(-infinity, -2) U (0, infinity)"
    );
    // Equations keep their owner and pinned representation.
    assert_eq!(
        solve_display("abs(x^2 - 1)", RelOp::Eq, "abs(x + 1)"),
        "{ -1, 2, 0 }"
    );
    // Constant-threshold abs: the threshold handler's job.
    assert_eq!(solve_display("abs(x - 1)", RelOp::Lt, "2"), "(-1, 3)");
}
