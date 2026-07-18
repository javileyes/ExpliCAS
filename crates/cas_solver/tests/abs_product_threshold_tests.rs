//! Contract tests for `|f(x)|·|g(x)| {op} c` (2026-07-10). The simplifier folds the
//! product of two abs into a single `|f·g|`, whose owner is the single-abs
//! sign-split handler — but the RAW unexpanded Mul branch (`−x·(x−1) − 2 < 0`)
//! defeated the recursive solver with a mangled residual, and the set algebra
//! silently swallowed it, dropping the whole between-the-zeros region:
//! `|x|·|x−1| < 2` returned `(−1, 0] ∪ [1, 2)` instead of `(−1, 2)`. Family F17 of
//! docs/AUDITORIA_FRONTERA_2026-07-09.md; the fix falls back to the already-parsed
//! branch POLYNOMIAL (canonical expanded form) when the raw solve is not concrete.

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
fn product_of_two_abs_below_threshold_keeps_the_interior_interval() {
    assert_eq!(
        solve_display("abs(x)*abs(x - 1)", RelOp::Lt, "2"),
        "(-1, 2)"
    );
    assert_eq!(
        solve_display("abs(x)*abs(x - 1)", RelOp::Leq, "2"),
        "[-1, 2]"
    );
    assert_eq!(
        solve_display("abs(x)*abs(x + 2)", RelOp::Lt, "3"),
        "(-3, 1)"
    );
    assert_eq!(
        solve_display("abs(x)*abs(x - 2)", RelOp::Lt, "3"),
        "(-1, 3)"
    );
}

#[test]
fn product_of_two_abs_above_threshold_directions() {
    assert_eq!(
        solve_display("abs(x)*abs(x - 1)", RelOp::Gt, "2"),
        "(-infinity, -1) U (2, infinity)"
    );
    assert_eq!(
        solve_display("abs(x)*abs(x - 1)", RelOp::Geq, "2"),
        "(-infinity, -1] U [2, infinity)"
    );
}

#[test]
fn sibling_abs_owners_are_unchanged() {
    // Direct single-abs quadratic (already-expanded argument).
    assert_eq!(solve_display("abs(x^2 - x)", RelOp::Lt, "2"), "(-1, 2)");
    // Difference-of-squares fold.
    assert_eq!(
        solve_display("abs(x - 1)*abs(x + 1)", RelOp::Lt, "3"),
        "(-2, 2)"
    );
    // The equation form keeps its owner and roots.
    assert_eq!(
        solve_display("abs(x)*abs(x - 1)", RelOp::Eq, "2"),
        "{ -1, 2 }"
    );
    // Entangled quadratic-plus-abs (historic family, same handler).
    assert_eq!(
        solve_display("x^2 - 2*abs(x) - 3", RelOp::Gt, "0"),
        "(-infinity, -3) U (3, infinity)"
    );
}
