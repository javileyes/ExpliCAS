//! Contract tests for nested absolute-value equations `||x|-c| = |g(x)|`
//! (2026-07-13b, family F2 of docs/AUDITORIA_FRONTERA_2026-07-13b.md). The
//! nested-abs handler already had the correct breakpoint/segment enumeration
//! core, but its claim gate only admitted the "variable-remainder" family
//! (`||x|-2| = x`) and declined the abs-vs-abs shape (`||x|-5| = |x|`), which
//! then fell to a generic path returning the maximally-wrong empty set. The gate
//! now also claims a var-free remainder when two or more outermost abs carry the
//! variable. Answers verified by direct substitution + a fine grid (sympy's
//! `solveset` spuriously returns EmptySet for nested Abs and is NOT an oracle here).

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
fn nested_abs_vs_abs_enumerates_all_segments() {
    assert_eq!(solve_display("abs(abs(x) - 5)", "abs(x)"), "{ -5/2, 5/2 }");
    assert_eq!(solve_display("abs(abs(x) - 4)", "abs(x)"), "{ -2, 2 }");
    assert_eq!(solve_display("abs(x)", "abs(abs(x) - 5)"), "{ -5/2, 5/2 }");
    // Different RHS abs (breakpoint x=1) — the per-segment affine abs-vs-abs sub-solve handles it.
    assert_eq!(
        solve_display("abs(abs(x) - 2)", "abs(x - 1)"),
        "{ -1/2, 3/2 }"
    );
    assert_eq!(solve_display("abs(abs(x) - 5)", "abs(x - 1)"), "{ -2, 3 }");
    // Inner shift and coefficient.
    assert_eq!(
        solve_display("abs(abs(x - 2) - 5)", "abs(x)"),
        "{ -3/2, 7/2 }"
    );
    assert_eq!(
        solve_display("abs(abs(x) - 5)", "2*abs(x)"),
        "{ -5/3, 5/3 }"
    );
    // Inner coefficient produces four roots.
    assert_eq!(
        solve_display("abs(abs(2*x) - 5)", "abs(x)"),
        "{ -5/3, -5, 5, 5/3 }"
    );
}

#[test]
fn nested_abs_degenerate_and_constant_cases_unchanged() {
    // c = 0: `||x|-0| = |x|` is the identity `|x| = |x|`.
    assert_eq!(
        solve_display("abs(abs(x) - 0)", "abs(x)"),
        "All real numbers"
    );
    // `||x|+1| = |x|` is `|x|+1 = |x|`, unsatisfiable.
    assert_eq!(solve_display("abs(abs(x) + 1)", "abs(x)"), "No solution");
    // Nested-vs-CONSTANT keeps its existing owner and root ordering.
    assert_eq!(solve_display("abs(abs(x) - 2)", "1"), "{ 3, -3, 1, -1 }");
    assert_eq!(solve_display("abs(abs(x) - 4)", "2"), "{ 6, -6, 2, -2 }");
    // Variable-remainder family (single outer abs, remainder has the var) unchanged.
    assert_eq!(solve_display("abs(abs(x) - 3)", "x"), "{ 3/2 }");
    // Affine abs-vs-abs (no nesting) unchanged.
    assert_eq!(solve_display("abs(x - 1)", "abs(x + 1)"), "{ 0 }");
}
