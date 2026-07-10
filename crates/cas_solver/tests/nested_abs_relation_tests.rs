//! Contract tests for NESTED abs vs a variable remainder (2026-07-10).
//! `||x|−2| > x` returned "No solution" for every relation direction (truth
//! `(−∞, 1)` for `>`): all dedicated abs handlers decline the nested shape and the
//! generic isolation's inner sub-solves came back as unresolved Conditional sets
//! that the outer union swallowed. Family F16 of
//! docs/AUDITORIA_FRONTERA_2026-07-09.md. The fix partitions ℝ at the INNER abs
//! zeros, reduces each region to a plain abs relation, clips and unions.

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
fn nested_abs_vs_variable_rhs_partitions_at_inner_zeros() {
    assert_eq!(
        solve_display("abs(abs(x) - 2)", RelOp::Gt, "x"),
        "(-infinity, 1)"
    );
    assert_eq!(
        solve_display("abs(abs(x) - 2)", RelOp::Lt, "x"),
        "(1, infinity)"
    );
    assert_eq!(solve_display("abs(abs(x) - 2)", RelOp::Eq, "x"), "{ 1 }");
    assert_eq!(
        solve_display("abs(abs(x) - 2)", RelOp::Geq, "x"),
        "(-infinity, 1]"
    );
    assert_eq!(
        solve_display("abs(abs(x) - 3)", RelOp::Gt, "x"),
        "(-infinity, 3/2)"
    );
    assert_eq!(
        solve_display("abs(abs(x) - 1)", RelOp::Gt, "x"),
        "(-infinity, 1/2)"
    );
    // Shifted inner argument: breakpoint moves off the origin.
    assert_eq!(
        solve_display("abs(abs(x - 1) - 2)", RelOp::Gt, "x"),
        "(-infinity, 3/2)"
    );
}

#[test]
fn sibling_abs_owners_are_unchanged() {
    // Nested abs vs CONSTANT: the existing owner keeps its pinned root ordering.
    assert_eq!(
        solve_display("abs(abs(x) - 2)", RelOp::Eq, "1"),
        "{ 3, -3, 1, -1 }"
    );
    assert_eq!(
        solve_display("abs(abs(x) - 2)", RelOp::Lt, "1"),
        "(-3, -1) U (1, 3)"
    );
    // Simple abs vs variable RHS (no nesting): existing owners.
    assert_eq!(
        solve_display("abs(x - 2)", RelOp::Gt, "x"),
        "(-infinity, 1)"
    );
    // Sum-of-abs and multi-abs owners.
    assert_eq!(
        solve_display("abs(x - 1) + abs(x + 1)", RelOp::Lt, "4"),
        "(-2, 2)"
    );
    assert_eq!(
        solve_display("x^2 + abs(x - 1) + abs(x + 1)", RelOp::Lt, "5"),
        "(1 - sqrt(6), sqrt(6) - 1)"
    );
}
