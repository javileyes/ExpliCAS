//! Contract tests for `A·arcfun(x) = k` / `arcfun(x)/A = k` / `−arcfun(x) = k`
//! (2026-07-10). The inverse-trig/hyperbolic handler matched only a BARE call —
//! the historic coefficient≠1 blind spot — so the generic isolation peeled the
//! coefficient into an UNFOLDED `pi/3/2` and leaked the reduced equation
//! un-dispatched (`UnaryInverseKind` has no arc/hyp inverses). Family F22 of
//! docs/AUDITORIA_FRONTERA_2026-07-09.md, plus the hyperbolic sibling.

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
fn rational_coefficient_wrappers_fold_into_the_constant() {
    assert_eq!(solve_display("2*arcsin(x)", "pi/3"), "{ 1/2 }");
    assert_eq!(solve_display("3*arcsin(x)", "pi/2"), "{ 1/2 }");
    assert_eq!(solve_display("2*arctan(x)", "pi/2"), "{ 1 }");
    assert_eq!(solve_display("2*arccos(x)", "pi/2"), "{ 2^(-1/2) }");
    // Already-folded constant, still un-dispatched before the fix.
    assert_eq!(solve_display("4*arctan(x)", "pi"), "{ 1 }");
    // Division and negation wrappers.
    assert_eq!(solve_display("arcsin(x)/2", "pi/12"), "{ 1/2 }");
    assert_eq!(solve_display("-arcsin(x)", "pi/6"), "{ -1/2 }");
    // The hyperbolic sibling shares the handler.
    assert_eq!(solve_display("2*sinh(x)", "3"), "{ asinh(3/2) }");
}

#[test]
fn range_gate_fires_after_the_peel() {
    // 5/2 > pi/2: provably outside arcsin's range - never the spurious sin(5/2).
    assert_eq!(solve_display("2*arcsin(x)", "5"), "No solution");
}

#[test]
fn bare_forms_and_other_owners_are_unchanged() {
    assert_eq!(solve_display("arcsin(x)", "pi/6"), "{ 1/2 }");
    assert_eq!(solve_display("arctan(x)", "pi/4"), "{ 1 }");
    assert_eq!(solve_display("arcsin(x)", "5"), "No solution");
    // ln keeps its own isolation owner.
    assert_eq!(solve_display("2*ln(x)", "4"), "{ e^2 }");
    // cosh keeps its two-branch reduction.
    assert_eq!(solve_display("cosh(x)", "2"), "{ acosh(2), -acosh(2) }");
}
