//! Contract tests for two-sided implicit-domain intersection in inequalities (2026-07-10).
//!
//! `solve(ln(x) > ln(3-x))` used to return `(3/2, infinity)`: the RHS argument-domain
//! `3 - x > 0` was only recorded as a `Requires` side-condition and never intersected
//! into the emitted set, so the ray kept points where the relation is UNDEFINED. A point
//! where EITHER side is undefined cannot satisfy the relation; both sides' implicit
//! domains must shrink interval results (family F9, docs/AUDITORIA_FRONTERA_2026-07-09.md).

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
fn ln_vs_ln_inequality_intersects_rhs_domain() {
    assert_eq!(solve_display("ln(x)", RelOp::Gt, "ln(3 - x)"), "(3/2, 3)");
    assert_eq!(solve_display("ln(x)", RelOp::Geq, "ln(3 - x)"), "[3/2, 3)");
    assert_eq!(solve_display("ln(x - 1)", RelOp::Gt, "ln(5 - x)"), "(3, 5)");
    assert_eq!(solve_display("ln(2*x)", RelOp::Gt, "ln(6 - x)"), "(2, 6)");
}

#[test]
fn ln_vs_ln_inequality_keeps_previously_correct_orientations() {
    // Bounded domain already on the LHS (worked before the fix): must stay green.
    assert_eq!(solve_display("ln(3 - x)", RelOp::Lt, "ln(x)"), "(3/2, 3)");
    // RHS domain subsumed by the solved interval: unchanged.
    assert_eq!(solve_display("ln(x)", RelOp::Lt, "ln(3 - x)"), "(0, 3/2)");
}

#[test]
fn sqrt_vs_sqrt_inequality_keeps_nonstrict_rhs_domain_edge() {
    // The RHS domain is NON-NEGATIVE (`6 - x >= 0`): x = 6 stays INCLUDED, since
    // sqrt(6) > sqrt(0) = 0 holds — the intersection must not turn the edge strict.
    assert_eq!(solve_display("sqrt(x)", RelOp::Gt, "sqrt(6 - x)"), "(3, 6]");
}

#[test]
fn rhs_without_domain_conditions_keeps_ray() {
    assert_eq!(solve_display("ln(x)", RelOp::Gt, "2"), "(e^2, infinity)");
}
