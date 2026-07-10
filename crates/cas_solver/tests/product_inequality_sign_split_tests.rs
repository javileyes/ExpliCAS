//! Contract tests for `f(x)·g(x) {op} 0` with a non-polynomial factor (2026-07-10).
//!
//! The solve prepass DISTRIBUTES the raw product, after which the Mul-isolation
//! fallback divides by the variable-carrying factor KEEPING the operator direction
//! (`is_known_negative` is a constant oracle, so an unproven sign silently meant
//! "assume positive"): `(x-1)*ln(x) < 0` → `(0, 1)` when the truth is NO solution
//! (shared root x = 1, same factor signs elsewhere on the domain). Family F11 of
//! docs/AUDITORIA_FRONTERA_2026-07-09.md; the factor-sign split on the RAW tree
//! also closes the `(1-x)*ln(x) > 0` P1 echo member.

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
fn shared_root_product_with_ln_factor_splits_on_factor_signs() {
    // Both factors share the root x = 1 and the same sign on each side of it.
    assert_eq!(solve_display("(x - 1)*ln(x)", RelOp::Lt, "0"), "No solution");
    assert_eq!(
        solve_display("(x - 1)*ln(x)", RelOp::Gt, "0"),
        "(0, 1) U (1, infinity)"
    );
    // Non-strict boundaries flow through the CLOSED factor sub-solves: the shared
    // root bridges the two open intervals for >=, and is the whole set for <=.
    assert_eq!(solve_display("(x - 1)*ln(x)", RelOp::Geq, "0"), "(0, infinity)");
    assert_eq!(solve_display("(x - 1)*ln(x)", RelOp::Leq, "0"), "[1, 1]");
}

#[test]
fn distinct_root_products_keep_both_nonstrict_boundaries() {
    assert_eq!(
        solve_display("(x - 2)*ln(x)", RelOp::Geq, "0"),
        "(0, 1] U [2, infinity)"
    );
    assert_eq!(solve_display("(x - 2)*ln(x)", RelOp::Leq, "0"), "[1, 2]");
}

#[test]
fn negation_and_swapped_factor_orientations() {
    // (1-x)*ln(x) has OPPOSITE factor signs everywhere on the domain: never positive.
    assert_eq!(solve_display("(1 - x)*ln(x)", RelOp::Gt, "0"), "No solution");
    assert_eq!(
        solve_display("-(x - 1)*ln(x)", RelOp::Gt, "0"),
        "No solution"
    );
}

#[test]
fn sibling_owners_are_unchanged() {
    // Distinct-root mixed product (previously correct route).
    assert_eq!(solve_display("(x - 2)*ln(x)", RelOp::Lt, "0"), "(1, 2)");
    assert_eq!(solve_display("x*ln(x)", RelOp::Lt, "0"), "(0, 1)");
    assert_eq!(solve_display("x*e^x", RelOp::Gt, "0"), "(0, infinity)");
    // Polynomial products: owned by the polynomial sign analysis.
    assert_eq!(solve_display("(x - 1)*(x - 2)", RelOp::Lt, "0"), "(1, 2)");
    // Constant coefficient: ordinary isolation.
    assert_eq!(
        solve_display("2*(x - 1)", RelOp::Lt, "0"),
        "(-infinity, 1)"
    );
    // The product EQUATION keeps its owner.
    assert_eq!(solve_display("(x - 1)*ln(x)", RelOp::Eq, "0"), "{ 1 }");
}
