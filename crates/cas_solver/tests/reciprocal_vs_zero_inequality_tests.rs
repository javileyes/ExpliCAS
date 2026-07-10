//! Contract tests for `c/g(x) ⋚ 0` with a NON-POLYNOMIAL denominator (2026-07-10).
//!
//! `solve(1/ln(x) > 0)` used to emit the garbage interval `(e^infinity, infinity)`:
//! the exact reduction `c/g (op) 0 ⟺ g (strict op') 0` existed but was gated to
//! affine denominators, so a transcendental `g` fell through to the legacy path,
//! which inverted the boundary equation into a non-real endpoint. Family F6 of
//! docs/AUDITORIA_FRONTERA_2026-07-09.md.

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
fn reciprocal_of_ln_vs_zero_reduces_to_denominator_sign() {
    assert_eq!(solve_display("1/ln(x)", RelOp::Gt, "0"), "(1, infinity)");
    assert_eq!(solve_display("1/ln(x)", RelOp::Lt, "0"), "(0, 1)");
    // The value `c/g` is never zero, so `>=`/`<=` collapse to the STRICT reduction.
    assert_eq!(solve_display("1/ln(x)", RelOp::Geq, "0"), "(1, infinity)");
    assert_eq!(solve_display("1/ln(x)", RelOp::Leq, "0"), "(0, 1)");
}

#[test]
fn coefficient_and_negation_fold_into_the_sign() {
    assert_eq!(solve_display("2/ln(x)", RelOp::Lt, "0"), "(0, 1)");
    assert_eq!(solve_display("-1/ln(x)", RelOp::Gt, "0"), "(0, 1)");
    assert_eq!(
        solve_display("1/(ln(x) - 1)", RelOp::Gt, "0"),
        "(e, infinity)"
    );
}

#[test]
fn unfolded_constant_rhs_is_normalized_for_the_gate() {
    // e^(1/x) < 1 log-transforms to `1/x < ln(1)`; the RHS must fold to 0 for the
    // reduction gate instead of leaking a `(undefined, infinity)` legacy interval.
    assert_eq!(solve_display("e^(1/x)", RelOp::Lt, "1"), "(-infinity, 0)");
}

#[test]
fn sibling_owners_are_unchanged() {
    // Polynomial denominator: owned by the rational-inequality path.
    assert_eq!(solve_display("1/(x - 2)", RelOp::Gt, "0"), "(2, infinity)");
    // Surd-affine denominator: the previously-fixed exact reduction.
    assert_eq!(
        solve_display("1/(x - sqrt(2))", RelOp::Gt, "0"),
        "(sqrt(2), infinity)"
    );
    // Exponential interior: owned by the exponential substitution.
    assert_eq!(
        solve_display("1/(e^x - 1)", RelOp::Gt, "0"),
        "(0, infinity)"
    );
}
