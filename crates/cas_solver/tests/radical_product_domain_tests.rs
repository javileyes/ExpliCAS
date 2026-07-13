//! Contract tests for products of square roots `√A·√B = c` (2026-07-13b, family
//! F1 of docs/AUDITORIA_FRONTERA_2026-07-13b.md). The simplifier merges
//! `√A·√B → √(A·B)`, widening the real domain from `{A≥0 ∧ B≥0}` to `{A·B≥0}`;
//! the merged equation squares to a quadratic whose extraneous root (e.g. `x=-1`
//! of `√x·√(x-3)=2`, where `(-1)·(-4)=4≥0`) then verifies against the squared
//! form and survives. `solve_local_core` re-verifies discrete candidates against
//! the ORIGINAL radical equation for the level plus the per-radicand domain
//! conditions, dropping the domain-violating roots.

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
fn radical_product_drops_the_domain_violating_root() {
    // Rational extraneous root x=-1 / x=-4 / etc. dropped (negative radicand).
    assert_eq!(solve_display("sqrt(x)*sqrt(x - 3)", "2"), "{ 4 }");
    assert_eq!(solve_display("sqrt(x)*sqrt(x + 3)", "2"), "{ 1 }");
    assert_eq!(solve_display("sqrt(x)*sqrt(x - 5)", "6"), "{ 9 }");
    // Surd extraneous root dropped by the exact surd-sign prover: only the root
    // with x ≥ 2 survives.
    assert_eq!(
        solve_display("sqrt(x - 1)*sqrt(x - 2)", "2"),
        "{ 1/2 * (sqrt(17) + 3) }"
    );
    // Domain on the x ≤ 0 side: only (1-√17)/2 survives.
    assert_eq!(
        solve_display("sqrt(-x)*sqrt(1 - x)", "2"),
        "{ 1/2 * (1 - sqrt(17)) }"
    );
    // Zero-product with a domain gate: x=0 needs x-3≥0 (violated), only x=3 survives.
    assert_eq!(solve_display("sqrt(x)*sqrt(x - 3)", "0"), "{ 3 }");
    // Outer coefficient does not hide the radical product.
    assert_eq!(solve_display("2*sqrt(x)*sqrt(x - 3)", "4"), "{ 4 }");
    // Three-radical zero product: only x=2 satisfies all three radicands.
    assert_eq!(
        solve_display("sqrt(x)*sqrt(x - 1)*sqrt(x - 2)", "0"),
        "{ 2 }"
    );
}

#[test]
fn single_radicals_and_non_radicals_are_unchanged() {
    // Single radical equations keep their existing (correct) behavior.
    assert_eq!(solve_display("sqrt(x)", "2"), "{ 4 }");
    assert_eq!(solve_display("sqrt(x + 1)", "x - 1"), "{ 3 }");
    assert_eq!(solve_display("sqrt(x)", "x - 2"), "{ 4 }");
    // √A·√A = A: genuine root kept (x ≥ 0 satisfied).
    assert_eq!(solve_display("sqrt(x)*sqrt(x)", "4"), "{ 4 }");
    // Radical sums untouched.
    assert_eq!(solve_display("sqrt(x - 1) + sqrt(x + 4)", "5"), "{ 5 }");
    // Non-radical polynomials keep BOTH roots (no domain condition to violate).
    assert_eq!(solve_display("x^2", "4"), "{ -2, 2 }");
    assert_eq!(solve_display("x^2 - 3*x - 4", "0"), "{ -1, 4 }");
}
