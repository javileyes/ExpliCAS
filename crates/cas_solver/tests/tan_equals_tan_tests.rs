//! Contract tests for `tan(u) = tan(v)` with affine rational-coefficient
//! arguments (2026-07-14, re-cycle C: the multiple-angle tan subfamily of F5 +
//! the `tan(2x)=tan(3x)` hang member of F8, docs/AUDITORIA_FRONTERA_2026-07-13b.md).
//! `tan(u) = tan(v) ⟺ u ≡ v (mod π)` where both sides are defined; the handler
//! solves the progression `w·x = kπ − Δb` and excludes the tan-pole progressions
//! with EXACT rational arithmetic (π-irrationality separates rational offsets;
//! rational gcd decides progression intersections). The generic paths previously
//! produced garbage residuals or hung (~216s for tan(2x)=tan(3x)).

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
        .replace(" * ", "·")
        .replace('*', "·")
}

#[test]
fn tan_equals_tan_solves_the_congruence_minus_poles() {
    // u−v = x: sin-roots {kπ}, no pole hits.
    assert_eq!(solve_display("tan(2*x)", "tan(x)"), "{ k·pi : k ∈ ℤ }");
    assert_eq!(solve_display("tan(x)", "tan(2*x)"), "{ k·pi : k ∈ ℤ }");
    // The former ~216s HANG: now instant.
    assert_eq!(solve_display("tan(2*x)", "tan(3*x)"), "{ k·pi : k ∈ ℤ }");
    // u−v = 2x: raw roots {kπ/2}, but the ODD multiples are tan poles (both
    // tan(x) and tan(3x) undefined at π/2+kπ) — the exclusion keeps {kπ}.
    assert_eq!(solve_display("tan(3*x)", "tan(x)"), "{ k·pi : k ∈ ℤ }");
    // Diff form matched on the RAW tree (simplify collapses it to sin/cos).
    assert_eq!(solve_display("tan(2*x) - tan(x)", "0"), "{ k·pi : k ∈ ℤ }");
}

#[test]
fn tan_equals_tan_rational_offsets_and_degenerate_slope() {
    // Rational offset: x = 1 + kπ; the pole progressions have DIFFERENT rational
    // parts, so they never intersect the solution family (π is irrational).
    assert_eq!(
        solve_display("tan(x + 1)", "tan(2*x)"),
        "{ 1 + k·pi : k ∈ ℤ }"
    );
    assert_eq!(
        solve_display("tan(x + 1/2)", "tan(3*x)"),
        "{ 1/4 + k·1/2·pi : k ∈ ℤ }"
    );
    // Same slope, distinct rational offsets: Δb ≡ 0 (mod π) is impossible.
    assert_eq!(solve_display("tan(x)", "tan(x + 1)"), "No solution");
}

#[test]
fn single_tan_owners_are_unchanged() {
    assert_eq!(solve_display("tan(x)", "1"), "{ 1/4·pi + k·pi : k ∈ ℤ }");
    assert_eq!(solve_display("tan(x)", "0"), "{ k·pi : k ∈ ℤ }");
    assert_eq!(
        solve_display("tan(2*x)", "1"),
        "{ 1/8·pi + k·1/2·pi : k ∈ ℤ }"
    );
    // Non-affine / transcendental RHS keeps the honest residual.
    let out = solve_display("tan(x)", "x");
    assert!(out.contains("solve("), "{out}");
}
