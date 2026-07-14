//! Contract tests for high-degree polynomials whose rational-root deflation
//! leaves a degree>=3 residual factor (2026-07-13b, family F4). The strategy
//! used to emit the deflated rational roots as a CONCRETE set, silently
//! dropping every real root of the residual (`solve(x^5-5x^3+5x-1=0)` -> `{1}`,
//! dropping 4 real roots). Now:
//! - a BIQUADRATIC residual is solved exactly (z = x^2, Vieta signs, ±√z),
//! - any other residual is counted with an EXACT Sturm chain: count 0 keeps the
//!   deflated set (it IS complete: `x^5-1` -> `{1}`), count > 0 declines to the
//!   honest operator-preserving echo.

use cas_ast::{Equation, RelOp};
use cas_parser::parse;
use cas_solver::api::solve;
use cas_solver::command_api::solve::display_solution_set;
use cas_solver::runtime::Simplifier;

fn solve_display(lhs: &str) -> String {
    let mut simplifier = Simplifier::with_default_rules();
    let lhs = parse(lhs, &mut simplifier.context).expect("parse lhs");
    let rhs = parse("0", &mut simplifier.context).expect("parse rhs");
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let (set, _) = solve(&eq, "x", &mut simplifier).expect("solve");
    display_solution_set(&simplifier.context, &set)
}

#[test]
fn biquadratic_residual_yields_all_five_real_roots() {
    // (x-1)(x^4-4x^2+1): 1 rational + 4 surd roots, all real (verified by
    // substitution and against sympy).
    let out = solve_display("x^5 - x^4 - 4*x^3 + 4*x^2 + x - 1");
    for want in ["1", "2 - sqrt(3)", "sqrt(1/2) + sqrt(3/2)"] {
        assert!(out.contains(want), "missing {want} in {out}");
    }
    // Five roots: the set has exactly 4 commas.
    assert_eq!(out.matches(',').count(), 4, "{out}");
}

#[test]
fn real_rooted_nonbiquadratic_residual_declines_honestly() {
    // Chebyshev-type: (x-1)(x^4+x^3-4x^2-4x+1), the quartic has 4 real roots and
    // no exact solver -> honest echo, NOT the silently-incomplete {1}.
    for poly in [
        "x^5 - 5*x^3 + 5*x - 1",
        "x^5 - 5*x^3 + 5*x + 1",
        "x^7 - 7*x^5 + 14*x^3 - 7*x - 1",
    ] {
        let out = solve_display(poly);
        assert!(
            out.contains("solve(") && !out.contains("{"),
            "{poly} must echo, got {out}"
        );
    }
}

#[test]
fn sturm_zero_residual_keeps_the_complete_deflated_set() {
    // x^5-1 = (x-1)(x^4+x^3+x^2+x+1): the quartic has NO real roots, so {1} is
    // the complete real solution set and must stay concrete.
    assert_eq!(solve_display("x^5 - 1"), "{ 1 }");
    // Existing owners unchanged.
    assert_eq!(solve_display("x^3 - 6*x^2 + 11*x - 6"), "{ 1, 2, 3 }");
    assert_eq!(solve_display("x^2 - 3*x - 4"), "{ -1, 4 }");
    let biq = solve_display("x^4 - 8*x^2 + 15");
    assert!(biq.contains("sqrt(3)") && biq.contains("sqrt(5)"), "{biq}");
}
