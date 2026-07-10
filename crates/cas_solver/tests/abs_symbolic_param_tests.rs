//! Contract tests for `|affine(x)| {op} c` with an UNDECIDABLE-sign parameter
//! (2026-07-11). The unconditional split assumed `c >= 0` for equations
//! (`abs(x) = a` -> the spurious-for-negative `{a, -a}`) while the inequality
//! paths assumed `c < 0` (`> a` -> AllReals, `< a` -> "No solution", `<= a` ->
//! the degenerate `[a,a] U [-a,-a]`). Family F2 of
//! docs/AUDITORIA_FRONTERA_2026-07-09.md. The forms are built DIRECTLY (the set
//! algebra cannot order symbolic endpoints): two-ray unions are universally
//! correct for `>` / `>=`; `=`, `<`, `<=` carry the sign guard as the
//! established single-case Conditional.

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
fn symbolic_threshold_equation_and_bounded_forms_carry_the_sign_guard() {
    assert_eq!(
        solve_display("abs(x)", RelOp::Eq, "a"),
        "if a >= 0: { a, -a }"
    );
    assert_eq!(solve_display("abs(x)", RelOp::Lt, "a"), "if a > 0: (-a, a)");
    assert_eq!(
        solve_display("abs(x)", RelOp::Leq, "a"),
        "if a >= 0: [-a, a]"
    );
    assert_eq!(
        solve_display("abs(x - 1)", RelOp::Eq, "a"),
        "if a >= 0: { a + 1, 1 - a }"
    );
    assert_eq!(
        solve_display("abs(x)", RelOp::Eq, "y"),
        "if y >= 0: { y, -y }"
    );
}

#[test]
fn symbolic_threshold_ray_unions_are_universal() {
    // For c < 0 the rays overlap and cover the line: correct for EVERY real a.
    assert_eq!(
        solve_display("abs(x)", RelOp::Gt, "a"),
        "(-infinity, -a) U (a, infinity)"
    );
    assert_eq!(
        solve_display("abs(x)", RelOp::Geq, "a"),
        "(-infinity, -a] U [a, infinity)"
    );
}

#[test]
fn decidable_parameters_keep_their_owners() {
    // Structurally provable positive: unconditional split stays.
    assert_eq!(
        solve_display("abs(x)", RelOp::Eq, "a^2 + 1"),
        "{ a^2 + 1, -a^2 - 1 }"
    );
    // Rational thresholds: historical routes.
    assert_eq!(solve_display("abs(x)", RelOp::Eq, "2"), "{ 2, -2 }");
    assert_eq!(solve_display("abs(x)", RelOp::Lt, "2"), "(-2, 2)");
    // Provably negative transcendental: the const-sign owner.
    assert_eq!(solve_display("abs(x)", RelOp::Eq, "ln(1/2)"), "No solution");
    assert_eq!(solve_display("abs(x)", RelOp::Eq, "0"), "{ 0 }");
}
